import os
import asyncio
import base64
import shutil
import json
import faiss
from pathlib import Path
from typing import List, Sequence, Set, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm
from io import BytesIO
import yaml
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel
from dotenv import load_dotenv

from prompt.prompt import batch_combine_prompts, location_prompt, verification_prompt
from prompt.preprocess.keyframe_extract import extract_keyframes
from prompt.preprocess.video_transcribe import transcribe_video_directory
from prompt.search.image_search import image_search_directory
from prompt.search.index_search import search_index_directory, save_results_to_json
from prompt.search.text_search import text_search_image, text_search_link
from prompt.fetch.satellite_fetch import fetch_satellite_image
from prompt.fetch.content_fetch import fetch_links_to_json

from utils import get_gps_from_location, calculate_similarity_scores
from g3.G3 import G3

load_dotenv()

# Pydantic models for structured output
class Evidence(BaseModel):
    analysis: str
    references: Optional[List[str]] = []

class LocationPrediction(BaseModel):
    latitude: float
    longitude: float
    location: str
    evidence: List[Evidence]

class GPSPrediction(BaseModel):
    latitude: float
    longitude: float
    analysis: Optional[str] = ""
    references: Optional[List[str]] = []

class G3BatchPredictor:
    """
    Batch prediction class for processing all images and videos in a directory.

    This class:
    1. Preprocesses all images and videos in a directory.
    2. Extracts keyframes from videos and combines them with images.
    3. Passes all keyframes and images to the Gemini model for prediction.
    """

    def __init__(
        self, 
        device: str = "cuda", 
        input_dir: str = "g3/data/input_data",
        prompt_dir: str = "g3/data/prompt_data",
        index_path: str = "g3/index/G3.index",
        checkpoint_path: str = "g3/checkpoints/mercator_finetune_weight.pth"
    ):
        """
        Initialize the BatchKeyframePredictor.

        Args:
            checkpoint_path (str): Path to G3 model checkpoint
            device (str): Device to run model on ("cuda" or "cpu")
            index_path (str): Path to FAISS index for RAG (required)
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        self.input_dir = Path(input_dir) 
        self.prompt_dir = Path(prompt_dir) 
        self.image_dir = self.prompt_dir / "images"
        self.audio_dir = self.prompt_dir / "audio"

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.prompt_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

        # Initialize G3 model
        base_path = Path(__file__).parent
        hparams = yaml.safe_load(open(base_path / "hparams.yaml", "r"))
        pe = "projection_mercator"
        nn = "rffmlp"

        self.model = G3(
            device=device,
            positional_encoding_type=pe,
            neural_network_type=nn,
            hparams=hparams[f"{pe}_{nn}"],
        )
        self.__load_checkpoint__()

        # Load FAISS index for RAG (required)
        try:
            self.index = faiss.read_index(index_path)
            print(f"✅ Successfully loaded FAISS index from: {index_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index from {index_path}: {e}")

        # Get API key
        self.GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
        self.GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX")
        self.IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
        self.SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY")

        self.image_extension = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        self.video_extension = {'.mp4', '.avi', '.mov', '.mkv'}

    def __load_checkpoint__(self):
        """
        Load the G3 model checkpoint.
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Successfully loaded G3 model checkpoint from: {self.checkpoint_path}")
    
    def __extract_keyframes__(self):
        """
        Extract keyframes from all videos in the input directory.
        Put all images and keyframes into the prompt directory.
        """
        output_dir = self.image_dir

        # Determine starting index based on existing files
        current_files = list(output_dir.glob("image_*.*"))
        idx = len(current_files)

        # Process images
        for file_name in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(tuple(self.image_extension)):
                out_path = output_dir / f"image_{idx:03d}.jpg"
                Image.open(file_path).convert("RGB").save(out_path)
                idx += 1

        # Process videos
        for file_name in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(tuple(self.video_extension)):
                if idx is None:
                    idx = 0
                idx = extract_keyframes(file_path, images_dir=str(output_dir), start_index=idx)
        print(f"✅ Extracted keyframes and images to: {output_dir}")

    def __transcribe_videos__(self):
        """
        Transcribe all videos in the input directory.
        Save transcripts into the prompt directory.
        """
        audio_dir = self.audio_dir

        if audio_dir.is_dir() and any(audio_dir.iterdir()):
            print(f"🔄 Found existing transcripts in directory: {audio_dir}")
            return

        transcribe_video_directory(
            video_dir=str(self.input_dir),
            output_dir=str(audio_dir),
            model_name="base"  # Use the base Whisper model for transcription
        )
        print(f"✅ Successfully transcribed videos to: {audio_dir}")

    def __image_search__(self):
        """
        Perform image search on all images in the input directory.
        Save search results into the prompt directory.
        """
        image_dir = self.image_dir

        if self.IMGBB_API_KEY is None:
            raise ValueError("IMGBB_API_KEY environment variable is not set or is None.")
        if self.SCRAPINGDOG_API_KEY is None:
            raise ValueError("SCRAPINGDOG_API_KEY environment variable is not set or is None.")
        image_search_directory(
            directory=str(image_dir),
            output_dir=str(self.prompt_dir),
            filename="image_search.json",
            imgbb_key=self.IMGBB_API_KEY,
            scrapingdog_key=self.SCRAPINGDOG_API_KEY
        )   
        print(f"✅ Successfully performed image search on: {image_dir}")

    def __text_search__(self):
        """
        Perform text search with metadata to get related links.
        """
        query = ""
        metadata_file = self.prompt_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            description = metadata.get("description", "")
            location = metadata.get("location", "")
            query = f"{description} in {location}".strip()
        
        text_search_link(
            query=query,
            output_dir=str(self.prompt_dir),
            filename="text_search.json",
            num_results=10,
            api_key=self.GOOGLE_CLOUD_API_KEY,
            cx=self.GOOGLE_CSE_CX
        )

    async def __fetch_related_link_content__(self, image_prediction: bool = True, text_prediction: bool = True):
        """
        Fetch related link content for all images in the prompt directory.
        """
        # Fetch image search results
        image_links = set()
        image_search_file = self.prompt_dir / "image_search.json"
        if image_prediction:
            if not image_search_file.exists():
                self.__image_search__()
            
            with open(image_search_file, 'r') as f:
                image_search_data = json.load(f)
                for item in image_search_data:
                    vision_links = item.get("vision_result", [])
                    scrapingdog_links = item.get("scrapingdog_result", [])
                    for link in vision_links:
                        image_links.add(link)
                    for link in scrapingdog_links:
                        image_links.add(link)
            print(f"Found {len(image_links)} image links to fetch content from.")
            if image_links:
                # Fetch content for each link
                await fetch_links_to_json(
                    links=list(image_links),
                    output_path=str(self.prompt_dir / "image_search_content.json"),
                    max_content_length=5000,  # Limit content to 5000 characters per link
                )

        # Fetch text search results
        text_links = set()
        text_search_file = self.prompt_dir / "text_search.json"
        if text_prediction:
            if not text_search_file.exists():
                self.__text_search__()
            
            with open(text_search_file, 'r') as f:
                text_search_data = json.load(f)
                text_links_list = text_search_data.get("links", [])  # Get as list from JSON
                for link in text_links_list:  # Iterate through the list
                    if link:
                        text_links.add(link)  # Add to the set
            print(f"Found {len(text_links)} text links to fetch content from.")
            if text_links:
                # Fetch content for each link
                await fetch_links_to_json(
                    links=list(text_links),
                    output_path=str(self.prompt_dir / "text_search_content.json"),
                    max_content_length=5000,  # Limit content to 5000 characters per link
                )

        if not image_links and not text_links:
            print("No links found in image search results.")
            return

    def __index_search__(self):
        """
        Perform FAISS index search on all images in the prompt directory.
        Save search results into the report directory.
        """
        if not self.index:
            raise RuntimeError("FAISS index is not loaded. Cannot perform index search.")
        
        output_path = self.prompt_dir / "index_search.json"
        if output_path.exists():
            print(f"Index search results already exist at {output_path}, skipping search.")
            return

        database_csv_path = "g3/data/mp16/MP16_Pro_filtered.csv"
        if not os.path.exists(database_csv_path):
            raise FileNotFoundError(f"Database CSV file not found: {database_csv_path}")

        candidates_gps, reverse_gps = search_index_directory(
            model=self.model,
            device=self.device,
            index=self.index,
            image_dir=str(self.image_dir),
            database_csv_path=database_csv_path,
            top_k=20,  # Default top_k value
            max_elements=20  # Default max_elements value
        )

        save_results_to_json(candidates_gps, reverse_gps, str(output_path))
        print(f"✅ Successfully performed index search. Results saved to: {output_path}")

    async def preprocess_input_data(self, image_prediction: bool = True, text_prediction: bool = True):
        """
        Preprocess all input data:
        - Extract keyframes from videos.
        - Transcribe videos.
        - Fetch related link content from images.
        Save images and extracted keyframes into the output directory
        """
        metadata_dest = self.prompt_dir / "metadata.json"
        if not metadata_dest.exists():
            for file in os.listdir(self.input_dir):
                if file.endswith(".json"):
                    file_path = os.path.join(self.input_dir, file)
                    with open(file_path, 'r') as src_file:
                        with open(metadata_dest, 'w') as dest_file:
                            dest_file.write(src_file.read())
                    break

        self.__extract_keyframes__()
        self.__transcribe_videos__()
        await self.__fetch_related_link_content__(
            image_prediction=image_prediction,
            text_prediction=text_prediction
        )
        self.__index_search__()

    async def llm_predict(self, model_name: str = "gemini-2.5-pro", n_search: Optional[int] = None, n_coords: Optional[int] = None, image_prediction: bool = True, text_prediction: bool = True) -> dict:
        """
        Generate a prediction using the Gemini LLM with Pydantic structured output.

        Args:
            model_name: LLM model name to use
            n_search: Number of search results to include
            n_coords: Number of coordinates to include
            image_prediction: Whether to use images in prediction
            text_prediction: Whether to use text in prediction

        Returns:
            dict: Parsed prediction response
        """
        prompt = batch_combine_prompts(
            prompt_dir=str(self.prompt_dir),
            n_coords=n_coords,
            n_search=n_search,
            image_prediction=image_prediction,
            text_prediction=text_prediction
        )

        images = []
        if image_prediction:
            image_dir = self.image_dir
            if not image_dir.exists():
                raise ValueError(f"Image directory does not exist: {image_dir}")

            for image_file in image_dir.glob("*.jpg"):
                with open(image_file, "rb") as f:
                    image = types.Part.from_bytes(
                        data=f.read(),
                        mime_type="image/jpeg"
                    )
                images.append(image)

        client = genai.Client(api_key=self.GOOGLE_CLOUD_API_KEY)

        async def api_call():
            # Run the synchronous API call in a thread executor to make it truly async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=[*images, prompt],
                    config=GenerateContentConfig(
                        tools=[
                            types.Tool(url_context=types.UrlContext()),
                        ],
                        response_mime_type="application/json",
                        response_schema=LocationPrediction,
                        temperature=0.1,
                        top_p=0.95,
                    )
                )
            )

            # Use the parsed response directly
            if response.parsed and isinstance(response.parsed, LocationPrediction):
                return response.parsed.dict()
            else:
                print("⚠️ Failed to get valid structured response, returning empty dict for retry")
                if response.text:
                    print(f"Raw response (first 1000 chars): {response.text[:1000]}")
                return {}

        return await self.handle_async_api_call_with_retry(
            api_call,
            fallback_result={},
            error_context=f"LLM prediction with {model_name}"
        )

    async def diversification_predict(
        self,
        model_name: str = "gemini-2.5-flash",
        image_prediction: bool = True,
        text_prediction: bool = True
    ) -> dict:
        """
        Diversification prediction without preprocessing (assumes preprocessing already done).
        Runs different sample sizes in parallel for faster execution.

        Args:
            model_name (str): LLM model name to use
            image_prediction (bool): Whether to use images in prediction
            text_prediction (bool): Whether to use text in prediction

        Returns:
            dict: Best prediction with latitude, longitude, location, reason, and metadata
        """

        # Function to try a specific sample size with retry logic
        async def try_sample_size(num_sample):
            print(f"Starting prediction with {num_sample} samples...")
            while True:
                prediction = await self.llm_predict(
                    model_name=model_name,
                    n_search=num_sample,
                    n_coords=num_sample,
                    image_prediction=image_prediction,
                    text_prediction=text_prediction
                )

                if prediction:
                    coords = (prediction["latitude"], prediction["longitude"])
                    # print(f"✅ Prediction with {num_sample} samples successful: {coords}")
                    return (num_sample, coords, prediction)
                else:
                    print(f"Invalid or empty prediction format with {num_sample} samples, retrying...")

        # Run all sample sizes in parallel
        num_samples = [10, 15, 20]
        print(f"🚀 Running {len(num_samples)} sample sizes in parallel: {num_samples}")
        
        tasks = [try_sample_size(num_sample) for num_sample in num_samples]
        results = await atqdm.gather(*tasks, desc="🔄 Running parallel predictions")
        
        # Build predictions dictionary from parallel results
        predictions_dict = {}
        for num_sample, coords, prediction in results:
            predictions_dict[coords] = prediction
            print(f"✅ Collected prediction with {num_sample} samples: {coords}")

        # Convert predictions to coordinate list for similarity scoring
        predicted_coords = list(predictions_dict.keys())
        print(f"Predicted coordinates: {predicted_coords}")

        if not predicted_coords:
            raise ValueError("No valid predictions obtained from any sample size")

        # Calculate similarity scores
        avg_similarities = calculate_similarity_scores(
            model=self.model,
            device=self.device,
            predicted_coords=predicted_coords,
            image_dir=self.image_dir
        )

        # Find best prediction
        best_idx = np.argmax(avg_similarities)
        best_coords = predicted_coords[best_idx]
        best_prediction = predictions_dict[best_coords]

        print(f"🎯 Best prediction selected: {best_coords}")
        print(f"   Similarity scores: {avg_similarities}")
        print(f"   Best index: {best_idx}")

        # print(json.dumps(best_prediction, indent=2))  # Commented out verbose output

        return best_prediction

    async def location_predict(
        self,
        model_name: str = "gemini-2.5-flash",
        location: str = "specified location",
    ) -> dict:
        """
        Generate a location-based prediction using the Gemini LLM with centralized retry logic.

        Args:
            model_name (str): LLM model name to use
            location (str): Location to use in the prompt

        Returns:
            dict: Parsed JSON prediction response
        """
        if not location:
            raise ValueError("Location must be specified for location-based prediction")
        
        lat, lon = get_gps_from_location(location)
        if lat is not None or lon is not None:
            print(f"Using GPS coordinates for location '{location}': ({lat}, {lon})")
            return {
                "latitude": lat,
                "longitude": lon,   
            }

        # Create location prompt
        prompt = location_prompt(location)

        client = genai.Client(api_key=self.GOOGLE_CLOUD_API_KEY)

        async def api_call():
            # Run the synchronous API call in a thread executor to make it truly async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=[prompt],
                    config=GenerateContentConfig(
                        tools=[
                            types.Tool(google_search=types.GoogleSearch()),
                        ],
                        response_mime_type="application/json",
                        response_schema=GPSPrediction,
                        temperature=0.1,
                        top_p=0.95,
                    )
                )
            )

            # Use the parsed response directly
            if response.parsed and isinstance(response.parsed, GPSPrediction):
                return response.parsed.dict()
            else:
                print("⚠️ Failed to get valid structured location response, returning empty dict for retry")
                if response.text:
                    print(f"Raw response (first 1000 chars): {response.text[:1000]}")
                return {}

        return await self.handle_async_api_call_with_retry(
            api_call,
            fallback_result={},
            error_context=f"Location prediction for '{location}' with {model_name}"
        )

    async def prepare_verification_data(self, prediction: dict, image_prediction: bool = True, text_prediction: bool = True) -> int:
        """
        Prepare verification data from the prediction with parallel fetching.

        Args:
            prediction (dict): Prediction dictionary with latitude, longitude, location, reason, and metadata
            image_prediction (bool): Whether to include original images in verification
            text_prediction (bool): Whether to include text-based verification

        Returns:
            int: Satellite image ID for reference in prompts
        """
        image_dir = self.image_dir
        satellite_image_id = len(list(self.image_dir.glob("image_*.*")))

        # Run satellite image fetching and text search in parallel
        async def fetch_satellite_async():
            """Async wrapper for satellite image fetching"""
            return await asyncio.get_event_loop().run_in_executor(
                None,
                fetch_satellite_image,
                prediction["latitude"],
                prediction["longitude"],
                200,
                str(image_dir / f"image_{satellite_image_id:03d}.jpg")
            )
        
        async def search_images_async():
            """Async wrapper for text-based image search"""
            return await asyncio.get_event_loop().run_in_executor(
                None,
                text_search_image,
                prediction["location"],
                5,
                self.GOOGLE_CLOUD_API_KEY,
                self.GOOGLE_CSE_CX,
                str(image_dir),
                satellite_image_id + 1
            )
        
        # Execute both operations in parallel
        print(f"🔄 Fetching satellite image and location images in parallel...")
        await asyncio.gather(
            fetch_satellite_async(),
            search_images_async()
        )
        print(f"✅ Verification data preparation completed")
        
        return satellite_image_id

    async def verification_predict(
        self,
        prediction: dict,
        model_name: str = "gemini-2.5-flash",
        image_prediction: bool = True,
        text_prediction: bool = True
    ) -> dict:
        """
        Generate verification prediction based on the provided prediction.

        Args:
            prediction (dict): Prediction dictionary with latitude, longitude, location, reason, and metadata
            model_name (str): LLM model name to use for verification

        Returns:
            dict: Verification prediction with latitude, longitude, location, reason, and evidence
        """
        if not is_valid_enhanced_gps_dict(prediction):
            raise ValueError("Invalid prediction format for verification")
        
        # Prepare verification data (now async)
        satellite_image_id = await self.prepare_verification_data(
            prediction=prediction, 
            image_prediction=image_prediction, 
            text_prediction=text_prediction
        )

        image_dir = self.image_dir

        images = []
        if image_prediction:
            if not image_dir.exists():
                raise ValueError(f"Image directory does not exist: {image_dir}")

            for image_file in image_dir.glob("*.jpg"):
                with open(image_file, "rb") as f:
                    image = types.Part.from_bytes(
                        data=f.read(),
                        mime_type="image/jpeg"
                    )
                images.append(image)

        # Prepare verification prompt
        prompt = verification_prompt(
            satellite_image_id=satellite_image_id, 
            prediction=prediction, 
            prompt_dir=str(self.prompt_dir),
            image_prediction=image_prediction,
            text_prediction=text_prediction
        )

        client = genai.Client(api_key=self.GOOGLE_CLOUD_API_KEY)

        async def api_call():
            # Run the synchronous API call in a thread executor to make it truly async  
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=[*images, prompt],
                    config=GenerateContentConfig(
                        tools=[
                            types.Tool(url_context=types.UrlContext()),
                        ],
                        response_mime_type="application/json",
                        response_schema=LocationPrediction,
                        temperature=0.1,
                        top_p=0.95,
                    )
                )
            )
            
            # Use the parsed response directly
            if response.parsed and isinstance(response.parsed, LocationPrediction):
                print("✅ Verification prediction successful")
                return response.parsed.dict()
            else:
                print("⚠️ Invalid or empty verification response format, retrying...")
                if response.text:
                    print(f"Raw response (first 1000 chars): {response.text[:1000]}")
                return {}  # Return empty dict to trigger retry

        return await self.handle_async_api_call_with_retry(
            api_call,
            fallback_result={},
            error_context=f"Verification prediction with {model_name}"
        )
    
    async def predict(
        self,
        model_name: str = "gemini-2.5-flash",
        image_prediction: bool = True,
        text_prediction: bool = True
    ) -> dict:
        """
        Complete prediction pipeline without preprocessing (assumes preprocessing already done).
        Used for parallel execution where preprocessing is done once beforehand.
        All major steps run in parallel for maximum speed.

        Args:
            model_name (str): LLM model name to use
            image_prediction (bool): Whether to use images in prediction
            text_prediction (bool): Whether to use text in prediction

        Returns:
            dict: Final prediction with latitude, longitude, location, reason, and evidence
        """
        
        # Step 1: Run diversification prediction (this is already parallel internally)
        print(f"\n🔄 Running diversification prediction for Image={image_prediction}, Text={text_prediction}...")
        diversification_result = await self.diversification_predict(
            model_name=model_name,
            image_prediction=image_prediction,
            text_prediction=text_prediction
        )

        # Step 2: Run location prediction in parallel with preparing for verification
        async def run_location_prediction():
            """Run location prediction with retry logic"""
            while True:
                location_prediction = await self.location_predict(
                    model_name=model_name,
                    location=diversification_result.get("location", "specified location")
                )
                if location_prediction:  # Check if we got a valid response
                    return location_prediction
                else:
                    print("Location prediction returned empty, retrying...")

        async def prepare_for_verification():
            """Prepare verification data in parallel"""
            # Create a copy of diversification result for verification
            verification_input = diversification_result.copy()
            return verification_input

        # Run location prediction and verification preparation in parallel
        print(f"\n🔄 Running location prediction and verification prep in parallel...")
        location_prediction, verification_input = await asyncio.gather(
            run_location_prediction(),
            prepare_for_verification()
        )

        print("✅ Location prediction completed:")
        # print(json.dumps(location_prediction, indent=2))  # Commented out verbose output

        # Step 3: Update coordinates and evidence from location prediction
        result = verification_input.copy()
        result["longitude"] = location_prediction.get("longitude", result.get("longitude"))
        result["latitude"] = location_prediction.get("latitude", result.get("latitude"))

        # Step 4: Normalize and append location evidence
        if "analysis" in location_prediction and "references" in location_prediction:
            location_evidence = [{
                "analysis": location_prediction["analysis"],
                "references": location_prediction["references"]
            }]
        else:
            location_evidence = location_prediction.get("evidence", [])

        # Append to result evidence
        result.setdefault("evidence", []).extend(location_evidence)

        # Step 5: Run verification prediction
        print(f"\n🔄 Running verification prediction for Image={image_prediction}, Text={text_prediction}...")
        result = await self.verification_predict(
            prediction=result,
            model_name=model_name,
            image_prediction=image_prediction,
            text_prediction=text_prediction
        )

        print(f"\n🎯 Final prediction for Image={image_prediction}, Text={text_prediction}:")
        # print(json.dumps(result, indent=2))  # Commented out verbose output

        return result

    def is_retryable_error(self, error: Exception) -> bool:
        """
        Check if an error is retryable (server errors, connection issues, etc.)
        
        Args:
            error (Exception): The exception to check
            
        Returns:
            bool: True if the error should be retried, False otherwise
        """
        error_str = str(error).lower()
        
        # Check for various retryable error patterns
        retryable_patterns = [
            # Server errors
            "503", "500", "502", "504",
            "overloaded", "unavailable", "internal",
            
            # Connection errors  
            "disconnected", "connection", "timeout",
            "remoteprotocolerror", "remote protocol error",
            
            # Network errors
            "network", "socket", "ssl", "tls",
            
            # Rate limiting
            "rate limit", "too many requests", "429",
            
            # Service unavailable
            "service unavailable", "temporarily unavailable"
        ]
        
        # Check if any retryable pattern is found in the error string
        for pattern in retryable_patterns:
            if pattern in error_str:
                return True
                
        # Additional checks for specific error types
        error_type = type(error).__name__.lower()
        retryable_error_types = [
            "connectionerror", "timeout", "httperror", 
            "remoteclosederror", "remoteprotocolerror",
            "sslerror", "tlserror"
        ]
        
        return error_type in retryable_error_types

    async def handle_async_api_call_with_retry(
        self, 
        api_call_func, 
        max_retries: int = 10,
        base_delay: float = 2.0,
        fallback_result: Optional[dict] = None,
        error_context: str = "API call"
    ) -> dict:
        """
        Centralized async API call with retry logic for Google API errors.
        
        Args:
            api_call_func: Async function to call (should return the API response)
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            fallback_result: Result to return if all retries fail
            error_context: Description of the operation for logging
            
        Returns:
            API response or fallback_result if all attempts fail
        """
        for attempt in range(max_retries):
            try:
                return await api_call_func()
                
            except Exception as e:
                error_str = str(e).lower()
                print(f"{error_context} error (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Check if error is retryable using our centralized function
                if self.is_retryable_error(e):
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"🔄 Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        print(f"❌ Max retries ({max_retries}) exceeded. Giving up.")
                        break
                else:
                    print(f"❌ Non-retryable error: {e}")
                    break
        
        # All attempts failed
        if fallback_result is not None:
            print(f"⚠️ Returning fallback result for {error_context}")
            return fallback_result
        else:
            print(f"❌ No fallback available for {error_context}")
            return {}

if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "acmmm2025-grand-challenge-gg-credentials.json"
    import argparse

    parser = argparse.ArgumentParser(description="G3 Batch Predictor Test")
    parser.add_argument(
        "--sample_id",
        type=str,
        default="ID243",
        help="Sample ID for data organization"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="g3/checkpoints/mercator_finetune_weight.pth",
        help="Path to G3 model checkpoint"
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="g3/index/G3.index",
        help="Path to FAISS index for RAG"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-pro",
        help="LLM model name to use"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="diversification",
        choices=["diversification"],
        help="Prediction mode: diversification (multiple predictions from 3 modalities with similarity scoring)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="batch_prediction_result.json",
        help="Output file to save prediction result"
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Generate comprehensive markdown and JSON reports after prediction"
    )

    args = parser.parse_args()

    async def main():
        try:
            print(f"🚀 Starting G3 Batch Predictor in {args.mode} mode...")
            print(f"🎯 Model: {args.model_name}")
            
            # Initialize predictor
            predictor = G3BatchPredictor(
                device="cuda" if torch.cuda.is_available() else "cpu",
                checkpoint_path=args.checkpoint_path,
                index_path=args.index_path
            )

            # Run prediction - always use the predict method for 3 modalities
            print("\n🔄 Running complete multi-modal prediction pipeline...")
            result = await predictor.predict(model_name=args.model_name)

            print(json.dumps(result, indent=2))
            print("\n🎉 Multi-modal prediction completed successfully!")

        except Exception as e:
            print(f"❌ Multi-modal prediction failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)

    # Run the async main function
    asyncio.run(main())

