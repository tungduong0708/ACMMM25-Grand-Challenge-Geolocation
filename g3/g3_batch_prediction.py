import os
import asyncio
import base64
import shutil
import json
import faiss
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Union
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
from dotenv import load_dotenv
from utils.extract_keyframe import extract_keyframes
from utils.prompt import batch_combine_prompts, location_prompt, verification_prompt, ranking_prompt
from utils.video_transcribe import transcribe_video_directory
from utils.image_search import image_search_directory
from utils.index_search import search_index_directory, save_results_to_json
from utils.utils import get_gps_from_location
from utils.fetch_satellite import fetch_satellite_image
from utils.fetch_content import fetch_links_to_json
from utils.text_search import text_search_image, text_search_link

# Import required utilities
from utils.G3 import G3

load_dotenv()

def extract_and_parse_json(raw_text: str) -> Union[dict, list]:
    """
    Extract JSON content between first { and last } or first [ and last ] and parse it.
    Returns empty dict if parsing fails to allow retry logic in calling methods.
    
    Args:
        raw_text (str): Raw response text from LLM
        
    Returns:
        Union[dict, list]: Parsed JSON data, or empty dict if parsing fails
    """
    # Try to find JSON object first (between { and })
    first_brace = raw_text.find('{')
    last_brace = raw_text.rfind('}')
    
    # Try to find JSON array (between [ and ])
    first_bracket = raw_text.find('[')
    last_bracket = raw_text.rfind(']')
    
    json_str = None
    
    # Determine which JSON format to use based on what appears first
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        # Use JSON object format
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_str = raw_text[first_brace:last_brace + 1]
    elif first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
        # Use JSON array format
        json_str = raw_text[first_bracket:last_bracket + 1]
    
    if not json_str:
        print(f"⚠️ No valid JSON braces/brackets found in response. Raw text (first 500 chars): {raw_text[:500]}...")
        return {}
    
    try:
        # Parse JSON
        parsed_data = json.loads(json_str)
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse JSON: {e}. Raw text (first 500 chars): {raw_text[:500]}...")
        return {}
    
def is_valid_enhanced_gps_dict(gps_data):
        """
        Check if GPS data dict has valid enhanced format with latitude, longitude, location, and evidence.
        """
        if not isinstance(gps_data, dict):
            return False
            
        required_fields = ["latitude", "longitude", "location", "evidence"]
        if not all(field in gps_data for field in required_fields):
            return False
            
        # Check if evidence is a list
        if not isinstance(gps_data["evidence"], list):
            return False
        
        # Check if evidence contains valid analysis and links
        for item in gps_data["evidence"]:
            if not isinstance(item, dict):
                return False
            if "analysis" not in item:
                return False
                
        try:
            float(gps_data["latitude"])
            float(gps_data["longitude"])
        except (ValueError, TypeError):
            return False
            
        return True

def is_valid_ranking_prediction_dict(ranking_data):
    """
    Check if ranking prediction data dict has valid format with best_prediction and all_predictions.
    """
    if not isinstance(ranking_data, dict):
        return False
        
    # Check for required top-level fields
    required_fields = ["best_prediction", "all_predictions"]
    if not all(field in ranking_data for field in required_fields):
        return False
    
    # Validate best_prediction structure
    best_pred = ranking_data["best_prediction"]
    if not isinstance(best_pred, dict):
        return False
    
    best_pred_required = ["latitude", "longitude", "location", "score"]
    if not all(field in best_pred for field in best_pred_required):
        return False
    
    try:
        float(best_pred["latitude"])
        float(best_pred["longitude"])
        int(best_pred["score"])
    except (ValueError, TypeError):
        return False
    
    # Validate all_predictions structure
    all_preds = ranking_data["all_predictions"]
    if not isinstance(all_preds, list):
        return False
    
    for pred in all_preds:
        if not isinstance(pred, dict):
            return False
        
        # Check required fields for each prediction
        pred_required = ["image_prediction", "text_prediction", "score", "breakdown", "prediction"]
        if not all(field in pred for field in pred_required):
            return False
        
        # Validate modality booleans and score
        try:
            bool(pred["image_prediction"])
            bool(pred["text_prediction"])
            int(pred["score"])
        except (ValueError, TypeError):
            return False
        
        # Validate breakdown structure
        breakdown = pred["breakdown"]
        if not isinstance(breakdown, dict):
            return False
        
        breakdown_required = ["location_plausibility", "location_name_specificity", "evidence_quality", "modality_use", "justification_clarity"]
        if not all(field in breakdown for field in breakdown_required):
            return False
        
        try:
            for field in breakdown_required:
                int(breakdown[field])
        except (ValueError, TypeError):
            return False
        
        # Validate prediction structure (reuse existing function)
        if not is_valid_enhanced_gps_dict(pred["prediction"]):
            return False
    
    return True

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
        sample_id: str = "sample_001",
        device: str = "cuda", 
        input_dir: str = "g3/data/input_data",
        prompt_dir: str = "g3/data/prompt_data",
        report_dir: str = "g3/data/report_data", 
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
        self.sample_id = sample_id
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        self.input_dir = Path(input_dir) / self.sample_id
        self.prompt_dir = Path(prompt_dir) / self.sample_id
        self.report_dir = Path(report_dir) / self.sample_id

        self.sample_image_dir = self.prompt_dir / "sample_images"
        self.image_location_image_dir = self.prompt_dir / "image_location_images"
        self.text_location_image_dir = self.prompt_dir / "text_location_images"
        self.image_text_location_image_dir = self.prompt_dir / "image_text_location_images"

        self.audio_dir = self.prompt_dir / "audio"

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.prompt_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.sample_image_dir, exist_ok=True)
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
        output_dir = self.sample_image_dir
        os.makedirs(output_dir, exist_ok=True)

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
        os.makedirs(audio_dir, exist_ok=True)

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
        image_dir = self.sample_image_dir
        os.makedirs(image_dir, exist_ok=True)

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
            image_dir=str(self.sample_image_dir),
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

    def calculate_similarity_scores(
        self,
        predicted_coords: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Calculate similarity scores between images and predicted coordinates.

        Args:
            rgb_images: List of PIL Images
            predicted_coords: List of (lat, lon) tuples

        Returns:
            np.ndarray: Average similarity scores across all images for each coordinate
        """
        all_similarities = []

        image_dir = self.sample_image_dir
        if not image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")

        for image_file in image_dir.glob("*.jpg"):
            # Load image as PIL Image first
            pil_image = Image.open(image_file).convert("RGB")
            
            # Process the PIL image
            image = self.model.vision_processor(images=pil_image, return_tensors="pt")[
                "pixel_values"
            ].reshape(-1, 224, 224)
            image = image.unsqueeze(0).to(self.device)

            with torch.no_grad():
                vision_output = self.model.vision_model(image)[1]

                image_embeds = self.model.vision_projection_else_2(
                    self.model.vision_projection(vision_output)
                )
                image_embeds = image_embeds / image_embeds.norm(
                    p=2, dim=-1, keepdim=True
                )  # b, 768

                # Process coordinates
                gps_batch = torch.tensor(predicted_coords, dtype=torch.float32).to(self.device)
                gps_input = gps_batch.clone().detach().unsqueeze(0)  # Add batch dimension
                b, c, _ = gps_input.shape
                gps_input = gps_input.reshape(b * c, 2)
                location_embeds = self.model.location_encoder(gps_input)
                location_embeds = self.model.location_projection_else(
                    location_embeds.reshape(b * c, -1)
                )
                location_embeds = location_embeds / location_embeds.norm(
                    p=2, dim=-1, keepdim=True
                )
                location_embeds = location_embeds.reshape(b, c, -1)  # b, c, 768

                similarity = torch.matmul(
                    image_embeds.unsqueeze(1), location_embeds.permute(0, 2, 1)
                )  # b, 1, c
                similarity = similarity.squeeze(1).cpu().detach().numpy()
                all_similarities.append(similarity[0])  # Remove batch dimension

        # Calculate average similarity across all images
        avg_similarities = np.mean(all_similarities, axis=0)
        return avg_similarities

    async def llm_predict(self, model_name: str = "gemini-2.5-pro", n_search: Optional[int] = None, n_coords: Optional[int] = None, image_prediction: bool = True, text_prediction: bool = True) -> dict:
        """
        Generate a prediction using the Gemini LLM with centralized retry logic.

        Args:
            model_name: LLM model name to use
            n_search: Number of search results to include
            n_coords: Number of coordinates to include
            image_prediction: Whether to use images in prediction
            text_prediction: Whether to use text in prediction

        Returns:
            dict: Parsed JSON prediction response
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
            image_dir = self.sample_image_dir
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

        tools = [
            types.Tool(url_context=types.UrlContext())
        ]

        config = types.GenerateContentConfig(
            tools=tools,
            response_modalities=["TEXT"],
            temperature=0.1,
            top_p=0.95,
        )

        async def api_call():
            # Run the synchronous API call in a thread executor to make it truly async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=[*images, prompt],
                    config=config
                )
            )

            raw_text = response.text.strip() if response.text is not None else ""

            # Extract and parse JSON from response
            parsed_json = extract_and_parse_json(raw_text)
            if not parsed_json or not isinstance(parsed_json, dict):  # Empty or wrong type means parsing failed
                print(f"⚠️ Failed to parse LLM response, returning empty dict for retry")
                print(f"Raw response (first 1000 chars): {raw_text[:1000]}")
                return {}
            
            return parsed_json

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

                # print(f"Sample {num_sample} result:")
                # print(json.dumps(prediction, indent=2))  # Commented out verbose output

                if prediction and is_valid_enhanced_gps_dict(prediction):
                    coords = (prediction["latitude"], prediction["longitude"])
                    print(f"✅ Prediction with {num_sample} samples successful: {coords}")
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
        avg_similarities = self.calculate_similarity_scores(predicted_coords=predicted_coords)

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

        tools = [
            types.Tool(google_search=types.GoogleSearch())
        ]

        config = types.GenerateContentConfig(
            tools=tools,
            response_modalities=["TEXT"],
            temperature=0.1,
            top_p=0.95,
        )

        async def api_call():
            # Run the synchronous API call in a thread executor to make it truly async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=[prompt],
                    config=config
                )
            )

            raw_text = response.text.strip() if response.text is not None else ""

            # Extract and parse JSON from response
            parsed_json = extract_and_parse_json(raw_text)
            if not parsed_json or not isinstance(parsed_json, dict):  # Empty or wrong type means parsing failed
                print(f"⚠️ Failed to parse location response, returning empty dict for retry")
                print(f"Raw response (first 1000 chars): {raw_text[:1000]}")
                return {}
            
            return parsed_json

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
        if not is_valid_enhanced_gps_dict(prediction):
            raise ValueError("Invalid prediction format for verification")
        
        if image_prediction and text_prediction:
            image_dir = self.image_text_location_image_dir
            os.makedirs(image_dir, exist_ok=True)
            satellite_image_id = len(list(self.sample_image_dir.glob("image_*.*")))
        elif image_prediction:
            image_dir = self.image_location_image_dir
            os.makedirs(image_dir, exist_ok=True)
            satellite_image_id = len(list(self.sample_image_dir.glob("image_*.*")))
        else:
            image_dir = self.text_location_image_dir
            os.makedirs(image_dir, exist_ok=True)
            satellite_image_id = 0
        
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

        sample_image_dir = self.sample_image_dir
        if image_prediction and text_prediction:
            location_image_dir = self.image_text_location_image_dir
        elif image_prediction:
            location_image_dir = self.image_location_image_dir
        else:
            location_image_dir = self.text_location_image_dir

        if not sample_image_dir.exists():
            raise ValueError(f"Image directory does not exist: {sample_image_dir}")
        if not location_image_dir.exists():
            raise ValueError(f"Location image directory does not exist: {location_image_dir}")

        images = []
        if image_prediction:
            if not sample_image_dir.exists():
                raise ValueError(f"Image directory does not exist: {sample_image_dir}")

            for image_file in sample_image_dir.glob("*.jpg"):
                with open(image_file, "rb") as f:
                    image = types.Part.from_bytes(
                        data=f.read(),
                        mime_type="image/jpeg"
                    )
                images.append(image)

        for file_name in os.listdir(location_image_dir):
            if file_name.lower().endswith(tuple(ext.lower() for ext in self.image_extension)):
                image_file = location_image_dir / file_name
                with open(image_file, "rb") as f:
                    ext = image_file.suffix.lower().lstrip(".")
                    if ext == "jpg":
                        mime_type = "image/jpeg"
                    else:
                        mime_type = f"image/{ext}"
                    
                    image = types.Part.from_bytes(
                        data=f.read(),
                        mime_type=mime_type
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

        tools = [
            # types.Tool(google_search=types.GoogleSearch()),
            types.Tool(url_context=types.UrlContext())
        ]

        config = types.GenerateContentConfig(
            tools=tools,
            response_modalities=["TEXT"],
            temperature=0.1,
            top_p=0.95,
        )

        async def api_call():
            # Run the synchronous API call in a thread executor to make it truly async  
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=[*images, prompt],
                    config=config
                )
            )
            
            raw_text = response.text.strip() if response.text is not None else ""

            # Extract and parse JSON from response
            parsed_json = extract_and_parse_json(raw_text)
            if parsed_json and isinstance(parsed_json, dict) and is_valid_enhanced_gps_dict(parsed_json):
                print("✅ Verification prediction successful")
                return parsed_json
            else:
                print("⚠️ Invalid or empty verification response format, retrying...")
                return {}  # Return empty dict to trigger retry

        return await self.handle_async_api_call_with_retry(
            api_call,
            fallback_result={},
            error_context=f"Verification prediction with {model_name}"
        )
    
    async def single_predict(
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
    
    async def ranking_predict(
        self,
        predictions: List[dict],
        model_name: str = "gemini-2.5-flash",
    ) -> dict:
        """
        Rank all predictions at once based on their confidence scores.

        Args:
            predictions (List[dict]): List of prediction dictionaries to rank
            model_name (str): LLM model name to use

        Returns:
            dict: Dictionary containing best prediction GPS/location and list of all ranked predictions
        """
        if not predictions or len(predictions) == 0:
            raise ValueError("Predictions list is empty")

        prompt = ranking_prompt(
            predictions=predictions,
        )
        images = []
        
        # First add sample images
        if self.sample_image_dir.exists():
            for image_file in sorted(self.sample_image_dir.iterdir()):
                if image_file.is_file() and image_file.suffix.lower() in self.image_extension:
                    with open(image_file, "rb") as f:
                        image = types.Part.from_bytes(
                            data=f.read(),
                            mime_type="image/jpeg"
                        )
                    images.append(image)
        
        # Then add location images from each modality in order
        modality_dirs = [
            self.image_text_location_image_dir,
            self.image_location_image_dir,
            self.text_location_image_dir, 
        ]
        
        for modality_dir in modality_dirs:
            if modality_dir.exists():
                for image_file in sorted(modality_dir.iterdir()):
                    if image_file.is_file() and image_file.suffix.lower() in self.image_extension:
                        with open(image_file, "rb") as f:
                            ext = image_file.suffix.lower().lstrip(".")
                            if ext == "jpg":
                                mime_type = "image/jpeg"
                            else:
                                mime_type = f"image/{ext}"
                            
                            image = types.Part.from_bytes(
                                data=f.read(),
                                mime_type=mime_type
                            )
                        images.append(image)

        tools = [
            types.Tool(url_context=types.UrlContext())
        ]

        config = types.GenerateContentConfig(
            tools=tools,
            response_modalities=["TEXT"],
            temperature=0.1,
            top_p=0.95,
        )

        client = genai.Client(api_key=self.GOOGLE_CLOUD_API_KEY)

        # Retry logic with exponential backoff
        max_retries = 10
        base_delay = 2.0  

        for attempt in range(max_retries):
            print(f"🔄 Sending batch ranking request to LLM (attempt {attempt + 1}/{max_retries})...")
            try:
                # Run the synchronous API call in a thread executor to make it truly async
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model_name,
                        contents=[*images, prompt],
                        config=config
                    )
                )
                
                raw_text = response.text.strip() if response.text is not None else ""

                # Extract and parse JSON from response
                parsed_json = extract_and_parse_json(raw_text)
                if parsed_json and isinstance(parsed_json, dict) and is_valid_ranking_prediction_dict(parsed_json):
                    all_predictions = parsed_json.get("all_predictions", [])
                    if len(all_predictions) == len(predictions):
                        print("✅ Batch ranking prediction successful and validated")
                        return parsed_json
                    else:
                        print(f"Invalid predictions count: expected {len(predictions)}, got {len(all_predictions)}, retrying...")
                        if attempt < max_retries - 1:  # Don't sleep on the last attempt
                            await asyncio.sleep(1)  # Brief pause for invalid response format
                else:
                    print("Invalid or empty ranking response format (failed validation), retrying...")
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        await asyncio.sleep(1)  # Brief pause for invalid response format
                    
            except Exception as e:
                # Check if it's a retryable server error (503 or 500)
                if ("503" in str(e) or "overloaded" in str(e).lower() or "unavailable" in str(e).lower() or
                    "500" in str(e) or "internal" in str(e).lower()):
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        delay = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s, 8s
                        print(f"🔄 Ranking model error (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        print(f"❌ Max retries ({max_retries}) exceeded for ranking server error. Giving up.")
                        raise e
                else:
                    # For non-retryable errors, short delay and retry
                    print(f"⚠️ Ranking request failed: {e}, retrying...")
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        await asyncio.sleep(1)  # Brief pause for other errors
        
        # If we get here, all retries failed
        raise RuntimeError(f"Ranking prediction failed after {max_retries} attempts")



    async def predict(
        self,
        model_name: str = "gemini-2.5-flash"
    ) -> dict:
        """
        Run the complete prediction pipeline with parallel execution of modalities.

        Args:
            model_name (str): LLM model name to use

        Returns:
            dict: Dictionary containing best prediction GPS/location and list of all ranked predictions
        """
        modalities = [(True, True), (True, False), (False, True)] # (image, text) combinations
        
        # Preprocess input data once for all modalities (includes all image and text preprocessing)
        print("🔄 Preprocessing input data for all modalities...")
        await self.preprocess_input_data(image_prediction=True, text_prediction=True)
        print("✅ Preprocessing completed!")
        
        # Create async tasks for all modalities to run in parallel
        async def run_modality_prediction(image_prediction: bool, text_prediction: bool):
            print(f"\n🔄 Starting prediction with modalities: Image={image_prediction}, Text={text_prediction}")
            result = {}
            result["modalities"] = {
                "image_prediction": image_prediction,
                "text_prediction": text_prediction
            }
            
            # Run single predict without preprocessing (already done above)
            result["prediction"] = await self.single_predict(
                model_name=model_name,
                image_prediction=image_prediction,
                text_prediction=text_prediction
            )
            print(f"✅ Completed prediction for modalities: Image={image_prediction}, Text={text_prediction}")
            return result
        
        # Run all modality predictions concurrently
        print("🚀 Running all 3 modality predictions in parallel...")
        tasks = [
            run_modality_prediction(image_pred, text_pred) 
            for image_pred, text_pred in modalities
        ]
        
        # Wait for all predictions to complete
        results = await atqdm.gather(*tasks, desc="🔄 Processing modality predictions")
        
        print("✅ All modality predictions completed!")
        for result in results:
            print(f"📊 Result summary: {result['modalities']} -> {result['prediction'].get('location', 'N/A')}")

        # Rank all predictions at once for better comparative analysis
        print("🔄 Processing all predictions for ranking...")
        ranked_results = await self.ranking_predict(
            predictions=results,
            model_name=model_name
        )
        
        print(f"Best prediction: {ranked_results['best_prediction']}")
        print(f"Total predictions ranked: {len(ranked_results['all_predictions'])}")

        return ranked_results
    
    def save_predictions_to_json(
        self,
        predictions: dict
    ) -> None:
        """
        Save a single prediction result to a JSON file.

        Args:
            prediction (dict): The prediction result to save
        """
        # Create report images directory
        report_images_dir = self.report_dir / "images"
        os.makedirs(report_images_dir, exist_ok=True)
        
        image_mapping = {}
        all_dirs = [
            ("sample", self.sample_image_dir),
            ("image_text", self.image_text_location_image_dir),
            ("image", self.image_location_image_dir),
            ("text", self.text_location_image_dir), 
        ]

        num_sample_images = len(list(self.sample_image_dir.glob("image_*.*"))) - 1
        final_sample_image_name = f"image_{num_sample_images:03d}.jpg"
        sample_images = set()
        for prediction in predictions["all_predictions"]:
            if prediction["image_prediction"] and prediction["text_prediction"]:
                modality_name = "image_text"
            elif prediction["image_prediction"]:
                modality_name = "image"
            else:
                modality_name = "text"

            if modality_name not in image_mapping:
                image_mapping[modality_name] = {} 

            evidence_list = prediction["prediction"].get("evidence", [])
            for evidence in evidence_list:
                reference_list = evidence.get("references", [])
                for i, reference in enumerate(reference_list):
                    if isinstance(reference, str) and reference.startswith("image"):
                        if modality_name == "text":
                            image_mapping[modality_name][reference] = reference 
                        elif reference > final_sample_image_name:
                            image_mapping[modality_name][reference] = reference 
                        else:
                            sample_images.add(reference)
        image_mapping["sample"] = {img: img for img in sample_images}

        image_counter = 0
        for modality_name, source_dir in all_dirs:
            if source_dir.exists():
                old_image_list = set(image_mapping.get(modality_name, {}).keys()) 
                for image_file in sorted(source_dir.iterdir()):
                    image_name = os.path.splitext(image_file.name)[0] + ".jpg"
                    if image_file.is_file() and image_file.suffix.lower() in self.image_extension and image_name in old_image_list:
                        old_image_name = image_name
                        new_image_name = f"image_{image_counter:03d}.jpg"
                        image_mapping[modality_name][old_image_name] = new_image_name

                        dest_path = report_images_dir / new_image_name
                        shutil.copy2(image_file, dest_path)
                        image_counter += 1

        print(f"📸 Copied {image_counter} images to report directory: {report_images_dir}")
        # Update image references in predictions
        for prediction in predictions["all_predictions"]:
            if prediction["image_prediction"] and prediction["text_prediction"]:
                modality_name = "image_text"
            elif prediction["image_prediction"]:
                modality_name = "image"
            else:
                modality_name = "text"

            evidence_list = prediction["prediction"].get("evidence", [])
            for evidence in evidence_list:
                reference_list = evidence.get("references", [])
                for i, reference in enumerate(reference_list):
                    if isinstance(reference, str) and reference.startswith("image"):
                        if modality_name == "text":
                            if reference in image_mapping["sample"]:
                                reference_list[i] = image_mapping["sample"][reference]
                        elif reference > final_sample_image_name:
                            if modality_name in image_mapping and reference in image_mapping[modality_name]:
                                reference_list[i] = image_mapping[modality_name][reference]
                        else:
                            if reference in image_mapping["sample"]:
                                reference_list[i] = image_mapping["sample"][reference]
                        

        # Save JSON report
        json_report_path = self.report_dir / f"{self.sample_id}_prediction_result.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"📄 JSON report saved: {json_report_path}")

    async def generate_markdown_report(
        self,
        predictions: dict,
        model_name: str = "gemini-2.5-flash"
    ) -> str:
        """
        Generate a markdown report from multiple prediction results and copy images to report directory.

        Args:
            predictions (List[dict]): List of prediction dictionaries from different modalities
            model_name (str): LLM model name to use for report generation

        Returns:
            str: Path to the generated markdown report file
        """
        self.save_predictions_to_json(predictions)
        
        # Simple prompt - just display everything in the dict
        prompt = f"""
Generate a comprehensive markdown report analyzing geolocation predictions from multiple modalities.

Prediction data:
{json.dumps(predictions, indent=2)}

Instructions:
- Display all information from the prediction data
- If there are images mentioned, display them using ![Description](./images/image_XX.jpg) format
- If there are links in references, cite them properly
- Create a professional, detailed analysis of all modality predictions
- Include sections for Executive Summary, Modality Analysis, Best Prediction, Evidence Analysis, and References

Generate a beautiful markdown report that comprehensively covers all the prediction data.
"""

        print("🔄 Generating comprehensive markdown report...")
        # print(prompt)

        client = genai.Client(api_key=self.GOOGLE_CLOUD_API_KEY)

        config = types.GenerateContentConfig(
            response_modalities=["TEXT"],
            temperature=0.3,
            top_p=0.9,
        )

        # Retry logic with exponential backoff
        max_retries = 10
        base_delay = 2.0  
        
        for attempt in range(max_retries):
            print(f"🔄 Generating markdown report (attempt {attempt + 1}/{max_retries})...")
            try:
                # Run the synchronous API call in a thread executor to make it truly async
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model_name,
                        contents=[prompt],
                        config=config
                    )
                )
                break  # Exit loop if successful
                
            except Exception as e:
                # Check if it's a retryable server error (503 or 500)
                if ("503" in str(e) or "overloaded" in str(e).lower() or "unavailable" in str(e).lower() or
                    "500" in str(e) or "internal" in str(e).lower()):
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        delay = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s, 8s
                        print(f"🔄 Markdown generation model error (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        print(f"❌ Max retries ({max_retries}) exceeded for markdown generation server error. Giving up.")
                        raise e
                else:
                    # For non-retryable errors, short delay and retry
                    print(f"⚠️ Markdown generation request failed: {e}, retrying...")
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        await asyncio.sleep(1)  # Brief pause for other errors
        else:
            # If we get here, all retries failed
            raise RuntimeError(f"Markdown generation failed after {max_retries} attempts")
        
        raw_text = response.text.strip() if response.text is not None else ""
        
        # Save markdown report
        report_path = self.report_dir / f"{self.sample_id}_geolocation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        
        print(f"📄 Markdown report generated: {report_path}")
        return str(report_path)

    async def generate_comprehensive_report(
        self,
        predictions: dict,
        model_name: str = "gemini-2.5-flash"
    ) -> Dict[str, str]:
        """
        Generate both JSON and markdown reports from multiple prediction results.

        Args:
            predictions (List[dict]): List of prediction dictionaries from different modalities
            model_name (str): LLM model name to use for report generation

        Returns:
            dict: Dictionary with paths to generated report files
        """
        # Generate markdown report
        markdown_report_path = await self.generate_markdown_report(predictions, model_name)
        
        report_paths = {
            "markdown_report": markdown_report_path,
            "images_directory": str(self.report_dir / "images")
        }
        
        print(f"📋 Comprehensive reports generated:")
        print(f"   Markdown: {markdown_report_path}")
        print(f"   Images: {self.report_dir / 'images'}")
        
        return report_paths

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

    # async def handle_llm_request_with_retry(
    #     self,
    #     request_func,
    #     request_name: str,
    #     max_retries: int = 10,
    #     base_delay: float = 2.0,
    #     fallback_result: Optional[dict] = None
    # ) -> dict:
    #     """
    #     Handle LLM requests with centralized retry logic and error handling.
        
    #     Args:
    #         request_func: Async function that makes the LLM request
    #         request_name (str): Name of the request for logging
    #         max_retries (int): Maximum number of retry attempts
    #         base_delay (float): Base delay for exponential backoff
    #         fallback_result (dict): Fallback result if all retries fail (optional)
            
    #     Returns:
    #         dict: LLM response or fallback result if all attempts fail
    #     """
        
    #     for attempt in range(max_retries):
    #         try:
    #             result = await request_func()
    #             if result:  # If we got a valid result
    #                 return result
    #             else:
    #                 print(f"⚠️ Empty {request_name} response on attempt {attempt + 1}")
    #                 if attempt >= max_retries - 1 and fallback_result:
    #                     print(f"⚠️ Creating fallback {request_name} result")
    #                     return fallback_result
    #                 elif attempt < max_retries - 1:
    #                     await asyncio.sleep(1)  # Brief pause for empty responses
                        
    #         except Exception as e:
    #             if self.is_retryable_error(e):
    #                 if attempt < max_retries - 1:  # Don't sleep on the last attempt
    #                     delay = base_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s, 16s
    #                     print(f"🔄 {request_name} retryable error (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
    #                     print(f"   Error: {str(e)[:200]}...")
    #                     await asyncio.sleep(delay)
    #                     continue
    #                 else:
    #                     print(f"❌ Max retries ({max_retries}) exceeded for {request_name} retryable error.")
    #                     if fallback_result:
    #                         print(f"⚠️ Creating fallback {request_name} result due to repeated errors")
    #                         return fallback_result
    #                     else:
    #                         raise e
    #             else:
    #                 # For non-retryable errors, raise immediately
    #                 print(f"❌ Non-retryable error in {request_name}: {str(e)[:200]}...")
    #                 if fallback_result:
    #                     print(f"⚠️ Creating fallback {request_name} result due to non-retryable error")
    #                     return fallback_result
    #                 else:
    #                     raise e
        
        # # This should only be reached if all retries failed without exceptions
        # if fallback_result:
        #     print(f"⚠️ All {request_name} attempts failed, using fallback result")
        #     return fallback_result
        # else:
        #     raise RuntimeError(f"{request_name} failed after {max_retries} attempts")

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
            print(f"📁 Sample ID: {args.sample_id}")
            print(f"🎯 Model: {args.model_name}")
            
            # Initialize predictor
            predictor = G3BatchPredictor(
                sample_id=args.sample_id,
                device="cuda" if torch.cuda.is_available() else "cpu",
                checkpoint_path=args.checkpoint_path,
                index_path=args.index_path
            )

            # Run prediction - always use the predict method for 3 modalities
            print("\n🔄 Running complete multi-modal prediction pipeline...")
            result = await predictor.predict(model_name=args.model_name)

            print(json.dumps(result, indent=2))

            # Generate comprehensive reports if requested
            if args.generate_report:
                print("\n📋 Generating comprehensive reports...")
                report_paths = await predictor.generate_comprehensive_report(
                    predictions=result,  # Pass the full result dict instead of just all_predictions
                    model_name=args.model_name
                )
                print(f"\n📄 Reports generated successfully!")

            print("\n🎉 Multi-modal prediction completed successfully!")

        except Exception as e:
            print(f"❌ Multi-modal prediction failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)

    # Run the async main function
    asyncio.run(main())

