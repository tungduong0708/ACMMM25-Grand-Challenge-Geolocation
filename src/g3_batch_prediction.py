import asyncio
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from google import genai
from google.genai import types
from pydantic import ValidationError
from tqdm.asyncio import tqdm as atqdm

from .data_processor import DataProcessor
from .g3.G3 import G3
from .prompt import (
    Evidence,
    GPSPrediction,
    LocationPrediction,
    diversification_prompt,
    location_prompt,
    verification_prompt,
)
from .utils import (
    calculate_similarity_scores,
    extract_and_parse_json,
    get_gps_from_location,
    handle_async_api_call_with_retry,
    image_to_base64,
)

logger = logging.getLogger("uvicorn.error")


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
        input_dir: str = "data/input_data",
        prompt_dir: str = "data/prompt_data",
        cache_dir: str = "data/cache",
        index_path: str = "data/index/G3.index",
        hparams_path: str = "g3/hparams.yaml",
        database_csv_path: str = "data/dataset/mp16/MP16_Pro_filtered.csv",
        checkpoint_path: str = "data/checkpoints/mercator_finetune_weight.pth",
    ):
        """
        Initialize the BatchKeyframePredictor.

        Args:
            checkpoint_path (str): Path to G3 model checkpoint
            device (str): Device to run model on ("cuda" or "cpu")
            index_path (str): Path to FAISS index for RAG (required)
        """
        self.device = torch.device(device)
        self.base_path = Path(__file__).parent
        self.checkpoint_path = self.base_path / checkpoint_path

        self.input_dir = self.base_path / input_dir
        self.prompt_dir = self.base_path / prompt_dir
        self.cache_dir = self.base_path / cache_dir
        self.image_dir = self.prompt_dir / "images"
        self.audio_dir = self.prompt_dir / "audio"

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.prompt_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

        # Initialize G3 model
        hparams = yaml.safe_load(open(self.base_path / hparams_path, "r"))
        pe = "projection_mercator"
        nn = "rffmlp"

        self.model = G3(
            device=device,
            positional_encoding_type=pe,
            neural_network_type=nn,
            hparams=hparams[f"{pe}_{nn}"],
        )
        self.__load_checkpoint()

        self.data_processor = DataProcessor(
            model=self.model,
            input_dir=self.input_dir,
            prompt_dir=self.prompt_dir,
            cache_dir=self.cache_dir,
            image_dir=self.image_dir,
            audio_dir=self.audio_dir,
            index_path=self.base_path / index_path,
            database_csv_path=self.base_path / database_csv_path,
            device=self.device,
        )

        self.image_extension = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp",
        }
        self.video_extension = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
        }

    def __load_checkpoint(self):
        """
        Load the G3 model checkpoint.
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}"
            )
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info(
            f"✅ Successfully loaded G3 model checkpoint from: {self.checkpoint_path}"
        )

    async def llm_predict(
        self,
        model_name: str = "gemini-2.5-pro",
        n_search: int | None = None,
        n_coords: int | None = None,
        image_prediction: bool = True,
        text_prediction: bool = True,
    ) -> LocationPrediction:
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
        prompt = diversification_prompt(
            prompt_dir=str(self.prompt_dir),
            n_coords=n_coords,
            n_search=n_search,
            image_prediction=image_prediction,
            text_prediction=text_prediction,
        )

        images = []
        if image_prediction:
            image_dir = self.image_dir
            if not image_dir.exists():
                raise ValueError(f"Image directory does not exist: {image_dir}")

            for image_file in image_dir.glob("*.jpg"):
                with open(image_file, "rb") as f:
                    image = types.Part.from_bytes(data=f.read(), mime_type="image/jpeg")
                images.append(image)

        client = genai.Client(api_key=os.environ["GOOGLE_CLOUD_API_KEY"])

        async def api_call():
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=[*images, prompt],
                    config=types.GenerateContentConfig(
                        tools=[
                            types.Tool(url_context=types.UrlContext()),
                        ],
                        temperature=0.1,
                        top_p=0.95,
                    ),
                ),
            )

            raw_text = response.text.strip() if response.text is not None else ""
            parsed_json = extract_and_parse_json(raw_text)

            try:
                validated = LocationPrediction.model_validate(parsed_json)
                return validated
            except (ValidationError, ValueError):
                raise ValueError("Empty or invalid LLM response")

        return await handle_async_api_call_with_retry(
            api_call,
            fallback_result=LocationPrediction(
                latitude=0.0, longitude=0.0, location="", evidence=[]
            ),
            error_context=f"LLM prediction with {model_name}",
        )

    async def diversification_predict(
        self,
        model_name: str = "gemini-2.5-flash",
        image_prediction: bool = True,
        text_prediction: bool = True,
    ) -> LocationPrediction:
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
            while True:
                prediction = await self.llm_predict(
                    model_name=model_name,
                    n_search=num_sample,
                    n_coords=num_sample,
                    image_prediction=image_prediction,
                    text_prediction=text_prediction,
                )

                if prediction:
                    coords = (prediction.latitude, prediction.longitude)
                    return (num_sample, coords, prediction)
                else:
                    logger.info(
                        f"Invalid or empty prediction format with {num_sample} samples, retrying..."
                    )

        # Run all sample sizes in parallel
        num_samples = [10, 15, 20]
        logger.info(
            f"🚀 Running {len(num_samples)} sample sizes in parallel: {num_samples}"
        )

        tasks = [try_sample_size(num_sample) for num_sample in num_samples]

        class LW:
            def write(self, msg: str) -> int:
                logger.info(msg)
                return len(msg)

            def flush(self):
                pass

        results = await atqdm.gather(
            *tasks,
            desc="🔄 Running diversification predictions",
            file=LW(),
        )

        # Build predictions dictionary from parallel results
        predictions_dict = {}
        for num_sample, coords, prediction in results:
            predictions_dict[coords] = prediction
            logger.info(f"✅ Collected prediction with {num_sample} samples: {coords}")

        # Convert predictions to coordinate list for similarity scoring
        predicted_coords = list(predictions_dict.keys())
        logger.info(f"Predicted coordinates: {predicted_coords}")

        if not predicted_coords:
            raise ValueError("No valid predictions obtained from any sample size")

        # Calculate similarity scores
        avg_similarities = calculate_similarity_scores(
            model=self.model,
            device=self.device,
            predicted_coords=predicted_coords,
            image_dir=self.image_dir,
        )

        # Find best prediction
        best_idx = np.argmax(avg_similarities)
        best_coords = predicted_coords[best_idx]
        best_prediction = predictions_dict[best_coords]

        logger.info(f"🎯 Best prediction selected: {best_coords}")
        logger.info(f"   Similarity scores: {avg_similarities}")
        logger.info(f"   Best index: {best_idx}")

        # print(json.dumps(best_prediction, indent=2))  # Commented out verbose output

        return best_prediction

    async def location_predict(
        self,
        model_name: str = "gemini-2.5-flash",
        location: str = "specified location",
    ) -> GPSPrediction:
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
        if lat is not None and lon is not None:
            logger.info(
                f"Using GPS coordinates for location '{location}': ({lat}, {lon})"
            )
            return GPSPrediction(
                latitude=lat, longitude=lon, analysis="", references=[]
            )
        else:
            prompt = location_prompt(location)
            client = genai.Client(api_key=os.environ["GOOGLE_CLOUD_API_KEY"])

            async def api_call():
                # Run the synchronous API call in a thread executor to make it truly async
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model_name,
                        contents=[prompt],
                        config=types.GenerateContentConfig(
                            tools=[
                                types.Tool(google_search=types.GoogleSearch()),
                            ],
                            temperature=0.1,
                            top_p=0.95,
                        ),
                    ),
                )

                raw_text = response.text.strip() if response.text is not None else ""
                parsed_json = extract_and_parse_json(raw_text)

                try:
                    validated = GPSPrediction.model_validate(parsed_json)
                    return validated
                except (ValidationError, ValueError):
                    raise ValueError("Empty or invalid LLM response")

            return await handle_async_api_call_with_retry(
                api_call,
                fallback_result=GPSPrediction(
                    latitude=0.0, longitude=0.0, analysis="", references=[]
                ),
                error_context=f"Location prediction for '{location}' with {model_name}",
            )

    async def verification_predict(
        self,
        prediction: LocationPrediction,
        model_name: str = "gemini-2.5-flash",
        image_prediction: bool = True,
        text_prediction: bool = True,
    ) -> LocationPrediction:
        """
        Generate verification prediction based on the provided prediction.

        Args:
            prediction (dict): Prediction dictionary with latitude, longitude, location, reason, and metadata
            model_name (str): LLM model name to use for verification

        Returns:
            dict: Verification prediction with latitude, longitude, location, reason, and evidence
        """
        # Prepare verification data (now async)
        satellite_image_id = await self.data_processor.prepare_location_images(
            prediction=prediction.model_dump(),
            image_prediction=image_prediction,
            text_prediction=text_prediction,
        )

        image_dir = self.image_dir

        images = []
        if image_prediction:
            if not image_dir.exists():
                raise ValueError(f"Image directory does not exist: {image_dir}")

            for image_file in image_dir.glob("*.jpg"):
                with open(image_file, "rb") as f:
                    image = types.Part.from_bytes(data=f.read(), mime_type="image/jpeg")
                images.append(image)

        # Prepare verification prompt
        prompt = verification_prompt(
            satellite_image_id=satellite_image_id,
            prediction=prediction.model_dump(),
            prompt_dir=str(self.prompt_dir),
            image_prediction=image_prediction,
            text_prediction=text_prediction,
        )

        client = genai.Client(api_key=os.environ["GOOGLE_CLOUD_API_KEY"])

        async def api_call():
            # Run the synchronous API call in a thread executor to make it truly async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=[*images, prompt],
                    config=types.GenerateContentConfig(
                        tools=[
                            types.Tool(url_context=types.UrlContext()),
                        ],
                        temperature=0.1,
                        top_p=0.95,
                    ),
                ),
            )

            raw_text = response.text.strip() if response.text is not None else ""
            parsed_json = extract_and_parse_json(raw_text)

            try:
                validated = LocationPrediction.model_validate(parsed_json)
                return validated
            except (ValidationError, ValueError):
                raise ValueError("Empty or invalid LLM response")

        return await handle_async_api_call_with_retry(
            api_call,
            fallback_result=LocationPrediction(
                latitude=0.0, longitude=0.0, location="", evidence=[]
            ),
            error_context=f"Verification prediction with {model_name}",
        )

    async def predict(
        self,
        model_name: str = "gemini-2.5-flash",
        image_prediction: bool = True,
        text_prediction: bool = True,
    ) -> LocationPrediction:
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
        logger.info(
            f"🚀 Starting multi-modal prediction pipeline with model: {model_name}"
        )
        await self.data_processor.preprocess_input_data()
        # Step 1: Run diversification prediction (this is already parallel internally)
        logger.info(
            f"\n🔄 Running diversification prediction for Image={image_prediction}, Text={text_prediction}..."
        )
        diversification_result = await self.diversification_predict(
            model_name=model_name,
            image_prediction=image_prediction,
            text_prediction=text_prediction,
        )

        # Step 2: Run location prediction
        location_prediction = await self.location_predict(
            model_name=model_name, location=diversification_result.location
        )

        logger.info("✅ Location prediction completed:")

        # Step 3: Update coordinates and evidence from location prediction
        result = diversification_result.model_copy()
        result.longitude = location_prediction.longitude
        result.latitude = location_prediction.latitude

        # Step 4: Normalize and append location evidence
        if location_prediction.analysis and location_prediction.references:
            location_evidence = Evidence(
                analysis=location_prediction.analysis,
                references=location_prediction.references,
            )
        else:
            location_evidence = Evidence(
                analysis="No specific location analysis provided.",
                references=[],
            )

        # Append to result evidence
        result.evidence.append(location_evidence)

        # Step 5: Run verification prediction
        logger.info(
            f"\n🔄 Running verification prediction for Image={image_prediction}, Text={text_prediction}..."
        )
        result = await self.verification_predict(
            prediction=result,
            model_name=model_name,
            image_prediction=image_prediction,
            text_prediction=text_prediction,
        )

        logger.info(
            f"\n🎯 Final prediction for Image={image_prediction}, Text={text_prediction}:"
        )
        # print(json.dumps(result, indent=2))  # Commented out verbose output

        return result

    def get_response(self, prediction: LocationPrediction) -> LocationPrediction:
        """
        Convert image references in the prediction to base64 strings.
        """
        for evidence in prediction.evidence:
            for i, ref in enumerate(evidence.references):
                if ref.startswith("image"):
                    evidence.references[i] = image_to_base64(self.image_dir / ref)
        return prediction

    def get_transcript(self) -> str:
        """
        Get the transcript from the transcript files in the audio directory.
        """
        transcript = ""
        for transcript_file in self.audio_dir.glob("*.txt"):
            with open(transcript_file, "r", encoding="utf-8") as f:
                logger.info(f"Reading transcript from {transcript_file.name}")
                transcript_data = f.read().strip()
                if transcript_data:
                    transcript += f"Transcript for {transcript_file.name}\n"
                    transcript += transcript_data
        return transcript

    def clear_directories(self):
        """
        Clear the input and prompt directories.
        """
        delete_dirs = [self.input_dir, self.prompt_dir]
        for dir_path in delete_dirs:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.info(f"Deleted folder: {dir_path}")
            else:
                logger.info(f"Folder does not exist: {dir_path}")
