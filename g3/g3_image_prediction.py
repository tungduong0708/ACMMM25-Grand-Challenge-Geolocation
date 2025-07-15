import os
import asyncio
import base64
import json
import faiss
from pathlib import Path
from typing import List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import yaml
from google import genai
from google.genai import types
from dotenv import load_dotenv
from utils.image_search import image_search_directory
from utils.index_search import search_index_directory, save_results_to_json
from utils.fetch_satellite import fetch_satellite_image
from utils.utils import get_gps_from_location

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

class G3ImagePredictor:
    """
    Single image prediction class for processing one image using G3 model and similarity scoring.

    This class:
    1. Takes a single image path as input
    2. Performs image-based prediction without text analysis
    3. Uses G3 model for similarity scoring between image and predicted coordinates
    4. Returns the most accurate location prediction
    """

    def __init__(
        self, 
        image_path: str,
        device: str = "auto", 
        index_path: str = "g3/index/G3.index",
        checkpoint_path: str = "g3/checkpoints/mercator_finetune_weight.pth",
        temp_dir: str = "temp_prediction"
    ):
        """
        Initialize the G3ImagePredictor.

        Args:
            image_path (str): Path to the input image
            device (str): Device to run model on ("cuda" or "cpu")
            index_path (str): Path to FAISS index for RAG (required)
            checkpoint_path (str): Path to G3 model checkpoint
            temp_dir (str): Temporary directory for processing
        """
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Robust device detection
        try:
            if device == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    print(f"✅ Auto-detected CUDA device: {torch.cuda.get_device_name()}")
                else:
                    self.device = torch.device("cpu")
                    print(f"✅ Auto-detected CPU device (CUDA not available)")
            elif device == "cuda":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    print(f"✅ Using CUDA device: {torch.cuda.get_device_name()}")
                else:
                    print(f"⚠️ CUDA requested but not available. Falling back to CPU.")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
                print(f"✅ Using CPU device")
        except Exception as e:
            print(f"⚠️ Device detection failed: {e}. Falling back to CPU.")
            self.device = torch.device("cpu")
            
        self.checkpoint_path = checkpoint_path
        self.temp_dir = Path(temp_dir)
        
        # Create temporary directories
        self.temp_dir.mkdir(exist_ok=True)
        self.image_dir = self.temp_dir / "image"
        self.search_dir = self.temp_dir / "search"
        self.image_dir.mkdir(exist_ok=True)
        self.search_dir.mkdir(exist_ok=True)

        # Copy input image to temp directory
        self.processed_image_path = self.image_dir / f"image_001.jpg"
        Image.open(self.image_path).convert("RGB").save(self.processed_image_path)

        # Initialize G3 model
        base_path = Path(__file__).parent
        hparams = yaml.safe_load(open(base_path / "hparams.yaml", "r"))
        pe = "projection_mercator"
        nn = "rffmlp"

        self.model = G3(
            device=self.device,
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

        # Get API keys
        self.GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
        self.IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
        self.SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY")

        if not self.GOOGLE_CLOUD_API_KEY:
            raise ValueError("GOOGLE_CLOUD_API_KEY environment variable is required")

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

    def __image_search__(self):
        """
        Perform image search on the input image.
        Save search results into the search directory.
        """
        if not self.IMGBB_API_KEY or not self.SCRAPINGDOG_API_KEY:
            print("⚠️ Missing API keys for image search. Skipping image search.")
            return
            
        image_search_directory(
            directory=str(self.image_dir),
            output_dir=str(self.search_dir),
            filename="image_search.json",
            imgbb_key=self.IMGBB_API_KEY,
            scrapingdog_key=self.SCRAPINGDOG_API_KEY
        )   
        print(f"✅ Successfully performed image search")

    def __index_search__(self):
        """
        Perform FAISS index search on the input image.
        Save search results into the search directory.
        """
        output_path = self.search_dir / "index_search.json"
        if output_path.exists():
            print(f"🔄 Found existing index search results: {output_path}")
            return

        database_csv_path = "g3/data/mp16/MP16_Pro_filtered.csv"
        if not os.path.exists(database_csv_path):
            print(f"⚠️ Database CSV not found: {database_csv_path}")
            return

        candidates_gps, reverse_gps = search_index_directory(
            model=self.model,
            device=self.device,
            index=self.index,
            image_dir=str(self.image_dir),
            database_csv_path=database_csv_path,
            top_k=20,
            max_elements=20
        )

        save_results_to_json(candidates_gps, reverse_gps, str(output_path))
        print(f"✅ Successfully performed index search. Results saved to: {output_path}")

    def calculate_similarity_scores(self, predicted_coords: List[Tuple[float, float]]) -> np.ndarray:
        """
        Calculate similarity scores between the image and predicted coordinates.

        Args:
            predicted_coords: List of (lat, lon) tuples

        Returns:
            np.ndarray: Similarity scores for each coordinate
        """
        # Load and process the single image
        pil_image = Image.open(self.processed_image_path).convert("RGB")
        
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
            
            return similarity[0]  # Remove batch dimension

    async def llm_predict(self, model_name: str = "gemini-2.5-flash", n_coords: Optional[int] = None) -> dict:
        """
        Generate a prediction using the Gemini LLM with retry logic and exponential backoff.

        Args:
            model_name (str): Model name to use for prediction
            n_coords (int): Number of coordinates to include in prompt

        Returns:
            dict: Parsed JSON prediction response
        """
        # Create a simple image-only prompt
        prompt_data = f"Image path: {self.image_path.name}\n"
        
        # Add index search results if available
        index_search_file = self.search_dir / "index_search.json"
        if index_search_file.exists():
            with open(index_search_file, 'r') as f:
                index_data = json.load(f)
                if n_coords:
                    # Limit to specified number of coordinates
                    limited_data = {}
                    for key in list(index_data.keys())[:n_coords]:
                        limited_data[key] = index_data[key]
                    index_data = limited_data
                prompt_data += f"\nIndex search results (top similar locations):\n{json.dumps(index_data, indent=2)}\n"

        # Add image search results if available
        image_search_file = self.search_dir / "image_search.json"
        if image_search_file.exists():
            with open(image_search_file, 'r') as f:
                search_data = json.load(f)
                prompt_data += f"\nImage search results:\n{json.dumps(search_data, indent=2)}\n"

        prompt = f"""
You are an expert in geo-localization. Analyze the image and determine the most precise possible location—ideally identifying the exact building, landmark, or facility, not just the city. 
Examine all provided content links in detail, using both textual and visual clues to support your conclusion. 
Use only the provided links for evidence. Any additional links must directly support specific visual observations (e.g., satellite imagery or publicly available street-level photos of the same location). 
Return your final answer as geographic coordinates.

{prompt_data}

Respond with **only** the following JSON structure (no extra text, markdown, or comments):

{{
    "latitude": float,
    "longitude": float,
    "location": string,
    "evidence": [
        {{
            "analysis": string,
            "references": [string, …]
        }}
    ]
}}

**Guidelines:**
- Each object in the "evidence" list should explain a single textual or visual clue.  
- The "analysis" field must describe the clue and reference one or more supporting sources using bracketed indices like [1], [2], etc.  
- The corresponding URLs for those references must be included in the "references" list for that object.  
- Use textual/news URLs for contextual clues and satellite/image URLs for visual clues.  
- Do **not** include any links that are not explicitly cited in the "analysis".  
- Maintain the order of evidence to match the sequence in which clues are introduced.  
- The combination of all "analysis" fields should make it clear how the clues lead to the final coordinates, without revealing intermediate reasoning or metadata.  
- **Do not use metadata** (e.g., EXIF data, filenames, author handles, timestamps, or embedded properties) as part of the analysis or evidence.
"""

        # Convert image to base64
        with open(self.processed_image_path, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

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

        # Retry logic with exponential backoff
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempting LLM prediction (attempt {attempt + 1}/{max_retries})")
                
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(text=prompt),
                                types.Part(
                                    inline_data=types.Blob(
                                        mime_type="image/jpeg",
                                        data=image_base64
                                    )
                                )
                            ]
                        )
                    ],
                    config=config
                )

                if response and response.text:
                    parsed_response = extract_and_parse_json(response.text)
                    if parsed_response and is_valid_enhanced_gps_dict(parsed_response):
                        print(f"✅ Successfully generated LLM prediction")
                        return parsed_response
                    else:
                        print(f"⚠️ Invalid response format on attempt {attempt + 1}")
                else:
                    print(f"⚠️ Empty response on attempt {attempt + 1}")

            except Exception as e:
                print(f"⚠️ Error on attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"🔄 Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
        
        # If all retries failed, return empty dict
        print(f"❌ All {max_retries} attempts failed")
        return {}

    async def diversification_predict(self, model_name: str = "gemini-2.5-flash") -> dict:
        """
        Diversification prediction using different sample sizes and similarity scoring.

        Args:
            model_name (str): LLM model name to use

        Returns:
            dict: Best prediction with latitude, longitude, location, reason, and metadata
        """
        # Function to try a specific sample size with retry logic
        async def try_sample_size(num_coords):
            try:
                prediction = await self.llm_predict(model_name=model_name, n_coords=num_coords)
                if prediction and is_valid_enhanced_gps_dict(prediction):
                    coords = (float(prediction["latitude"]), float(prediction["longitude"]))
                    return num_coords, coords, prediction
                else:
                    print(f"⚠️ Invalid prediction for {num_coords} coordinates")
                    return num_coords, None, {}
            except Exception as e:
                print(f"⚠️ Error with {num_coords} coordinates: {e}")
                return num_coords, None, {}

        # Run different sample sizes
        num_samples = [5, 10, 15]
        print(f"🚀 Running {len(num_samples)} sample sizes: {num_samples}")
        
        tasks = [try_sample_size(num_coords) for num_coords in num_samples]
        results = []
        for task in tasks:
            result = await task
            results.append(result)
        
        # Build predictions dictionary from results
        predictions_dict = {}
        for num_coords, coords, prediction in results:
            if coords and prediction:
                predictions_dict[coords] = prediction
                print(f"✅ Valid prediction for {num_coords} coordinates: {coords}")
            else:
                print(f"❌ Failed prediction for {num_coords} coordinates")

        # # Convert predictions to coordinate list for similarity scoring
        predicted_coords = list(predictions_dict.keys())
        print(f"Predicted coordinates: {predicted_coords}")

        if not predicted_coords:
            print("❌ No valid predictions generated")
            return {}

        # Calculate similarity scores
        similarity_scores = self.calculate_similarity_scores(predicted_coords=predicted_coords)

        # Find best prediction
        best_idx = np.argmax(similarity_scores)
        best_coords = predicted_coords[best_idx]
        best_prediction = predictions_dict[best_coords]

        print(f"🎯 Best prediction selected: {best_coords}")
        print(f"   Similarity scores: {similarity_scores}")
        print(f"   Best index: {best_idx}")

        return best_prediction

    async def predict(self, model_name: str = "gemini-2.5-flash") -> dict:
        """
        Complete prediction pipeline for single image.

        Args:
            model_name (str): LLM model name to use

        Returns:
            dict: Final prediction with latitude, longitude, location, and evidence
        """
        print(f"🚀 Starting prediction for image: {self.image_path}")
        
        # Step 1: Perform searches
        print("🔄 Performing image and index searches...")
        self.__image_search__()
        self.__index_search__()
        
        # Step 2: Run diversification prediction
        print("🔄 Running diversification prediction...")
        prediction = await self.diversification_predict(model_name=model_name)
        
        if prediction:
            print("✅ Prediction completed successfully")
        else:
            print("❌ Prediction failed")
            
        return prediction

    def save_prediction_to_json(self, prediction: dict, output_path: str = None) -> None:
        """
        Save prediction result to JSON file.

        Args:
            prediction (dict): Prediction result
            output_path (str): Output file path (optional)
        """
        if output_path is None:
            output_path = self.temp_dir / "prediction_result.json"
        
        with open(output_path, 'w') as f:
            json.dump(prediction, f, indent=2)
        
        print(f"💾 Prediction saved to: {output_path}")

    def cleanup(self):
        """
        Clean up temporary files and directories.
        """
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "acmmm2025-grand-challenge-gg-credentials.json"
    import argparse

    parser = argparse.ArgumentParser(description="G3 Single Image Predictor")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image"
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
        default="gemini-2.5-flash",
        help="LLM model name to use"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="image_prediction_result.json",
        help="Output file to save prediction result"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run model on (cuda, cpu, or auto for automatic detection)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up temporary files after prediction"
    )

    args = parser.parse_args()

    async def main():
        try:
            # Initialize predictor
            predictor = G3ImagePredictor(
                image_path=args.image_path,
                device=args.device,
                index_path=args.index_path,
                checkpoint_path=args.checkpoint_path
            )

            # Run prediction
            prediction = await predictor.predict(model_name=args.model_name)

            # Save results
            predictor.save_prediction_to_json(prediction, args.output_file)

            # Print results
            if prediction:
                print("\n" + "="*50)
                print("PREDICTION RESULTS:")
                print("="*50)
                print(f"Location: {prediction.get('location', 'N/A')}")
                print(f"Latitude: {prediction.get('latitude', 'N/A')}")
                print(f"Longitude: {prediction.get('longitude', 'N/A')}")
                print(f"Evidence items: {len(prediction.get('evidence', []))}")
                print("="*50)
            else:
                print("\n❌ No prediction generated")

            # Cleanup if requested
            if args.cleanup:
                predictor.cleanup()

        except Exception as e:
            print(f"❌ Error: {e}")
            return 1

        return 0

    # Run the async main function
    exit_code = asyncio.run(main())
    exit(exit_code)
