import argparse
import base64
import json
import os
import re
import csv
import concurrent.futures
import threading
from io import BytesIO
from pathlib import Path
from functools import partial

import faiss
import numpy as np
import pandas as pd
from typing import Optional
import torch
import yaml
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

# Add the missing imports from simple version
from utils.G3 import G3
from utils.utils import search_with_image_and_text, extract_image_search_candidates

# Add base path setup like in simple version
base_path = Path(__file__).parent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

# Import the prompt functions
from utils.prompt import combine_prompts

def extract_and_parse_json(raw_text: str) -> dict:
    """
    Extract JSON content between first { and last } and parse it.
    
    Args:
        raw_text (str): Raw response text from LLM
        
    Returns:
        dict: Parsed JSON data
        
    Raises:
        ValueError: If no valid JSON found or parsing fails
    """
    # Find first { and last }
    first_brace = raw_text.find('{')
    last_brace = raw_text.rfind('}')
    
    if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
        raise ValueError(f"No valid JSON braces found in response: {raw_text}")
    
    # Extract JSON substring
    json_str = raw_text[first_brace:last_brace + 1]
    
    try:
        # Parse JSON
        parsed_data = json.loads(json_str)
        return parsed_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {json_str}, Error: {e}")

class G3Predictor:
    def __init__(self, checkpoint_path, device, index_path=None):
        # Use the same initialization as simple version
        hparams = yaml.safe_load(open(base_path / "hparams.yaml", "r"))
        pe = "projection_mercator"
        nn = "rffmlp"
        self.model = G3(
            device=device,
            positional_encoding_type=pe,
            neural_network_type=nn,
            hparams=hparams[f"{pe}_{nn}"],
        )
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = device
        self.index = faiss.read_index(index_path) if index_path else None

    def search_index(self, rgb_image, top_k=20):
        print("Searching index...")
        image = self.model.vision_processor(images=rgb_image, return_tensors="pt")[
            "pixel_values"
        ].reshape(-1, 224, 224)
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            vision_output = self.model.vision_model(image)[1]
            image_embeds = self.model.vision_projection(vision_output)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

            image_text_embeds = self.model.vision_projection_else_1(
                self.model.vision_projection(vision_output)
            )
            image_text_embeds = image_text_embeds / image_text_embeds.norm(
                p=2, dim=-1, keepdim=True
            )

            image_location_embeds = self.model.vision_projection_else_2(
                self.model.vision_projection(vision_output)
            )
            image_location_embeds = image_location_embeds / image_location_embeds.norm(
                p=2, dim=-1, keepdim=True
            )

            positive_image_embeds = torch.cat(
                [image_embeds, image_text_embeds, image_location_embeds], dim=1
            )
            positive_image_embeds = (
                positive_image_embeds.cpu().detach().numpy().astype(np.float32)
            )

            negative_image_embeds = positive_image_embeds * (-1.0)

        if self.index is None:
            raise ValueError("FAISS index is not loaded. Please provide a valid index_path.")
        D, I = self.index.search(positive_image_embeds, top_k)
        D_reverse, I_reverse = self.index.search(negative_image_embeds, top_k)
        return D, I, D_reverse, I_reverse

    def get_response_with_prompts(
        self, 
        base64_image: str, 
        api_key: str,
        image_path: str = "",
        transcript_file_path: str = "",
        metadata_file_path: str = "",
        candidates_gps: Optional[list[tuple]] = None,
        reverse_gps: Optional[list[tuple]] = None,
        search_candidates: Optional[list[str]] = None,
        n_coords: int = 15,
        n_search: int = 5,
        model_name: str = "gemini-2.5-flash"
    ) -> dict:
        """Enhanced response method using combined prompts."""
        
        # Create comprehensive prompt using prompt.py functions
        combined_prompt = combine_prompts(
            image_path=image_path,
            transcript_file_path=transcript_file_path,
            metadata_file_path=metadata_file_path,
            candidates_gps=candidates_gps,
            reverse_gps=reverse_gps,
            search_candidates=search_candidates,
            n_search= n_search,
            n_coords=n_coords,
        )

        print(combined_prompt)

        client = genai.Client(
            api_key=api_key
        )
        image = types.Part.from_bytes(
            data=base64.b64decode(base64_image), mime_type="image/jpeg"
        )

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

        response = client.models.generate_content(
            model=model_name,
            contents=[image, combined_prompt],
            config=config
        )
        
        raw_text = response.text.strip() if response.text is not None else ""        
        # Extract and parse JSON from response
        try:
            parsed_json = extract_and_parse_json(raw_text)
            return parsed_json
        except ValueError as e:
            print(f"Error parsing enhanced response: {e}")
            raise e

    def is_valid_enhanced_gps_dict(self, gps_data):
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
            if "analysis" not in item or "links" not in item:
                return False
            if not isinstance(item["links"], list):
                return False
            
        try:
            lat = float(gps_data["latitude"])
            lon = float(gps_data["longitude"])
            
            # Basic GPS coordinate validation
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return True
        except (ValueError, TypeError):
            pass
            
        return False

    def simple_predict(
        self,
        base64_image: str,
        database_csv_path: str,
        image_path: str = "",
        transcript_file_path: str = "",
        metadata_file_path: str = "",
        serpapi_key: str = "",
        imgbb_key: str = "",
        top_k: int = 20,
        no_results: int = 5,
        model_name: str = "gemini-2.5-flash",
    ) -> dict:
        """Enhanced predict method that returns detailed prediction with location and reason."""
        
        # Decode image for FAISS search
        image_bytes = base64.b64decode(base64_image)
        rgb_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Get similar/dissimilar coordinates using FAISS
        D, I, D_reverse, I_reverse = self.search_index(rgb_image, top_k)
        
        # Get GPS coordinates for RAG
        candidates_gps, reverse_gps = self._get_gps_coordinates(
            I, I_reverse, database_csv_path
        )
        
        api_key = os.getenv("API_KEY")
        if api_key is None:
            raise ValueError("API_KEY environment variable is not set.")
        
        # Get search results using image search
        search_results = search_with_image_and_text(
            image_path=image_path,
            search_text="",  # Image-only search
            serpapi_key=serpapi_key,
            imgbb_key=imgbb_key
        )
        # Extract candidate links
        search_candidates = extract_image_search_candidates(search_results, no_results)
        
        # Get enhanced prediction with location and reason
        while True:
            try:
                prediction = self.get_response_with_prompts(
                    base64_image=base64_image,
                    api_key=api_key,
                    image_path=image_path,
                    transcript_file_path=transcript_file_path,
                    metadata_file_path=metadata_file_path,
                    candidates_gps=candidates_gps[:15] if candidates_gps else [],
                    reverse_gps=reverse_gps[:15] if reverse_gps else [],
                    search_candidates=search_candidates[:15] if search_candidates else [],
                    n_coords=15,
                    n_search=no_results,
                    model_name=model_name
                )
                
                if self.is_valid_enhanced_gps_dict(prediction):
                    return prediction
                else:
                    print("Invalid enhanced prediction format, retrying...")
                    
            except Exception as e:
                print(f"Enhanced prediction failed: {e}, retrying...")

    def diversification_predict(
        self,
        base64_image: str,
        database_csv_path: str,
        image_path: str = "",
        transcript_file_path: str = "",
        metadata_file_path: str = "",
        serpapi_key: str = "",
        imgbb_key: str = "",
        top_k: int = 20,
        model_name: str = "gemini-2.5-flash",
    ) -> dict:
        """
        Enhanced predict method that follows the same pattern as simple_g3_prediction.
        Creates multiple predictions and uses similarity scoring to select the best one.
        """
        
        # Decode image for FAISS search
        image_bytes = base64.b64decode(base64_image)
        rgb_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Get similar/dissimilar coordinates using FAISS
        D, I, D_reverse, I_reverse = self.search_index(rgb_image, top_k)
        
        # Get GPS coordinates for RAG
        candidates_gps, reverse_gps = self._get_gps_coordinates(
            I, I_reverse, database_csv_path
        )

        search_results = search_with_image_and_text(
            image_path=image_path,
            search_text="",  # Image-only search
            serpapi_key=serpapi_key,
            imgbb_key=imgbb_key
        )
        # Extract candidate links
        search_candidates = extract_image_search_candidates(search_results, no_results=20)
        
        api_key = os.getenv("API_KEY")
        if api_key is None:
            raise ValueError("API_KEY environment variable is not set.")
        
        # Dictionary to store predictions: {(lat, lon): prediction_dict}
        predictions_dict = {}
        
        # Add a default/fallback prediction from database
        for chunk in pd.read_csv(database_csv_path, chunksize=10000, usecols=["LAT", "LON"]):
            default_coords = (float(chunk.loc[0, "LAT"]), float(chunk.loc[0, "LON"]))
            break
        
        # RAG predictions with different sample sizes
        num_samples = [5, 7, 10]
        for num_sample in num_samples:
            print(f"Starting prediction with {num_sample} samples...")
            while True:
                try:
                    prediction = self.get_response_with_prompts(
                        base64_image=base64_image,
                        api_key=api_key,
                        image_path=image_path,
                        transcript_file_path=transcript_file_path,
                        metadata_file_path=metadata_file_path,
                        candidates_gps=candidates_gps[:num_sample] if candidates_gps else [],
                        reverse_gps=reverse_gps[:num_sample] if reverse_gps else [],
                        search_candidates=search_candidates[:num_sample] if search_candidates else [],
                        n_search=num_sample,
                        n_coords=num_sample,
                        model_name=model_name
                    )

                    print(json.dumps(prediction, indent=2))

                    if self.is_valid_enhanced_gps_dict(prediction):
                        coords = (prediction["latitude"], prediction["longitude"])
                        predictions_dict[coords] = prediction
                        print(f"✅ Prediction with {num_sample} samples successful: {coords}")
                        break
                    else:
                        print(f"Invalid prediction format with {num_sample} samples, retrying...")

                except Exception as e:
                    print(f"Prediction with {num_sample} samples failed: {e}, retrying...")

        # Convert predictions to coordinate list for similarity scoring
        predicted_coords = list(predictions_dict.keys())
        print(f"Predicted coordinates: {predicted_coords}")
        
        # Similarity scoring (same as simple_g3_prediction)
        image = self.model.vision_processor(images=rgb_image, return_tensors="pt")[
            "pixel_values"
        ].reshape(-1, 224, 224)
        image = image.unsqueeze(0).to(self.device)

        vision_output = self.model.vision_model(image)[1]

        image_embeds = self.model.vision_projection_else_2(
            self.model.vision_projection(vision_output)
        )
        image_embeds = image_embeds / image_embeds.norm(
            p=2, dim=-1, keepdim=True
        )  # b, 768

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
        max_idx = np.argmax(similarity, axis=1)

        # Get the best prediction based on similarity score
        best_coords = predicted_coords[max_idx[0]]
        best_prediction = predictions_dict[best_coords]
        
        print(f"🎯 Best prediction selected: {best_coords}")
        print(f"   Similarity scores: {similarity[0]}")
        print(f"   Best index: {max_idx[0]}")
        
        return best_prediction

    def _get_gps_coordinates(self, I, I_reverse, database_csv_path):
        """Helper method to get GPS coordinates from database."""
        candidate_indices = I[0]
        reverse_indices = I_reverse[0]
        
        candidates_gps = []
        reverse_gps = []
        
        for chunk in pd.read_csv(database_csv_path, chunksize=10000, usecols=["LAT", "LON"]):
            for idx in candidate_indices:
                if idx in chunk.index:
                    lat = float(chunk.loc[idx, "LAT"])
                    lon = float(chunk.loc[idx, "LON"])
                    candidates_gps.append((lat, lon))

            for ridx in reverse_indices:
                if ridx in chunk.index:
                    lat = float(chunk.loc[ridx, "LAT"])
                    lon = float(chunk.loc[ridx, "LON"])
                    reverse_gps.append((lat, lon))
        
        return candidates_gps, reverse_gps

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

def process_single_image(predictor, image_path, transcript_path="", metadata_path="", serpapi_key="", imgbb_key="", thread_id=0):
    """Process a single image with thread-safe logging."""
    try:
        print(f"[Thread {thread_id}] 🖼️ Processing: {Path(image_path).name}")
        
        image_base64 = image_to_base64(image_path)
        
        prediction = predictor.diversification_predict(
            base64_image=image_base64,
            database_csv_path="g3/data/mp16/MP16_Pro_filtered.csv",  
            image_path=image_path,
            transcript_file_path=transcript_path,
            metadata_file_path=metadata_path,
            serpapi_key=serpapi_key,
            imgbb_key=imgbb_key,
            top_k=20,
            model_name="gemini-2.5-pro",
        )

        print(json.dumps(prediction, indent=2))

        # Extract ID and video name
        if metadata_path:
            prediction_id = Path(metadata_path).stem
        else:
            prediction_id = Path(image_path).stem
        
        vid_name = Path(image_path).stem
        
        # Extract evidence from the response
        evidence = prediction.get('evidence', [])
        
        # Combine all analysis into a single reason string
        reason_parts = []
        all_links = []
        
        for i, item in enumerate(evidence):
            if isinstance(item, dict):
                analysis = item.get('analysis', '')
                links = item.get('links', [])
                
                if analysis:
                    reason_parts.append(f"Evidence {i+1}: {analysis}")
                
                if isinstance(links, list):
                    all_links.extend(links)
                elif links:
                    all_links.append(str(links))
        
        # Join all analysis parts
        reason = ' | '.join(reason_parts) if reason_parts else 'No analysis provided'
        
        # Join all links
        links_str = '; '.join(all_links) if all_links else 'No links'
        
        result = {
            'id': prediction_id,
            'vid_name': vid_name,
            'image_path': image_path,
            'latitude': prediction['latitude'],
            'longitude': prediction['longitude'],
            'location': prediction['location'],
            'reason': reason,
            'links': links_str,
            'status': 'success',
            'thread_id': thread_id
        }
        
        print(f"[Thread {thread_id}] ✅ Success: {prediction_id} -> {prediction['latitude']}, {prediction['longitude']}")
        return result
        
    except Exception as e:
        print(f"[Thread {thread_id}] ❌ Error processing {image_path}: {e}")
        return {
            'id': Path(image_path).stem,
            'vid_name': Path(image_path).stem,
            'image_path': image_path,
            'latitude': 0.0,
            'longitude': 0.0,
            'location': 'Error',
            'reason': f'Processing failed: {str(e)}',
            'links': 'No links',
            'status': 'error',
            'thread_id': thread_id
        }

def process_images_parallel(predictor, image_paths, transcript_path="", metadata_path="", serpapi_key="", imgbb_key="", max_workers=4):
    """
    Process multiple images in parallel using ThreadPoolExecutor.
    
    Args:
        predictor: G3Predictor instance
        image_paths: List of image file paths
        transcript_path: Path to transcript file
        metadata_path: Path to metadata file
        serpapi_key: SerpAPI key
        imgbb_key: ImgBB key
        max_workers: Maximum number of parallel threads
    
    Returns:
        List of prediction results
    """
    results = []
    
    # Create a partial function with fixed parameters
    process_func = partial(
        process_single_image,
        predictor=predictor,
        transcript_path=transcript_path,
        metadata_path=metadata_path,
        serpapi_key=serpapi_key,
        imgbb_key=imgbb_key
    )
    
    print(f"🚀 Starting parallel processing with {max_workers} threads...")
    print(f"📊 Total images to process: {len(image_paths)}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_func, image_path=image_path, thread_id=i): (image_path, i) 
            for i, image_path in enumerate(image_paths)
        }
        
        # Collect results as they complete
        completed = 0
        total = len(image_paths)
        
        for future in concurrent.futures.as_completed(future_to_image):
            image_path, thread_id = future_to_image[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                print(f"📈 Progress: {completed}/{total} completed ({completed/total*100:.1f}%)")
            except Exception as e:
                print(f"❌ Thread {thread_id} failed for {image_path}: {e}")
                # Add error result
                results.append({
                    'id': Path(image_path).stem,
                    'vid_name': Path(image_path).stem,
                    'image_path': image_path,
                    'latitude': 0.0,
                    'longitude': 0.0,
                    'location': 'Error',
                    'reason': f'Thread execution failed: {str(e)}',
                    'links': 'No links',
                    'status': 'error',
                    'thread_id': thread_id
                })
    
    # Sort results by original order (based on image path)
    image_path_to_index = {path: i for i, path in enumerate(image_paths)}
    results.sort(key=lambda x: image_path_to_index.get(x['image_path'], float('inf')))
    
    return results

def save_results_to_csv(results, csv_path):
    """Save all results to CSV file."""
    # Ensure the results directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if CSV exists to determine if we need to write headers
    file_exists = os.path.exists(csv_path)
    
    # Write to CSV
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'vid_name', 'latitude', 'longitude', 'location', 'reason', 'links', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write all results
        for result in results:
            writer.writerow({
                'id': result['id'],
                'vid_name': result['vid_name'],
                'latitude': result['latitude'],
                'longitude': result['longitude'],
                'location': result['location'],
                'reason': result.get('reason', 'No reason'),
                'links': result.get('links', 'No links'),
                'status': result['status']
            })
    
    print(f"\n💾 Results saved to: {csv_path}")
    print(f"📊 Total processed: {len(results)}")
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Enhanced G3Predictor")

    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to a single image file to predict (use either this or --image_list)",
    )
    parser.add_argument(
        "--image_list",
        type=str,
        help="Path to a text file containing list of image paths (one per line)",
    )
    parser.add_argument(
        "--transcript_path",
        type=str,
        default="",
        help="Path to the transcript file (optional)",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="",
        help="Path to the metadata JSON file (optional)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for multiple images",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.image_path and not args.image_list:
        parser.error("Either --image_path or --image_list must be provided")
    
    if args.image_path and args.image_list:
        parser.error("Cannot use both --image_path and --image_list at the same time")

    # Use the same paths as simple version
    predictor = G3Predictor(
        checkpoint_path="g3/checkpoints/mercator_finetune_weight.pth",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        index_path="g3/index/G3.index",
    )
    
    # Get API keys from environment
    serpapi_key = os.getenv("SERPAPI_KEY", "")
    imgbb_key = os.getenv("IMGBB_KEY", "")
    
    # Prepare image list
    image_paths = []
    if args.image_path:
        # Single image mode
        image_paths = [args.image_path]
    elif args.image_list:
        # Batch mode - read image paths from file
        try:
            with open(args.image_list, 'r') as f:
                image_paths = [line.strip() for line in f if line.strip()]
            print(f"📁 Loaded {len(image_paths)} image paths from {args.image_list}")
        except Exception as e:
            print(f"❌ Error reading image list file: {e}")
            exit(1)
    
    # Process all images
    results = []
    
    if len(image_paths) > 1 and args.parallel:
        # Parallel processing mode
        print(f"\n🚀 Starting parallel processing of {len(image_paths)} images with {args.max_workers} workers...")
        results = process_images_parallel(
            predictor=predictor,
            image_paths=image_paths,
            transcript_path=args.transcript_path,
            metadata_path=args.metadata_path,
            serpapi_key=serpapi_key,
            imgbb_key=imgbb_key,
            max_workers=args.max_workers
        )
    else:
        # Sequential processing mode
        print(f"\n🔄 Starting sequential processing of {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {Path(image_path).name}")
            
            result = process_single_image(
                predictor=predictor,
                image_path=image_path,
                transcript_path=args.transcript_path,
                metadata_path=args.metadata_path,
                serpapi_key=serpapi_key,
                imgbb_key=imgbb_key,
                thread_id=0
            )
            results.append(result)
    
    # Save results to CSV
    csv_path = r"C:\Users\tungd\OneDrive - MSFT\Second Year\ML\ACMMM25 - Grand Challenge on Multimedia Verification\G3-Original\g3\results\validation.csv"
    save_results_to_csv(results, csv_path)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETED")
    print("="*60)
    
    if len(image_paths) == 1:
        # Single image - show detailed result
        result = results[0]
        if result['status'] == 'success':
            print(f"📍 Predicted Location: {result['latitude']}, {result['longitude']}")
            print(f"📋 Place: {result['location']}")
            print(f"💭 Analysis: {result['reason']}")
            print(f"🔗 Links: {result['links']}")
        else:
            print(f"❌ Processing failed: {result['reason']}")
    else:
        # Multiple images - show summary
        success_results = [r for r in results if r['status'] == 'success']
        error_results = [r for r in results if r['status'] == 'error']
        
        print(f"📊 Total images processed: {len(results)}")
        print(f"✅ Successful predictions: {len(success_results)}")
        print(f"❌ Failed predictions: {len(error_results)}")
        
        if success_results:
            print(f"\n🎯 Sample successful predictions:")
            for result in success_results[:3]:  # Show first 3 successful results
                print(f"   {result['vid_name']}: {result['latitude']}, {result['longitude']} - {result['location']}")
        
        if error_results:
            print(f"\n⚠️ Failed images:")
            for result in error_results[:5]:  # Show first 5 failed results
                print(f"   {result['vid_name']}: {result['reason']}")
    
    print("="*60)