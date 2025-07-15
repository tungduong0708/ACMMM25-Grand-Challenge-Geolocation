import argparse
import json
import os
from pathlib import Path
import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Union
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from google.cloud import vision

load_dotenv()

# GOOGLE CLOUD VISION API

def annotate(path: str) -> vision.WebDetection:
    """Returns web annotations given the path to an image.

    Args:
        path: path to the input image.

    Returns:
        An WebDetection object with relevant information of the
        image from the internet (i.e., the annotations).
    """
    client = vision.ImageAnnotatorClient()

    if path.startswith("http") or path.startswith("gs:"):
        image = vision.Image()
        image.source.image_uri = path

    else:
        with open(path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

    response = client.annotate_image({
        'image': image,
        'features': [{'type_': vision.Feature.Type.WEB_DETECTION}],
    })
    return response.web_detection

def annotate_directory(directory: str) -> list[vision.WebDetection]:
    """
    Perform web detection on all image files in the given directory in batches of 16.

    Args:
        directory (str): Path to the directory containing image files.

    Returns:
        list[vision.WebDetection]: List of WebDetection objects for each image.
    """
    client = vision.ImageAnnotatorClient()
    
    # Collect all image files first
    image_files = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            image_files.append(file_path)
    
    all_web_detections = []
    batch_size = 16  # Google Vision API batch limit
    
    # Process images in batches of 16
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size} ({len(batch_files)} images)...")
        
        # Prepare batch requests
        image_requests = []
        for file_path in batch_files:
            try:
                with open(file_path, "rb") as image_file:
                    content = image_file.read()
                    image = vision.Image(content=content)
                    image_requests.append(image)
            except Exception as e:
                print(f"⚠️ Warning: Failed to read image {file_path}: {e}")
                # Add a placeholder to maintain order
                image_requests.append(None)
        
        # Filter out None values and keep track of valid indices
        valid_requests = []
        valid_indices = []
        for idx, request in enumerate(image_requests):
            if request is not None:
                valid_requests.append(request)
                valid_indices.append(idx)
        
        if not valid_requests:
            print(f"⚠️ Warning: No valid images in batch {i//batch_size + 1}")
            continue
        
        try:
            # Make batch API call
            responses = client.batch_annotate_images(
                requests=[
                    vision.AnnotateImageRequest(
                        image=image, 
                        features=[vision.Feature(type=vision.Feature.Type.WEB_DETECTION)]
                    ) for image in valid_requests
                ]
            ).responses
            
            # Process responses and maintain order
            batch_detections: list[vision.WebDetection | None] = [None] * len(batch_files)
            for response_idx, global_idx in enumerate(valid_indices):
                if response_idx < len(responses) and responses[response_idx].web_detection:
                    batch_detections[global_idx] = responses[response_idx].web_detection
            
            # Add to results (filter out None values)
            all_web_detections.extend([det for det in batch_detections if det is not None])
            
        except Exception as e:
            print(f"⚠️ Warning: Batch {i//batch_size + 1} failed: {e}")
            continue
    
    print(f"✅ Successfully processed {len(all_web_detections)} images out of {len(image_files)} total")
    return all_web_detections


def parse_web_detection(annotations: vision.WebDetection) -> dict:
    """Returns detected features in the provided web annotations as a dict."""
    result = {
        "pages_with_matching_images": [],
        "full_matching_images": [],
        "partial_matching_images": [],
        "web_entities": []
    }
    if annotations.pages_with_matching_images:
        for page in annotations.pages_with_matching_images:
            result["pages_with_matching_images"].append(page.url)
    if annotations.full_matching_images:
        for image in annotations.full_matching_images:
            result["full_matching_images"].append(image.url)
    if annotations.partial_matching_images:
        for image in annotations.partial_matching_images:
            result["partial_matching_images"].append(image.url)
    if annotations.web_entities:
        for entity in annotations.web_entities:
            result["web_entities"].append({
                "score": entity.score,
                "description": entity.description
            })
    return result

def get_image_links_vision(annotations: vision.WebDetection) -> list[str]:
    """Extracts image links from web detection annotations."""
    links = []
    if annotations.pages_with_matching_images:
        for page in annotations.pages_with_matching_images:
            links.append(page.url)
    if not links and annotations.full_matching_images:
        # Fallback to full matching images if no pages found
        for image in annotations.full_matching_images:
            links.append(image.url)
    if not links and annotations.partial_matching_images:
        # Fallback to partial matching images if no full matches found
        for image in annotations.partial_matching_images:
            links.append(image.url)
    return links

# SCRAPING DOG API
def upload_image_to_imgbb(image_path: str, api_key: str) -> str:
    """Upload image to imgbb with automatic retry on transient errors."""

    # Encode the image
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise Exception(f"Error reading image file: {e}")

    payload = {'key': api_key, 'image': image_data}
    imgbb_url = "https://api.imgbb.com/1/upload"

    # Configure session with retry logic
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        resp = session.post(imgbb_url, data=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if result.get('success'):
            return result['data']['url']
        else:
            raise Exception(f"imgbb upload failed: {result.get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to upload after retries: {e}")

def search_with_scrapingdog_lens(
    image_path: str,
    imgbb_key: str,
    scrapingdog_key: str
) -> dict:
    """
    Uploads an image to imgbb, then queries ScrapingDog's Google Lens API with 3 retries.
    """
    try:
        image_url = upload_image_to_imgbb(image_path, imgbb_key)
        print(f"Image uploaded to ImgBB: {image_url}")

        lens_url = f"https://lens.google.com/uploadbyurl?url={image_url}"
        params = {
            "api_key": scrapingdog_key,
            "url": lens_url,
            "visual_matches": "true",
            "exact_matches": "true",
        }

        # Retry logic - 3 attempts
        for attempt in range(3):
            try:
                resp = requests.get("https://api.scrapingdog.com/google_lens", params=params, timeout=60)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                print(f"⚠️ ScrapingDog attempt {attempt + 1}/3 failed for {os.path.basename(image_path)}: {e}")
                if attempt < 2:  # Don't sleep on the last attempt
                    time.sleep(2)  # Wait 2 seconds before retrying
                continue
        
        # All retries failed
        print(f"❌ All 3 ScrapingDog attempts failed for {os.path.basename(image_path)}")
        return {"lens_results": []}
        
    except Exception as e:
        print(f"⚠️ ScrapingDog API unexpected error for {image_path}: {e}")
        return {"lens_results": []}


def get_image_links_scrapingdog(search_results: dict, n_results: int = 5) -> list[str]:
    """Get image links from Scrapingdog Lens API."""
    return [result['link'] for result in search_results.get('lens_results', [])][:n_results]

# Thread-safe print lock
print_lock = Lock()

def process_single_image(image_path: str, imgbb_key: str, scrapingdog_key: str) -> dict:
    """
    Process a single image with both Vision API and ScrapingDog API.
    
    Args:
        image_path: Path to the image file
        imgbb_key: ImgBB API key
        scrapingdog_key: ScrapingDog API key
        
    Returns:
        Dictionary containing the results for this image
    """
    try:
        # Vision API processing
        annotations = annotate(image_path)
        vision_result = get_image_links_vision(annotations)
        
        # ScrapingDog API processing
        scrapingdog_search_result = search_with_scrapingdog_lens(
            image_path=image_path, 
            imgbb_key=imgbb_key, 
            scrapingdog_key=scrapingdog_key
        )
        scrapingdog_result = get_image_links_scrapingdog(scrapingdog_search_result, n_results=5)
        # scrapingdog_result = []
        
        result = {
            "image_path": os.path.basename(image_path),
            "vision_result": vision_result,
            "scrapingdog_result": scrapingdog_result
        }
        
        with print_lock:
            print(f"✅ Completed processing {os.path.basename(image_path)}")
        
        return result
        
    except Exception as e:
        with print_lock:
            print(f"❌ Error processing {os.path.basename(image_path)}: {e}")
        return {
            "image_path": os.path.basename(image_path),
            "vision_result": [],
            "scrapingdog_result": [],
            "error": str(e)
        }

def image_search_directory(
        directory: str, 
        output_dir: str = "g3/data/prompt_data", 
        filename: str = "image_search.json",
        imgbb_key: str = "YOUR_IMGBB_API_KEY",
        scrapingdog_key: str = "YOUR_SCRAPINGDOG_API_KEY",
        max_workers: int = 4
    ) -> None:
    """
    Perform web detection on all image files in the given directory in parallel,
    and save them to a single JSON file in the specified output directory.

    Args:
        directory (str): Path to the directory containing image files.
        output_dir (str): Directory to save the JSON output.
        filename (str): Name of the JSON file to save the results.
        imgbb_key (str): ImgBB API key for image uploading.
        scrapingdog_key (str): ScrapingDog API key for lens search.
        max_workers (int): Maximum number of parallel workers.

    Returns:
        None
    """
    # Get all image files
    image_files = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            image_files.append(file_path)
    
    if not image_files:
        print("No image files found in the directory.")
        return
    
    print(f"Found {len(image_files)} image files. Processing with {max_workers} parallel workers...")
    
    search_results = []
    completed_count = 0
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_single_image, image_path, imgbb_key, scrapingdog_key): image_path 
            for image_path in image_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                result = future.result()
                search_results.append(result)
                completed_count += 1
                
                with print_lock:
                    print(f"Progress: {completed_count}/{len(image_files)} images completed")
                    
            except Exception as e:
                with print_lock:
                    print(f"❌ Failed to process {os.path.basename(image_path)}: {e}")
                # Add error result
                search_results.append({
                    "image_path": os.path.basename(image_path),
                    "vision_result": [],
                    "scrapingdog_result": [],
                    "error": str(e)
                })
    
    # Sort results by image path for consistent ordering
    search_results.sort(key=lambda x: x["image_path"])

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save parsed results to JSON file
    out_path = Path(output_dir) / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(search_results, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(search_results)} results to {out_path}")


if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\tungd\\OneDrive - MSFT\\Second Year\\ML\\ACMMM25 - Grand Challenge on Multimedia Verification\\G3-Original\\acmmm2025-grand-challenge-gg-credentials.json"
    parser = argparse.ArgumentParser(
        description="Perform web detection on a single image or all images in a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_path", help="Path to an image file or a directory containing image files.")
    parser.add_argument("--output_dir", type=str, default="g3/data/prompt_data", help="Directory to save JSON output")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers (default: 4)")
    args = parser.parse_args()

    if os.path.isdir(args.input_path):
        imgbb_key = os.getenv("IMGBB_API_KEY", "YOUR_IMGBB_API_KEY")
        scrapingdog_key = os.getenv("SCRAPINGDOG_API_KEY", "YOUR_SCRAPINGDOG_API_KEY")

        # Check if API keys are available
        if imgbb_key == "YOUR_IMGBB_API_KEY" or scrapingdog_key == "YOUR_SCRAPINGDOG_API_KEY":
            print("Warning: ImgBB and/or ScrapingDog API keys not found in environment variables.")
            print("ScrapingDog search will be skipped. Only Google Vision API results will be available.")
            print("To enable ScrapingDog search, set IMGBB_API_KEY and SCRAPINGDOG_API_KEY environment variables.")
        
        image_search_directory(
            directory=args.input_path,
            output_dir=args.output_dir,
            imgbb_key=imgbb_key,
            scrapingdog_key=scrapingdog_key,
            max_workers=args.max_workers
        )
    else:
        annotations = annotate(args.input_path)
        parsed_result = parse_web_detection(annotations)

        # Ensure the output directory exists
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        print(json.dumps(parsed_result, indent=2, ensure_ascii=False))

        # # Save parsed result to JSON file
        # out_path = Path(args.output_dir) / "image_search.json"
        # with open(out_path, "w", encoding="utf-8") as f:
        #     json.dump([parsed_result], f, ensure_ascii=False, indent=2)

        # print(f"Saved result to {out_path}")