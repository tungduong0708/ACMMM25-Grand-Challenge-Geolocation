import base64
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests
from google.cloud import vision
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("uvicorn.error")

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

    response = client.annotate_image(
        {
            "image": image,
            "features": [{"type_": vision.Feature.Type.WEB_DETECTION}],
        }
    )
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
        if os.path.isfile(file_path) and file_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        ):
            image_files.append(file_path)

    all_web_detections = []
    batch_size = 16  # Google Vision API batch limit

    # Process images in batches of 16
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i : i + batch_size]
        logger.info(
            f"Processing batch {i // batch_size + 1}/{(len(image_files) + batch_size - 1) // batch_size} ({len(batch_files)} images)..."
        )

        # Prepare batch requests
        image_requests = []
        for file_path in batch_files:
            try:
                with open(file_path, "rb") as image_file:
                    content = image_file.read()
                    image = vision.Image(content=content)
                    image_requests.append(image)
            except Exception as e:
                logger.warning(f"⚠️ Failed to read image {file_path}: {e}")
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
            logger.warning(f"⚠️ No valid images in batch {i // batch_size + 1}")
            continue

        try:
            # Make batch API call
            responses = client.batch_annotate_images(
                requests=[
                    vision.AnnotateImageRequest(
                        image=image,
                        features=[
                            vision.Feature(type=vision.Feature.Type.WEB_DETECTION)
                        ],
                    )
                    for image in valid_requests
                ]
            ).responses

            # Process responses and maintain order
            batch_detections: list[vision.WebDetection | None] = [None] * len(
                batch_files
            )
            for response_idx, global_idx in enumerate(valid_indices):
                if (
                    response_idx < len(responses)
                    and responses[response_idx].web_detection
                ):
                    batch_detections[global_idx] = responses[response_idx].web_detection

            # Add to results (filter out None values)
            all_web_detections.extend(
                [det for det in batch_detections if det is not None]
            )

        except Exception as e:
            logger.warning(f"⚠️ Batch {i // batch_size + 1} failed: {e}")
            continue

    logger.info(
        f"✅ Successfully processed {len(all_web_detections)} images out of {len(image_files)} total"
    )
    return all_web_detections


def parse_web_detection(annotations: vision.WebDetection) -> dict:
    """Returns detected features in the provided web annotations as a dict."""
    result = {
        "pages_with_matching_images": [],
        "full_matching_images": [],
        "partial_matching_images": [],
        "web_entities": [],
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
            result["web_entities"].append(
                {"score": entity.score, "description": entity.description}
            )
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
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise Exception(f"Error reading image file: {e}")

    payload = {"key": api_key, "image": image_data}
    imgbb_url = "https://api.imgbb.com/1/upload"

    # Configure session with retry logic
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        resp = session.post(imgbb_url, data=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if result.get("success"):
            return result["data"]["url"]
        else:
            raise Exception(
                f"imgbb upload failed: {result.get('error', 'Unknown error')}"
            )
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to upload after retries: {e}")


def search_with_scrapingdog_lens(
    image_path: str, imgbb_key: str, scrapingdog_key: str
) -> dict:
    """
    Uploads an image to imgbb, then queries ScrapingDog's Google Lens API with 3 retries.
    """
    try:
        image_url = upload_image_to_imgbb(image_path, imgbb_key)
        logger.info(f"Image uploaded to ImgBB: {image_url}")

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
                resp = requests.get(
                    "https://api.scrapingdog.com/google_lens", params=params, timeout=60
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"⚠️ ScrapingDog attempt {attempt + 1}/3 failed for {os.path.basename(image_path)}: {e}"
                )
                if attempt < 2:  # Don't sleep on the last attempt
                    time.sleep(2)  # Wait 2 seconds before retrying
                continue

        # All retries failed
        logger.error(
            f"❌ All 3 ScrapingDog attempts failed for {os.path.basename(image_path)}"
        )
        return {"lens_results": []}

    except Exception as e:
        logger.warning(f"⚠️ ScrapingDog API unexpected error for {image_path}: {e}")
        return {"lens_results": []}


def get_image_links_scrapingdog(search_results: dict, n_results: int = 5) -> list[str]:
    """Get image links from Scrapingdog Lens API."""
    return [result["link"] for result in search_results.get("lens_results", [])][
        :n_results
    ]


def process_scrapingdog_only(image_path: str) -> list[str]:
    """Process a single image with ScrapingDog API only."""
    try:
        scrapingdog_search_result = search_with_scrapingdog_lens(
            image_path=image_path,
            imgbb_key=os.environ["IMGBB_API_KEY"],
            scrapingdog_key=os.environ["SCRAPINGDOG_API_KEY"],
        )
        scrapingdog_result = get_image_links_scrapingdog(
            scrapingdog_search_result, n_results=5
        )

        with print_lock:
            logger.info(
                f"✅ ScrapingDog completed for {os.path.basename(image_path)} - {len(scrapingdog_result)} links"
            )

        return scrapingdog_result
    except Exception as e:
        with print_lock:
            logger.error(
                f"❌ ScrapingDog error for {os.path.basename(image_path)}: {e}"
            )
        return []


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
            image_path=image_path, imgbb_key=imgbb_key, scrapingdog_key=scrapingdog_key
        )
        scrapingdog_result = get_image_links_scrapingdog(
            scrapingdog_search_result, n_results=5
        )
        # scrapingdog_result = []

        result = {
            "image_path": os.path.basename(image_path),
            "vision_result": vision_result,
            "scrapingdog_result": scrapingdog_result,
        }

        with print_lock:
            logger.info(f"✅ Completed processing {os.path.basename(image_path)}")

        return result

    except Exception as e:
        with print_lock:
            logger.error(f"❌ Error processing {os.path.basename(image_path)}: {e}")
        return {
            "image_path": os.path.basename(image_path),
            "vision_result": [],
            "scrapingdog_result": [],
            "error": str(e),
        }


def image_search_directory(
    directory: str,
    output_dir: str = "g3/data/prompt_data",
    filename: str = "image_search.json",
    imgbb_key: str = "YOUR_IMGBB_API_KEY",
    scrapingdog_key: str = "YOUR_SCRAPINGDOG_API_KEY",
    max_workers: int = 4,
    target_links: int = 20,
) -> None:
    """
    Perform web detection with a two-phase approach:
    1. Run Vision API on all images first using annotate_directory
    2. If total unique links < target_links, run ScrapingDog on images until target is reached

    Args:
        directory (str): Path to the directory containing image files.
        output_dir (str): Directory to save the JSON output.
        filename (str): Name of the JSON file to save the results.
        imgbb_key (str): ImgBB API key for image uploading.
        scrapingdog_key (str): ScrapingDog API key for lens search.
        max_workers (int): Maximum number of parallel workers.
        target_links (int): Target number of unique links to collect.

    Returns:
        None
    """
    EXCLUDE_DOMAIN = [
        "youtube.com",
    ]
    # Get all image files
    image_files = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        ):
            image_files.append(file_path)

    if not image_files:
        logger.info("No image files found in the directory.")
        return

    logger.info(
        f"Found {len(image_files)} image files. Target: {target_links} unique links"
    )

    # Phase 1: Run Vision API on all images using annotate_directory
    logger.info("🔍 Phase 1: Running Vision API on all images...")
    all_links = set()
    vision_links_count = 0

    try:
        # Use the existing annotate_directory function for batch processing
        web_detections = annotate_directory(directory)

        # Extract links from all web detections
        for detection in web_detections:
            links = get_image_links_vision(detection)
            # Filter out links from excluded domains
            links = [
                link for link in links if not any(domain in link for domain in EXCLUDE_DOMAIN)
            ]
            all_links.update(links)  # Add to set (automatically deduplicates)

        vision_links_count = len(all_links)
        logger.info(
            f"✅ Phase 1 complete: {vision_links_count} unique links from Vision API"
        )

    except Exception as e:
        logger.error(f"❌ Vision API processing failed: {e}")
        all_links = set()
        vision_links_count = 0

    # Phase 2: Run ScrapingDog if needed
    scrapingdog_links_count = 0

    if len(all_links) < target_links:
        needed_links = target_links - len(all_links)
        logger.info(
            f"🔍 Phase 2: Need {needed_links} more links. Running ScrapingDog..."
        )

        # Check if API keys are available
        if (
            imgbb_key == "YOUR_IMGBB_API_KEY"
            or scrapingdog_key == "YOUR_SCRAPINGDOG_API_KEY"
        ):
            logger.warning("⚠️ ScrapingDog API keys not available. Skipping Phase 2.")
        else:
            scrapingdog_completed = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit ScrapingDog tasks for all images
                future_to_image = {
                    executor.submit(process_scrapingdog_only, image_path): image_path
                    for image_path in image_files
                }

                # Collect ScrapingDog results until we have enough links
                for future in as_completed(future_to_image):
                    image_path = future_to_image[future]
                    try:
                        result_links = future.result()
                        filtered_links = [
                            link for link in result_links if not any(domain in link for domain in EXCLUDE_DOMAIN)
                        ]
                        initial_count = len(filtered_links)
                        all_links.update(filtered_links)  # Add new links to the main set
                        scrapingdog_links_count += (
                            len(all_links) - initial_count
                        )  # Count new unique links added
                        scrapingdog_completed += 1

                        with print_lock:
                            logger.info(
                                f"ScrapingDog Progress: {scrapingdog_completed}/{len(image_files)} images, "
                                f"{scrapingdog_links_count} new ScrapingDog links, {len(all_links)} total unique"
                            )

                        # Stop early if we have enough links
                        if len(all_links) >= target_links:
                            logger.info(
                                f"🎯 Target reached! {len(all_links)} >= {target_links} links"
                            )
                            # Cancel remaining futures
                            for remaining_future in future_to_image:
                                if not remaining_future.done():
                                    remaining_future.cancel()
                            break

                    except Exception as e:
                        with print_lock:
                            logger.error(
                                f"❌ Failed ScrapingDog for {os.path.basename(image_path)}: {e}"
                            )
                        scrapingdog_completed += 1

    # Prepare final results
    total_unique_links = len(all_links)
    all_links = list(all_links)[:target_links]
    results = {
        "all_links": all_links,
        "total_unique_links": total_unique_links,
        "target_achieved": total_unique_links >= target_links,
        "summary": {
            "images_processed": len(image_files),
            "vision_links": vision_links_count,
            "scrapingdog_links": scrapingdog_links_count,
            "total_unique_links": total_unique_links,
            "target_links": target_links,
        },
    }

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save results to JSON file
    out_path = Path(output_dir) / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(
        f"✅ Saved results to {out_path}\n"
        f"📊 Summary: {vision_links_count} Vision + {scrapingdog_links_count} ScrapingDog = {total_unique_links} total unique links"
    )
