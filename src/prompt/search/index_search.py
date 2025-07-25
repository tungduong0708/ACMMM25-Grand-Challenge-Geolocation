import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import torch
from PIL import Image

logger = logging.getLogger("uvicorn.error")

# Thread-safe lock for logging
print_lock = Lock()


def search_index(model, rgb_image, device, index, top_k=20):
    """
    Search FAISS index for similar and dissimilar coordinates using image embeddings.

    Args:
        model: Vision model used for embedding generation.
        rgb_image: PIL RGB Image.
        device: Device to run the model on (e.g., "cuda" or "cpu").
        index: FAISS index for searching.
        top_k (int): Number of top results to return.

    Returns:
        tuple: (D, I, D_reverse, I_reverse) - distances and indices for positive and negative embeddings.
    """
    # logger.info("Searching FAISS index...")
    image = model.vision_processor(images=rgb_image, return_tensors="pt")[
        "pixel_values"
    ].reshape(-1, 224, 224)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        vision_output = model.vision_model(image)[1]
        image_embeds = model.vision_projection(vision_output)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        image_text_embeds = model.vision_projection_else_1(
            model.vision_projection(vision_output)
        )
        image_text_embeds = image_text_embeds / image_text_embeds.norm(
            p=2, dim=-1, keepdim=True
        )

        image_location_embeds = model.vision_projection_else_2(
            model.vision_projection(vision_output)
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

    # Search FAISS index
    D, I = index.search(positive_image_embeds, top_k)
    D_reverse, I_reverse = index.search(negative_image_embeds, top_k)
    return D, I, D_reverse, I_reverse


def get_gps_coordinates(I, I_reverse, database_csv_path):
    """
    Helper method to get GPS coordinates from database using FAISS indices.

    Args:
        I: FAISS indices for positive embeddings
        I_reverse: FAISS indices for negative embeddings
        database_csv_path (str): Path to GPS coordinates database CSV

    Returns:
        tuple: (candidates_gps, reverse_gps) - lists of (lat, lon) tuples
    """
    if I is None or I_reverse is None:
        return [], []

    candidate_indices = I[0]
    reverse_indices = I_reverse[0]

    candidates_gps = []
    reverse_gps = []

    try:
        for chunk in pd.read_csv(
            database_csv_path, chunksize=10000, usecols=["LAT", "LON"]
        ):
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
    except Exception as e:
        logger.error(f"⚠️ Error loading GPS coordinates from database: {e}")

    return candidates_gps, reverse_gps


def save_results_to_json(candidates_gps: list, reverse_gps: list, output_path: str):
    """
    Save search results to a JSON file.

    Args:
        results (dict): Search results to save.
        output_path (str): Path to the output JSON file.
    """
    results = {"candidates_gps": candidates_gps, "reverse_gps": reverse_gps}
    with open(output_path, "w") as json_file:
        json.dump(results, json_file, indent=4)


def process_single_image(image_path, model, device, index, database_csv_path, top_k=20):
    """
    Process a single image for index search.

    Args:
        image_path: Path to the image file
        model: Vision model used for embedding generation
        device: Device to run the model on
        index: FAISS index for searching
        database_csv_path: Path to GPS coordinates database CSV
        top_k: Number of top results to return

    Returns:
        tuple: (candidates_gps, reverse_gps) for this image
    """
    try:
        rgb_image = Image.open(image_path).convert("RGB")
        D, I, D_reverse, I_reverse = search_index(
            model, rgb_image, device, index, top_k
        )
        candidates_gps, reverse_gps = get_gps_coordinates(
            I, I_reverse, database_csv_path
        )

        # with print_lock:
        #     logger.info(
        #         f"✅ Processed {os.path.basename(image_path)}: {len(candidates_gps)} candidates, {len(reverse_gps)} reverse"
        #     )

        return candidates_gps, reverse_gps
    except Exception as e:
        with print_lock:
            logger.error(f"❌ Error processing {os.path.basename(image_path)}: {e}")
        return [], []


def search_index_directory(
    model,
    device,
    index,
    image_dir,
    database_csv_path,
    top_k=20,
    max_elements=20,
    max_workers=4,
):
    """
    Perform FAISS index search for all images in a directory in parallel and gradually build a prioritized set of candidates.

    Args:
        model: Vision model used for embedding generation.
        device: Device to run the model on (e.g., "cuda" or "cpu").
        index: FAISS index for searching.
        image_dir (str): Path to the directory containing images.
        database_csv_path (str): Path to GPS coordinates database CSV.
        top_k (int): Number of top results to return for each image.
        max_elements (int): Maximum number of elements in the final candidates set.
        max_workers (int): Maximum number of parallel workers.

    Returns:
        tuple: (candidates_gps, reverse_gps) - lists of (lat, lon) tuples.
    """
    # Get all image paths
    image_paths = [
        Path(image_dir) / img
        for img in os.listdir(image_dir)
        if img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not image_paths:
        logger.warning("No images found in directory")
        return [], []

    logger.info(
        f"🚀 Processing {len(image_paths)} images with {max_workers} parallel workers..."
    )

    all_candidates_gps = []
    all_reverse_gps = []
    completed_count = 0

    # Process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(
                process_single_image,
                image_path,
                model,
                device,
                index,
                database_csv_path,
                top_k,
            ): image_path
            for image_path in image_paths
        }

        # Collect results as they complete
        for future in as_completed(future_to_path):
            image_path = future_to_path[future]
            try:
                candidates_gps, reverse_gps = future.result()
                all_candidates_gps.append(candidates_gps)
                all_reverse_gps.append(reverse_gps)
                completed_count += 1

                with print_lock:
                    logger.info(
                        f"Progress: {completed_count}/{len(image_paths)} images completed"
                    )

            except Exception as e:
                with print_lock:
                    logger.error(
                        f"❌ Failed to process {os.path.basename(image_path)}: {e}"
                    )
                # Add empty results for failed images
                all_candidates_gps.append([])
                all_reverse_gps.append([])
                completed_count += 1

    # Build prioritized sets from all results
    candidates_gps = set()
    reverse_gps = set()

    for priority in range(top_k):
        for image_candidates_gps, image_reverse_gps in zip(
            all_candidates_gps, all_reverse_gps
        ):
            if len(candidates_gps) < max_elements and priority < len(
                image_candidates_gps
            ):
                candidates_gps.add(image_candidates_gps[priority])
            if len(reverse_gps) < max_elements and priority < len(image_reverse_gps):
                reverse_gps.add(image_reverse_gps[priority])

            if len(candidates_gps) >= max_elements and len(reverse_gps) >= max_elements:
                break

    logger.info(
        f"🎯 Final results: {len(candidates_gps)} candidates, {len(reverse_gps)} reverse GPS coordinates"
    )

    return list(candidates_gps), list(reverse_gps)
