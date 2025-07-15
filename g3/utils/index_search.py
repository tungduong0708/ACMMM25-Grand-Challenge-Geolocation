import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import os
from PIL import Image

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
    print("Searching FAISS index...")
    image = model.vision_processor(images=rgb_image, return_tensors="pt")["pixel_values"].reshape(-1, 224, 224)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        vision_output = model.vision_model(image)[1]
        image_embeds = model.vision_projection(vision_output)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        image_text_embeds = model.vision_projection_else_1(model.vision_projection(vision_output))
        image_text_embeds = image_text_embeds / image_text_embeds.norm(p=2, dim=-1, keepdim=True)

        image_location_embeds = model.vision_projection_else_2(model.vision_projection(vision_output))
        image_location_embeds = image_location_embeds / image_location_embeds.norm(p=2, dim=-1, keepdim=True)

        positive_image_embeds = torch.cat([image_embeds, image_text_embeds, image_location_embeds], dim=1)
        positive_image_embeds = positive_image_embeds.cpu().detach().numpy().astype(np.float32)

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
    except Exception as e:
        print(f"⚠️ Error loading GPS coordinates from database: {e}")

    return candidates_gps, reverse_gps


def save_results_to_json(candidates_gps: list, reverse_gps: list, output_path: str):
    """
    Save search results to a JSON file.

    Args:
        results (dict): Search results to save.
        output_path (str): Path to the output JSON file.
    """
    results = {
        "candidates_gps": candidates_gps,
        "reverse_gps": reverse_gps
    }
    with open(output_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

def search_index_directory(model, device, index, image_dir, database_csv_path, top_k=20, max_elements=20):
    """
    Perform FAISS index search for all images in a directory and gradually build a prioritized set of candidates.

    Args:
        model: Vision model used for embedding generation.
        device: Device to run the model on (e.g., "cuda" or "cpu").
        index: FAISS index for searching.
        image_dir (str): Path to the directory containing images.
        database_csv_path (str): Path to GPS coordinates database CSV.
        top_k (int): Number of top results to return for each image.
        max_elements (int): Maximum number of elements in the final candidates set.

    Returns:
        tuple: (candidates_gps, reverse_gps) - lists of (lat, lon) tuples.
    """
    all_candidates_gps = []
    all_reverse_gps = []

    image_paths = [Path(image_dir) / img for img in os.listdir(image_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for image_path in image_paths:
        rgb_image = Image.open(image_path).convert("RGB")
        D, I, D_reverse, I_reverse = search_index(model, rgb_image, device, index, top_k)
        image_candidates_gps, image_reverse_gps = get_gps_coordinates(I, I_reverse, database_csv_path)
        all_candidates_gps.append(image_candidates_gps)
        all_reverse_gps.append(image_reverse_gps)

    candidates_gps = set()
    reverse_gps = set()

    for priority in range(top_k):
        for image_candidates_gps, image_reverse_gps in zip(all_candidates_gps, all_reverse_gps):
            if len(candidates_gps) < max_elements and priority < len(image_candidates_gps):
                candidates_gps.add(image_candidates_gps[priority])
            if len(reverse_gps) < max_elements and priority < len(image_reverse_gps):
                reverse_gps.add(image_reverse_gps[priority])

            if len(candidates_gps) >= max_elements and len(reverse_gps) >= max_elements:
                break

    return list(candidates_gps), list(reverse_gps)

if __name__ == "__main__":
    # Example usage
    model = None  # Replace with actual model instance
    rgb_img = None  # Replace with actual PIL RGB image
    device = "cuda"  # Replace with actual device
    index = None  # Replace with actual FAISS index
    top_k = 20
    database_csv_path = "path/to/database.csv"

    D, I, D_reverse, I_reverse = search_index(model, rgb_img, device, index, top_k)
    candidates_gps, reverse_gps = get_gps_coordinates(I, I_reverse, database_csv_path)

    results = {
        "candidates_gps": candidates_gps,
        "reverse_gps": reverse_gps
    }

    output_path = "search_results.json"
    save_results_to_json(results, output_path)

    # Directory search with priority example
    prioritized_candidates_gps, prioritized_reverse_gps = search_index_directory(model, device, index, image_directory, database_csv_path, top_k, max_elements=100)

    prioritized_results = {
        "candidates_gps": prioritized_candidates_gps,
        "reverse_gps": prioritized_reverse_gps
    }

    prioritized_output_path = "prioritized_directory_search_results.json"
    save_results_to_json(prioritized_results, prioritized_output_path)