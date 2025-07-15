import json
import os
from typing import Optional

SINGLE_PROMPT = (
    '''
    You are an expert in geo-localization. Analyze the image and determine the most precise possible location—ideally identifying the exact building, landmark, or facility, not just the city. 
    Examine all provided content links in detail, using both textual and visual clues to support your conclusion. 
    Use only the provided links for evidence. Any additional links must directly support specific visual observations (e.g., satellite imagery or publicly available street-level photos of the same location). 
    Return your final answer as geographic coordinates.

    {prompt_data}

    Respond with **only** the following JSON structure (no extra text, markdown, or comments):

    {
        "latitude": float,
        "longitude": float,
        "location": string,
        "evidence": [
            {
                "analysis": string,
                "references": [string, …]
            }
        ]
    }

    **Guidelines:**
    - Each object in the "evidence" list should explain a single textual or visual clue.  
    - The "analysis" field must describe the clue and reference one or more supporting sources using bracketed indices like [1], [2], etc.  
    - The corresponding URLs for those references must be included in the "links" list for that object.  
    - Use textual/news URLs for contextual clues and satellite/image URLs for visual clues.  
    - Do **not** include any links that are not explicitly cited in the "analysis".  
    - Maintain the order of evidence to match the sequence in which clues are introduced.  
    - The combination of all "analysis" fields should make it clear how the clues lead to the final coordinates, without revealing intermediate reasoning or metadata.  
    - **Do not use metadata** (e.g., EXIF data, filenames, author handles, timestamps, or embedded properties) as part of the analysis or evidence.
    '''
)

BATCH_IMAGE_TEXT_PROMPT = (
    '''
    You are an expert in geo-localization. Analyze the image and determine the most precise possible location—ideally identifying the exact building, landmark, or facility, not just the city. 
    Examine all provided content links in detail, using both textual and visual clues to support your conclusion. 
    Use only the provided links for evidence. Any additional links must directly support specific visual observations (e.g., satellite imagery or publicly available street-level photos of the same location). 
    Return your final answer as geographic coordinates.

    {prompt_data}

    Respond with **only** the following JSON structure (no extra text, markdown, or comments):

    {
        "latitude": float,
        "longitude": float,
        "location": string,
        "evidence": [
            {
                "analysis": string,
                "references": [string, …]
            }
        ]
    }

    **Guidelines:**
    - One entry per clue (visual and textual).  
    - Each object in the "evidence" list should explain a single textual or visual clue and be as many as possible. All image in the prompt follow the format: "image_{idx:03d}.jpg", starting from image_000.jpg.
    - In the "references" list, each element must be a URL or an image file name (e.g., "image_000.jpg"). They are marked with indices like [1], [2], etc in order of appearance in "references" list. "Analysis" must use these indices to cite the corresponding references.
    - The "analysis" field must describe the clue and cite reference in its corresponding "references" using bracketed indices like [1], [2], etc. The corresponding URLs or images for those references must be included in the "references" list for that object.  
        + For contextual evidence, must cite textual/news URLs.
        + For visual clues, cite `image_{idx:03d}.jpg` in `references` and any satellite/map URLs as needed.
    - MUST use given links to support the analysis.
    - If you can’t identify a specific building, give the city‑center coordinates.
    '''
)

BATCH_TEXT_PROMPT = (
    '''
    You are an expert in geo-localization. Analyze the data and determine the most precise possible location—ideally identifying the exact building, landmark, or facility, not just the city. 
    Examine all provided content links in detail, using textual clues to support your conclusion. 
    Use only the provided links for evidence. 
    Return your final answer as geographic coordinates.

    {prompt_data}

    Respond with **only** the following JSON structure (no extra text, markdown, or comments):

    {
        "latitude": float,
        "longitude": float,
        "location": string,
        "evidence": [
            {
                "analysis": string,
                "references": [string, …]
            }
        ]
    }

    **Guidelines:**
    - One entry per textual clue.  
    - Each object in the "evidence" list should explain a single textual clue and be as many as possible.
    - In the "references" list, each element must be a URL. They are marked with indices like [1], [2], etc in order of appearance in "references" list. "Analysis" must use these indices to cite the corresponding references.
    - The "analysis" field must describe the clue and cite reference in its corresponding "references" using bracketed indices like [1], [2], etc. The corresponding URLs for those references must be included in the "references" list for that object. For contextual evidence, must cite textual/news URLs.
    - MUST use given links to support the analysis.
    - If you can’t identify a specific building, give the city‑center coordinates.
    '''
)

BATCH_IMAGE_PROMPT = (
    '''
    You are an expert in geo-localization. Analyze the image and determine the most precise possible location—ideally identifying the exact building, landmark, or facility, not just the city. 
    Examine all provided content links in detail, using both textual and visual clues to support your conclusion. 
    Use only the provided links for evidence. Any additional links must directly support specific visual observations (e.g., satellite imagery or publicly available street-level photos of the same location). 
    Return your final answer as geographic coordinates.

    {prompt_data}

    Respond with **only** the following JSON structure (no extra text, markdown, or comments):

    {
        "latitude": float,
        "longitude": float,
        "location": string,
        "evidence": [
            {
                "analysis": string,
                "references": [string, …]
            }
        ]
    }

    **Guidelines:**
    - One entry per clue (visual and textual).  
    - Each object in the "evidence" list should explain a single textual or visual clue and be as many as possible. All image in the prompt follow the format: "image_{idx:03d}.jpg", starting from image_000.jpg.
    - In the "references" list, each element must be a URL or an image file name (e.g., "image_000.jpg"). They are marked with indices like [1], [2], etc in order of appearance in "references" list. "Analysis" must use these indices to cite the corresponding references.
    - The "analysis" field must describe the clue and cite reference in its corresponding "references" using bracketed indices like [1], [2], etc. The corresponding URLs or images for those references must be included in the "references" list for that object.  
        + For contextual evidence, must cite textual/news URLs.
        + For visual clues, cite `image_{idx:03d}.jpg` in `references` and any satellite/map URLs as needed.
    - MUST use given links to support the analysis.
    - If you can’t identify a specific building, give the city‑center coordinates.
    '''
)

LOCATION_PROMPT = (
    """
    Location: {location}

    Your task is to determine the geographic coordinates (latitude and longitude) of the specified location by following these steps:

    1. Attempt to find the exact GPS coordinates using reliable online sources such as maps or satellite imagery.

    2. If the exact location is not available, find the coordinates of a nearby or adjacent place (e.g., a recognizable landmark, building, road, or intersection).

    3. If no specific nearby location can be found, use the coordinates of the broader area (e.g., the center of Khan Younis or Gaza).

    4. In the "references" list, each element must be a URL or an image file name (e.g., "image_000.jpg"). They are marked with indices like [1], [2], etc in order of appearance in "references" list. "Analysis" must use these indices to cite the corresponding references.

    Return your answer in the following JSON format:

    {
      "latitude": float,
      "longitude": float,
      "analysis": "Describe how the coordinates were identified or approximated, including any visual or textual clues used.",
      "references": ["URL1", "URL2", ...]
    }

    - The "analysis" must clearly explain the reasoning behind the chosen coordinates.
    - The "references" list must include all URLs cited in the analysis.
    - Do not include any text outside of the JSON structure.
    """
)

VERIFICATION_IMAGE_TEXT_PROMPT = (
    """
    You are an expert in multimedia verification. Analyze the provided content and decide if it’s authentic or fabricated. Support your conclusion with detailed, verifiable evidence.

    {prompt_data}

    Prediction to verify:
    {prediction}

    Guidelines:
    1. Output only a JSON object with these fields:
    {
        "latitude": float,
        "longitude": float,
        "location": string,
        "evidence": [
            {
                "analysis": string,
                "references": [string, …]
            }
        ]
    }

    2. Images are named “image_{idx:03d}.jpg”:
    - Images up to “image_{satellite_image_id}.jpg” were used to generate the prediction.
    - “image_{satellite_image_id}.jpg” is the satellite reference.
    - Images after that show the claimed location’s landmarks—use them only to confirm buildings or landmarks.

    3. In the "references" field of response, each element must be a URL or an image file name (e.g., "image_000.jpg"). They are marked with indices like [1], [2], etc in order of appearance in "references" list. "Analysis" must use these indices to cite the corresponding references.

    4. There must be both visual and contextual evidences. For each evidence entry:
        a. **Visual evidence**: cross‑check the original images against the satellite view.
            - When citing original images (those before `image_{satellite_image_id}.jpg`), **do not** list them alone: each must be accompanied by at least one supporting satellite image, street‑view photo, or map URL in the same reference list.
            - If confirmed, **rewrite and enrich** your analysis with additional visual details (textures, angles, shadows) and cite any new image or map references.
            - If it can’t be verified, **remove** that entry entirely.

        b. **Contextual evidence**: verify against the provided URLs.
            - If confirmed, **rewrite and expand** your analysis with deeper context (dates, sources, related events) and cite any new supporting links.
            - If it can’t be verified, **remove** that entry.

        c. Analyze but **do not** need cite transcript and metadata.

    5. All evidence must directly support the predicted latitude/longitude. Do not include analysis or references unrelated to verifying that specific location.

    6. Do **not** include any metadata (EXIF, timestamps, filenames) as evidence.

    Return only the JSON—no extra text, markdown, or comments.
    """
)

VERIFICATION_TEXT_PROMPT = (
    """
    You are an expert in multimedia verification. Analyze the provided content and decide if it’s authentic or fabricated. Support your conclusion with detailed, verifiable evidence.

    {prompt_data}

    Prediction to verify:
    {prediction}

    Guidelines:
    1. Output only a JSON object with these fields:
    {
        "latitude": float,
        "longitude": float,
        "location": string,
        "evidence": [
            {
                "analysis": string,
                "references": [string, …]
            }
        ]
    }

    2. Images are named “image_{idx:03d}.jpg”:
    - “image_{satellite_image_id}.jpg” is the satellite reference.
    - Images after that show the claimed location’s landmarks—use them only to confirm buildings or landmarks.

    3. In the "references" field of response, each element must be a URL or an image file name (e.g., "image_000.jpg"). They are marked with indices like [1], [2], etc in order of appearance in "references" list. "Analysis" must use these indices to cite the corresponding references.

    4. There must be both visual and contextual evidences. For each evidence entry:
        a. **Visual evidence**: cross‑check the landmark images against the satellite view.
            - When citing landmark images (those after `image_{satellite_image_id}.jpg`), **do not** list them alone: each must be accompanied by at least one supporting satellite image, street‑view photo, or map URL in the same reference list.
            - If confirmed, **rewrite and enrich** your analysis with additional visual details (textures, angles, shadows) and cite any new image or map references.
            - If it can’t be verified, **remove** that entry entirely.

        b. **Contextual evidence**: verify against the provided URLs.
            - If confirmed, **rewrite and expand** your analysis with deeper context (dates, sources, related events) and cite any new supporting links.
            - If it can’t be verified, **remove** that entry.

        c. Analyze but **do not** need cite transcript and metadata.

    5. All evidence must directly support the predicted latitude/longitude. Do not include analysis or references unrelated to verifying that specific location.

    6. Do **not** include any metadata (EXIF, timestamps, filenames) as evidence.

    Return only the JSON—no extra text, markdown, or comments.
    """
)

VERIFICATION_IMAGE_PROMPT = (
    """
    You are an expert in multimedia verification. Analyze the provided content and decide if it’s authentic or fabricated. Support your conclusion with detailed, verifiable evidence.

    {prompt_data}

    Prediction to verify:
    {prediction}

    Guidelines:
    1. Output only a JSON object with these fields:
    {
        "latitude": float,
        "longitude": float,
        "location": string,
        "evidence": [
            {
                "analysis": string,
                "references": [string, …]
            }
        ]
    }

    2. Images are named “image_{idx:03d}.jpg”:
    - Images up to “image_{satellite_image_id}.jpg” were used to generate the prediction.
    - “image_{satellite_image_id}.jpg” is the satellite reference.
    - Images after that show the claimed location’s landmarks—use them only to confirm buildings or landmarks.

    3. In the "references" field of response, each element must be a URL or an image file name (e.g., "image_000.jpg"). They are marked with indices like [1], [2], etc in order of appearance in "references" list. "Analysis" must use these indices to cite the corresponding references.

    4. There must be both visual and contextual evidences. For each evidence entry:
        a. **Visual evidence**: cross‑check the original images against the satellite view.
            - When citing original images (those before `image_{satellite_image_id}.jpg`), **do not** list them alone: each must be accompanied by at least one supporting satellite image, street‑view photo, or map URL in the same reference list.
            - If confirmed, **rewrite and enrich** your analysis with additional visual details (textures, angles, shadows) and cite any new image or map references.
            - If it can’t be verified, **remove** that entry entirely.

        b. **Contextual evidence**: verify against the provided URLs.
            - If confirmed, **rewrite and expand** your analysis with deeper context (dates, sources, related events) and cite any new supporting links.
            - If it can’t be verified, **remove** that entry.

        c. Analyze but **do not** need cite transcript and metadata.

    5. All evidence must directly support the predicted latitude/longitude. Do not include analysis or references unrelated to verifying that specific location.

    Return only the JSON—no extra text, markdown, or comments.
    """
)

RANKING_PROMPT = (
    """
    You are evaluating multiple geolocation predictions from different modalities on their analysis and logic leading to the prediction. Analyze and score each prediction individually, then provide a comparative ranking.

    Predictions to evaluate:
    {predictions}

    For each prediction, evaluate on a 50-point scale using these criteria (each 0-10 points):

    1. **Plausibility of the Location (0–10)**
    2. **Specificity and Relevance of the Location Name (0–10)**  
    3. **Depth and Validity of the Evidence (0–10)**
    4. **Use and Integration of Modalities (0–10)**
    5. **Confidence and Clarity of Justification (0–10)**

    Return a JSON object with the best prediction and all scored predictions:

    {
        "best_prediction": {
            "latitude": float,
            "longitude": float,
            "location": string,
            "score": int
        },
        "all_predictions": [
            {
                "image_prediction": bool,
                "text_prediction": bool,
                "score": int,
                "breakdown": {
                    "location_plausibility": int,
                    "location_name_specificity": int,
                    "evidence_quality": int,
                    "modality_use": int,
                    "justification_clarity": int
                },
                "prediction": {
                    "latitude": float,
                    "longitude": float,
                    "location": string,
                    "evidence": [
                        {
                            "analysis": string,
                            "references": [string, …]
                        }
                    ]
                }
            }
        ]
    }

    The best_prediction should contain the GPS coordinates and location from the highest-scoring prediction.
    Provide detailed scoring that reflects the quality and plausibility of each prediction, not correctness. Usually, the best prediction is the one having the most context.
    """
)



def rag_prompt(index_search_json: str, n_coords: Optional[int] = None) -> str:
    """
    Creates a formatted string with GPS coordinates for similar and dissimilar images.

    Args:
        candidates_gps (list[tuple]): List of (lat, lon) tuples for similar images.
        reverse_gps (list[tuple]): List of (lat, lon) tuples for dissimilar images.
        n_coords (int, optional): Number of coords to include from each list. Defaults to all.

    Returns:
        str: Formatted string with coordinates for reference.
    """
    if not os.path.exists(index_search_json):
        return ""
    
    with open(index_search_json, 'r', encoding='utf-8') as file:
        data = json.load(file)

    candidates_gps = data.get("candidates_gps", [])
    reverse_gps = data.get("reverse_gps", [])

    if n_coords is not None:
        candidates_gps = candidates_gps[:min(n_coords, len(candidates_gps))]
        reverse_gps = reverse_gps[:min(n_coords, len(reverse_gps))]
    else:
        candidates_gps = candidates_gps
        reverse_gps = reverse_gps

    candidates_str = (
        "[" + ", ".join(f"[{lat}, {lon}]" for (lat, lon) in candidates_gps) + "]"
    )
    reverse_str = (
        "[" + ", ".join(f"[{lat}, {lon}]" for (lat, lon) in reverse_gps) + "]"
    )
    return f"For your reference, these are coordinates of some similar images: {candidates_str}, and these are coordinates of some dissimilar images: {reverse_str}."

def metadata_prompt(metadata_file_path: str) -> str:
    """
    Reads a metadata JSON file and returns a formatted string combining all fields.

    Args:
        metadata_file_path (str): Path to the metadata JSON file

    Returns:
        str: Formatted string with all metadata fields combined
    """
    if not metadata_file_path or not os.path.exists(metadata_file_path):
        return ""

    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as file:
            metadata = json.load(file)

        if not metadata:
            return ""

        metadata_parts = []

        if "location" in metadata and metadata["location"]:
            metadata_parts.append(f"Location: {metadata['location']}")

        if "violence level" in metadata and metadata["violence level"]:
            metadata_parts.append(f"Violence level: {metadata['violence level']}")

        if "title" in metadata and metadata["title"]:
            metadata_parts.append(f"Title: {metadata['title']}")

        if "social media link" in metadata and metadata["social media link"]:
            metadata_parts.append(f"Social media link: {metadata['social media link']}")

        if "description" in metadata and metadata["description"]:
            metadata_parts.append(f"Description: {metadata['description']}")

        if "category" in metadata and metadata["category"]:
            metadata_parts.append(f"Category: {metadata['category']}")

        if not metadata_parts:
            return ""

        metadata_string = "Metadata for the image is: "
        return metadata_string + ". ".join(metadata_parts) + "."

    except Exception:
        return ""

def search_prompt(search_candidates: list[str], n_search: Optional[int] = None) -> str:
    """
    Formats search candidate links into a prompt string.
    
    Args:
        search_candidates (list[str]): List of candidate URLs from image search
        n_search (int): Number of results to include (default: 5)
        
    Returns:
        str: Formatted string with candidate links, each on a new line
        
    Example:
        >>> candidates = search_prompt(["https://example1.com", "https://example2.com"], n_search=3)
        >>> print(candidates)
        Similar image can be found in those links:
        https://example1.com
        https://example2.com
    """
    
    if not search_candidates or not isinstance(search_candidates, list):
        return ""
    
    EXCLUDE_DOMAINS =[
        "x.com",
        "twitter.com",
        "linkedin.com",
        "bbc.com",
        "bbc.co.uk",
        "instagram.com",
        "tiktok.com",
    ]

    for domain in EXCLUDE_DOMAINS:
        search_candidates = [url for url in search_candidates if domain not in url]
    
    if n_search is not None:
        search_candidates = search_candidates[:min(n_search, len(search_candidates))]
    
    try:
        prompt = "\n".join(search_candidates)
        return prompt

    except Exception:
        return ""

def image_search_prompt(image_search_json: str, n_search: Optional[int] = None) -> str:
    """
    Reads all JSON files in the base directory's image_search folder and combines links.

    Args:
        base_dir (str): Path to the base directory containing image search JSON files

    Returns:
        str: Combined search prompt string
    """
    pages_with_matching_images = set()
    full_matching_images = set()
    partial_matching_images = set()

    with open(image_search_json, "r", encoding="utf-8") as file:
        data_list = json.load(file)
        for json_data in data_list:
            if "pages_with_matching_images" in json_data:
                pages_with_matching_images.update(json_data["pages_with_matching_images"])
            elif "full_matching_images" in json_data:
                full_matching_images.update(json_data["full_matching_images"])
            elif "partial_matching_images" in json_data:
                partial_matching_images.update(json_data["partial_matching_images"])

    if not pages_with_matching_images and not full_matching_images and not partial_matching_images:
        return ""
    
    prompt = "Those are pages with matching images:\n"
    prompt += search_prompt(list(pages_with_matching_images), n_search=n_search)
    # prompt += "\n\nThose are full matching images:\n"
    # prompt += search_prompt(list(full_matching_images), n_search=n_search)
    # prompt += "\n\nThose are partial matching images:\n"
    # prompt += search_prompt(list(partial_matching_images), n_search=n_search)

    return prompt

def search_content_prompt(search_content_json: str) -> str:
    """
    Reads a JSON file containing search content and returns a formatted string.

    Args:
        search_content_json (str): Path to the JSON file with search content

    Returns:
        str: Formatted string with all search content links
    """
    if not os.path.exists(search_content_json):
        return ""

    try:
        with open(search_content_json, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if not data or not isinstance(data, list):
            return ""

        prompt = json.dumps(data, indent=2)
        return prompt

    except Exception:
        return ""

def transcript_prompt(audio_dir: str) -> str:
    """
    Reads all transcript text files in the audio directory and returns a formatted string.

    Args:
        audio_dir (str): Path to the audio directory containing transcript files

    Returns:
        str: Combined transcript content formatted as a prompt
    """
    if not os.path.exists(audio_dir):
        return ""
    
    transcript_content = []

    for txt_file in os.listdir(audio_dir):
        if txt_file.endswith(".txt"):
            txt_path = os.path.join(audio_dir, txt_file)
            with open(txt_path, "r", encoding="utf-8") as file:
                transcript_content.append(file.read().strip())

    combined_transcript = "\n".join(transcript_content)
    return f"This is the transcript of the video: {combined_transcript}" if combined_transcript else ""

def combine_prompt_data(prompt_dir: str, n_search: Optional[int] = None, n_coords: Optional[int] = None, image_prediction: bool = True,  text_prediction: bool = True) -> str:
    """
    Combines all prompt data into one comprehensive prompt string.
    
    Args:
        base_dir (str): Path to the base directory
        candidates_gps (list[tuple]): GPS coordinates for similar images (for RAG)
        reverse_gps (list[tuple]): GPS coordinates for dissimilar images (for RAG)
        n_search (int): Number of search results to include (default: 5)
        n_coords (int, optional): Number of coordinates to include in RAG
        
    Returns:
        str: Combined prompt string
        
    Example:
        >>> prompt = combine_prompts(
        ...     base_dir="path/to/base_dir",
        ...     candidates_gps=[(40.7128, -74.0060)],
        ...     reverse_gps=[(51.5074, -0.1278)]
        ... )
    """
    
    prompt_parts = []
    
    # 1. RAG prompt (optional)
    if n_coords is not None:
        rag_text = rag_prompt(os.path.join(prompt_dir, "index_search.json"), n_coords)
        prompt_parts.append(rag_text)
    
    # 2. Metadata prompt
    if text_prediction:
        metadata_text = metadata_prompt(os.path.join(prompt_dir, "metadata.json"))
        if metadata_text:
            prompt_parts.append(metadata_text)

    # 3. Search prompt
    if image_prediction:
        image_search_text = search_content_prompt(os.path.join(prompt_dir, "image_search_content.json"))
        if image_search_text:
            prompt_parts.append(image_search_text)
    
    if text_prediction:
        search_content_text = search_content_prompt(os.path.join(prompt_dir, "text_search_content.json"))
        if search_content_text:
            prompt_parts.append(search_content_text)

    # 4. Transcript prompt
    transcript_text = transcript_prompt(os.path.join(prompt_dir, "audio"))
    if transcript_text:
        prompt_parts.append(transcript_text)
    
    # Combine all parts with double newlines for readability
    combined_prompt = "\n\n".join(part for part in prompt_parts if part.strip())
    
    return combined_prompt

def batch_combine_prompts(prompt_dir: str, n_search: Optional[int] = None, n_coords: Optional[int] = None, image_prediction: bool = True, text_prediction: bool = True) -> str:
    """
    Combines all prompts into one comprehensive prompt string.
    
    Args:
        base_dir (str): Path to the base directory
        candidates_gps (list[tuple]): GPS coordinates for similar images (for RAG)
        reverse_gps (list[tuple]): GPS coordinates for dissimilar images (for RAG)
        n_search (int): Number of search results to include (default: 5)
        n_coords (int, optional): Number of coordinates to include in RAG
        
    Returns:
        str: Combined prompt string
        
    Example:
        >>> prompt = combine_prompts(
        ...     base_dir="path/to/base_dir",
        ...     candidates_gps=[(40.7128, -74.0060)],
        ...     reverse_gps=[(51.5074, -0.1278)]
        ... )
    """

    prompt_data = combine_prompt_data(prompt_dir, n_search=n_search, n_coords=n_coords, image_prediction=image_prediction, text_prediction=text_prediction)

    if image_prediction and text_prediction:
        prompt = BATCH_IMAGE_TEXT_PROMPT.strip()
    elif image_prediction:
        prompt = BATCH_IMAGE_PROMPT.strip()
    else:
        prompt = BATCH_TEXT_PROMPT.strip()

    prompt = prompt.replace("{prompt_data}", prompt_data)

    return prompt

def location_prompt(location: str) -> str:
    """
    Creates a prompt string for the given location.

    Args:
        location (str): The location to include in the prompt.

    Returns:
        str: Formatted string with the location.
    """
    if not location:
        return ""
    
    prompt = LOCATION_PROMPT.strip()
    prompt = prompt.replace("{location}", location)

    return prompt

def verification_prompt(satellite_image_id: int, prediction: dict, prompt_dir: str, n_search: Optional[int] = None, n_coords: Optional[int] = None, image_prediction: bool = True, text_prediction: bool = True) -> str:
    """
    Creates a verification prompt string with the provided data and prediction.

    Args:
        prompt_data (str): The prompt data to include.
        prediction (str): The prediction to verify.

    Returns:
        str: Formatted verification prompt string.
    """
    prompt_data = combine_prompt_data(prompt_dir, n_search=n_search, n_coords=n_coords, image_prediction=image_prediction, text_prediction=text_prediction)
    
    if image_prediction and text_prediction:
        prompt = VERIFICATION_IMAGE_TEXT_PROMPT.strip()
    elif image_prediction:
        prompt = VERIFICATION_IMAGE_PROMPT.strip()
    else:
        prompt = VERIFICATION_TEXT_PROMPT.strip()

    prompt = prompt.replace("{prompt_data}", prompt_data)
    prompt = prompt.replace("{prediction}", json.dumps(prediction, indent=2))
    prompt = prompt.replace("{satellite_image_id}", f"{satellite_image_id:03d}")

    return prompt

def ranking_prompt(predictions: list[dict]) -> str:
    """
    Creates a ranking prompt string with the provided prediction.

    Args:
        prediction (dict): The prediction data to include.

    Returns:
        str: Formatted ranking prompt string.
    """
    if not predictions:
        return ""

    prompt = RANKING_PROMPT.strip()
    prompt = prompt.replace("{predictions}", json.dumps(predictions, indent=2))

    return prompt

# Example usage
if __name__ == "__main__":
    prompt_dir = "g3/data/prompt_data"
    prompt = batch_combine_prompts(prompt_dir=prompt_dir, n_search=5, n_coords=10)
    print(prompt)