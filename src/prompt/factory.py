import json
import os

from .template import DIVERSIFICATION_PROMPT, LOCATION_PROMPT, VERIFICATION_PROMPT


def rag_prompt(index_search_json: str, n_coords: int | None = None) -> str:
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

    with open(index_search_json, "r", encoding="utf-8") as file:
        data = json.load(file)

    candidates_gps = data.get("candidates_gps", [])
    reverse_gps = data.get("reverse_gps", [])

    if n_coords is not None:
        candidates_gps = candidates_gps[: min(n_coords, len(candidates_gps))]
        reverse_gps = reverse_gps[: min(n_coords, len(reverse_gps))]
    else:
        candidates_gps = candidates_gps
        reverse_gps = reverse_gps

    candidates_str = (
        "[" + ", ".join(f"[{lat}, {lon}]" for (lat, lon) in candidates_gps) + "]"
    )
    reverse_str = "[" + ", ".join(f"[{lat}, {lon}]" for (lat, lon) in reverse_gps) + "]"
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
        with open(metadata_file_path, "r", encoding="utf-8") as file:
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


def search_prompt(search_candidates: list[str], n_search: int | None = None) -> str:
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

    EXCLUDE_DOMAINS = [
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
        search_candidates = search_candidates[: min(n_search, len(search_candidates))]

    try:
        prompt = "\n".join(search_candidates)
        return prompt

    except Exception:
        return ""


def image_search_prompt(image_search_json: str, n_search: int | None = None) -> str:
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
                pages_with_matching_images.update(
                    json_data["pages_with_matching_images"]
                )
            elif "full_matching_images" in json_data:
                full_matching_images.update(json_data["full_matching_images"])
            elif "partial_matching_images" in json_data:
                partial_matching_images.update(json_data["partial_matching_images"])

    if (
        not pages_with_matching_images
        and not full_matching_images
        and not partial_matching_images
    ):
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
        with open(search_content_json, "r", encoding="utf-8") as file:
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
    return (
        f"This is the transcript of the video: {combined_transcript}"
        if combined_transcript
        else ""
    )


def combine_prompt_data(
    prompt_dir: str,
    n_search: int | None = None,
    n_coords: int | None = None,
    image_prediction: bool = True,
    text_prediction: bool = True,
) -> str:
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
        image_search_text = search_content_prompt(
            os.path.join(prompt_dir, "image_search_content.json")
        )
        if image_search_text:
            prompt_parts.append(image_search_text)

    if text_prediction:
        search_content_text = search_content_prompt(
            os.path.join(prompt_dir, "text_search_content.json")
        )
        if search_content_text:
            prompt_parts.append(search_content_text)

    # 4. Transcript prompt
    transcript_text = transcript_prompt(os.path.join(prompt_dir, "audio"))
    if transcript_text:
        prompt_parts.append(transcript_text)

    # Combine all parts with double newlines for readability
    combined_prompt = "\n\n".join(part for part in prompt_parts if part.strip())

    return combined_prompt


def diversification_prompt(
    prompt_dir: str,
    n_search: int | None = None,
    n_coords: int | None = None,
    image_prediction: bool = True,
    text_prediction: bool = True,
) -> str:
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

    prompt_data = combine_prompt_data(
        prompt_dir,
        n_search=n_search,
        n_coords=n_coords,
        image_prediction=image_prediction,
        text_prediction=text_prediction,
    )

    prompt = DIVERSIFICATION_PROMPT.strip().format(prompt_data=prompt_data)

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
    prompt = prompt.format(location=location)

    return prompt


def verification_prompt(
    satellite_image_id: int,
    prediction: dict,
    prompt_dir: str,
    n_search: int | None = None,
    n_coords: int | None = None,
    image_prediction: bool = True,
    text_prediction: bool = True,
) -> str:
    """
    Creates a verification prompt string with the provided data and prediction.

    Args:
        prompt_data (str): The prompt data to include.
        prediction (str): The prediction to verify.

    Returns:
        str: Formatted verification prompt string.
    """
    prompt_data = combine_prompt_data(
        prompt_dir,
        n_search=n_search,
        n_coords=n_coords,
        image_prediction=image_prediction,
        text_prediction=text_prediction,
    )

    prompt = VERIFICATION_PROMPT.strip().format(
        prompt_data=prompt_data,
        prediction=json.dumps(prediction, indent=2),
        satellite_image_id=f"{satellite_image_id:03d}",
    )

    return prompt


# Example usage
if __name__ == "__main__":
    prompt_dir = "g3/data/prompt_data"
    prompt = diversification_prompt(prompt_dir=prompt_dir, n_search=5, n_coords=10)
    print(prompt)
