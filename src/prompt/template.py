DIVERSIFICATION_PROMPT = """
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
    - Each object in the "evidence" list should explain a single textual or visual clue and be as many as possible. All image in the prompt follow the format: "image_{{idx:03d}}.jpg", starting from image_000.jpg.
    - In the "references" list, each element must be a URL or an image file name (e.g., "image_000.jpg"). They are marked with indices like [1], [2], etc in order of appearance in "references" list. "Analysis" must use these indices to cite the corresponding references.
    - The "analysis" field must describe the clue and cite reference in its corresponding "references" using bracketed indices like [1], [2], etc. The corresponding URLs or images for those references must be included in the "references" list for that object.  
        + For contextual evidence, must cite textual/news URLs.
        + For visual clues, cite `image_{{idx:03d}}.jpg` in `references` and any satellite/map URLs as needed.
    - MUST use given links to support the analysis.
    - If you can’t identify a specific building, give the city‑center coordinates.
    """

LOCATION_PROMPT = """
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

VERIFICATION_PROMPT = """
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

    2. Images are named “image_{{idx:03d}}.jpg”:
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
