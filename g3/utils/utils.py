import requests
import time
import base64
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv() 

def upload_image_to_imgbb(image_path: str, api_key: str) -> str:
    """Upload image to imgbb with automatic retry on transient errors."""
    print(api_key)
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

def search_with_image_and_text(image_path: Optional[str] = None, 
                              search_text: str = "",
                              serpapi_key: str = "",
                              imgbb_key: str = "",
                              engine: str = "auto") -> dict:
    """
    Search using image and/or text with SerpAPI. 
    - Uses Google Lens if image is provided
    - Uses Google search if no image provided
    - Uploads image to imgbb first to get URL when needed
    
    Args:
        image_path (str, optional): Path to the image file to search with
        search_text (str): Text query to search for
        serpapi_key (str): SerpAPI key for search
        imgbb_key (str): imgbb API key for image upload (only needed if image provided)
        engine (str): Search engine to use ("auto", "google_lens", "google")
        
    Returns:
        dict: JSON response from SerpAPI containing search results
        
    Example:
        # Image + text search (uses Google Lens)
        >>> results = search_with_image_and_text(
        ...     image_path="photo.jpg", 
        ...     search_text="landmark location", 
        ...     serpapi_key="your_serpapi_key",
        ...     imgbb_key="your_imgbb_key"
        ... )
        
        # Text only search (uses Google)
        >>> results = search_with_image_and_text(
        ...     search_text="Gaza hospital press conference children",
        ...     serpapi_key="your_serpapi_key"
        ... )
    """
    
    # Determine which engine to use
    has_image = image_path and image_path.strip() and os.path.exists(image_path)
    
    if engine == "auto":
        selected_engine = "google_lens" if has_image else "google"
    else:
        selected_engine = engine
    
    print(f"🔍 Search mode: {'Image + Text' if has_image else 'Text only'}")
    print(f"🚀 Using engine: {selected_engine}")
    
    # Step 1: Upload image to imgbb to get URL (if image provided)
    
    
    # Step 2: Search with SerpAPI using image URL and/or text
    def search_with_serpapi(image_url: Optional[str], text_query: str, api_key: str, engine: str) -> dict:
        """Search using SerpAPI with image URL and/or text query"""
        
        serpapi_url = "https://serpapi.com/search"
        
        # Parameters for SerpAPI - different for different engines
        if engine == "google_lens" and image_url:
            params = {
                'engine': 'google_lens',
                'api_key': api_key,
                'url': image_url,
            }
            # Add text query if provided
            if text_query:
                params['q'] = text_query
                
        elif engine == "google":
            params = {
                'engine': 'google',
                'api_key': api_key,
                'q': text_query,
            }
            # Add image URL if provided for Google Images
            if image_url:
                params['tbm'] = 'isch'  # Google Images search
                
        else:
            # Fallback to basic search
            params = {
                'engine': engine,
                'api_key': api_key,
            }
            if image_url:
                params['url'] = image_url
            if text_query:
                params['q'] = text_query
        
        # Remove None/empty values
        params = {k: v for k, v in params.items() if v}
        
        try:
            print(f"🌐 SerpAPI request params: {params}")
            response = requests.get(serpapi_url, params=params, timeout=60)
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            raise Exception(f"Error with SerpAPI request: {e}")
    
    try:
        image_url = None
        
        # Upload image if provided
        if has_image and image_path:  # Add explicit None check
            if not imgbb_key:
                raise Exception("imgbb_key is required when image_path is provided")
                
            print(f"📤 Uploading image to imgbb: {image_path}")
            image_url = upload_image_to_imgbb(image_path, imgbb_key)
            print(f"✅ Image uploaded successfully: {image_url}")
        
        # Validate inputs
        if not has_image and not search_text:
            raise Exception("Either image_path or search_text must be provided")
            
        if not serpapi_key:
            raise Exception("serpapi_key is required")
        
        # Search with SerpAPI
        print(f"🔍 Searching with SerpAPI...")
        print(f"   Engine: {selected_engine}")
        print(f"   Text query: {search_text}")
        if image_url:
            print(f"   Image URL: {image_url}")
        
        search_results = search_with_serpapi(image_url, search_text, serpapi_key, selected_engine)
        
        print(f"✅ Search completed successfully")
        return search_results
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "success": False,
            "message": f"Search failed: {e}"
        }
        print(f"❌ Search failed: {e}")
        return error_result

def search_with_image_and_text_sd(image_path: Optional[str] = None,
                                  search_text: str = "",
                                  imgbb_key: str = "") -> dict:
    if not image_path:
        raise ValueError("Scrapingdog Lens requires image input")

    # 1️⃣ Upload image to imgbb as before
    print(imgbb_key)
    image_url = upload_image_to_imgbb(image_path, imgbb_key)

    # 2️⃣ Make GET request to Scrapingdog
    resp = requests.get(
        "https://api.scrapingdog.com/google_lens",
        params={
            "api_key": os.getenv("SCRAPINGDOG_API_KEY"),
            "url": image_url
        },
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()  # contains 'lens_results'


def get_location_from_gps(latitude: float, longitude: float, 
                         language: str = 'en', 
                         timeout: int = 10,
                         user_agent: str = 'keyframe_extraction_app') -> str:
    """
    Get location information from GPS coordinates using Nominatim API (OpenStreetMap).
    
    Args:
        latitude (float): Latitude coordinate (-90 to 90)
        longitude (float): Longitude coordinate (-180 to 180)
        language (str): Language for results (default: 'en')
        timeout (int): Request timeout in seconds (default: 10)
        user_agent (str): User agent string for API requests
        
    Returns:
        str: Location address string, or "Unknown location" if failed
        
    Example:
        >>> location = get_location_from_gps(40.7128, -74.0060)
        >>> print(location)
        'New York, NY, United States'
    """
    
    # Validate coordinates
    if not (-90 <= latitude <= 90):
        return f"Invalid latitude: {latitude}"
    
    if not (-180 <= longitude <= 180):
        return f"Invalid longitude: {longitude}"
    
    # Nominatim API endpoint
    base_url = "https://nominatim.openstreetmap.org/reverse"
    
    # Parameters for the API request
    params = {
        'lat': latitude,
        'lon': longitude,
        'format': 'json',
        'addressdetails': 1,
        'accept-language': language,
        'zoom': 18  # Detail level (18 = building level)
    }
    
    # Headers (User-Agent is required by Nominatim)
    headers = {
        'User-Agent': user_agent
    }
    
    try:
        # Make the API request
        response = requests.get(
            base_url, 
            params=params, 
            headers=headers, 
            timeout=timeout
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        if 'error' in data:
            return "Unknown location"
        
        # Return the display name (full address)
        return data.get('display_name', 'Unknown location')
        
    except:
        return "Unknown location"

def get_gps_from_location(location: str,
                         language: str = 'en',
                         timeout: int = 10,
                         user_agent: str = 'keyframe_extraction_app') -> Tuple[Optional[float], Optional[float]]:
    """
    Get GPS coordinates from a location string using Nominatim API (OpenStreetMap).
    
    Args:
        location (str): Location string (address, place name, city, etc.)
        language (str): Language for search (default: 'en')
        timeout (int): Request timeout in seconds (default: 10)
        user_agent (str): User agent string for API requests
        
    Returns:
        Tuple[float, float]: (latitude, longitude) coordinates, or (None, None) if failed
        
    Example:
        >>> lat, lon = get_gps_from_location("New York, NY, USA")
        >>> print(f"NYC coordinates: {lat}, {lon}")
        'NYC coordinates: 40.7127281, -74.0060152'
    """
    
    # Validate input
    if not location or not isinstance(location, str):
        return (None, None)
    
    location = location.strip()
    if not location:
        return (None, None)
    
    # Nominatim API endpoint for forward geocoding
    base_url = "https://nominatim.openstreetmap.org/search"
    
    # Parameters for the API request
    params = {
        'q': location,
        'format': 'json',
        'addressdetails': 1,
        'accept-language': language,
        'limit': 1  # Only return the best match
    }
    
    # Headers (User-Agent is required by Nominatim)
    headers = {
        'User-Agent': user_agent
    }
    
    try:
        # Make the API request
        response = requests.get(
            base_url,
            params=params,
            headers=headers,
            timeout=timeout
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        if not data:
            return (None, None)
        
        # Get the first (best) result
        result = data[0]
        
        # Extract latitude and longitude
        lat = float(result.get('lat'))
        lon = float(result.get('lon'))
        
        return (lat, lon)
        
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return (None, None)


def extract_image_search_candidates(search_results: dict, no_results: int) -> List[str]:
    """
    Extract candidates from image search results (Google Lens or Google Images).
    Filters out social media links (YouTube, TikTok, Instagram).
    
    Args:
        search_results (dict): Search results from SerpAPI image search
        no_results (int): Maximum number of results to return
        
    Returns:
        List[str]: List of up to no_results candidate strings with link and title
        
    Example:
        >>> candidates = extract_image_search_candidates(lens_results, 5)
        >>> for candidate in candidates:
        ...     print(candidate)
        'Link: https://example.com | Title: Example Title'
    """
    candidates = []
    
    # Social media domains to filter out
    excluded_domains = [
        # "youtube.com",
        # "instagram.com",
        # "tiktok.com",
        # "twitter.com",
        # "x.com",
        # "facebook.com",
    ]
    
    def is_excluded_link(link):
        """Check if a link should be excluded based on domain."""
        if not link or link == 'No link':
            return False
        
        # Convert to lowercase for case-insensitive comparison
        link_lower = link.lower()
        
        # Check if any excluded domain is in the link
        for domain in excluded_domains:
            if domain in link_lower:
                return True
        return False
    
    try:
        # Handle Google Lens results
        if 'visual_matches' in search_results:
            visual_matches = search_results['visual_matches']
            for match in visual_matches:
                if len(candidates) >= no_results:
                    break
                    
                link = match.get('link', 'No link')
                title = match.get('title', 'No title')
                
                # Skip if link is from excluded domains
                if is_excluded_link(link):
                    continue
                
                candidate = f"Link: {link} | Title: {title}"
                candidates.append(candidate)
        
        # Handle Google Images results  
        elif 'images_results' in search_results:
            images_results = search_results['images_results']
            for result in images_results:
                if len(candidates) >= no_results:
                    break
                    
                link = result.get('link', result.get('original', 'No link'))
                title = result.get('title', result.get('alt', 'No title'))
                
                # Skip if link is from excluded domains
                if is_excluded_link(link):
                    continue
                
                candidate = f"Link: {link} | Title: {title}"
                candidates.append(candidate)
        
        # Handle other possible image result formats
        elif 'organic_results' in search_results:
            organic_results = search_results['organic_results']
            for result in organic_results:
                if len(candidates) >= no_results:
                    break
                    
                link = result.get('link', 'No link')
                title = result.get('title', 'No title')
                
                # Skip if link is from excluded domains
                if is_excluded_link(link):
                    continue
                
                candidate = f"Link: {link} | Title: {title}"
                candidates.append(candidate)
        
        # Handle error cases
        elif 'error' in search_results:
            candidates.append(f"Error: {search_results['error']}")
        
        else:
            # Try to extract from any available results
            for key in ['results', 'visual_matches', 'images_results']:
                if key in search_results and isinstance(search_results[key], list):
                    for item in search_results[key]:
                        if len(candidates) >= no_results:
                            break
                            
                        if isinstance(item, dict):
                            link = item.get('link', item.get('url', 'No link'))
                            title = item.get('title', item.get('name', 'No title'))
                            
                            # Skip if link is from excluded domains
                            if is_excluded_link(link):
                                continue
                            
                            candidate = f"Link: {link} | Title: {title}"
                            candidates.append(candidate)
                    break
        
        # If no candidates found, add a message
        if not candidates:
            candidates.append("No image search results found (after filtering)")
            
    except Exception as e:
        candidates.append(f"Error processing image search results: {str(e)}")
    
    return candidates[:no_results]


def extract_text_search_candidates(search_results: dict, no_results: int) -> List[str]:
    """
    Extract candidates from text search results (Google search).
    Filters out social media links (YouTube, TikTok, Instagram).
    
    Args:
        search_results (dict): Search results from SerpAPI text search
        no_results (int): Maximum number of results to return
        
    Returns:
        List[str]: List of up to no_results candidate strings with link, title, and snippet
        
    Example:
        >>> candidates = extract_text_search_candidates(google_results, 5)
        >>> for candidate in candidates:
        ...     print(candidate)
        'Link: https://example.com | Title: Example Title | Snippet: Example description...'
    """
    candidates = []
    
    # Social media domains to filter out
    excluded_domains = [
        'youtube.com', 'www.youtube.com', 'youtu.be',
        'tiktok.com', 'www.tiktok.com', 'vm.tiktok.com',
        'instagram.com', 'www.instagram.com', 'instagr.am'
    ]
    
    def is_excluded_link(link):
        """Check if a link should be excluded based on domain."""
        if not link or link == 'No link':
            return False
        
        # Convert to lowercase for case-insensitive comparison
        link_lower = link.lower()
        
        # Check if any excluded domain is in the link
        for domain in excluded_domains:
            if domain in link_lower:
                return True
        return False
    
    try:
        # Handle Google search results
        if 'organic_results' in search_results:
            organic_results = search_results['organic_results']
            for result in organic_results:
                if len(candidates) >= no_results:
                    break
                    
                link = result.get('link', 'No link')
                title = result.get('title', 'No title')
                snippet = result.get('snippet', result.get('description', 'No snippet'))
                
                # Skip if link is from excluded domains
                if is_excluded_link(link):
                    continue
                
                candidate = f"Link: {link} | Title: {title} | Snippet: {snippet}"
                candidates.append(candidate)
        
        # Handle news results if present
        elif 'news_results' in search_results:
            news_results = search_results['news_results']
            for result in news_results:
                if len(candidates) >= no_results:
                    break
                    
                link = result.get('link', 'No link')
                title = result.get('title', 'No title')
                snippet = result.get('snippet', result.get('summary', 'No snippet'))
                
                # Skip if link is from excluded domains
                if is_excluded_link(link):
                    continue
                
                candidate = f"Link: {link} | Title: {title} | Snippet: {snippet}"
                candidates.append(candidate)
        
        # Handle answer box or featured snippet
        elif 'answer_box' in search_results:
            answer_box = search_results['answer_box']
            link = answer_box.get('link', 'No link')
            title = answer_box.get('title', 'Featured Answer')
            snippet = answer_box.get('answer', answer_box.get('snippet', 'No snippet'))
            
            # Only add if not from excluded domains
            if not is_excluded_link(link):
                candidate = f"Link: {link} | Title: {title} | Snippet: {snippet}"
                candidates.append(candidate)
            
            # Also try to get organic results
            if 'organic_results' in search_results:
                organic_results = search_results['organic_results']
                for result in organic_results:
                    if len(candidates) >= no_results:
                        break
                        
                    link = result.get('link', 'No link')
                    title = result.get('title', 'No title')
                    snippet = result.get('snippet', result.get('description', 'No snippet'))
                    
                    # Skip if link is from excluded domains
                    if is_excluded_link(link):
                        continue
                    
                    candidate = f"Link: {link} | Title: {title} | Snippet: {snippet}"
                    candidates.append(candidate)
        
        # Handle error cases
        elif 'error' in search_results:
            candidates.append(f"Error: {search_results['error']}")
        
        else:
            # Try to extract from any available results
            for key in ['results', 'items', 'data']:
                if key in search_results and isinstance(search_results[key], list):
                    for item in search_results[key]:
                        if len(candidates) >= no_results:
                            break
                            
                        if isinstance(item, dict):
                            link = item.get('link', item.get('url', 'No link'))
                            title = item.get('title', item.get('name', 'No title'))
                            snippet = item.get('snippet', item.get('description', item.get('summary', 'No snippet')))
                            
                            # Skip if link is from excluded domains
                            if is_excluded_link(link):
                                continue
                            
                            candidate = f"Link: {link} | Title: {title} | Snippet: {snippet}"
                            candidates.append(candidate)
                    break
        
        # If no candidates found, add a message
        if not candidates:
            candidates.append("No text search results found (after filtering)")
            
    except Exception as e:
        candidates.append(f"Error processing text search results: {str(e)}")
    
    return candidates[:no_results]  # Ensure max no_results results





# Example usage and testing
if __name__ == "__main__":
    # Test coordinates (some famous landmarks)
    # test_coordinates = [
    #     (31.420044, 34.360174)
    # ]
    
    # print("Testing single coordinate lookup (reverse geocoding):")
    # print("=" * 60)

    # # Test single coordinate lookup
    # lat, lon = test_coordinates[0]
    # location = get_location_from_gps(lat, lon)
    
    # print(f"Coordinates: ({lat}, {lon})")
    # print(f"Location: {location}")
    
    # print("\nTesting round-trip conversion:")
    # print("=" * 60)
    
    # # Forward geocoding
    # lat, lon = get_gps_from_location("Pobeda Stadium, Kadijivka, Luhansk Oblast, Ukraine")
    # print(f"GPS coordinates: ({lat}, {lon})")
    
    # if lat is not None and lon is not None:
    #     # Reverse geocoding
    #     converted_location = get_location_from_gps(lat, lon)
    #     print(f"Converted back to: {converted_location}")
    # else:
    #     print("Forward geocoding failed")
    
    print("\nTesting SerpAPI search:")
    print("=" * 60)
    
    scrapingdog_key = os.getenv("SCRAPINGDOG_API_KEY")
    imgbb_key = os.getenv("IMGBB_API_KEY")
    image_path = "C:\\Users\\tungd\\OneDrive - MSFT\\Second Year\\ML\\ACMMM25 - Grand Challenge on Multimedia Verification\\G3-Original\\g3\\data\\prompt_data\\ID111\\images\\image_001.jpg"

    if not scrapingdog_key:
        print("❌ Error: SCRAPINGDOG_API_KEY environment variable is not set.")
    else:
        # # Test 1: Text-only search (uses Google)
        # print("🔍 Test 1: Text-only search (Google engine)")
        # try:
        #     results = search_with_image_and_text(
        #         search_text="Ghassan Al-Salem Pharmacy, Beit Lahia, a town in the northern Gaza Strip",
        #         serpapi_key=serpapi_key
        #     )
        #     print(f"✅ Text search successful")

        #     output_path = "image_text_search_results.json"
        #     with open(output_path, "w", encoding="utf-8") as fp:
        #         json.dump(results, fp, ensure_ascii=False, indent=2)
        #     print(f"📄  Full results saved to {output_path}")

        #     text_results = extract_text_search_candidates(results, no_results=5)
        #     print(text_results)
        # except Exception as e:
        #     print(f"❌ Text search failed: {e}")
        
        # Test 2: Image + text search (uses Google Lens)
        if imgbb_key and os.path.exists(image_path):
            print("\n🔍 Test 2: Image + text search (Google Lens engine)")
            try:
                results = search_with_image_and_text_sd(
                    image_path=image_path,
                    search_text="",
                    imgbb_key=imgbb_key,
                )
                print(f"✅ Image + text search successful")

                output_path = "image_text_search_results.json"
                with open(output_path, "w", encoding="utf-8") as fp:
                    json.dump(results, fp, ensure_ascii=False, indent=2)
                print(f"📄  Full results saved to {output_path}")

                print(json.dumps(results, indent=2))
                image_results = extract_image_search_candidates(results, no_results=5)
                print(image_results)
            except Exception as e:
                print(f"❌ Image + text search failed: {e}")
        else:
            print("\n⚠️ Skipping image + text test: IMGBB_KEY not set or image file not found")
