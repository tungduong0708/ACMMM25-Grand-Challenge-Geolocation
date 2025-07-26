import asyncio
import json
import logging
import os
import hashlib
import shutil
from pathlib import Path

import faiss
import torch
from PIL import Image
from torch import nn

from .prompt.fetch.content_fetch import fetch_links_to_json
from .prompt.fetch.satellite_fetch import fetch_satellite_image
from .prompt.preprocess.keyframe_extract import extract_and_save_keyframes
from .prompt.preprocess.video_transcribe import transcribe_video_directory
from .prompt.search.image_search import image_search_directory
from .prompt.search.index_search import save_results_to_json, search_index_directory
from .prompt.search.text_search import text_search_image, text_search_link

logger = logging.getLogger("uvicorn.error")


class DataProcessor:
    def __init__(
        self,
        model: nn.Module,
        input_dir: Path,
        prompt_dir: Path,
        cache_dir: Path,
        image_dir: Path,
        audio_dir: Path,
        index_path: Path,
        database_csv_path: Path,
        device: torch.device,
    ):
        self.input_dir = input_dir
        self.prompt_dir = prompt_dir
        self.cache_dir = cache_dir
        self.image_dir = image_dir
        self.audio_dir = audio_dir
        self.model = model
        self.device = device
        self.database_csv_path = database_csv_path

        try:
            self.index = faiss.read_index(str(index_path))
            logger.info(f"✅ Successfully loaded FAISS index from: {index_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index from {index_path}: {e}")

        self.image_extension = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp",
        }
        self.video_extension = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
        }

    def __extract_keyframes(self):
        """
        Extract keyframes from all videos in the input directory.
        Put all images and keyframes into the prompt directory.
        """
        output_dir = self.image_dir
        os.makedirs(output_dir, exist_ok=True)

        # Determine starting index based on existing files
        current_files = list(output_dir.glob("image_*.*"))
        idx = len(current_files)

        # Process images
        for file_name in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(
                tuple(self.image_extension)
            ):
                out_path = output_dir / f"image_{idx:03d}.jpg"
                Image.open(file_path).convert("RGB").save(out_path)
                idx += 1

        # Process videos
        for file_name in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(
                tuple(self.video_extension)
            ):
                if idx is None:
                    idx = 0
                idx = extract_and_save_keyframes(
                    video_path=file_path, output_dir=str(output_dir), start_index=idx
                )
        logger.info(f"✅ Extracted keyframes and images to: {output_dir}")

    def __transcribe_videos(self):
        """
        Transcribe all videos in the input directory.
        Save transcripts into the prompt directory.
        """
        audio_dir = self.audio_dir
        os.makedirs(audio_dir, exist_ok=True)

        if audio_dir.is_dir() and any(audio_dir.iterdir()):
            logger.info(f"🔄 Found existing transcripts in directory: {audio_dir}")
            return

        transcribe_video_directory(
            video_dir=str(self.input_dir),
            output_dir=str(audio_dir),
            model_name="base",  # Use the base Whisper model for transcription
        )
        logger.info(f"✅ Successfully transcribed videos to: {audio_dir}")

    def __image_search(self):
        """
        Perform image search on all images in the input directory.
        Save search results into the prompt directory.
        """
        image_dir = self.image_dir

        if os.environ["IMGBB_API_KEY"] is None:
            raise ValueError(
                "IMGBB_API_KEY environment variable is not set or is None."
            )
        if os.environ["SCRAPINGDOG_API_KEY"] is None:
            raise ValueError(
                "SCRAPINGDOG_API_KEY environment variable is not set or is None."
            )
        image_search_directory(
            directory=str(image_dir),
            output_dir=str(self.prompt_dir),
            filename="image_search.json",
            imgbb_key=os.environ["IMGBB_API_KEY"],
            scrapingdog_key=os.environ["SCRAPINGDOG_API_KEY"],
            max_workers=4,
            target_links=20,
        )
        logger.info(f"✅ Successfully performed image search on: {image_dir}")

    def __text_search(self):
        """
        Perform text search with metadata to get related links.
        """
        query = ""
        metadata_file = self.prompt_dir / "metadata.json"
        if not metadata_file.exists():
            query = ""
        else:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                description = metadata.get("description", "")
                location = metadata.get("location", "")
                query = f"{description} in {location}".strip()

        text_search_link(
            query=query,
            output_dir=str(self.prompt_dir),
            filename="text_search.json",
            num_results=10,
            api_key=os.environ["GOOGLE_CLOUD_API_KEY"],
            cx=os.environ["GOOGLE_CSE_CX"],
        )

    async def __fetch_related_link_content(
        self, image_prediction: bool = True, text_prediction: bool = True
    ):
        """
        Fetch related link content for all images and text in the prompt directory.
        """

        async def fetch_and_save_links(links, output_filename):
            if links:
                await fetch_links_to_json(
                    links=list(links),
                    output_path=str(self.prompt_dir / output_filename),
                    max_content_length=5000,
                )
                logger.info(
                    f"Fetched content for {len(links)} links into {output_filename}"
                )

        # Image links
        image_links = set()
        image_search_file = self.prompt_dir / "image_search.json"
        if image_prediction:
            if not image_search_file.exists():
                self.__image_search()
            with open(image_search_file, "r") as f:
                image_search_data = json.load(f)
                image_links.update(image_search_data.get("all_links", []))
            logger.info(f"Found {len(image_links)} image links to fetch content from.")
            await fetch_and_save_links(image_links, "image_search_content.json")

        # Text links
        text_links = set()
        text_search_file = self.prompt_dir / "text_search.json"
        if text_prediction:
            if not text_search_file.exists():
                self.__text_search()
            with open(text_search_file, "r") as f:
                text_search_data = json.load(f)
                text_links.update(filter(None, text_search_data.get("links", [])))
            logger.info(f"Found {len(text_links)} text links to fetch content from.")
            await fetch_and_save_links(text_links, "text_search_content.json")

        if not image_links and not text_links:
            logger.info("No links found in image or text search results.")

    def __index_search(self):
        """
        Perform FAISS index search on all images in the prompt directory.
        Save search results into the report directory.
        """
        if not self.index:
            raise RuntimeError(
                "FAISS index is not loaded. Cannot perform index search."
            )

        output_path = self.prompt_dir / "index_search.json"
        if output_path.exists():
            logger.info(
                f"Index search results already exist at {output_path}, skipping search."
            )
            return

        if not os.path.exists(self.database_csv_path):
            raise FileNotFoundError(
                f"Database CSV file not found: {self.database_csv_path}"
            )

        candidates_gps, reverse_gps = search_index_directory(
            model=self.model,
            device=self.device,
            index=self.index,
            image_dir=str(self.image_dir),
            database_csv_path=str(self.database_csv_path),
            top_k=20,
            max_elements=20,
        )

        save_results_to_json(candidates_gps, reverse_gps, str(output_path))
        logger.info(
            f"✅ Successfully performed index search. Results saved to: {output_path}"
        )

    async def __fetch_satellite_image_async(
        self,
        latitude: float,
        longitude: float,
        zoom: int,
        output_path: Path,
    ) -> None:
        """
        Asynchronously fetches a satellite image without blocking the event loop.

        Runs the synchronous `fetch_satellite_image` function in a background thread.

        Args:
            latitude (float): Latitude of the location.
            longitude (float): Longitude of the location.
            zoom (int): Zoom level of the satellite image.
            output_path (Path): Path to save the image file.
        """
        await asyncio.to_thread(
            fetch_satellite_image,
            latitude,
            longitude,
            zoom,
            str(output_path),
        )

    async def __search_images_async(
        self,
        location: str,
        num_images: int,
        api_key: str | None,
        cse_cx: str | None,
        output_dir: Path,
        image_id_offset: int,
    ) -> list[str]:
        """
        Asynchronously searches for images based on a text location query.

        Args:
            location (str): Text location to search.
            num_images (int): Number of images to fetch.
            api_key (str): Google Cloud API key.
            cse_cx (str): Google Custom Search Engine ID.
            output_dir (Path): Directory where images will be saved.
            image_id_offset (int): Offset for image filenames.

        Returns:
            Any: The result of `text_search_image`, if it returns a value.
        """
        return await asyncio.to_thread(
            text_search_image,
            location,
            num_images,
            api_key,
            cse_cx,
            str(output_dir),
            image_id_offset,
        )

    def __compute_sha256(self, filepath: Path) -> str:
        """
        Compute the SHA-256 hash of a file.
        """
        if not filepath.is_file():
            raise ValueError(f"File does not exist: {filepath}")

        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def __compare_directories(self, dir1: Path, dir2: Path) -> bool:
        """
        Compare two directories to check if they contain the same files with identical content.
        Args:
            dir1 (Path): First directory to compare.
            dir2 (Path): Second directory to compare.
        Returns:
            bool: True if both directories contain the same files with identical content, False otherwise.
        """
        if not dir1.is_dir() or not dir2.is_dir():
            return False

        files1 = sorted(p for p in dir1.iterdir() if p.is_file())
        files2 = sorted(p for p in dir2.iterdir() if p.is_file())

        # Check if filenames match exactly
        names1 = {p.name for p in files1}
        names2 = {p.name for p in files2}
        if names1 != names2:
            return False

        # Compare each matching file
        for filename in names1:
            path1 = dir1 / filename
            path2 = dir2 / filename

            # Skip directories
            if not path1.is_file() or not path2.is_file():
                continue

            hash1 = self.__compute_sha256(path1)
            hash2 = self.__compute_sha256(path2)

            if hash1 != hash2:
                return False  # Found mismatch
        return True  # All matching files are identical

    def __copy_directory(self, src: Path, dest: Path):
        """
        Recursively copy all files from src to dest.
        """
        if not src.is_dir():
            raise ValueError(f"Source path is not a directory: {src}")

        # Delete everything in dest first
        if dest.exists():
            for item in dest.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

        # Ensure dest exists
        dest.mkdir(parents=True, exist_ok=True)

        for item in src.iterdir():
            if item.is_dir():
                self.__copy_directory(item, dest / item.name)
            else:
                dest_file = dest / item.name
                if not dest_file.exists() or not self.__compare_directories(
                    item, dest_file
                ):
                    shutil.copy2(item, dest_file)

    async def preprocess_input_data(
        self,
        image_prediction: bool = True,
        text_prediction: bool = True,
    ):
        """
        Preprocess all input data:
        - Extract keyframes from videos.
        - Transcribe videos.
        - Fetch related link content from images.
        Save images and extracted keyframes into the output directory
        """
        os.makedirs(self.prompt_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        cache_dir_input = self.cache_dir / "input_data"
        cache_dir_prompt = self.cache_dir / "prompt_data"
        if self.__compare_directories(self.input_dir, cache_dir_input):
            logger.info("Input data already processed, skipping...")
            self.__copy_directory(cache_dir_prompt, self.prompt_dir)
            return
        else:
            logger.info("Processing input data...")

        metadata_dest = self.prompt_dir / "metadata.json"
        if not metadata_dest.exists():
            for file in os.listdir(self.input_dir):
                if file.endswith(".json"):
                    file_path = os.path.join(self.input_dir, file)
                    with open(file_path, "r") as src_file:
                        with open(metadata_dest, "w") as dest_file:
                            dest_file.write(src_file.read())
                    break

        self.__extract_keyframes()
        self.__transcribe_videos()
        await self.__fetch_related_link_content(
            image_prediction=image_prediction, text_prediction=text_prediction
        )
        self.__index_search()

        logger.info("✅ Preprocessing completed")
        logger.info(f"Saving processed data to cache directory: {self.cache_dir}")
        self.__copy_directory(self.input_dir, cache_dir_input)
        self.__copy_directory(self.prompt_dir, cache_dir_prompt)

    async def prepare_location_images(
        self,
        prediction: dict,
        image_prediction: bool = True,
        text_prediction: bool = True,
    ) -> int:
        """
        Prepare verification data from the prediction with parallel fetching.

        Args:
            prediction (dict): Prediction dictionary with latitude, longitude, location, reason, and metadata
            image_prediction (bool): Whether to include original images in verification
            text_prediction (bool): Whether to include text-based verification

        Returns:
            int: Satellite image ID for reference in prompts
        """
        image_dir = self.image_dir
        satellite_image_id = len(list(self.image_dir.glob("image_*.*")))

        # Execute both operations in parallel
        logger.info("🔄 Fetching satellite image and location images in parallel...")

        # Ensure required API keys are present
        if not os.environ.get("GOOGLE_CLOUD_API_KEY"):
            raise ValueError(
                "GOOGLE_CLOUD_API_KEY environment variable is not set or is None."
            )
        if not os.environ.get("GOOGLE_CSE_CX"):
            raise ValueError(
                "GOOGLE_CSE_CX environment variable is not set or is None."
            )

        await asyncio.gather(
            self.__fetch_satellite_image_async(
                prediction["latitude"],
                prediction["longitude"],
                zoom=200,
                output_path=image_dir / f"image_{satellite_image_id:03d}.jpg",
            ),
            self.__search_images_async(
                location=prediction["location"],
                num_images=5,
                api_key=os.environ["GOOGLE_CLOUD_API_KEY"],
                cse_cx=os.environ["GOOGLE_CSE_CX"],
                output_dir=image_dir,
                image_id_offset=satellite_image_id + 1,
            ),
        )
        logger.info("✅ Verification data preparation completed")
        return satellite_image_id
