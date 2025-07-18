import os
import pickle
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import transformers
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, get_worker_info
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPModel,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModel,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow truncated images to be loaded

from io import BytesIO
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torchvision.transforms as T
from datasets import load_dataset
from huggingface_hub import login
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

__all__ = [
    "MP16StreamingDataset",
    "mp16_collate",
]


class MP16StreamingDataset(IterableDataset):
    """Stream **MP‑16** samples from the HuggingFace Hub and yield a simple
    tuple per example::

        (image, text, longitude, latitude)

    * **image**  – either a tensor (``C×H×W``) if *vision_processor* is set or if
      the fallback transform is used, otherwise a PIL image.
    * **text**   – caption string (either provided by the dataset or generated
      from location fields).
    * **longitude**, **latitude** – floats.

    The class is an :class:`torch.utils.data.IterableDataset`, so wrap it in a
    :class:`~torch.utils.data.DataLoader` for batching.
    """

    def __init__(
        self,
        repo_id: str = "tduongvn/MP16-Pro-shards",
        split: str = "train",
        vision_processor: Optional[Any] = None,
        shuffle_buffer: int = 10_000,
        HF_TOKEN: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.split = split
        self.vision_processor = vision_processor
        self.shuffle_buffer = shuffle_buffer
        self.HF_TOKEN = HF_TOKEN

        # Base transform when we *don't* have a fancy processor
        self.fallback_transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(size=224),
                T.ToTensor(),
            ]
        )

        # Prepare an initial dataset iterator for the main process
        self._base_iter = self._new_iterator()

    # ──────────────────────────────────────────────────────────────────────────
    # Internals                                                               ─┘

    def _new_iterator(self):
        if self.HF_TOKEN is not None:
            login(token=self.HF_TOKEN)
        return (
            load_dataset(self.repo_id, split=self.split, streaming=True)
            .shuffle(buffer_size=self.shuffle_buffer)
            .__iter__()
        )

    def _decode_image(self, img_bytes):
        """bytes → PIL.Image or tensor (if processor is set)."""
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        if self.vision_processor is not None:
            return self.vision_processor(images=img, return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)
        return self.fallback_transform(img)

    def _caption(self, ex_json: Dict[str, Any]) -> str:
        parts = [ex_json.get(k) for k in ("city", "state", "country") if ex_json.get(k)]
        return "A street view photo taken in " + ", ".join(parts)

    # ──────────────────────────────────────────────────────────────────────────
    # IterableDataset API                                                     ─┘

    def __iter__(self) -> Iterator[Tuple[Any, str, float, float]]:
        # Each DataLoader worker gets its own iterator to avoid state clashes.
        worker = get_worker_info()
        iterator = self._new_iterator() if worker is not None else self._base_iter

        for ex in iterator:
            # Dataset structure: {'jpg': <PIL or bytes>, 'json': {...}, ...}
            img_field = ex["jpg"]
            if isinstance(img_field, Image.Image):
                img = img_field.convert("RGB")
                if self.vision_processor is not None:
                    img = self.vision_processor(images=img, return_tensors="pt")[
                        "pixel_values"
                    ].squeeze(0)
                else:
                    img = self.fallback_transform(img)
            else:  # bytes
                img = self._decode_image(img_field)

            meta = ex["json"] if "json" in ex else {}
            lon = float(meta.get("lon", meta.get("LON")))
            lat = float(meta.get("lat", meta.get("LAT")))
            text = meta.get("text") or self._caption(meta)

            yield img, text, lon, lat

    # No __len__ – this is a stream.


# ─────────────────────────────────────────────────────────────────────────────
# Collate                                                                     ─┘


def make_mp16_collate(text_processor):
    def collate(batch):
        images, texts, lons, lats = zip(*batch)

        images = torch.stack(images)  # (B, C, H, W)

        token_out = text_processor(
            list(texts),
            padding="longest",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        lons = torch.tensor(lons, dtype=torch.float32)
        lats = torch.tensor(lats, dtype=torch.float32)

        return images, token_out, lons, lats

    return collate


class MP16Dataset(VisionDataset):
    def __init__(
        self,
        root_path="data/mp16/",
        text_data_path="MP16_Pro_places365.csv",
        image_data_path="mp-16-images.tar",
        member_info_path="tar_index.pkl",
        vision_processor=None,
        text_processor=None,
    ):
        super().__init__(self)
        self.root_path = root_path
        self.text_data_path = text_data_path
        self.image_data_path = image_data_path
        self.text_data = pd.read_csv(os.path.join(self.root_path, self.text_data_path))
        self.text_data["IMG_ID"] = self.text_data["IMG_ID"].apply(
            lambda x: x.replace("/", "_")
        )
        # self.text_data = self.text_data[self.text_data['IMG_ID'].str.endswith('.jpg')] # only keep jpg images
        print("read text data success")
        worker = get_worker_info()
        worker = worker.id if worker else None
        self.tar_obj = {worker: tarfile.open(os.path.join(root_path, image_data_path))}
        # self.tar = tarfile.open(os.path.join(root_path, image_data_path))

        if os.path.exists(os.path.join(self.root_path, member_info_path)):
            with open(os.path.join(self.root_path, member_info_path), "rb") as f:
                self.tar_index = pickle.load(f)
            all_image_names = list(self.tar_index.keys())
            print("load tar index success")
        else:
            print("no exist tar index success, need building...")
            self.tar_index = {}
            all_image_names = []
            for member in tqdm(self.tar_obj[worker]):
                if member.name.endswith(".jpg") and member.size > 5120:
                    self.tar_index[member.name.split("/")[1]] = member
                    all_image_names.append(member.name.split("/")[1])
            print("tar index buidling success")
            with open(os.path.join(self.root_path, member_info_path), "wb") as f:
                pickle.dump(self.tar_index, f)
        all_image_names = set(all_image_names)

        self.text_data = self.text_data[self.text_data["country"].notnull()]
        self.text_data = self.text_data[self.text_data["IMG_ID"].isin(all_image_names)]
        print("data columns: ", self.text_data.shape[0])

        # location from str to float
        self.text_data.loc[:, "LON"] = self.text_data["LON"].astype(float)
        self.text_data.loc[:, "LAT"] = self.text_data["LAT"].astype(float)
        print("location from str to float success")

        # image transform
        self.transform = T.Resize(size=(512, 512))
        self.transform_totensor = T.ToTensor()

        self.vision_processor = vision_processor
        self.text_processor = text_processor

        # Define the contrast transforms here
        self.contrast_transforms = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(size=224),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=9),
                T.ToTensor(),
                # T.Normalize((0.5,), (0.5,))
            ]
        )

        # self.text_data.to_csv('/data/mp-16/MP16_Pro_filtered.csv', index=False)

    def caption_generation(self, row):
        pass

    def __getitem__(self, index):
        image_path = self.text_data.iloc[index]["IMG_ID"]
        text = ""
        neighbourhood, city, county, state, region, country, continent = (
            self.text_data.iloc[index][
                [
                    "neighbourhood",
                    "city",
                    "county",
                    "state",
                    "region",
                    "country",
                    "continent",
                ]
            ]
        )
        # location_elements = [element for element in [neighbourhood, city, state, country] if element is not np.nan and str(element) != 'nan']
        location_elements = [
            element
            for element in [city, state, country]
            if element is not np.nan and str(element) != "nan"
        ]
        text = "A street view photo taken in " + ", ".join(location_elements)

        longitude = self.text_data.iloc[index]["LON"]
        latitude = self.text_data.iloc[index]["LAT"]
        # read the image from self.tar
        worker = get_worker_info()
        worker = worker.id if worker else None
        if worker not in self.tar_obj:
            self.tar_obj[worker] = tarfile.open(
                os.path.join(self.root_path, self.image_data_path)
            )
        image = self.tar_obj[worker].extractfile(self.tar_index[image_path])
        image = Image.open(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.vision_processor:
            image = self.vision_processor(images=image, return_tensors="pt")[
                "pixel_values"
            ].reshape(3, 224, 224)

        return image, text, longitude, latitude

    def __len__(self):
        return len(self.text_data)


class im2gps3kDataset(VisionDataset):
    def __init__(
        self,
        root_path="./data/im2gps3k",
        text_data_path="im2gps3k_places365.csv",
        image_data_path="images/",
        vision_processor=None,
        text_processor=None,
    ):
        super().__init__(self)
        print("start loading im2gps...")
        self.root_path = root_path
        self.text_data_path = text_data_path
        self.image_data_path = image_data_path
        self.text_data = pd.read_csv(os.path.join(self.root_path, self.text_data_path))
        # self.text_data = self.text_data[self.text_data['IMG_ID'].str.endswith('.jpg')] # only keep jpg images
        print("read text data success")

        # location from str to float
        self.text_data.loc[:, "LAT"] = self.text_data["LAT"].astype(float)
        self.text_data.loc[:, "LON"] = self.text_data["LON"].astype(float)
        print("location from str to float success")

        self.vision_processor = vision_processor
        self.text_processor = text_processor

        self.tencrop = T.TenCrop(224)

    def __getitem__(self, index):
        image_path = self.text_data.iloc[index]["IMG_ID"]
        text = image_path

        longitude = self.text_data.iloc[index]["LON"]
        latitude = self.text_data.iloc[index]["LAT"]

        image = Image.open(
            os.path.join(self.root_path, self.image_data_path, image_path)
        )

        if image.mode != "RGB":
            image = image.convert("RGB")

        # image = self.tencrop(image) # for tencrop

        if self.vision_processor:
            image = self.vision_processor(images=image, return_tensors="pt")[
                "pixel_values"
            ].reshape(-1, 224, 224)

        return image, text, longitude, latitude

    def __len__(self):
        return len(self.text_data)


class yfcc4kDataset(VisionDataset):
    def __init__(
        self,
        root_path="./data/yfcc4k",
        text_data_path="yfcc4k_places365.csv",
        image_data_path="images/",
        vision_processor=None,
        text_processor=None,
    ):
        super().__init__(self)
        print("start loading yfcc4k...")
        self.root_path = root_path
        self.text_data_path = text_data_path
        self.image_data_path = image_data_path
        self.text_data = pd.read_csv(os.path.join(self.root_path, self.text_data_path))
        # self.text_data = self.text_data[self.text_data['IMG_ID'].str.endswith('.jpg')] # only keep jpg images
        print("read text data success")

        # location from str to float
        self.text_data.loc[:, "LAT"] = self.text_data["LAT"].astype(float)
        self.text_data.loc[:, "LON"] = self.text_data["LON"].astype(float)
        print("location from str to float success")

        self.vision_processor = vision_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        image_path = self.text_data.iloc[index]["IMG_ID"]
        text = image_path

        longitude = self.text_data.iloc[index]["LON"]
        latitude = self.text_data.iloc[index]["LAT"]

        image = Image.open(
            os.path.join(self.root_path, self.image_data_path, image_path)
        )

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.vision_processor:
            image = self.vision_processor(images=image, return_tensors="pt")[
                "pixel_values"
            ].reshape(-1, 224, 224)

        return image, text, longitude, latitude

    def __len__(self):
        return len(self.text_data)
