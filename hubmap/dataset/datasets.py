from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cv2
import json
import pandas as pd
import os
from abc import ABC, abstractmethod


id2label = {0: "blood_vessel", 1: "glomerulus", 2: "unsure", 3: "background"}
label2id = {"blood_vessel": 0, "glomerulus": 1, "unsure": 2, "background": 3}

def generate_mask(img_data, with_background=False, as_id_mask=False):
    if as_id_mask:
        mask = np.full((512, 512, 1), 3, dtype=np.uint8)
    elif with_background:
        mask = np.zeros((512, 512, 4), dtype=np.uint8)
    else:
        mask = np.zeros((512, 512, 3), dtype=np.uint8)

    for group in img_data["annotations"]:
        coordinates = group["coordinates"][0]
        points = np.array(coordinates, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        temp = np.zeros((512,512), dtype=np.uint8)

        if as_id_mask:
            cv2.fillPoly(mask[:,:,0], [points], color=label2id[group["type"]])
        else:
            cv2.fillPoly(temp, [points], color=(255))
            channel = label2id[group["type"]]
            mask[:, :, channel] += temp
    if with_background and not as_id_mask:
        background_channel = label2id["background"]
        mask[:, :, background_channel] = 255 - np.sum(mask, axis=2)

    return mask


class AbstractDataset(ABC, Dataset):
    def __init__(self, image_dir, transform=None, with_background=False, as_id_mask=False):
        self.image_dir = image_dir
        self.transform = transform
        self.tiles_dicts = self._load_polygons()
        self.meta_df = pd.read_csv(f"{image_dir}/tile_meta.csv")
        self.with_background=with_background
        self.as_id_mask=as_id_mask

    def _load_polygons(self):
        with open(f"{self.image_dir}/polygons.jsonl", "r") as polygons:
            json_list = list(polygons)
        tiles_dicts = [json.loads(json_str) for json_str in json_list]
        return tiles_dicts

    def plot_base(self, idx):
        plt.figure(figsize=(10, 10))

        img_data = self.tiles_dicts[idx]
        image_path = f'{self.image_dir}/train/{img_data["id"]}.tif'
        image = Image.open(image_path)

        legend_elements = [
            mpatches.Patch(color="green", label="glomerulus"),
            mpatches.Patch(color="red", label="blood vessel"),
            mpatches.Patch(color="yellow", label="unsure"),
        ]

        for entry in img_data["annotations"]:
            if entry["type"] == "glomerulus":
                color = "green"
            elif entry["type"] == "blood_vessel":
                color = "red"
            else:
                color = "yellow"

            sublist = entry["coordinates"][0]
            x = []
            y = []

            for datapoint in sublist:
                x.append(datapoint[0])
                y.append(datapoint[1])

            plt.scatter(x, y, s=0)
            plt.fill(x, y, color, alpha=0.5)

        plt.imshow(image)
        plt.title("Tile with annotations")
        plt.legend(handles=legend_elements, loc="upper right")

    # plots a example after transformation i.e. the transformed image and all three masksk
    def plot_example(self, idx):
        img, mask = self[idx]
        img = img.permute(1, 2, 0).numpy()
        mask = mask.permute(1, 2, 0).numpy()
        if self.as_id_mask:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            axs[0].imshow(img)
            axs[0].set_title("Image")
            axs[1].imshow(mask[:, :, 0]*80, cmap="gray")
            axs[1].set_title("id_mask")
        else:
            if self.with_background:
                fig, axs = plt.subplots(1, 5, figsize=(15, 5))
            else:
                fig, axs = plt.subplots(1, 4, figsize=(15, 5))
            axs[0].imshow(img)
            axs[0].set_title("Image")
            axs[1].imshow(mask[:, :, 0], cmap="gray")
            axs[1].set_title("blood_vessel mask")
            axs[2].imshow(mask[:, :, 1], cmap="gray")
            axs[2].set_title("glomerulus mask")
            axs[3].imshow(mask[:, :, 2], cmap="gray")
            axs[3].set_title("unsure mask")
            if self.with_background:
                axs[4].imshow(mask[:,:,3], cmap='gray')
                axs[4].set_title("background mask")

        plt.tight_layout()
        plt.show()

    def get(self, idx: int, transform=None):
        img_data = self.tiles_dicts[idx]
        image_path = f'{self.image_dir}/train/{img_data["id"]}.tif'
        image = Image.open(image_path)
        mask = generate_mask(img_data, with_background=self.with_background, as_id_mask=self.as_id_mask)
        if transform:
            image, mask = transform(image, mask)
        return image, mask

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


# Dataset for all annotated images,
# index operations return mask in the following format:
# [mask_idx, h, w] where mask_dix coresponds to:
# mask_idx: 0 => blood_vessel
# mask_idx:1 => glomerulus
# mask_idx:2 => unsure
# mask_idx:3 => background
class BaseDataset(AbstractDataset):
    def __init__(self, image_dir, transform=None, with_background=False, as_id_mask=False):
        super().__init__(image_dir, transform, with_background, as_id_mask)

    def __len__(self):
        return len(self.tiles_dicts)

    def __getitem__(self, index):
        img_data = self.tiles_dicts[index]
        image_path = f'{self.image_dir}/train/{img_data["id"]}.tif'
        image = np.asarray(Image.open(image_path))
        mask = generate_mask(img_data, with_background=self.with_background, as_id_mask=self.as_id_mask)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
    

class TrainTestValBaseDataset(AbstractDataset):
    def __init__(self, image_dir, sub_dir, transform=None, with_background=False, as_id_mask=False):
        super().__init__(image_dir, transform, with_background, as_id_mask)
        self.sub_dir = sub_dir
        self.ids = os.listdir(image_dir + sub_dir)
        self.id_dict = {}
        for dict in self.tiles_dicts:
            self.id_dict[dict["id"]] = dict
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        image_id = self.ids[index]
        img_data = self.id_dict[str(os.path.splitext(image_id)[0])]
        image_path = f'{self.image_dir + self.sub_dir}/{image_id}'
        image = np.asarray(Image.open(image_path))
        mask = generate_mask(img_data, with_background=self.with_background, as_id_mask=self.as_id_mask)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
    
    def get(self, idx: int, transform=None):
        image_id = self.ids[idx]
        img_data = self.id_dict[str(os.path.splitext(image_id)[0])]
        image_path = f'{self.image_dir + self.sub_dir}/{image_id}'
        image = np.asarray(Image.open(image_path))
        mask = generate_mask(img_data, with_background=self.with_background, as_id_mask=self.as_id_mask)

        if transform is not None:
            image, mask = transform(image, mask)

        return image, mask
    

    
class TrainDataset(TrainTestValBaseDataset):
    def __init__(self, image_dir, transform=None, with_background=False, as_id_mask=False):
        super().__init__(image_dir, 'train/', transform, with_background, as_id_mask)

class TestDataset(TrainTestValBaseDataset):
    def __init__(self, image_dir, transform=None, with_background=False, as_id_mask=False):
        super().__init__(image_dir, 'test/', transform, with_background, as_id_mask)

class ValDataset(TrainTestValBaseDataset):
    def __init__(self, image_dir, transform=None, with_background=False, as_id_mask=False):
        super().__init__(image_dir, 'val/', transform, with_background, as_id_mask)



def _gen_dict_from_json_list(lst):
    out = dict()
    for jsn in lst:
        out[jsn["id"]] = jsn
    return out


# Dataset for all EXPERT annotated images (422 tiles),
# index operations return mask in the following format:
# [mask_idx, h, w] where mask_dix coresponds to:
# mask_idx: 0 => blood_vessel
# mask_idx:1 => glomerulus
# mask_idx:2 => unsure
class ExpertDataset(AbstractDataset):
    def __init__(self, image_dir, transform=None):
        super().__init__(image_dir, transform)
        self.d1_ids = self.meta_df.loc[self.meta_df["dataset"] == 1]["id"].tolist()
        self.id_map = _gen_dict_from_json_list(self.tiles_dicts)

    def __len__(self):
        return len(self.d1_ids)

    def __getitem__(self, index):
        img_data = self.id_map[self.d1_ids[index]]
        image_path = f'{self.image_dir}/train/{img_data["id"]}.tif'
        image = np.asarray(Image.open(image_path))
        mask = generate_mask(img_data)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


# ich mache nicht die regeln :D
# https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209
class DatasetFromSubset(AbstractDataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x, y = self.transform(x, y)
        return x, y

    def __len__(self):
        return len(self.subset)