from torchvision.transforms import functional as F
from torchvision import transforms
import random
import torch
import numpy as np


# Das ist natürlich super bad von pytorch, aber anders kann ich random transformations nicht für
# image und mask machen, keine ahnung warum es keinen standart dafür gibt
# https://discuss.huggingface.co/t/apply-same-transform-to-pixel-values-and-labels-for-semantic-segmentation/16267


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

    def __iter__(self):
        return iter(self.transforms)


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(
            image,
            self.size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
        target = F.resize(
            target,
            self.size,
            interpolation=transforms.InterpolationMode.NEAREST,
            antialias=True,
        )
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = transforms.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class RandomRotate90:
    # TODO: this is not working properly
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            turns = random.choice([0, 1, 2, 3, 4])
            image = image.rotate(90 * turns)
            mask = mask.rotate(90 * turns)
        return image, mask


class RandomRotation:
    # TODO: this is not working properly
    def __init__(self, degrees: int, p=0.5):
        self.p = p
        self.degree = degrees

    def __call__(self, image, mask):
        if random.random() < self.p:
            turns = random.choice([0, 1, 2, 3, 4])
            image = image.rotate(self.degrees * turns)
            mask = mask.rotate(self.degrees * turns)
        return image, mask


class RandomHueSaturationValue:
    def __init__(
        self, hue_shift=(-0.05, 0.05), sat_shift=(0.8, 1.2), val_shift=(0.8, 1.2)
    ):
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift

    def __call__(self, image, mask):
        h_shift = np.random.uniform(self.hue_shift[0], self.hue_shift[1])
        s_shift = np.random.uniform(self.sat_shift[0], self.sat_shift[1])
        v_shift = np.random.uniform(self.val_shift[0], self.val_shift[1])

        image = F.adjust_hue(image, h_shift)
        image = F.adjust_saturation(image, s_shift)
        image = F.adjust_brightness(image, v_shift)

        return image, mask


class RandomGamma:
    def __init__(self, gamma=(0.2, 2.0)):
        self.gamma = gamma

    def __call__(self, image, mask):
        gamma_value = np.random.uniform(self.gamma[0], self.gamma[1])
        image = F.adjust_gamma(image, gamma_value)

        return image, mask


class RandomBrightness:
    def __init__(self, brightness=(0.5, 2.0)):
        self.brightness = brightness

    def __call__(self, image, mask):
        brightness_factor = np.random.uniform(self.brightness[0], self.brightness[1])
        image = F.adjust_brightness(image, brightness_factor)

        return image, mask


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor:
    def __init__(self, mask_as_integer=False):
        self.mask_as_integer = mask_as_integer

    def __call__(self, image, mask):
        if self.mask_as_integer:
            return F.to_tensor(np.array(image)), torch.tensor(
                mask, dtype=torch.uint8
            ).permute(2, 0, 1)
        else:
            return F.to_tensor(np.array(image)), F.to_tensor(np.array(mask))


class Grayscale:
    def __init__(self):
        pass

    def __call__(self, image, mask):
        image_gray = transforms.Grayscale()(image)
        return image_gray, mask


class CustomOnImage:
    def __init__(self, inner):
        self._inner = inner

    def __call__(self, image, mask):
        out = self._inner(image)
        return out, mask
