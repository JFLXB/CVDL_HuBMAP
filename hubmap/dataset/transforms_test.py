import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import transforms as tr
from torchvision import transforms


# use this if u dont have an example image
def create_test_image(size):
    quadrant_size = size // 2
    red_quadrant = np.full(
        (quadrant_size, quadrant_size, 3), [255, 0, 0], dtype=np.uint8
    )
    green_quadrant = np.full(
        (quadrant_size, quadrant_size, 3), [0, 255, 0], dtype=np.uint8
    )
    blue_quadrant = np.full(
        (quadrant_size, quadrant_size, 3), [0, 0, 255], dtype=np.uint8
    )
    white_quadrant = np.full(
        (quadrant_size, quadrant_size, 3), [255, 255, 255], dtype=np.uint8
    )

    top_half = np.concatenate([red_quadrant, green_quadrant], axis=1)
    bottom_half = np.concatenate([blue_quadrant, white_quadrant], axis=1)

    img = np.concatenate([top_half, bottom_half], axis=0)

    return Image.fromarray(img, "RGB")


def visualize_transformation(transformation, trans_name, size=256):
    # img = create_test_image(size)
    img = Image.open("dog.jpeg")
    mask = img.copy()

    img_transformed, mask_transformed = transformation(img, mask)

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[1].imshow(img_transformed)
    ax[1].set_title(f"{trans_name} Image")
    ax[2].imshow(mask_transformed)
    ax[2].set_title(f"{trans_name} Mask")
    plt.show()


resize = tr.Resize(128)
visualize_transformation(resize, "Resize")

flip = tr.RandomHorizontalFlip(1)
visualize_transformation(flip, "RandomHorizontalFlip")

vflip = tr.RandomVerticalFlip(1)
visualize_transformation(vflip, "RandomVerticalFlip")

crop = tr.RandomCrop(128)
visualize_transformation(crop, "RandomCrop")

hue_sat = tr.RandomHueSaturationValue()
visualize_transformation(hue_sat, "HueSat")

gamma = tr.RandomGamma()
visualize_transformation(gamma, "Gamma")

