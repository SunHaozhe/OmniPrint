import cv2
import math
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from utils import different_random_color, gaussian_blur_RGB


def gaussian_noise(height, width):
    """
        Create a background with Gaussian noise (to mimic paper)
    """

    # We create an all white image
    image = np.ones((height, width)) * 255

    # We add gaussian noise
    cv2.randn(image, 235, 10)

    return Image.fromarray(image).convert("RGB")


def plain_white(height, width):
    """
        Create a plain white background
    """

    return Image.new("RGB", (width, height), color=(255, 255, 255))

def plain_color(height, width, color):
    """
        Create a plain color background
        color: iterable of 3 integers representing RGB value 
    """

    return Image.new("RGB", (width, height), color=color)


def random_color_composition(height, width, foreground_color=None, 
                             background_random_color_composition_params=None):
    if foreground_color is None:
        foreground_color = (0, 0, 0)
    background_color = different_random_color(foreground_color, method="randomcolor")
    background_polygon_fill_color = different_random_color(foreground_color, method="randomcolor")
    if np.random.uniform(0, 1) < 0.9:
        background_polygon_outline_color = background_polygon_fill_color
    else:
        background_polygon_outline_color = different_random_color(foreground_color, method="randomcolor")

    if background_random_color_composition_params is None:
        background_random_color_composition_params = np.random.uniform(0, 1.5, size=4)
        background_random_color_composition_params[2] = np.random.uniform(0.5, 0.9)
        background_random_color_composition_params[3] = np.random.uniform(0, 90)

    img = plain_color(height, width, background_color)
    draw = ImageDraw.Draw(img)
    draw.regular_polygon((img.size[0] * background_random_color_composition_params[0], 
                          img.size[1] * background_random_color_composition_params[1], 
                          min(img.size[0], img.size[1]) * background_random_color_composition_params[2]), 
                          n_sides=4, 
                          fill=background_polygon_fill_color, 
                          outline=background_polygon_outline_color, 
                          rotation=background_random_color_composition_params[3])
    img = gaussian_blur_RGB(img, sigma=2)
    return img, (background_color, background_polygon_fill_color, 
        background_polygon_outline_color, background_random_color_composition_params)


def quasicrystal(height, width):
    """
        Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
    """

    image = Image.new("L", (width, height))
    pixels = image.load()

    frequency = random.random() * 30 + 20  # frequency
    phase = random.random() * 2 * math.pi  # phase
    rotation_count = random.randint(10, 20)  # of rotations

    for kw in range(width):
        y = float(kw) / (width - 1) * 4 * math.pi - 2 * math.pi
        for kh in range(height):
            x = float(kh) / (height - 1) * 4 * math.pi - 2 * math.pi
            z = 0.0
            for i in range(rotation_count):
                r = math.hypot(x, y)
                a = math.atan2(y, x) + i * math.pi * 2.0 / rotation_count
                z += math.cos(r * math.sin(a) * frequency + phase)
            c = int(255 - round(255 * z / rotation_count))
            pixels[kw, kh] = c  # grayscale
    return image.convert("RGB")


def image(height, width, image_dir):
    """
        Create a background with a image
    """
    images = [xx for xx in os.listdir(image_dir) \
              if xx.endswith(".jpeg") or xx.endswith(".jpg") or xx.endswith(".png")]

    if len(images) > 0:
        image_name = images[random.randint(0, len(images) - 1)]
        pic = Image.open(os.path.join(image_dir, image_name))
        pic_original_width = pic.size[0]
        pic_original_height = pic.size[1]

        if pic.size[0] < width:
            pic = pic.resize([width, int(pic.size[1] * (width / pic.size[0]))], Image.ANTIALIAS)
        if pic.size[1] < height:
            pic = pic.resize([int(pic.size[0] * (height / pic.size[1])), height], Image.ANTIALIAS)

        pic_final_width = pic.size[0]
        pic_final_height = pic.size[1]

        if pic.size[0] == width:
            x = 0
        else:
            x = random.randint(0, pic.size[0] - width)
        if pic.size[1] == height:
            y = 0
        else:
            y = random.randint(0, pic.size[1] - height)

        return pic.crop((x, y, x + width, y + height)), (image_name, pic_original_width, pic_original_height, 
            pic_final_width, pic_final_height, x, y, x + width, y + height)
    else:
        raise Exception("No images where found in the images folder!")
