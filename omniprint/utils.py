"""
Utility functions
"""

import os
import glob
import pandas as pd 
import numpy as np 
import PIL
from PIL import Image, ImageColor, ImageOps
import cv2
import fontTools.ttLib 
import skimage.filters

import randomcolor
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color as colormath_convert
import colormath.color_diff

from poisson_image_editing import blit_images as poisson_editing
import transforms


def load_dict(lang):
    """Read the dictionnary file and returns all words in it.
    """

    lang_dict = []
    with open(
        os.path.join(os.path.dirname(__file__), "dicts", lang + ".txt"),
        "r",
        encoding="utf8",
        errors="ignore",
    ) as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
    return lang_dict


def load_fonts(lang):
    """Load all fonts in the fonts directories
    """
    
    font_index = "latin.txt"
    for p in glob.glob(os.path.join(os.path.dirname(__file__), "fonts", "index", "*.txt")):
        basename = os.path.basename(p)
        if lang == os.path.splitext(basename)[0]:
            font_index = basename
            break 
    with open(os.path.join(os.path.dirname(__file__), "fonts", "index", font_index), "r") as f:
        fonts = [os.path.join(os.path.dirname(__file__), "fonts", "fonts", p) for p in f.read().split("\n")] 
    return fonts 


def add_txt_extension(file_name):
    if "." not in os.path.basename(file_name):
        file_name = file_name + ".txt"
    return file_name 


def generate_label_dataframe(labels, external_dataframes=None, save_path=None):
    """
    labels:
        list of dict 
    save_path:
        If save_path is None, the generated dataframe will not be saved to disk
    """
    columns = set()
    columns.update(*[set(label.keys()) for label in labels])
    columns = sorted(list(columns))

    ordered_keys = ["image_name", "text", "unicode_code_point", "font_file"]

    for key in reversed(ordered_keys):
        try:
            index_ = columns.index(key)
        except ValueError:
            pass
        else:
            columns.insert(0, columns.pop(index_))
    
    df = pd.DataFrame(labels, columns=columns)
    if external_dataframes is not None:
        for external_dataframe in external_dataframes:
            df = df.merge(external_dataframe, how="left")
        
    
    if save_path is not None:
        df.to_csv(os.path.join(save_path, "raw_labels.csv"), sep="\t", encoding="utf-8")
    return df 

def float2int_image(img):
    """
    convert values 0-1 float to 0-255 int format
    img:
        np.array of dtype float 
    """
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def int2float_image(img):
    """
    convert values 0-255 int to 0-1 float format
    img:
        np.array of dtype uint8
    """ 
    return (np.clip(img, 0, 255) / 255).astype(np.float32)

def get_font_weight_range(font_file_path):
    try:
        ttfont = fontTools.ttLib.TTFont(font_file_path) 
        for axis in ttfont["fvar"].axes:
            if axis.axisTag == "wght":
                return axis.minValue, axis.maxValue 
    except Exception:
        return None, None 


def generate_random_color(method="randomcolor"):
    if method == "randomcolor":
        return ImageColor.getcolor(randomcolor.RandomColor().generate()[0], "RGB") 
    elif method == "uniform":
        return (np.random.randint(0, 256, None), 
                np.random.randint(0, 256, None), 
                np.random.randint(0, 256, None))
    else:
        raise Exception("Not implemented method {}".format(method))


def gaussian_blur_RGB(img, sigma):
    """
    img: PIL.Image (RGB)
    sigma: int
    returns PIL.Image
    """
    img = skimage.filters.gaussian(np.array(img), sigma=sigma, 
                                   multichannel=True, 
                                   preserve_range=True)
    return Image.fromarray(img.astype(np.uint8))


def compute_color_difference(color1, color2):
    """
    Computes the Delta E value (CIE2000) which ranges from 0 to 100
        smaller values correspond to similar colors
    
    color1: tuple of 3 integers in [0, 255]
    color2: tuple of 3 integers in [0, 255]
    returns float 
    """
    color1 = colormath_convert(sRGBColor(*color1, is_upscaled=True), LabColor)
    color2 = colormath_convert(sRGBColor(*color2, is_upscaled=True), LabColor)
    return colormath.color_diff.delta_e_cie2000(color1, color2)

def different_random_color(previous_color, threshold=10, 
                           maximum_trials=10, method="randomcolor"):
    """
    returns a random color that is not that 
    visually similar to previous_color 
    
    previous_color: tuple of 3 integers in [0, 255]
    threshold: int
        Threshold of the Delta E value (CIE2000)
        The Delta E value ranges from 0 to 100
        <= 1.0 Not perceptible by human eyes
        1 - 2 Perceptible through close observation
        2 - 10 Perceptible at a glance
        11 - 49 Colors are more similar than opposite
        100 Colors are exact opposite
    maximum_trials: int
        Maximum number of trials to generate "different" color.
        
        According to a test of 10000 runs, if the color is 
        generated by generate_random_color("randomcolor") and 
        threshold=10, then 95.45% of trials broke in the first 
        iteration, 4.2% of trials broke in the second iteration, 
        0.31% of trials broke in the third iteration, 0.04% of trials 
        broke in the fourth iteration, no trials needed five 
        iterations or more.
    
    returns a tuple of 3 integers in [0, 255]
    """
    for _ in range(maximum_trials):
        color = generate_random_color(method)
        if compute_color_difference(previous_color, color) > threshold:
            break
    return color


def fill_foreground_color(mask, new_color):
    """
    new_color: iterable of 3 integers representing RGB value 
    """
    if isinstance(new_color, str):
        new_color = tuple([int(xx) for xx in new_color.split(",")])
    width, height = mask.size
    background = Image.new("RGB", (width, height), color=new_color)
    foreground = Image.new("RGB", (width, height), color=(255, 255, 255)) # white
    background.paste(foreground, (0, 0), ImageOps.invert(mask))
    return background, mask  


def fill_foreground_image(mask, external_image):
    """
    external_image is a crop of an external image, 
    its size must match mask
    """
    width, height = mask.size
    foreground = Image.new("RGB", (width, height), color=(255, 255, 255)) # white
    external_image.paste(foreground, (0, 0), ImageOps.invert(mask))
    return external_image, mask  


def generate_text_outline(img, mask, outline, outline_size=5):
    """
    If outline is a PIL.Image.Image (external image crop), 
    then the text outline will be filled by the external image crop.
    
    If outline is a comma-separated str or a tuple of 3 integers, 
    then the text outline will be filled by the chosen color.
    
    If outline is a str and equals "random_color", then a random 
    color will be sampled and used to fill the text outline. Note 
    that there is no guarantee that the randomly sampled color will 
    be visually distinguishable from the foreground color or 
    background color.
    """
    _, mask_hollow = transforms.morph_gradient_transform(img, mask, 
        kernel_size=outline_size)
    if isinstance(outline, tuple) and len(outline) == 3:
        outline, _ = fill_foreground_color(mask_hollow, outline)
    elif isinstance(outline, str):
        # fill text outline with color
        if outline == "random_color":
            outline = generate_random_color()
        else:
            outline = tuple([int(xx) for xx in outline.split(",")])
        outline, _ = fill_foreground_color(mask_hollow, outline)
    elif isinstance(outline, PIL.Image.Image):
        outline, _ = fill_foreground_image(mask_hollow, outline)
    else:
        raise Exception("Invalid outline: {}".format(outline))
    
    outline.paste(img, (0, 0), mask)
    mask = Image.fromarray(np.maximum(np.array(mask_hollow), np.array(mask)))
    return outline, mask