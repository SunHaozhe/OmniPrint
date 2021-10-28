import argparse
import os
import errno
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import random 
import string
import sys
import numpy as np 
import datetime
import pandas as pd 

import tqdm
from omniprint.string_generator import create_strings_from_dict
from omniprint.string_generator import create_strings_from_file
from omniprint.string_generator import create_strings_from_dict_equal

from omniprint.utils import load_dict, load_fonts 
from omniprint.utils import add_txt_extension, generate_label_dataframe 
from omniprint.data_generator import TextDataGenerator
import multiprocessing



def parse_margins(x):
    x = x.split(",")
    if len(x) == 1:
        return [float(x[0])] * 4
    return [float(el) for el in x]


def parse_affine_perspective_transformations(x):
    x = x.split(",")
    if len(x) == 1:
        return [float(x[0])] * 6
    return [float(el) for el in x]


def parse_linear_transform(x):
    x = x.split(",")
    if len(x) in [4, 5, 9]:
        return [float(el) for el in x]
    else:
        raise Exception("The length of --linear_transform must be 4, 5 or 9.")


def parse_perspective_transform(x):
    x = x.split(",")
    if len(x) == 8:
        return [float(el) for el in x]
    else:
        raise Exception("The length of --perspective_transform must be 8.")

def parse_morphological_image_processing_iteration(x):
    x = x.split(",") 
    assert len(x) in [2, 3]
    x_0 = int(x[0])
    assert x_0 >= 1 
    x_1 = int(x[1])
    assert x_1 >= 1 
    if len(x) == 2:
        return x_0, x_1, None 
    else:
        return x_0, x_1, x[2] 

def parse_morphological_image_processing(x):
    x = x.split(",") 
    assert len(x) in [1, 2]
    x_0 = int(x[0])
    assert x_0 >= 1 
    if len(x) == 1:
        return x_0, None 
    else:
        return x_0, x[1] 


def parse_arguments():
    """
        Parse the command line arguments of the program.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic text data for text recognition."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        nargs="?", 
        help="The output directory", 
        default="out"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="""The total number of images to be created. In this situation, 
        the number of images for each character can differ. The semantic of 
        this argument can be modified by the argument --equal_char is enabled.""",
        default=10
    )
    parser.add_argument(
        "-eqchr", 
        "--equal_char",
        action="store_true",
        help="""If enabled, the argument --count will be considered as the number of images 
        for each character, rather than the total number of images to be created. In 
        this situation, the total number of images to be created will be the number of 
        characters of this alphabet multiplied by the number of the value of the argument 
        --count.""",
        default=False
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        nargs="?",
        help="""Define the height of the produced images. If the option 
        --ensure_square_layout is activated, then this will also be the 
        width of the produced images, otherwise the width will be determined 
        by both the length of the text and the height.""",
        default=32
    )
    parser.add_argument(
        "-p",
        "--nb_processes",
        type=int,
        nargs="?",
        help="""Define the number of processes to use for image generation. 
        If not provided, this equals to the number of CPU cores""", 
        default=None
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="?",
        help="Define the extension to save the image with",
        default="png"
    )
    parser.add_argument(
        "--output_subdir", 
        type=str, 
        default=None, 
        help="""The output subdirectory. This option is reserved for 
        multilingual/meta-learning dataset generation. In general, 
        it is not recommended to manually set this option as the user."""
    )
    parser.add_argument(
        "-bl",
        "--blur",
        type=int,
        nargs="+",
        help="""Apply gaussian blur to the resulting sample. 
        Should be an integer defining the blur radius, 0 by default. If this argument 
        receives two space-seperated integers, a random value will be used, the first 
        value will be the lower bound, the second value will be the upper bound.""",
        default=None
    )
    parser.add_argument(
        "-b",
        "--background",
        type=str,
        default="plain_white",
        help="""Define what background to use. Possible values include: 
        plain_white, random_plain_color, image, random_color_composition, 
        gaussian_noise, quasicrystal or 3 comma-separated integers 
        representing a RGB color."""
    )
    parser.add_argument(
        "-blend", 
        "--image_blending_method", 
        type=str, 
        default="trivial", 
        help="""How to blend foreground text with background. 
        Current implementation includes trivial and poisson.""")
    parser.add_argument(
        "-om",
        "--output_mask",
        action="store_true",
        help="Define if the generator will return masks for the text",
        default=False
    )
    parser.add_argument(
        "-m",
        "--margins",
        type=parse_margins,
        nargs="?",
        help="""Define the margins (percentage) around the text when rendered. 
        Each element (top, left, bottom and right) should be a float""",
        default=[0, 0, 0, 0]
    )
    parser.add_argument(
        "-ft", 
        "--font", 
        type=str, 
        nargs="?", 
        help="Define font to be used"
    )
    parser.add_argument(
        "-fd",
        "--font_dir",
        type=str,
        nargs="?",
        help="Define a font directory to be used"
    )
    parser.add_argument(
        "-fidx",
        "--font_index",
        type=str,
        nargs="?",
        help="Define the font index file to be used, an example is fonts{}latin.txt".format(os.sep)
    )
    parser.add_argument(
        "-id",
        "--image_dir",
        type=str,
        nargs="?",
        help="""Define an image directory to use when background is 
        set to image or the option foreground_image is set.""",
        default=os.path.join(os.path.split(os.path.realpath(__file__))[0], "images")
    )
    parser.add_argument(
        "-fgdid",
        "--foreground_image_dir",
        type=str,
        nargs="?",
        help="""Define an image directory to use when the option foreground_image is set.""",
        default=os.path.join(os.path.split(os.path.realpath(__file__))[0], "images")
    )
    parser.add_argument(
        "-otlnid",
        "--outline_image_dir",
        type=str,
        nargs="?",
        help="""Define an image directory to use if text outline exists and needs to 
        be filled by natural image/texture.""",
        default=os.path.join(os.path.split(os.path.realpath(__file__))[0], "images")
    )
    parser.add_argument(
        "-otln",
        "--outline",
        type=str,
        help="""If this option is used, text will have outline effect. The possible 
        values include: "image", "random_color" or 3 comma-separated integers 
        representing a RGB color.""",
        default=None
    )
    parser.add_argument(
        "-otlnsz",
        "--outline_size",
        type=int,
        help="""The size of text outline.""",
        default=5
    )
    parser.add_argument(
        "-dt", 
        "--dict", 
        type=str, 
        nargs="?", 
        help="""Define the dictionary to be used. 
        Example: --dict alphabets/fine/basic_latin_lowercase""",
        default="alphabets/***EMPTY***"
    )
    parser.add_argument(
        "-txtf", 
        "--textfile", 
        type=str, 
        nargs="?", 
        help="""Define the text file to be used, 
        each line of this text file will be used following the order. 
        This option is used when the option --dict is not set.""",
        default="textfiles/***EMPTY***"
    )
    parser.add_argument(
        "-fwt",
        "--font_weight",
        type=float, 
        nargs="+",
        help="""Define the stroke width (font weight). If two space-seperated float are presented, 
        then a random value will be used. The first value will be the lower bound, the second 
        value will be the upper bound.""",
        default=400
    )
    parser.add_argument(
        "-stf",
        "--stroke_fill",
        type=str, 
        help="Define the color of the strokes",
        default=None
    )
    parser.add_argument(
        "-rstf",
        "--random_stroke_fill", 
        action="store_true",
        help="Use random color to fill strokes.", 
        default=False
    )
    parser.add_argument(
        "-fgdimg",
        "--foreground_image", 
        action="store_true",
        help="""When set, use random crop of images to fill foreground text, 
        the options random_stroke_fill and stroke_fill will be ignored if 
        this option is set. The images to crop are from the value of the 
        option foreground_image_dir.""", 
        default=False
    )
    parser.add_argument(
        "-im",
        "--image_mode",
        type=str,
        nargs="?",
        help="""Define the image mode to be used. RGB is default, L means 8-bit grayscale 
                images, 1 means 1-bit binary images stored with one pixel per byte, etc.""",
        default="RGB"
    )
    parser.add_argument(
        "-rsd",
        "--random_seed",
        type=int,
        help="Random seed",
        default=None
    )
    parser.add_argument(
        "-esl",
        "--ensure_square_layout",
        action="store_true", 
        help="Whether the width should be the same as the height",
        default=False
    )
    """
    parser.add_argument(
        "-otlwd",
        "--outline_width",
        type=int,
        help="Width of stroke outline. Not yet implemented",
        default=None
    )
    """
    parser.add_argument(
        "-fsz",
        "--font_size",
        type=int,
        help="Font size in point",
        default=192
    )
    parser.add_argument(
        "-lt",
        "--linear_transform",
        type=parse_linear_transform,
        nargs="?",
        help="""The parameter for linear transform. The length must be 4, 5 or 9. 
        Length 4 corresponds low level parameterization, which means a, b, d, e, this 
        is the most customizable parameterization. Length 5 and length 9 correspond to 
        high level parameterization. Length 5 means rotation, shear_x, shear_y, 
        scale_x, scale_y. Length 9 means rotation, shear_x, shear_y, 
        scale_x, scale_y, alpha_, beta_, gamma_, delta_. If this parameter is set, 
        other linear transform parameters like rotation, shear_x, etc. will be ignored""",
        default=None
    )
    parser.add_argument(
        "-rtn",
        "--rotation",
        type=float,
        nargs="+",
        help="""Define rotation angle (in degree) of the generated text. 
        Used only when --linear_transform is not set. When two space-separated float 
        are present, a random value will be used. The first value will be used as the 
        lower bound, the second value will be used as the upper bound.""",
        default=None
    )
    parser.add_argument(
        "-shrx",
        "--shear_x",
        type=float,
        nargs="+",
        help="""High level linear transform parameter, horizontal shear. 
        Used only when --linear_transform is not set. When two space-separated float 
        are present, a random value will be used. The first value will be used as the 
        lower bound, the second value will be used as the upper bound.""",
        default=None
    )
    parser.add_argument(
        "-shry",
        "--shear_y",
        type=float,
        nargs="+",
        help="""High level linear transform parameter, vertical shear. 
        Used only when --linear_transform is not set. When two space-separated float 
        are present, a random value will be used. The first value will be used as the 
        lower bound, the second value will be used as the upper bound.""",
        default=None
    )
    parser.add_argument(
        "-sclx",
        "--scale_x",
        type=float,
        nargs="+",
        help="""High level linear transform parameter, horizontal scale. 
        Used only when --linear_transform is not set. When two space-separated float 
        are present, a random value will be used. The first value will be used as the 
        lower bound, the second value will be used as the upper bound.""",
        default=None
    )
    parser.add_argument(
        "-scly",
        "--scale_y",
        type=float,
        nargs="+",
        help="""High level linear transform parameter, vertical scale. 
        Used only when --linear_transform is not set.  horizontal scale. 
        Used only when --linear_transform is not set. When two space-separated float 
        are present, a random value will be used. The first value will be used as the 
        lower bound, the second value will be used as the upper bound.""",
        default=None
    )
    parser.add_argument(
        "-alpha",
        "--alpha",
        type=float,
        nargs="+",
        help="""Customizable high level linear transform parameter, top left 
        element in the 2x2 matrix. Used only when --linear_transform is not set. 
        When two space-separated float 
        are present, a random value will be used. The first value will be used as the 
        lower bound, the second value will be used as the upper bound.""",
        default=None
    )
    parser.add_argument(
        "-beta",
        "--beta",
        type=float,
        nargs="+",
        help="""Customizable high level linear transform parameter, top right 
        element in the 2x2 matrix. Used only when --linear_transform is not set. 
        When two space-separated float 
        are present, a random value will be used. The first value will be used as the 
        lower bound, the second value will be used as the upper bound.""",
        default=None
    )
    parser.add_argument(
        "-gamma",
        "--gamma",
        type=float,
        nargs="+",
        help="""Customizable high level linear transform parameter, bottom left 
        element in the 2x2 matrix. Used only when --linear_transform is not set. 
        When two space-separated float 
        are present, a random value will be used. The first value will be used as the 
        lower bound, the second value will be used as the upper bound.""",
        default=None
    )
    parser.add_argument(
        "-delta",
        "--delta",
        type=float,
        nargs="+",
        help="""Customizable high level linear transform parameter, bottom right 
        element in the 2x2 matrix. Used only when --linear_transform is not set. 
        When two space-separated float 
        are present, a random value will be used. The first value will be used as the 
        lower bound, the second value will be used as the upper bound.""",
        default=None
    )
    parser.add_argument(
        "-rtslnx",
        "--random_translation_x",
        action="store_true",
        help="""Uniformly sample the value of horizontal translation. 
        This will have no effect if horizontal margins are 0""", 
        default=False
    )
    parser.add_argument(
        "-rtslny",
        "--random_translation_y",
        action="store_true",
        help="""Uniformly sample the value of vertical translation. 
        This will have no effect if vertical margins are 0""", 
        default=False
    )
    parser.add_argument(
        "-pt",
        "--perspective_transform",
        type=parse_perspective_transform,
        nargs="?",
        help="""Apply a perspective transformation. 
        Given the coordinates of the four corners of the first quadrilateral 
        and the coordinates of the four corners of the second quadrilateral, 
        quadrilateral onto the appropriate position on the second quadrilateral. 
        Perspective transformation simulates different angle of view. 
        Enter 8 real numbers (float) which correspond to the 4 corner points (2D coordinates) 
        of the target quadrilateral, these 4 corner points which be respectively 
        mapped to [[0, 0], [1, 0], [0, 1], [1, 1]] in the source quadrilateral. 
        [0, 0] is the top left corner, [1, 0] is the top left corner, [0, 1] is 
        the bottom left corner, [1, 1] is the bottom right corner.
        These coordinates have been normalized to unit square [0, 1]^2. Thus, 
        the entered corner points should match the order of magnitude and must be convex. 
        For example, 0,0,1,0,0,1,1,1 will produce identity transform. 
        This option will have no effect if --random_perspective_transform is set. 
        It is recommended to use appropriate value of the 
        option --margins, otherwise part of transformed text may fall out 
        of the image boundary, which can lead to incomplete text.""",
        default=None
    )
    parser.add_argument(
        "-rpt",
        "--random_perspective_transform",
        type=float,
        nargs="?",
        help="""Randomly use a perspective transformation. 
        Randomly generate a convex quadrilateral which will be mapped to the normalized unit square, 
        the value of each axis is independently sampled from the gaussian distribution, 
        the standard deviation of the gaussian distribution is given by --random_perspective_transform. 
        If this option is present but not followed by a command-line argument, the standard deviation 
        0.05 will be used by default. It is recommended to use appropriate value of the 
        option --margins, otherwise part of transformed text may fall out 
        of the image boundary, which can lead to incomplete text.""",
        default=None,
        const=0.05
    )
    parser.add_argument(
        "-preelas",
        "--pre_elastic",
        type=float,
        help="""Pre-rasterization elastic transformation, also known as 
        random vibration of individual anchor points. 
        Value should be float in the range [0, 1). 
        The actual elastic variation range depends on the product of 
        the range of original anchor points and the value of pre_elastic. 
        If this value is too high, the visual effect may not be good. 
        It is recommended that pre_elastic <= 0.04.
        pre_elastic=0.1 can already lead to unrecognizable text.""",
        default=None
    )
    parser.add_argument(
        "-postelas",
        "--post_elastic",
        type=float,
        help="""Post-rasterization elastic transformation. 
        Define the parameter sigma of elastic transformation, it needs to be a 
        real positive scalar. If the image size is 128x128, 5 will be a good choice 
        for this parameter, which leads to a water-like effect.""",
        default=None
    )
    parser.add_argument(
        "-ascender",
        "--stretch_ascender",
        type=int,
        help="""Elongate (positive value) or contract (negative value) 
        the ascender part of the text (the corresponding anchor points). 
        The value should be integer.
        The effect of hundreds is sometimes invisible.""",
        default=None
    )
    parser.add_argument(
        "-descender",
        "--stretch_descender",
        type=int,
        help="""Elongate (positive value) or contract (negative value) 
        the descender part of the text (the corresponding anchor points). 
        The value should be integer.
        The effect of hundreds is sometimes invisible.""",
        default=None
    )
    parser.add_argument(
        "-gpr",
        "--gaussian_prior_resizing",
        type=float,
        help="""If not None, apply Gaussian filter to smooth image prior to resizing, 
        the argument of this parameter needs to be a float, which will be used as the 
        standard deviation of Gaussian filter. Default is None, which means Gaussian 
        filter is not used before resizing.""",
        default=None
    )
    parser.add_argument(
        "-morphero",
        "--morph_erosion",
        type=parse_morphological_image_processing_iteration,
        help="""Morphological image processing - erosion. The argument must be a tuple 
        separated by comma without space, the first element is the kernel 
        size, the second element is the number of iterations. For example, 3,2 means 
        kernel_size=3x3, iterations=2. 3,2,ellipse (3,2,cross) means using 
        elliptical (cross-shaped) kernel respectively. If the third argument is not given, 
        the default kernel shape will be rectangle. 
        Please be aware that morphological image processing methods can generate 
        unwelcome artifacts if text is close to the image boundary, please always consider 
        adding margins. If --random_translation_x or --random_translation_y moves the text 
        near the boundary, there will be artifacts.""",
        default=None
    )
    parser.add_argument(
        "-rmorphero",
        "--random_morph_erosion",
        action="store_true",
        help="""Uniformly sample the value of morphological erosion, the parameter 
        --morph_erosion needs to be set. 
        The range is [1, kernel_size] ([1, iterations]) 
        kernel_shape is randomly chosen among [rectangle, ellipse, cross].""",  
        default=False
    )
    parser.add_argument(
        "-morphdil",
        "--morph_dilation",
        type=parse_morphological_image_processing_iteration,
        help="""Morphological image processing - dilation. The argument must be a tuple 
        separated by comma without space, the first element is the kernel 
        size, the second element is the number of iterations. For example, 3,2 means 
        kernel_size=3x3, iterations=2. 3,2,ellipse (3,2,cross) means using 
        elliptical (cross-shaped) kernel respectively. If the third argument is not given, 
        the default kernel shape will be rectangle. 
        Please be aware that morphological image processing methods can generate 
        unwelcome artifacts if text is close to the image boundary, please always consider 
        adding margins. If --random_translation_x or --random_translation_y moves the text 
        near the boundary, there will be artifacts.""",
        default=None
    )
    parser.add_argument(
        "-rmorphdil",
        "--random_morph_dilation",
        action="store_true",
        help="""Uniformly sample the value of morphological dilation, the parameter 
        --morph_dilation needs to be set. 
        The range is [1, kernel_size] ([1, iterations]) 
        kernel_shape is randomly chosen among [rectangle, ellipse, cross].""", 
        default=False
    )
    parser.add_argument(
        "-morphope",
        "--morph_opening",
        type=parse_morphological_image_processing,
        help="""Morphological image processing - opening. The argument must be a tuple 
        separated by comma without space, the first element is the kernel 
        size, the second element is the kernel shape. For example, 3 means 
        kernel_size=3x3. 3,ellipse (3,cross) means using 
        elliptical (cross-shaped) kernel respectively. If the second argument is not given, 
        the default kernel shape will be rectangle. 
        Please be aware that morphological image processing methods can generate 
        unwelcome artifacts if text is close to the image boundary, please always consider 
        adding margins. If --random_translation_x or --random_translation_y moves the text 
        near the boundary, there will be artifacts.""",
        default=None
    )
    parser.add_argument(
        "-rmorphope",
        "--random_morph_opening",
        action="store_true",
        help="""Uniformly sample the value of morphological opening, the parameter 
        --morph_opening needs to be set. 
        The range is [1, kernel_size] 
        kernel_shape is randomly chosen among [rectangle, ellipse, cross].""",  
        default=False
    )
    parser.add_argument(
        "-morphclo",
        "--morph_closing",
        type=parse_morphological_image_processing,
        help="""Morphological image processing - closing. The argument must be a tuple 
        separated by comma without space, the first element is the kernel 
        size, the second element is the kernel shape. For example, 3 means 
        kernel_size=3x3. 3,ellipse (3,cross) means using 
        elliptical (cross-shaped) kernel respectively. If the second argument is not given, 
        the default kernel shape will be rectangle. 
        Please be aware that morphological image processing methods can generate 
        unwelcome artifacts if text is close to the image boundary, please always consider 
        adding margins. If --random_translation_x or --random_translation_y moves the text 
        near the boundary, there will be artifacts.""",
        default=None
    )
    parser.add_argument(
        "-rmorphclo",
        "--random_morph_closing",
        action="store_true",
        help="""Uniformly sample the value of morphological closing, the parameter 
        --morph_closing needs to be set. 
        The range is [1, kernel_size] 
        kernel_shape is randomly chosen among [rectangle, ellipse, cross].""",  
        default=False
    )
    parser.add_argument(
        "-morphgra",
        "--morph_gradient",
        type=parse_morphological_image_processing,
        help="""Morphological image processing - gradient. The argument must be a tuple 
        separated by comma without space, the first element is the kernel 
        size, the second element is the kernel shape. For example, 3 means 
        kernel_size=3x3. 3,ellipse (3,cross) means using 
        elliptical (cross-shaped) kernel respectively. If the second argument is not given, 
        the default kernel shape will be rectangle. 
        Please be aware that morphological image processing methods can generate 
        unwelcome artifacts if text is close to the image boundary, please always consider 
        adding margins. If --random_translation_x or --random_translation_y moves the text 
        near the boundary, there will be artifacts.""",
        default=None
    )
    parser.add_argument(
        "-rmorphgra",
        "--random_morph_gradient",
        action="store_true",
        help="""Uniformly sample the value of morphological gradient, the parameter 
        --morph_gradient needs to be set. 
        The range is [1, kernel_size] 
        kernel_shape is randomly chosen among [rectangle, ellipse, cross].""",  
        default=False
    )
    parser.add_argument(
        "-morphtoph",
        "--morph_tophat",
        type=parse_morphological_image_processing,
        help="""Morphological image processing - Top Hat. The argument must be a tuple 
        separated by comma without space, the first element is the kernel 
        size, the second element is the kernel shape. For example, 3 means 
        kernel_size=3x3. 3,ellipse (3,cross) means using 
        elliptical (cross-shaped) kernel respectively. If the second argument is not given, 
        the default kernel shape will be rectangle. 
        Please be aware that morphological image processing methods can generate 
        unwelcome artifacts if text is close to the image boundary, please always consider 
        adding margins. If --random_translation_x or --random_translation_y moves the text 
        near the boundary, there will be artifacts.""",
        default=None
    )
    parser.add_argument(
        "-rmorphtoph",
        "--random_morph_tophat",
        action="store_true",
        help="""Uniformly sample the value of morphological tophat, the parameter 
        --morph_tophat needs to be set. 
        The range is [1, kernel_size] 
        kernel_shape is randomly chosen among [rectangle, ellipse, cross].""",  
        default=False
    )
    parser.add_argument(
        "-morphblah",
        "--morph_blackhat",
        type=parse_morphological_image_processing,
        help="""Morphological image processing - Black Hat. The argument must be a tuple 
        separated by comma without space, the first element is the kernel 
        size, the second element is the kernel shape. For example, 3 means 
        kernel_size=3x3. 3,ellipse (3,cross) means using 
        elliptical (cross-shaped) kernel respectively. If the second argument is not given, 
        the default kernel shape will be rectangle. 
        Please be aware that morphological image processing methods can generate 
        unwelcome artifacts if text is close to the image boundary, please always consider 
        adding margins. If --random_translation_x or --random_translation_y moves the text 
        near the boundary, there will be artifacts.""",
        default=None
    )
    parser.add_argument(
        "-rmorphblah",
        "--random_morph_blackhat",
        action="store_true",
        help="""Uniformly sample the value of morphological blackhat, the parameter 
        --morph_blackhat needs to be set. 
        The range is [1, kernel_size] 
        kernel_shape is randomly chosen among [rectangle, ellipse, cross].""", 
        default=False
    )
    parser.add_argument(
        "-brtn",
        "--brightness",
        type=float,
        nargs="+",
        help="""Brightness enhancement. Needs to be a float. Values greater than 1 make 
        the image brighter. Values less than 1 make the image darker. The value 0 makes the 
        image completely black. The value 1 leaves the image unchanged. 
        If two space-separated float are present, then a random value will be used. The 
        first value will be used as the lower bound, the second value will be used as the 
        upper bound.""",
        default=None
    )
    parser.add_argument(
        "-ctrst",
        "--contrast",
        type=float,
        nargs="+",
        help="""Contrast enhancement. Needs to be a float. Values greater than 1 increase 
        brightness range, making bright color brigher, dark color darker. 
        Values less than 1 push colors towards gray. The value 0 makes the 
        image completely gray. The value 1 leaves the image unchanged. 
        If two space-separated float are present, then a random value will be used. The 
        first value will be used as the lower bound, the second value will be used as the 
        upper bound.""",
        default=None
    )
    parser.add_argument(
        "-clrenhc",
        "--color_enhance",
        type=float,
        nargs="+",
        help="""Color enhancement. Needs to be a float. Values greater than 1 make 
        the color stronger. The value 0 makes the image grayscale. 
        The value 1 leaves the image unchanged. 
        If two space-separated float are present, then a random value will be used. The 
        first value will be used as the lower bound, the second value will be used as the 
        upper bound.""",
        default=None
    )
    parser.add_argument(
        "-shrpns",
        "--sharpness",
        type=float,
        nargs="+",
        help="""Sharpness enhancement. Needs to be a float. Values greater than 1 apply 
        a sharpening filter to the image. Values less than 1 blur the image. 
        The value 1 leaves the image unchanged. 
        If two space-separated float are present, then a random value will be used. The 
        first value will be used as the lower bound, the second value will be used as the 
        upper bound.""",
        default=None
    )
    parser.add_argument(
        "-silt",
        "--silent",
        action="store_true",
        help="""If True, no verbose information will be printed during generation process.""",
        default=False
    )
    return parser.parse_args()


def create_strings(args):
    if args.dict != "alphabets/***EMPTY***":
        lang_dict = []
        args.dict = add_txt_extension(args.dict)
        if os.path.isfile(args.dict):
            with open(args.dict, "r", encoding="utf8", errors="ignore") as d:
                lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
        else:
            sys.exit("Cannot open dict")
        if args.equal_char:
            strings = create_strings_from_dict_equal(1, False, args.count, lang_dict)
        else:
            strings = create_strings_from_dict(1, False, args.count, lang_dict)
    else:
        assert args.textfile != "textfiles/***EMPTY***", \
            "Either the option --dict or the option --textfile should be used."
        args.textfile = add_txt_extension(args.textfile)
        strings = create_strings_from_file(args.textfile, args.count)
    return strings

def determine_fonts(args, string_count):
    # Creating font (path) list
    if args.font_index:
        font_index = args.font_index.split(os.sep)
        if len(font_index) == 1: # only the text set file
            font_index_dir = "fonts"
            font_index_file = font_index[0]
        elif len(font_index) == 2:
            font_index_dir, font_index_file = font_index
        elif len(font_index) > 2:
            font_index_dir = os.sep.join(font_index[:-1])
            font_index_file = font_index[-1]
        else:
            raise Exception("Wrong --font_index format, a correct example fonts{}latin.txt".format(os.sep)) 
        font_index_file = add_txt_extension(font_index_file)  
        with open(os.path.join(font_index_dir, "index", font_index_file), "r") as f:
            fonts = [os.path.join(font_index_dir, "fonts", p) for p in f.read().split("\n")] 
    elif args.font_dir:
        fonts = []
        for p in glob.glob(os.path.join(args.font_dir, "*.ttf")):
            fonts.append(p) 
        for p in glob.glob(os.path.join(args.font_dir, "*.otf")):
            fonts.append(p) 
    elif args.font:
        # Here, we make a magic utility option which automatically uses
        # the first valid font from the alphabet's font index.
        # "use_the_first_valid_font_from_index" is a magic option. 
        if args.font == "use_the_first_valid_font_from_index":
            font_index_file = None
            if args.dict != "alphabets/***EMPTY***":
                font_index_file = os.path.basename(args.dict) 
            elif args.textfile != "textfiles/***EMPTY***":
                font_index_file = os.path.basename(args.textfile) 
            assert font_index_file is not None, \
                "Specify either --dict or --textfile when using " +\
                "--font use_the_first_valid_font_from_index"
            font_index_file = add_txt_extension(font_index_file)
            font_index_dir = "fonts"
            with open(os.path.join(font_index_dir, "index", font_index_file), "r") as f:
                args.font = [os.path.join(font_index_dir, "fonts", p) for p in f.read().split("\n")][0]

        if os.path.isfile(args.font):
            fonts = [args.font]
        else:
            sys.exit("Cannot open font")
    else:
        font_index_dir = "fonts"
        font_index_file = os.path.basename(args.dict) 
        font_index_file = add_txt_extension(font_index_file)  
        with open(os.path.join(font_index_dir, "index", font_index_file), "r") as f:
            fonts = [os.path.join(font_index_dir, "fonts", p) for p in f.read().split("\n")] 
    
    # verify that fonts is not empty
    if len(fonts) == 0:
        raise Exception("No fonts are available for the chosen text set.")
    if len(fonts) == 1 and fonts[0][-1] == os.sep:
        raise Exception("No fonts are available for the chosen text set.")
    return [fonts[random.randrange(0, len(fonts))] for _ in range(0, string_count)]


def make_directories(args_dict):
    if args_dict["output_subdir"] is None:
        # use UTC time as the id of the generated dataset 
        args_dict["dataset_id"] = str(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"))
    else:
        args_dict["dataset_id"] = args_dict["output_subdir"]
    dataset_dir = os.path.join(args.output_dir, args_dict["dataset_id"])
    # create data subdirectory
    output_data_dir = os.path.join(dataset_dir, "data")
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    # create label subdirectory
    output_label_dir = os.path.join(dataset_dir, "label")
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
    args_dict["output_data_dir"] = output_data_dir
    return args_dict, output_label_dir

def create_raw_labels(args, output_label_dir):
    external_dataframes = []
    
    path = os.path.join("fonts", "metadata", "font_description.csv")
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    df = df.loc[:, ["font_file", "family_name", "style_name", "postscript_name", 
                    "variable_font_weight"]]
    external_dataframes.append(df)

    if len(external_dataframes) == 0:
        external_dataframes = None 
    generate_label_dataframe(labels, external_dataframes, save_path=output_label_dir)


if __name__ == "__main__":
    args = parse_arguments()

    # determine the number of processes to use 
    nb_processes = args.nb_processes
    if nb_processes is None:
        nb_processes = multiprocessing.cpu_count() 
    if not args.silent:
        print("Using {} processes.".format(nb_processes))

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(2 * args.random_seed + 1)

    strings = create_strings(args)
    string_count = len(strings)
    fonts = determine_fonts(args, string_count) 

    # create the dictionary of args 
    args_dict = vars(args)

    args_dict, output_label_dir = make_directories(args_dict)
    
    labels = []
    # generate images using multiprocessing 
    with multiprocessing.Pool(nb_processes) as pool:
        imap_it = list(tqdm.tqdm(pool.imap(TextDataGenerator.generate_from_tuple, 
                                           zip([i for i in range(0, string_count)], 
                                                strings, 
                                                fonts, 
                                                [args_dict] * string_count, 
                                                [False] * string_count)), 
                                total=string_count, disable=args.silent))
    for label in imap_it:
        labels.append(label)
    create_raw_labels(args, output_label_dir)

