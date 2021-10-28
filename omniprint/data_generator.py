import os
import logging
import random 
import math 
import numpy as np 
import scipy.ndimage
from collections.abc import Iterable

import PIL
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2 

from omniprint import freetype_text_generator, background_generator
import transforms
from utils import get_font_weight_range, generate_random_color 
from utils import gaussian_blur_RGB, different_random_color
from utils import fill_foreground_color, fill_foreground_image
from utils import generate_text_outline
from poisson_image_editing import blit_images as poisson_editing


_high_level_lt_params = ["rotation", "shear_x", "shear_y", "scale_x", 
                         "scale_y", "alpha", "beta", "gamma", "delta"]
_random_high_level_lt_params = ["random_rotation", "random_shear_x", "random_shear_y", 
                                "random_scale_x", "random_scale_y", "random_alpha", 
                                "random_beta", "random_gamma", "random_delta"]
_background_image_labels = ["background_image_name", "background_image_original_width", 
                            "background_image_original_height", "background_image_resized_width", 
                            "background_image_resized_height", "background_image_crop_x", 
                            "background_image_crop_y", "background_image_crop_x_plus_width", 
                            "background_image_crop_y_plus_height"] 
_foreground_image_labels = ["foreground_image_name", "foreground_image_original_width", 
                            "foreground_image_original_height", "foreground_image_resized_width", 
                            "foreground_image_resized_height", "foreground_image_crop_x", 
                            "foreground_image_crop_y", "foreground_image_crop_x_plus_width", 
                            "foreground_image_crop_y_plus_height"] 
_outline_image_labels = ["outline_image_name", "outline_image_original_width", 
                         "outline_image_original_height", "outline_image_resized_width", 
                         "outline_image_resized_height", "outline_image_crop_x", 
                         "outline_image_crop_y", "outline_image_crop_x_plus_width", 
                         "outline_image_crop_y_plus_height"] 
_background_random_color_composition_labels = ["background_color", 
                                               "background_polygon_fill_color", 
                                               "background_polygon_outline_color", 
                                               "background_random_color_composition_params"]



class TextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        return cls.generate(*t)

    @classmethod
    def generate(cls, index, text, font_file_path, args, returns_img=True):
        # dictionary to store all kinds of labels 
        label = {}
        
        if args.get("random_seed") is not None:
            random.seed(3 * args.get("random_seed") + 2 + 2 * index)
            np.random.seed(4 * args.get("random_seed") + 3 + 3 * index)

        margin_top, margin_left, margin_bottom, margin_right = args.get("margins")  
        assert margin_top >= 0, "Margins cannot be negative." 
        assert margin_left >= 0, "Margins cannot be negative." 
        assert margin_bottom >= 0, "Margins cannot be negative." 
        assert margin_right >= 0, "Margins cannot be negative." 
        assert margin_top + margin_bottom < 1, "Sum of vertical margins exceeds limit."
        assert margin_left + margin_right < 1, "Sum of horizontal margins exceeds limit."
        if args.get("ensure_square_layout"):
            assert margin_top + margin_bottom == margin_left + margin_right

        # collect labels
        label["text"] = text
        if len(text) == 1:
            label["unicode_code_point"] = ord(text)
        label["font_file"] = os.path.basename(font_file_path)
        label["margin_top"] = margin_top
        label["margin_left"] = margin_left
        label["margin_bottom"] = margin_bottom
        label["margin_right"] = margin_right
        args, label = log_text_set(args, label)
        
        img, mask, label, args = generate_initial_image(text, font_file_path, args, label)

        img, mask, label, args = add_image_margins(img, mask, label, args)

        img, mask, label, args = apply_morphological_transformations(img, mask, label, args)

        img, mask, label, args = apply_post_rasterization_elastic_transformation(img, mask, label, args)

        img, mask, label, args = apply_perspective_transformation(img, mask, label, args)
        
        if args.get("background") == "image":
            img, mask, label, args = resize_image(img, mask, label, args)
            img, mask, label, args = fill_foreground(img, mask, label, args)
            img, mask, label, args = fill_outline(img, mask, label, args)
            img, mask, label, args = add_background(img, mask, label, args)
            img, label = image_enhancement(img, label, args)
        else:
            img, label = image_enhancement(img, label, args)
            img, mask, label, args = resize_image(img, mask, label, args)
            img, mask, label, args = fill_foreground(img, mask, label, args)
            img, mask, label, args = fill_outline(img, mask, label, args)
            img, mask, label, args = add_background(img, mask, label, args)
        
        img, mask, label, args = apply_gaussian_blur(img, mask, label, args)
        
        img, label, args = change_image_mode(img, label, args)
        
        save_image_(img, mask, label, args, index)
        
        if returns_img:  
            if args.get("output_mask"):
                return img, mask, label
            return img, label
        else:
            return label  


def gaussian_lanczos(img, size, sigma):
    """
    first apply Gaussian filter to smooth image, 
    then resize image using Lanczos filter with reducing_gap=4 
    
    img:
        PIL.Image.Image or np.array
    size:
        tuple of size 2
    sigma:
        scalar 
    """
    img = gaussian_blur_RGB(img, sigma=sigma)
    return img.resize(size, resample=Image.LANCZOS, reducing_gap=4)


def image_enhancement(img, label, args):
    for name_, func_ in zip(["brightness", "contrast", "color_enhance", "sharpness"], 
        [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color, ImageEnhance.Sharpness]):
        if args.get(name_) is not None:
            factor = args.get(name_)
            if isinstance(factor, Iterable):
                if len(factor) == 2:
                    factor = np.random.uniform(factor[0], factor[1], None)
                elif len(factor) == 1:
                    factor = factor[0]
                else:
                    raise Exception("More than two values received.")
            img = func_(img).enhance(factor) 
            label[name_] = factor
    return img, label


def factor2magnitude(factor):
    """legacy function"""
    if factor == 0:
        return 0.01
    if factor < 1:
        return 1 / factor 
    return factor 


def add_image_margins(img, mask, label, args):
    margin_top = label.get("margin_top") 
    margin_left = label.get("margin_left") 
    margin_bottom = label.get("margin_bottom") 
    margin_right = label.get("margin_right") 

    if args.get("ensure_square_layout"):
        max_size = max(img.size[0], img.size[1]) 
        background_w = math.ceil(max_size / (1 - margin_left - margin_right))
        background_h = math.ceil(max_size / (1 - margin_top - margin_bottom)) 
        offset_x = (max_size - img.size[0]) // 2 + math.floor(background_w * margin_left)
        offset_y = (max_size - img.size[1]) // 2 + math.floor(background_h * margin_top)
    else:
        background_w = math.ceil(img.size[0] / (1 - margin_left - margin_right))
        background_h = math.ceil(img.size[1] / (1 - margin_top - margin_bottom)) 
        offset_x = math.floor(background_w * margin_left)
        offset_y = math.floor(background_h * margin_top)
    if args.get("random_translation_x"):
        offset_x = random.randint(0, math.floor(background_w - img.size[0]))
    if args.get("random_translation_y"):
        offset_y = random.randint(0, math.floor(background_h - img.size[1]))
    background = Image.new("RGB", (background_w, background_h), (255, 255, 255))    
    background.paste(img, (offset_x, offset_y), mask)
    background_mask = Image.new("L", (background_w, background_h), 0)    
    background_mask.paste(mask, (offset_x, offset_y), mask)
    img = background 
    mask = background_mask 

    # collect labels
    label["offset_horizontal"] = offset_x
    label["offset_vertical"] = offset_y
    label["original_image_width_resolution"] = background_w
    label["original_image_height_resolution"] = background_h

    return img, mask, label, args

def resize_image(img, mask, label, args):
    final_h = args.get("size") 
    if args.get("ensure_square_layout"):
        final_w = args.get("size")
    else:
        final_w = math.ceil(final_h * img.size[0] / img.size[1])

    # resize img and mask 
    gaussian_prior_resizing = args.get("gaussian_prior_resizing") 
    if gaussian_prior_resizing is None:
        # directly resize
        img = img.resize((final_w, final_h), resample=Image.LANCZOS, reducing_gap=4) 
        mask = mask.resize((final_w, final_h), resample=Image.LANCZOS, reducing_gap=4)
    else:
        # apply Gaussian filter before resizing 
        img = gaussian_lanczos(img, size=(final_w, final_h), 
                               sigma=gaussian_prior_resizing)
        mask = gaussian_lanczos(mask, size=(final_w, final_h), 
                                sigma=gaussian_prior_resizing)
        label["gaussian_prior_resizing"] = gaussian_prior_resizing
    
    # collect labels
    label["image_width_resolution"] = final_w
    label["image_height_resolution"] = final_h

    return img, mask, label, args


def image_blending(img, mask, background, method="poisson"):
    if method == "trivial":
        background.paste(img, (0, 0), mask)
        img = background
    elif method == "poisson":
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        background = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)
        img = poisson_editing(img, background)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        raise Exception("Not implemented method {}".format(method)) 
    return img, mask 

def determine_image_blending_method(background_type):
    """
    Not used at this stage 
    """
    if background_type in ["image"]:
        return "poisson"
    else:
        # The "poisson" method can render false image in 
        # some cases, e.g. white text on black background 
        return "trivial"

def get_foreground_color(args):
    if args.get("stroke_fill") is None:
        foreground_color = (0, 0, 0)
    else:
        foreground_color = args.get("stroke_fill") 
    return foreground_color

def add_background(img, mask, label, args):
    background_type = args.get("background")
    final_w = label.get("image_width_resolution") 
    final_h = label.get("image_height_resolution") 

    rgb_value = background_type.split(",")

    if len(rgb_value) == 3:
        rgb_value = [int(xx) for xx in rgb_value]
        assert isinstance(rgb_value[0], int) and rgb_value[0] >= 0 and rgb_value[0] <= 255
        assert isinstance(rgb_value[1], int) and rgb_value[1] >= 0 and rgb_value[1] <= 255
        assert isinstance(rgb_value[2], int) and rgb_value[2] >= 0 and rgb_value[2] <= 255
        color = (rgb_value[0], rgb_value[1], rgb_value[2])
        background_img = background_generator.plain_color(final_h, final_w, color)
        label["background_color"] = color
    elif background_type == "plain_white":
        background_img = background_generator.plain_white(final_h, final_w)
        label["background_color"] = (255, 255, 255)
    elif background_type == "random_plain_color":
        # by default, the background color will not be too similar to the foreground color 
        color = different_random_color(get_foreground_color(args), method="randomcolor")
        background_img = background_generator.plain_color(final_h, final_w, color)
        label["background_color"] = color
    elif background_type == "image":
        background_img, label_info = background_generator.image(final_h, final_w, args.get("image_dir")) 
        for label_name, label_content in zip(_background_image_labels, label_info):
            label[label_name] = label_content
    elif background_type == "random_color_composition":
        background_img, label_info = background_generator.random_color_composition(final_h, final_w, 
            get_foreground_color(args), background_random_color_composition_params=None)
        for label_name, label_content in zip(_background_random_color_composition_labels, label_info):
            label[label_name] = label_content
    elif background_type == "gaussian_noise":
        background_img = background_generator.gaussian_noise(final_h, final_w)
    elif background_type == "quasicrystal":
        background_img = background_generator.quasicrystal(final_h, final_w)
    else:
        raise NotImplementedError
    label["background"] = background_type

    image_blending_method = args.get("image_blending_method")
    img, mask = image_blending(img, mask, background_img, method=image_blending_method)
    label["image_blending_method"] = image_blending_method

    return img, mask, label, args 

def apply_gaussian_blur(img, mask, label, args):
    blur = args.get("blur")
    if blur is not None:
        if isinstance(blur, Iterable):
            if len(blur) == 2:
                blur = random.randint(blur[0], blur[1])
            elif len(blur) == 1:
                blur = blur[0]
            else:
                raise Exception("More than two values received.")
        img = gaussian_blur_RGB(img, sigma=blur)
        mask = Image.fromarray(scipy.ndimage.gaussian_filter(mask, sigma=blur))

        # collect labels
        label["blur_radius"] = blur 

    return img, mask, label, args 

def change_image_mode(img, label, args):
    """
    Change image mode (RGB, grayscale, etc.)
    """
    img = img.convert(args.get("image_mode")) 
    label["image_mode"] = args.get("image_mode")
    return img, label, args

def save_image_(img, mask, label, args, index):
    if args.get("output_data_dir") is not None:
        # Generate name for resulting image
        extension = args.get("extension")
        file_prefix = args.get("dataset_id") + "_{}".format(index)
        image_name = "{}.{}".format(file_prefix, extension)
        mask_name = "{}_mask.{}".format(file_prefix, extension)
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        mask_name = os.path.join(args.get("output_data_dir"), mask_name)

        # save 
        img.save(image_name)
        label["image_name"] = os.path.basename(image_name)
        if args.get("output_mask"):
            mask.save(mask_name)
            label["mask_name"] = os.path.basename(mask_name)

def apply_perspective_transformation(img, mask, label, args):
    # perspective/projective transformation 
    if args.get("random_perspective_transform") is not None:
        if not all_margins_are_positive(label):
            logging.warning("""Using perspective transformation, however 
                some margins are zero, part of transformed text may fall out 
                of the image boundary, which can lead to incomplete text.""")
        img, mask, perspective_params = transforms.perspective_transform(img, mask, 
                                                     quadrilateral=None, 
                                                     gaussian_std=args.get("random_perspective_transform"),
                                                     return_perspective_params=True)
        # collect labels
        label["perspective_params"] = perspective_params 
    elif args.get("perspective_transform") is not None:
        if not all_margins_are_positive(label):
            logging.warning("""Using perspective transformation, however 
                some margins are zero, part of transformed text may fall out 
                of the image boundary, which can lead to incomplete text.""")
        perspective_transform = np.asarray(args.get("perspective_transform")).reshape((4, 2))
        img, mask, perspective_params = transforms.perspective_transform(img, mask, 
                                                     quadrilateral=perspective_transform, 
                                                     gaussian_std=None,
                                                     return_perspective_params=True)
        # collect labels
        label["perspective_params"] = perspective_params 
    return img, mask, label, args

def generate_initial_image(text, font_file_path, args, label):
    transform_param = {}
    if args.get("linear_transform") is not None:
        transform_param = args.get("linear_transform") 
        label["linear_transform"] = transform_param 
    else:
        for lt_param_ in _high_level_lt_params:
            if args.get(lt_param_) is not None:
                value_ = args.get(lt_param_)
                if isinstance(value_, Iterable):
                    if len(value_) == 2:
                        transform_param[lt_param_] = random.uniform(value_[0], value_[1])
                    elif len(value_) == 1:
                        transform_param[lt_param_] = value_[0]
                    else:
                        raise Exception("More than two values received.")
                else:
                    transform_param[lt_param_] = value_

        # collect labels
        for lt_param_ in _high_level_lt_params:
            if args.get(lt_param_) is not None:
                label[lt_param_] = transform_param[lt_param_] 

    # sample random stroke width
    font_weight = args.get("font_weight")
    if font_weight is not None:
        if isinstance(font_weight, Iterable):
            if len(font_weight) == 2:
                min_font_weight, max_font_weight = get_font_weight_range(font_file_path)
                if min_font_weight is not None:
                    min_font_weight = max(min_font_weight, font_weight[0])
                else:
                    min_font_weight = font_weight[0]
                if max_font_weight is not None:
                    max_font_weight = min(max_font_weight, font_weight[1])
                else:
                    max_font_weight = font_weight[1]
                args["font_weight"] = np.random.uniform(min_font_weight, max_font_weight, None) 
            elif len(font_weight) == 1:
                args["font_weight"] = font_weight[0]
            else:
                raise Exception("More than two values received.")
    
    # generate initial text image 
    try:
        img, mask = freetype_text_generator.render_lt_text(text, 
                                                           font_file_path, 
                                                           transform_param=transform_param, 
                                                           font_size=args.get("font_size"), 
                                                           font_weight=args.get("font_weight"), 
                                                           stroke_radius=args.get("outline_width"),
                                                           pre_elastic=args.get("pre_elastic"),
                                                           stretch_ascender=args.get("stretch_ascender"),
                                                           stretch_descender=args.get("stretch_descender"))
    except Exception as exception_:
        raise Exception("""freetype_text_generator.render_lt_text failed with text {} and 
            font_file_path {}. The Exception is {}""".format(text, font_file_path, exception_))
    # collect labels
    for x in ["font_size", "font_weight", "pre_elastic", "stretch_ascender", "stretch_descender"]:
        if args.get(x) is not None:
            label[x] = args.get(x)
    return img, mask, label, args

def apply_morphological_transformations(img, mask, label, args):
    morph_operations = zip(["morph_erosion", 
                            "morph_dilation"], 
                           [transforms.morph_erosion_transform, 
                            transforms.morph_dilation_transform])
    for morph_operation, morph_func in morph_operations:
        if args.get(morph_operation) is not None:
            if not all_margins_are_positive(label):
                logging.warning("""Using morphological image processing {}, however 
                    some margins are zero, which can 
                    lead to unwelcome artifacts.""".format(args.get(morph_operation)))
            kernel_size, iterations, kernel_shape = args.get(morph_operation) 
            if args.get("random_{}".format(morph_operation)):
                kernel_size = np.random.randint(0, kernel_size + 1)
                iterations = np.random.randint(0, iterations + 1)
                kernel_shape = np.random.choice([None, "ellipse", "cross"], 
                                                 size=None, replace=True)
            img, mask = morph_func(img, mask, 
                                   kernel_size=kernel_size, 
                                   iterations=iterations, 
                                   kernel_shape=kernel_shape)
            label["{}_kernel_size".format(morph_operation)] = kernel_size
            if kernel_shape is None:
                kernel_shape = "rectangle"
            label["{}_kernel_shape".format(morph_operation)] = kernel_shape 
            label["{}_iterations".format(morph_operation)] = iterations 

    morph_operations = zip(["morph_opening",
                            "morph_closing",
                            "morph_gradient",
                            "morph_tophat",
                            "morph_blackhat"], 
                           [transforms.morph_opening_transform,
                            transforms.morph_closing_transform,
                            transforms.morph_gradient_transform,
                            transforms.morph_tophat_transform,
                            transforms.morph_blackhat_transform])
    for morph_operation, morph_func in morph_operations:
        if args.get(morph_operation) is not None: 
            if not all_margins_are_positive(label):
                logging.warning("""Using morphological image processing {}, however 
                    some margins are zero, which can 
                    lead to unwelcome artifacts.""".format(args.get(morph_operation)))
            kernel_size, kernel_shape = args.get(morph_operation) 
            if args.get("random_{}".format(morph_operation)): 
                kernel_size = np.random.randint(0, kernel_size + 1) 
                kernel_shape = np.random.choice([None, "ellipse", "cross"], 
                                                 size=None, replace=True)
            img, mask = morph_func(img, mask, 
                                   kernel_size=kernel_size, 
                                   kernel_shape=kernel_shape)
            label["{}_kernel_size".format(morph_operation)] = kernel_size
            if kernel_shape is None:
                kernel_shape = "rectangle"
            label["{}_kernel_shape".format(morph_operation)] = kernel_shape

    return img, mask, label, args

def apply_post_rasterization_elastic_transformation(img, mask, label, args):
    if args.get("post_elastic") is not None:
        img, mask = transforms.elastic_transform(img, mask, 
                                                 args.get("post_elastic"))
        label["post_elastic"] = args.get("post_elastic") 
    return img, mask, label, args

def fill_foreground(img, mask, label, args):
    """
    fill the foreground

    This function assumes that the (possibly anti-aliased) image (img) 
    contains black text on white background. The color of the text will 
    be replaced by another color while avoiding boundary anti-aliasing 
    artifacts 
    """
    if args.get("foreground_image"):
        label["foreground"] = "image"
        width, height = mask.size
        external_image, label_info = background_generator.image(height, width, args.get("foreground_image_dir"))
        img, mask = fill_foreground_image(mask, external_image)
        for label_name, label_content in zip(_foreground_image_labels, label_info):
            label[label_name] = label_content
    else:
        if args.get("random_stroke_fill"):
            args["stroke_fill"] = generate_random_color(method="randomcolor")
            label["foreground"] = "random_color"
        else:
            label["foreground"] = "others"
        img, mask = fill_foreground_color(mask, args.get("stroke_fill"))
    
        if args.get("stroke_fill") is not None:
            label["stroke_fill"] = args.get("stroke_fill")
    return img, mask, label, args


def fill_outline(img, mask, label, args):
    outline = args.get("outline")
    if outline is not None:
        if outline == "image":
            # fill text outline with natural image/texture
            label["outline"] = "image"
            width, height = mask.size
            outline, label_info = background_generator.image(height, width, args.get("outline_image_dir"))
            for label_name, label_content in zip(_outline_image_labels, label_info):
                label[label_name] = label_content
        elif isinstance(outline, str):
            # fill text outline with uniform color 
            if outline == "random_color":
                label["outline"] = outline
            else:
                outline = tuple([int(xx) for xx in outline.split(",")])
                label["outline"] = outline
        else:
            raise Exception("Invalid outline: {}".format(outline))
        img, mask = generate_text_outline(img, mask, outline, 
            outline_size=args.get("outline_size"))
        label["outline_size"] = args.get("outline_size")
    return img, mask, label, args


def all_margins_are_positive(label):
    if label.get("margin_top") > 0 and \
       label.get("margin_left") > 0 and \
       label.get("margin_bottom") > 0 and \
       label.get("margin_right") > 0:
        return True
    else:
        return False 


def log_text_set(args, label):
    if args.get("dict") != "alphabets/***EMPTY***":
        text_set = args.get("dict")
    else:
        assert args.get("textfile") != "textfiles/***EMPTY***"
        text_set = args.get("textfile")
    text_set = os.path.splitext(os.path.basename(text_set))[0]
    label["text_set"] = text_set
    return args, label







