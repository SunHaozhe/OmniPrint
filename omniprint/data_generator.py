import os
import random 
import math 
import numpy as np 
import scipy 

import PIL
from PIL import Image, ImageFilter

from omniprint import freetype_text_generator, background_generator, distorsion_generator
import transforms
from utils import fill_stroke_color 

try:
    from omniprint import handwritten_text_generator
except ImportError as e:
    print("Missing modules for handwritten text generation.")

_high_level_lt_params = ["rotation", "shear_x", "shear_y", "scale_x", 
                         "scale_y", "alpha", "beta", "gamma", "delta"]
_random_high_level_lt_params = ["random_rotation", "random_shear_x", "random_shear_y", 
                                "random_scale_x", "random_scale_y", "random_alpha", 
                                "random_beta", "random_gamma", "random_delta"]


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
            assert margin_top + margin_bottom == margin_left + margin_right, _warning_square_layout

        # collect labels
        label["text"] = text
        if len(text) == 1:
            label["unicode_code_point"] = ord(text)
        label["font_file"] = os.path.basename(font_file_path)
        label["margin_top"] = margin_top
        label["margin_left"] = margin_left
        label["margin_bottom"] = margin_bottom
        label["margin_right"] = margin_right

        ########################
        # Create image of text #
        ########################

        if args.get("handwritten"):
            if args.get("orientation") == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            img, mask = handwritten_text_generator.generate(text, "#282828") # black 
        else:
            transform_param = {}
            if args.get("linear_transform") is not None:
                transform_param = args.get("linear_transform") 
                label["linear_transform"] = transform_param 
            else:
                for lt_param_ in _high_level_lt_params:
                    if args.get(lt_param_) is not None:
                        transform_param[lt_param_] = args.get(lt_param_) 
                for random_lt_param_ in _random_high_level_lt_params:
                    if args.get(random_lt_param_):
                        lt_param_ = random_lt_param_[7:] 
                        limit_value = args.get(lt_param_) 
                        assert limit_value is not None, "The range of parameter is required."
                        limit_value = abs(limit_value) 
                        transform_param[lt_param_] = random.uniform(- limit_value, limit_value) 

                # collect labels
                for lt_param_ in _high_level_lt_params:
                    if args.get(lt_param_) is not None:
                        label[lt_param_] = transform_param[lt_param_] 

            img, mask = freetype_text_generator.render_lt_text(text, 
                                                               font_file_path, 
                                                               transform_param=transform_param, 
                                                               font_size=args.get("font_size"), 
                                                               font_weight=args.get("font_weight"), 
                                                               stroke_radius=args.get("outline_width"))
            

            # morphological image processing, applied before color filling 
            morph_operations = zip(["morph_erosion", 
                                    "morph_dilation"], 
                                   [transforms.morph_erosion_transform, 
                                    transforms.morph_dilation_transform])
            for morph_operation, morph_func in morph_operations:
                if args.get(morph_operation) is not None:
                    kernel_size, iterations, kernel_shape = args.get(morph_operation) 
                    if args.get("random_{}".format(morph_operation)):
                        kernel_size = np.random.randint(1, kernel_size + 1)
                        iterations = np.random.randint(1, iterations + 1)
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
                    kernel_size, kernel_shape = args.get(morph_operation) 
                    if args.get("random_{}".format(morph_operation)): 
                        kernel_size = np.random.randint(1, kernel_size + 1) 
                        kernel_shape = np.random.choice([None, "ellipse", "cross"], 
                                                         size=None, replace=True)
                    img, mask = morph_func(img, mask, 
                                           kernel_size=kernel_size, 
                                           kernel_shape=kernel_shape)
                    label["{}_kernel_size".format(morph_operation)] = kernel_size
                    if kernel_shape is None:
                        kernel_shape = "rectangle"
                    label["{}_kernel_shape".format(morph_operation)] = kernel_shape
                    


            # change the fill color of text stroke
            img = fill_stroke_color(img, args.get("stroke_fill"))

            # collect labels
            for x in ["font_weight", "stroke_fill"]:
                if args.get(x) is not None:
                    label[x] = args.get(x)


        ##############################
        # Place transformations here #
        ##############################


        # perspective/projective transformation 
        if args.get("random_perspective_transform") is not None:
            img, mask, perspective_params = transforms.perspective_transform(img, mask, 
                                                         quadrilateral=None, 
                                                         gaussian_std=args.get("random_perspective_transform"),
                                                         return_perspective_params=True)
            # collect labels
            label["perspective_params"] = perspective_params 
        elif args.get("perspective_transform") is not None:
            perspective_transform = np.asarray(args.get("perspective_transform")).reshape((4, 2))
            img, mask, perspective_params = transforms.perspective_transform(img, mask, 
                                                         quadrilateral=perspective_transform, 
                                                         gaussian_std=None,
                                                         return_perspective_params=True)
            # collect labels
            label["perspective_params"] = perspective_params 


        #############################
        # Apply distorsion to image #
        #############################
        distorsion_type = args.get("distorsion")
        distorsion_orientation = args.get("distorsion_orientation")
        if distorsion_type == 0:
            pass 
        elif distorsion_type == 1:
            img, mask = distorsion_generator.sin(
                img,
                mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            img, mask = distorsion_generator.cos(
                img,
                mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            img, mask = distorsion_generator.random(
                img,
                mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################

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
        
        #############################
        # Generate background image #
        #############################
        background_type = args.get("background")
        if background_type == 0:
            background_img = background_generator.gaussian_noise(final_h, final_w)
        elif background_type == 1:
            background_img = background_generator.plain_white(final_h, final_w)
        elif background_type == 2:
            background_img = background_generator.quasicrystal(final_h, final_w)
        else:
            background_img = background_generator.image(final_h, final_w, args.get("image_dir"))

        background_img.paste(img, (0, 0), mask)
        img = background_img
        
        #######################
        # Apply gaussian blur #
        #######################

        blur = args.get("blur")
        if blur is None:
            blur = 0 
        if args.get("random_blur"):
            blur = random.randint(0, blur)
        gaussian_filter = ImageFilter.GaussianBlur(radius=blur)
        img = img.filter(gaussian_filter)
        mask = mask.filter(gaussian_filter)

        # collect labels
        if args.get("blur") is not None:
            label["blur_radius"] = blur 
        
        ############################################
        # Change image mode (RGB, grayscale, etc.) #
        ############################################
        
        img = img.convert(args.get("image_mode")) 


        ##################
        # Save the image #
        ##################

        if args.get("output_data_dir") is not None:
            # Generate name for resulting image
            extension = args.get("extension")
            file_prefix = args.get("dataset_id") + "_{}".format(index)
            image_name = "{}.{}".format(file_prefix, extension)
            mask_name = "{}_mask.png".format(file_prefix)
            image_name = os.path.join(args.get("output_data_dir"), image_name)
            mask_name = os.path.join(args.get("output_data_dir"), mask_name)

            # save 
            img.save(image_name)
            label["image_name"] = os.path.basename(image_name)
            if args.get("output_mask"):
                mask.save(mask_name)
                label["mask_name"] = os.path.basename(mask_name)

        

        ##########
        # Return #
        ##########
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
    img = scipy.ndimage.gaussian_filter(img, sigma=sigma)
    return Image.fromarray(img).resize(size, resample=Image.LANCZOS, reducing_gap=4)




