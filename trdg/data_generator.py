import os
import random as rnd
import math 
import numpy as np 

import PIL
from PIL import Image, ImageFilter

from trdg import freetype_text_generator, background_generator, distorsion_generator

try:
    from trdg import handwritten_text_generator
except ImportError as e:
    print("Missing modules for handwritten text generation.")


class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    @classmethod
    def generate(
        cls,
        index,
        text,
        font,
        out_dir,
        size,
        extension,
        skewing_angle,
        random_skew,
        blur,
        random_blur,
        background_type,
        distorsion_type,
        distorsion_orientation,
        is_handwritten,
        name_format,
        width,
        alignment,
        text_color,
        orientation,
        space_width,
        character_spacing,
        margins,
        fit,
        output_mask,
        word_split,
        image_dir,
        stroke_width=0, 
        stroke_fill="#282828",
        image_mode="RGB"
    ):
        ensure_square_layout = True

        margin_top, margin_left, margin_bottom, margin_right = margins  
        assert margin_top >= 0, "Margins cannot be negative." 
        assert margin_left >= 0, "Margins cannot be negative." 
        assert margin_bottom >= 0, "Margins cannot be negative." 
        assert margin_right >= 0, "Margins cannot be negative." 
        assert margin_top + margin_bottom < 1, "Sum of vertical margins exceeds limit."
        assert margin_left + margin_right < 1, "Sum of horizontal margins exceeds limit."
        if ensure_square_layout:
            assert margin_top + margin_bottom == margin_left + margin_right, _warning_square_layout

        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            img, mask = handwritten_text_generator.generate(text, text_color)
        else:
            img, mask = freetype_text_generator.render_lt_text(text, 
                                                               font, 
                                                               transform_param=None, 
                                                               font_size=192, 
                                                               font_weight=None, 
                                                               stroke_radius=None, 
                                                               stroke_fill=None)

        ##############################
        # Place transformations here #
        ##############################



        #############################
        # Apply distorsion to image #
        #############################
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

        if ensure_square_layout:
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
        background = Image.new("RGB", (background_w, background_h), (255, 255, 255))    
        background.paste(img, (offset_x, offset_y), mask)
        background_mask = Image.new("L", (background_w, background_h), 0)    
        background_mask.paste(mask, (offset_x, offset_y), mask)
        img = background 
        mask = background_mask 

        final_h = size 
        if ensure_square_layout:
            final_w = size
        else:
            final_w = math.ceil(final_h * img.size[0] / img.size[1])
        img = img.resize((final_w, final_h), resample=Image.LANCZOS, reducing_gap=4)
        mask = mask.resize((final_w, final_h), resample=Image.LANCZOS, reducing_gap=4)

        # make mask binary (255 or 0)
        mask = np.array(mask)
        mask[mask != 0] = 255
        mask = Image.fromarray(mask)

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background_img = background_generator.gaussian_noise(final_h, final_w)
        elif background_type == 1:
            background_img = background_generator.plain_white(final_h, final_w)
        elif background_type == 2:
            background_img = background_generator.quasicrystal(final_h, final_w)
        else:
            background_img = background_generator.image(final_h, final_w, image_dir)

        background_img.paste(img, (0, 0), mask)
        img = background_img
        
        #######################
        # Apply gaussian blur #
        #######################

        gaussian_filter = ImageFilter.GaussianBlur(
            radius=blur if not random_blur else rnd.randint(0, blur)
        )
        img = img.filter(gaussian_filter)
        mask = mask.filter(gaussian_filter)
        
        ############################################
        # Change image mode (RGB, grayscale, etc.) #
        ############################################
        
        img = img.convert(image_mode) 

        #####################################
        # Generate name for resulting image #
        #####################################
        if name_format == 0:
            image_name = "{}_{}.{}".format(text, str(index), extension)
            mask_name = "{}_{}_mask.png".format(text, str(index))
        elif name_format == 1:
            image_name = "{}_{}.{}".format(str(index), text, extension)
            mask_name = "{}_{}_mask.png".format(str(index), text)
        elif name_format == 2:
            image_name = "{}.{}".format(str(index), extension)
            mask_name = "{}_mask.png".format(str(index))
        else:
            print("{} is not a valid name format. Using default.".format(name_format))
            image_name = "{}_{}.{}".format(text, str(index), extension)
            mask_name = "{}_{}_mask.png".format(text, str(index))

        # Save the image
        if out_dir is not None:
            img.save(os.path.join(out_dir, image_name))
            if output_mask == 1:
                mask.save(os.path.join(out_dir, mask_name))
        else:
            if output_mask == 1:
                return img, mask
            return img
