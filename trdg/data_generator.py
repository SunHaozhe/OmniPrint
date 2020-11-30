import os
import random as rnd

import PIL
from PIL import Image, ImageFilter

from trdg import computer_text_generator, background_generator, distorsion_generator

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
        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            img, mask = handwritten_text_generator.generate(text, text_color)
        else:
            img, mask = computer_text_generator.generate(
                text,
                font,
                text_color,
                size,
                orientation,
                space_width,
                character_spacing,
                fit,
                word_split,
                stroke_width, 
                stroke_fill,
            )
        random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        img = img.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

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

        # Horizontal text
        if orientation == 0:
            new_width = int(
                img.size[0]
                * (float(size - vertical_margin) / float(img.size[1]))
            )
            img = img.resize(
                (new_width, size - vertical_margin), Image.ANTIALIAS
            )
            mask = mask.resize((new_width, size - vertical_margin), Image.NEAREST)
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            new_height = int(
                float(img.size[1])
                * (float(size - horizontal_margin) / float(img.size[0]))
            )
            img = img.resize(
                (size - horizontal_margin, new_height), Image.ANTIALIAS
            )
            mask = mask.resize(
                (size - horizontal_margin, new_height), Image.NEAREST
            )
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background_img = background_generator.gaussian_noise(
                background_height, background_width
            )
        elif background_type == 1:
            background_img = background_generator.plain_white(
                background_height, background_width
            )
        elif background_type == 2:
            background_img = background_generator.quasicrystal(
                background_height, background_width
            )
        else:
            background_img = background_generator.image(
                background_height, background_width, image_dir
            )
        background_mask = Image.new(
            "RGB", (background_width, background_height), (0, 0, 0)
        )

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = img.size

        if alignment == 0 or width == -1:
            background_img.paste(img, (margin_left, margin_top), img)
            background_mask.paste(mask, (margin_left, margin_top))
        elif alignment == 1:
            background_img.paste(
                img,
                (int(background_width / 2 - new_text_width / 2), margin_top),
                img,
            )
            background_mask.paste(
                mask,
                (int(background_width / 2 - new_text_width / 2), margin_top),
            )
        else:
            background_img.paste(
                img,
                (background_width - new_text_width - margin_right, margin_top),
                img,
            )
            background_mask.paste(
                mask,
                (background_width - new_text_width - margin_right, margin_top),
            )

        img = background_img
        mask = background_mask

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
        mask = mask.convert(image_mode) 

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
