import os
import random 
import math 
import numpy as np 
import cv2
from skimage import transform as skimage_transform 
import scipy 


import PIL
from PIL import Image, ImageFilter

from omniprint import freetype_text_generator, background_generator, distorsion_generator
import transforms

try:
    from omniprint import handwritten_text_generator
except ImportError as e:
    print("Missing modules for handwritten text generation.")

_high_level_lt_params = ["rotation", "shear_x", "shear_y", "scale_x", 
                         "scale_y", "alpha", "beta", "gamma", "delta"]
_random_high_level_lt_params = ["random_rotation", "random_shear_x", "random_shear_y", 
                                "random_scale_x", "random_scale_y", "random_alpha", 
                                "random_beta", "random_gamma", "random_delta"]


def float2int_image(img):
    """
    https://stackoverflow.com/a/38869210/7636942
    
    convert values 0-1 float to 0-255 int format
    img:
        np.array of dtype float 
    """
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def int2float_image(img):
    """
    https://stackoverflow.com/a/38869210/7636942
    
    convert values 0-255 int to 0-1 float format
    img:
        np.array of dtype uint8
    """ 
    return (np.clip(img, 0, 255) / 255).astype(np.float32)

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

def gaussian_bilinear(img, size, sigma):
    """
    first apply Gaussian filter to smooth image, 
    then resize image using bilinear filter with reducing_gap=4 
    
    img:
        PIL.Image.Image or np.array
    size:
        tuple of size 2
    sigma:
        scalar 
    """
    img = scipy.ndimage.gaussian_filter(img, sigma=sigma)
    return Image.fromarray(img).resize(size, resample=Image.BILINEAR, reducing_gap=4)



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

        ##########################
        # Create picture of text #
        ##########################

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
                                                               stroke_radius=args.get("outline_width"), 
                                                               stroke_fill=args.get("stroke_fill"))
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
        
        # _0
        image_name = args.get("img_name") + "_0.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        img.save(image_name)

        # _1 resize 105x105
        tmp_img = img.resize((105, 105), resample=Image.LANCZOS, reducing_gap=4) 
        image_name = args.get("img_name") + "_1_0.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_lanczos(img, size=(105, 105), sigma=1)
        image_name = args.get("img_name") + "_1_1.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_lanczos(img, size=(105, 105), sigma=2)
        image_name = args.get("img_name") + "_1_2.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_lanczos(img, size=(105, 105), sigma=3)
        image_name = args.get("img_name") + "_1_3.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        ##### 
        tmp_img = float2int_image(cv2.resize(int2float_image(np.array(img)), (105, 105), 
                                           interpolation=cv2.INTER_AREA))
        tmp_img = Image.fromarray(tmp_img, mode="RGB")
        image_name = args.get("img_name") + "_1_4.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = float2int_image(cv2.resize(int2float_image(np.array(img)), (105, 105), 
                                           interpolation=cv2.INTER_LANCZOS4))
        tmp_img = Image.fromarray(tmp_img, mode="RGB")
        image_name = args.get("img_name") + "_1_5.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_bilinear(img, size=(105, 105), sigma=0.1)
        image_name = args.get("img_name") + "_1_6.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_bilinear(img, size=(105, 105), sigma=1)
        image_name = args.get("img_name") + "_1_7.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_bilinear(img, size=(105, 105), sigma=3)
        image_name = args.get("img_name") + "_1_8.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)


        # _2 resize 64x64
        tmp_img = img.resize((64, 64), resample=Image.LANCZOS, reducing_gap=4) 
        image_name = args.get("img_name") + "_2_0.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_lanczos(img, size=(64, 64), sigma=0.5)
        image_name = args.get("img_name") + "_2_1.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_lanczos(img, size=(64, 64), sigma=1)
        image_name = args.get("img_name") + "_2_2.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_lanczos(img, size=(64, 64), sigma=2)
        image_name = args.get("img_name") + "_2_3.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_lanczos(img, size=(64, 64), sigma=3)
        image_name = args.get("img_name") + "_2_4.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        ##### 
        tmp_img = float2int_image(cv2.resize(int2float_image(np.array(img)), (64, 64), 
                                           interpolation=cv2.INTER_AREA))
        tmp_img = Image.fromarray(tmp_img, mode="RGB")
        image_name = args.get("img_name") + "_2_5.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = float2int_image(cv2.resize(int2float_image(np.array(img)), (64, 64), 
                                           interpolation=cv2.INTER_LANCZOS4))
        tmp_img = Image.fromarray(tmp_img, mode="RGB")
        image_name = args.get("img_name") + "_2_6.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_bilinear(img, size=(64, 64), sigma=0.1)
        image_name = args.get("img_name") + "_2_7.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_bilinear(img, size=(64, 64), sigma=1)
        image_name = args.get("img_name") + "_2_8.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_bilinear(img, size=(64, 64), sigma=3)
        image_name = args.get("img_name") + "_2_9.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        # _3 resize 32x32
        final_h = args.get("size") 
        if args.get("ensure_square_layout"):
            final_w = args.get("size")
        else:
            final_w = math.ceil(final_h * img.size[0] / img.size[1])

        tmp_img = img.resize((32, 32), resample=Image.LANCZOS, reducing_gap=4) 
        image_name = args.get("img_name") + "_3_0.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_lanczos(img, size=(32, 32), sigma=0.5)
        image_name = args.get("img_name") + "_3_1.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_lanczos(img, size=(32, 32), sigma=1)
        image_name = args.get("img_name") + "_3_2.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_lanczos(img, size=(32, 32), sigma=2)
        image_name = args.get("img_name") + "_3_3.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_lanczos(img, size=(32, 32), sigma=3)
        image_name = args.get("img_name") + "_3_4.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        ##### 
        tmp_img = float2int_image(cv2.resize(int2float_image(np.array(img)), (32, 32), 
                                           interpolation=cv2.INTER_AREA))
        tmp_img = Image.fromarray(tmp_img, mode="RGB")
        image_name = args.get("img_name") + "_3_5.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = float2int_image(cv2.resize(int2float_image(np.array(img)), (32, 32), 
                                           interpolation=cv2.INTER_LANCZOS4))
        tmp_img = Image.fromarray(tmp_img, mode="RGB")
        image_name = args.get("img_name") + "_3_6.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_bilinear(img, size=(32, 32), sigma=0.1)
        image_name = args.get("img_name") + "_3_7.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_bilinear(img, size=(32, 32), sigma=1)
        image_name = args.get("img_name") + "_3_8.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)

        tmp_img = gaussian_bilinear(img, size=(32, 32), sigma=3)
        image_name = args.get("img_name") + "_3_9.png"
        image_name = os.path.join(args.get("output_data_dir"), image_name)
        tmp_img.save(image_name)
        
        
        ##########
        # Return #
        ##########
        if returns_img:  
            if args.get("output_mask"):
                return img, mask, label
            return img, label
        else:
            return label  








