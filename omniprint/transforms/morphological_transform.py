"""
Warning:

Sufficient margins need to be added before applying morphological image processing methods 
to images, otherwise the boundary effect will introduce artifacts. The required amount of 
margins is proportional to the parameters of morphological image processing methods, e.g. 
kernel size, iterations, etc. 

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html 
"""


import cv2
import numpy as np 
from PIL import Image 

def morph_erosion_transform(img, mask, kernel_size=2, iterations=1, kernel_shape=None):
    """
    The kernel slides through the image (as in 2D convolution). A pixel in the 
    original image (either 1 or 0) will be considered 1 only if all the pixels 
    under the kernel is 1, otherwise it is eroded (made to zero). 
    Depending on image mode, 1 is replaced by 255 in some cases.

    Current implementation does not support color images. If one treats each of the 
    three color channels independently in RGB color space, one can introduce colors 
    that are not present in the original image.
    
    img:
        PIL.Image.Image (RGB, uint8)
        text in black (0), background in white (255)
        Even if in RGB format, the value of R/G/B channels must be equal, i.e. gray image 
    mask:
        PIL.Image.Image (L, uint8)
        text in white (255), background in black (0) 
    returns 2 PIL.Image.Image
    """ 
    if kernel_size == 0 or iterations == 0:
        return img, mask
    
    img_mode = img.mode 
    mask_mode = mask.mode 

    kernel = _build_kernel(kernel_size, kernel_shape)

    img = cv2.erode(np.invert(np.array(img.convert("L"))), kernel, iterations=iterations) 
    img = Image.fromarray(np.invert(img), mode="L").convert(img_mode)

    mask = cv2.erode(np.array(mask), kernel, iterations=iterations)
    mask = Image.fromarray(mask, mode=mask_mode) 
    return img, mask


def morph_dilation_transform(img, mask, kernel_size=2, iterations=1, kernel_shape=None):
    """
    It is just opposite of erosion. Here, a pixel element is ‘1’ if at least one 
    pixel under the kernel is ‘1’. So it increases the white region in the image 
    or size of foreground object increases. Normally, in cases like noise removal, 
    erosion is followed by dilation. Because, erosion removes white noises, but 
    it also shrinks our object. So we dilate it. Since noise is gone, they won’t 
    come back, but our object area increases. It is also useful in joining broken 
    parts of an object. Depending on image mode, 1 is replaced by 255 in some cases.

    Current implementation does not support color images. If one treats each of the 
    three color channels independently in RGB color space, one can introduce colors 
    that are not present in the original image.
    
    img:
        PIL.Image.Image (RGB, uint8)
        text in black (0), background in white (255)
        Even if in RGB format, the value of R/G/B channels must be equal, i.e. gray image 
    mask:
        PIL.Image.Image (L, uint8)
        text in white (255), background in black (0) 
    returns 2 PIL.Image.Image
    """ 
    if kernel_size == 0 or iterations == 0:
        return img, mask

    img_mode = img.mode 
    mask_mode = mask.mode 

    kernel = _build_kernel(kernel_size, kernel_shape)

    img = cv2.dilate(np.invert(np.array(img.convert("L"))), kernel, iterations=iterations) 
    img = Image.fromarray(np.invert(img), mode="L").convert(img_mode)

    mask = cv2.dilate(np.array(mask), kernel, iterations=iterations)
    mask = Image.fromarray(mask, mode=mask_mode) 
    return img, mask


def morph_opening_transform(img, mask, kernel_size=2, kernel_shape=None):
    """
    Opening is just another name of erosion followed by dilation. 
    It is useful in removing noise.
    
    Current implementation does not support color images. If one treats each of the 
    three color channels independently in RGB color space, one can introduce colors 
    that are not present in the original image.
    
    img:
        PIL.Image.Image (RGB, uint8)
        text in black (0), background in white (255)
        Even if in RGB format, the value of R/G/B channels must be equal, i.e. gray image 
    mask:
        PIL.Image.Image (L, uint8)
        text in white (255), background in black (0) 
    returns 2 PIL.Image.Image
    """ 
    if kernel_size == 0:
        return img, mask
    
    img_mode = img.mode 
    mask_mode = mask.mode 

    kernel = _build_kernel(kernel_size, kernel_shape)

    img = cv2.morphologyEx(np.invert(np.array(img.convert("L"))), cv2.MORPH_OPEN, kernel)
    img = Image.fromarray(np.invert(img), mode="L").convert(img_mode)

    mask = cv2.morphologyEx(np.array(mask), cv2.MORPH_OPEN, kernel)
    mask = Image.fromarray(mask, mode=mask_mode) 
    return img, mask


def morph_closing_transform(img, mask, kernel_size=2, kernel_shape=None):
    """
    Closing is reverse of Opening, Dilation followed by Erosion. It is useful in closing 
    small holes inside the foreground objects, or small black points on the object.
    
    Current implementation does not support color images. If one treats each of the 
    three color channels independently in RGB color space, one can introduce colors 
    that are not present in the original image.
    
    img:
        PIL.Image.Image (RGB, uint8)
        text in black (0), background in white (255)
        Even if in RGB format, the value of R/G/B channels must be equal, i.e. gray image 
    mask:
        PIL.Image.Image (L, uint8)
        text in white (255), background in black (0) 
    returns 2 PIL.Image.Image
    """ 
    if kernel_size == 0:
        return img, mask
    
    img_mode = img.mode 
    mask_mode = mask.mode 

    kernel = _build_kernel(kernel_size, kernel_shape)

    img = cv2.morphologyEx(np.invert(np.array(img.convert("L"))), cv2.MORPH_CLOSE, kernel)
    img = Image.fromarray(np.invert(img), mode="L").convert(img_mode)

    mask = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, kernel)
    mask = Image.fromarray(mask, mode=mask_mode) 
    return img, mask


def morph_gradient_transform(img, mask, kernel_size=2, kernel_shape=None):
    """
    Morphological Gradient is the difference between dilation and erosion of an image. 
    
    Current implementation does not support color images. If one treats each of the 
    three color channels independently in RGB color space, one can introduce colors 
    that are not present in the original image.
    
    img:
        PIL.Image.Image (RGB, uint8)
        text in black (0), background in white (255)
        Even if in RGB format, the value of R/G/B channels must be equal, i.e. gray image 
    mask:
        PIL.Image.Image (L, uint8)
        text in white (255), background in black (0) 
    returns 2 PIL.Image.Image
    """ 
    if kernel_size == 0:
        return img, mask
    
    img_mode = img.mode 
    mask_mode = mask.mode 

    kernel = _build_kernel(kernel_size, kernel_shape)

    img = cv2.morphologyEx(np.invert(np.array(img.convert("L"))), cv2.MORPH_GRADIENT, kernel)
    img = Image.fromarray(np.invert(img), mode="L").convert(img_mode)

    mask = cv2.morphologyEx(np.array(mask), cv2.MORPH_GRADIENT, kernel)
    mask = Image.fromarray(mask, mode=mask_mode) 
    return img, mask


def morph_tophat_transform(img, mask, kernel_size=2, kernel_shape=None):
    """
    Top Hat is the difference between input image and Opening of the image. 
    
    Current implementation does not support color images. If one treats each of the 
    three color channels independently in RGB color space, one can introduce colors 
    that are not present in the original image.
    
    img:
        PIL.Image.Image (RGB, uint8)
        text in black (0), background in white (255)
        Even if in RGB format, the value of R/G/B channels must be equal, i.e. gray image 
    mask:
        PIL.Image.Image (L, uint8)
        text in white (255), background in black (0) 
    returns 2 PIL.Image.Image
    """ 
    if kernel_size == 0:
        return img, mask
    
    img_mode = img.mode 
    mask_mode = mask.mode 

    kernel = _build_kernel(kernel_size, kernel_shape)

    img = cv2.morphologyEx(np.invert(np.array(img.convert("L"))), cv2.MORPH_TOPHAT, kernel)
    img = Image.fromarray(np.invert(img), mode="L").convert(img_mode)

    mask = cv2.morphologyEx(np.array(mask), cv2.MORPH_TOPHAT, kernel)
    mask = Image.fromarray(mask, mode=mask_mode) 
    return img, mask


def morph_blackhat_transform(img, mask, kernel_size=2, kernel_shape=None):
    """
    Black Hat is the difference between the closing of the input image and input image. 
    
    Current implementation does not support color images. If one treats each of the 
    three color channels independently in RGB color space, one can introduce colors 
    that are not present in the original image.
    
    img:
        PIL.Image.Image (RGB, uint8)
        text in black (0), background in white (255)
        Even if in RGB format, the value of R/G/B channels must be equal, i.e. gray image 
    mask:
        PIL.Image.Image (L, uint8)
        text in white (255), background in black (0) 
    returns 2 PIL.Image.Image
    """ 
    if kernel_size == 0:
        return img, mask
    
    img_mode = img.mode 
    mask_mode = mask.mode 

    kernel = _build_kernel(kernel_size, kernel_shape)

    img = cv2.morphologyEx(np.invert(np.array(img.convert("L"))), cv2.MORPH_BLACKHAT, kernel)
    img = Image.fromarray(np.invert(img), mode="L").convert(img_mode)

    mask = cv2.morphologyEx(np.array(mask), cv2.MORPH_BLACKHAT, kernel)
    mask = Image.fromarray(mask, mode=mask_mode) 
    return img, mask


def _build_kernel(kernel_size, kernel_shape=None):
    """
    This function allows elliptical/circular shaped kernels 
    """
    if kernel_shape is None:
        return np.ones((kernel_size, kernel_size), np.uint8)
    elif kernel_shape == "ellipse":
        # elliptical kernel
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape == "cross":
        # cross-shaped kernel 
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size)) 
    else:
        raise NotImplementedError









