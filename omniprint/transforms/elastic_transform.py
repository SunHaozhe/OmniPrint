# https://imgaug.readthedocs.io/en/latest/source/api_augmenters_geometric.html?highlight=elastic#imgaug.augmenters.geometric.ElasticTransformation

"""
Simard, Steinkraus and Platt
Best Practices for Convolutional Neural Networks applied to Visual
Document Analysis
in Proc. of the International Conference on Document Analysis and
Recognition, 2003
"""

import numpy as np 
from PIL import Image 
from imgaug import augmenters as iaa 


def transform(img, mask, sigma=1):
    '''
    Transform images by moving pixels locally around using displacement fields.
    
    The augmenter has the parameters alpha and sigma. 
    alpha controls the strength of the displacement: higher values mean that pixels are moved further. 
    sigma controls the smoothness of the displacement: higher values lead to smoother patterns – 
    as if the image was below water – while low values will cause indivdual pixels to be moved very 
    differently from their neighbours, leading to noisy and pixelated images.

    A relation of 10:1 seems to be good for alpha and sigma, e.g. alpha=10 and sigma=1 or 
    alpha=50, sigma=5. For 128x128 a setting of alpha=(0, 70.0), sigma=(4.0, 6.0) may be a 
    good choice and will lead to a water-like effect. 
    '''
    img_mode = img.mode 
    mask_mode = mask.mode 

    aug = iaa.ElasticTransformation(alpha=10*sigma, sigma=sigma, order=5, mode="constant", cval=255)
    aug = aug.to_deterministic()

    img = aug(image=np.array(img))
    img = Image.fromarray(img, mode=img_mode) 

    mask = aug(image=np.array(mask))
    mask = Image.fromarray(mask, mode=mask_mode) 

    return img, mask



