"""
Utility functions
"""

import os
import glob
import pandas as pd 
import numpy as np 
from PIL import Image 


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


def fill_stroke_color(img, stroke_fill, black_=0):
    """
    change the fill color of text stroke

    img:
        PIL.Image.Image (RGB)
    return PIL.Image.Image 
    """
    if stroke_fill is not None: 
        assert len(stroke_fill) == 3, "stroke_fill must be a iterable of length 3."
        img_mode = img.mode 
        img = np.array(img)
        black_areas = (img[:, :, 0] == black_) & (img[:, :, 1] == black_) & (img[:, :, 2] == black_)
        img[black_areas, :] = stroke_fill 
        img = Image.fromarray(img, mode=img_mode)  
    return img 


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




