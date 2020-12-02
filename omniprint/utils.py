"""
Utility functions
"""

import os
import glob


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



