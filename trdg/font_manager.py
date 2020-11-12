import os
import sys
import glob
import argparse
import pandas as pd 

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) 

from trdg.utils import add_txt_extension

def parse_arguments():

	parser = argparse.ArgumentParser(
		description="Font management."
	)
	parser.add_argument(
		"--dir", 
		type=str, 
		nargs="?", 
		help="The font directory where all font files are stored together with index files", 
		default="fonts"
	)
	return parser.parse_args()


def count_available_fonts(dir_, metadata_dir_name="metadata"):
	columns=["text_set", "available_fonts"]
	df = [] 
	font_file_set = set()
	for path in sorted(glob.glob(os.path.join(dir_, "index", "*.txt"))):
		with open(path, "r") as f:
			available_fonts = f.read().split("\n")
			font_file_set.update(available_fonts)
			nb_available_fonts = len(available_fonts)
		df.append([os.path.basename(path), nb_available_fonts])
	df = pd.DataFrame(df, columns=columns)
	df.sort_values("text_set", ascending=True, inplace=True)
	
	total_count = [["text_sets_count", df.shape[0]], ["distinct_fonts_count", len(font_file_set)]]
	df = df.append(pd.DataFrame(total_count, columns=columns))
	df.reset_index(drop=True, inplace=True) 

	metadata_path = os.path.join(dir_, metadata_dir_name)
	if not os.path.exists(metadata_path):
		os.makedirs(metadata_path)
	df.to_csv(os.path.join(metadata_path, "available_fonts.csv"), sep="\t", encoding="utf-8")


def get_available_fonts(text_set_path, font_index):
	font_index = font_index.split(os.sep)
	if len(font_index) == 1:
		font_index_dir = "fonts"
		font_index_file = font_index[0]
	elif len(font_index) == 2:
		font_index_dir, font_index_file = font_index
	else:
		raise Exception("Wrong font_index format, a correct example fonts{}latin.txt".format(os.sep)) 
	font_index_file = add_txt_extension(font_index_file) 
	with open(os.path.join(font_index_dir, "index", font_index_file), "r") as f:
		fonts = f.read().split("\n") 
	return fonts 


if __name__ == "__main__":
	
	# Argument parsing
	args = parse_arguments()

	metadata_dir_name = "metadata"

	count_available_fonts(args.dir, metadata_dir_name)
	
















