import time
import os
import glob
import argparse
import shutil
from font_utils import build_index_files, count_available_fonts, detect_unwanted_fonts 


def transfer_font_files(src_font_paths, target_dir_path, unwanted_keywords, 
						track_src_paths=False, verbose=False):
	if len(src_font_paths) == 0:
		return  
	if not os.path.exists(target_dir_path):
		os.makedirs(target_dir_path)
	if track_src_paths:
		src_path_records = []
	
	font_set = set()
	for font_path in src_font_paths:
		basename = os.path.basename(font_path)
		if detect_unwanted_fonts(basename, keywords=unwanted_keywords):
			continue
		if basename in font_set:
			continue
		font_set.add(basename)
		shutil.copy(font_path, target_dir_path) 
		if track_src_paths:
			src_path_records.append(basename + "\t" + font_path)
	if track_src_paths:
		with open("src_path_records.txt", "w") as f:
			f.write("\n".join(src_path_records))
	if verbose:
		print("Total number of fonts {}".format(len(font_set)))


t0 = time.time()

# read command line arguments
parser = argparse.ArgumentParser(description="Build the working directory") 
parser.add_argument("-d", "--text_set_directory", type=str, 
					default=os.path.join(os.pardir, "alphabets", "fine"))
parser.add_argument("-m", "--metadata_dir_name", type=str, default="metadata")
parser.add_argument("-k", "--keep_temporary_directory", action="store_true", default=False)
parser.add_argument("-i", "--include_ttc", action="store_true", default=False)
args = parser.parse_args() 

print("This can take a while. Please be patient...")


tmp_save_dir = "temporary_save_directory"
target_directory = "fonts"


extensions = [".ttf", ".otf"]
if args.include_ttc:
	extensions += [".ttc"]

# copy all font files to the new directory
paths = []
for extension in ["*" + xx for xx in extensions]:
	for path in glob.glob(os.path.join(tmp_save_dir, "**", extension), recursive=True):
		paths.append(path)
paths = sorted(paths)

unwanted_keywords = ["bold", "heavy", "cuti", "italic", "italique", "xieti", "yidali", 
					 "light", "oblique", "medium", "extra", "ultra", "slant", "skew", 
					 "thin"]
transfer_font_files(paths, target_directory, unwanted_keywords)

# build index files which are necessary for data generation
build_index_files(target_directory, args.text_set_directory, extensions=extensions)

# build available_fonts.csv which consists of useful metadata
count_available_fonts(target_directory, args.metadata_dir_name)

if not args.keep_temporary_directory:
	shutil.rmtree(tmp_save_dir) 

print("Done in {:.3f} s.".format(time.time() - t0)) 

