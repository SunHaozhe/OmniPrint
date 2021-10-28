import time
import os
import glob
import argparse
import shutil
from font_utils import build_index_files, count_available_fonts, delete_unused_fonts 
from font_utils import detect_unwanted_fonts, configure_NotoCJK_fonts, generate_font_description 
from font_utils import generate_variable_weight_font_index
from font_utils import build_coarse_index_files_from_fine_index_files
from font_utils import generate_representative_font_index


def transfer_font_files(src_font_paths, target_dir_path, unwanted_keywords, 
                        track_src_paths=False, verbose=False):
    if len(src_font_paths) == 0:
        return  
    target_dir_path = os.path.join(target_dir_path, "fonts")
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


_non_license_suffixes = [".ttf", ".otf", ".ttc", ".woff", ".css", ".html", 
                         ".pdf", ".odt", ".conf"]
_license_file_name_patterns = ['license', 'licence', 'readme', 'copyright', 
                               'author', 'release', 'ofl', 'fontlog', 'faq', 
                               'documentation']


def is_a_license_file(path):
    if os.path.isdir(path):
        return False 
    lowered_base_name = os.path.basename(path).lower()
    if os.path.splitext(lowered_base_name)[1] in _non_license_suffixes:
        return False 
    for pattern_ in _license_file_name_patterns:
        if pattern_ in lowered_base_name:
            return True 
    return False 


def transfer_license_files(source_dir_path, target_dir_path):
    """
    copy license files and other related statements 
    """
    target_dir_path = os.path.join(target_dir_path, "licenses")
    for subdir_name in os.listdir(source_dir_path):
        if not os.path.isdir(os.path.join(source_dir_path, subdir_name)):
            continue
        license_file_id = 0 
        current_dir = os.path.join(target_dir_path, subdir_name) 
        for path in glob.glob(os.path.join(source_dir_path, subdir_name, "**", "*"), recursive=True):
            if is_a_license_file(path): 
                dir_path = os.path.join(current_dir, "license_file_{}".format(license_file_id))
                license_file_id += 1
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path) 
                shutil.copy(path, dir_path) 


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

# copy license files to the new directory 
transfer_license_files(tmp_save_dir, target_directory)

# build index files which are necessary for data generation
build_index_files(target_directory, args.text_set_directory, extensions=extensions)

# configure NotoCJK fonts 
configure_NotoCJK_fonts(target_directory)

# delete unused fonts in the target font directory
delete_unused_fonts(target_directory, extensions)

# build index files for coarse-grained alphabets using that of fine-grained alphabets
build_coarse_index_files_from_fine_index_files(os.path.join(target_directory, "index"))

# generate description metadata for each font
generate_font_description(target_directory, extensions)

# find variable fonts which have weight axes 
generate_variable_weight_font_index(target_directory)

# pick one representative font for each selected alphabet 
generate_representative_font_index(target_directory)

# build available_fonts.csv which consists of useful metadata
count_available_fonts(target_directory, args.metadata_dir_name)

# move metadata files to the dedicated directory
files_to_move = ["log_download.csv", "cjk_file_name_table.csv", 
                 "font2url_id.pkl", "url_id2font.pkl"] 
metadata_dir = os.path.join(target_directory, args.metadata_dir_name)
if not os.path.exists(metadata_dir):
    os.makedirs(metadata_dir) 
for file_ in files_to_move:
    shutil.copy(file_, metadata_dir)
    os.remove(file_)

if not args.keep_temporary_directory:
    shutil.rmtree(tmp_save_dir) 


print("Done in {:.3f} s.".format(time.time() - t0)) 

