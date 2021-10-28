import os
import sys
import glob
import collections
import re
import argparse
import pandas as pd 
import fontTools.ttLib 
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from freetype import Face

sys.path.append(os.pardir) 

from utils import add_txt_extension

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


def count_available_fonts(font_directory, metadata_dir_name="metadata"):
    """
    build metadata/available_fonts.csv and save it under font_directory directory  
    """
    columns=["text_set", "available_fonts"]
    df = [] 
    font_file_set = set()
    for path in sorted(glob.glob(os.path.join(font_directory, "index", "*.txt"))):
        with open(path, "r") as f:
            available_fonts = f.read().split("\n")
            if len(available_fonts) == 1 and available_fonts[0].strip() == "":
                nb_available_fonts = 0
            else:
                font_file_set.update(available_fonts)
                nb_available_fonts = len(available_fonts)
        df.append([os.path.basename(path), nb_available_fonts])
    df = pd.DataFrame(df, columns=columns)
    df.sort_values("text_set", ascending=True, inplace=True)
    
    # count the number of index files for variable font weight 
    variable_weight_prefix = "variable_weight_"
    variable_weight_count = 0
    for text_set in df["text_set"].tolist():
        if text_set.startswith(variable_weight_prefix):
            variable_weight_count += 1

    total_count = [["distinct_text_sets_count", df.shape[0] - variable_weight_count], 
                   ["distinct_fonts_count", len(font_file_set)]] 
    df = df.append(pd.DataFrame(total_count, columns=columns))
    df.reset_index(drop=True, inplace=True) 

    metadata_path = os.path.join(font_directory, metadata_dir_name)
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)
    df.to_csv(os.path.join(metadata_path, "available_fonts.csv"), sep="\t", encoding="utf-8")


def get_available_fonts(text_set_path, font_index):
    font_index = font_index.split(os.sep)
    if len(font_index) == 1:
        font_index_dir = os.path.join("fonts", "fonts")
        font_index_file = font_index[0]
    elif len(font_index) == 2:
        font_index_dir, font_index_file = font_index
        font_index_dir = os.path.join(font_index_dir, "fonts")
    else:
        raise Exception("Wrong font_index format, a correct example fonts{}latin.txt".format(os.sep)) 
    font_index_file = add_txt_extension(font_index_file) 
    with open(os.path.join(font_index_dir, "index", font_index_file), "r") as f:
        fonts = f.read().split("\n") 
    return fonts 


def normalize_font_name_for_detection(basename):
    """
    normalization before detecting last resort fonts and some other fonts 
    
    keywords: 
    bold, black, italic, light, oblique, medium, extra
    """
    file_name = os.path.splitext(basename)[0].lower()
    return re.sub(r"[\d\s\-_.]", "", file_name) 


def detect_unwanted_fonts(basename, keywords=[]):
    """
    return bool
    
    check if basename is the "last resort" font, 
    this font introduces a box around the character 
    
    Possible unwanted keywords include: 
    bold, heavy, cuti, italic, italique, xieti, yidali, light, 
    oblique, medium, extra, ultra, slant, skew, thin, 
    condensed, regular, etc.
    """
    normalized_name = normalize_font_name_for_detection(basename)
    # check if basename is the "last resort" font
    if "lastresort" in normalized_name:
        return True
    # check if basename contains unwanted keywords
    for keyword in keywords:
        if keyword in normalized_name:
            return True 
    return False 

def ttf_supports_char(ttf, char_): 
    """
    ttf: fontTools.ttLib.ttFont.TTFont object 
    returns bool 
    
    Assuming that font_path ends with .ttf (or .otf), then 
    ttf = fontTools.ttLib.TTFont(font_path) 
    """ 
    for table in ttf["cmap"].tables:
        try:
            if ord(char_) in table.cmap.keys():
                return True
        except AttributeError:
            return False 
    return False


def test_pil_compatibility(font_file_path, text_):
    """
    check whether Pillow library can correctly render text_ 
    with the font defined by font_file_path 
    """
    try:
        # test stroke_width and size 64
        size_ = 64
        img = Image.new("RGB", (size_, size_), 255)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_file_path, size=size_)
        draw.text((0, 0), text_, font=font, stroke_width=2)
        # test size 32 
        size_ = 32
        img = Image.new("L", (size_, size_), 255)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_file_path, size=size_)
        draw.text((0, 0), text_, font=font)
    except Exception as e:
        return False
    else:
        # test whether img is falsely blank/constant 
        arr = np.array(img)
        first_element = arr[0, 0]
        if np.all(arr == first_element):
            if ord(text_) == 32: # 32 is the space character
                return True
            else:
                return False 
        else:
            return True  


def test_freetype_compatibility(face, text_):
    """
    This is the simplified version of test_freetype_compatibility, 
    the complete verification consumes way too much time. 
    """
    try:
        face.load_char(text_)
    except Exception as e:
        return False
    else:
        return True 
    

# black list of font files, it contains problematic font files 
_hard_coded_black_list = ['albayan', 'baghdad', 'capture_it_2', 'courier new', 'gurumaa-2.04', 
                          'hoefler text ornaments', 'keyboard', 'kufistandardgk', 'lohit_as', 
                          'lohit_bn', 'nadeem', 'notoserifthai-black', 'notoserifthai-condensed', 
                          'notoserifthai-condensedblack', 'notoserifthai-regular', 
                          'notoserifthai-semicondensed', 'notoserifthai-semicondensedblack', 
                          'osho_v125', 'samyak-devanagari', 'samyak-gujarati', 'utibetan', 
                          'webdings', 'wingdings', 'wingdings 2', 'wingdings 3', 'zapfdingbats', 
                          'symbol', 'siddhibeta1', 'kai', 
                          'applegothic', 'applemyungjo', 'arimamadurai-black', 'arimamadurai-regular', 
                          'biaukai', 'chalkboard', 'chalkduster', 'gungseouche', 'hei', 'helvetica', 
                          'jaipur__', 'khand-regular', 'notosanskaithi-regular', 'pcmyoungjo', 
                          'pilgiche', 'proggysmall', 'proggysquare', 'rajdhani-regular', 
                          'rozhaone-regular', 'sarpanch-black', 'sarpanch-regular', 'tamu_kadampari', 
                          'tamu_kalyani', 'tamu_maduram', 'tscu_paranari', 'teko-regular', 'monof55', 
                          'monof56', 'nakula', 'sahadeva', 'zcoolqingkehuangyou-regular']


# text set file names that do not correspond to languages
_non_language_text_set_file_name_list = ["ascii_digits", "common_punctuations_symbols", 
                                         "mathematical_operators", "musical_symbols"]
# black list of font files, this black list will be applied 
# when the text set does not correspond to languages
_non_language_symbol_font_black_list = ['tamu_kadampari', 'tamu_kalyani', 'tamu_maduram', 
                                        'tscu_comic', 'tscu_paranar', 'tscu_paranarb', 
                                        'tscu_paranari', 'mitra']


def filter_fonts(font_paths, text_set_file_path, extensions=[".ttf", ".otf"], verbose=False):
    """
    return a subset of font_paths, this subset of font paths fully support 
    the character set defined by text_set_file_path 
    
    Example:
    font_paths = ["xxx/yyy/Arial.ttf", "xxx/yyy/Yahei.otf", "xxx/yyy/Freemono.ttf"]
    text_set_file_path = "../alphabets/fine/arabic.txt"
    """
    with open(text_set_file_path, "r") as f:
        chars = [str(xx) for xx in f.read().split("\n")] 
    len_chars = len(chars)
    filtered_font_paths = []
    lowered_text_set_file_name = os.path.splitext(os.path.basename(text_set_file_path))[0].lower()
    for font_path in font_paths:
        # test if this is a font file 
        if os.path.splitext(font_path)[1] not in extensions:
            continue 
        # filter out problematic font files (manually selected) 
        lowered_font_name = os.path.splitext(os.path.basename(font_path))[0].lower()
        if lowered_font_name in _hard_coded_black_list:
            continue 
        if lowered_text_set_file_name in _non_language_text_set_file_name_list:
            if lowered_font_name in _non_language_symbol_font_black_list:
                continue 
        # transforms font_path to a list of TTFont objects
        try:
            if font_path.endswith(".ttc"): 
                ttfonts = fontTools.ttLib.TTCollection(font_path).fonts 
            elif font_path.endswith(".ttf") or font_path.endswith(".otf"):
                ttfonts = [fontTools.ttLib.TTFont(font_path)]
            else:
                raise Exception 
        except Exception as e:
            if verbose:
                print(type(e), e, font_path) 
            continue
        # verification step: 
        # 1. verify if all TTFont objects support all characters from chars 
        # 2. double check with Pillow library 
        # 3. double check with freetype library 
        flag = True 
        try:
            face = Face(font_path) 
            face.set_char_size(12288) 
        except Exception:
            flag = False 
        for char_ in chars:
            for ttfont in ttfonts:
                flag = flag and ttf_supports_char(ttfont, char_)
            flag = flag and test_pil_compatibility(font_path, char_) 
            flag = flag and test_freetype_compatibility(face, char_)
        if flag:
            filtered_font_paths.append(font_path)
    return sorted(list(set(filtered_font_paths))) 


def build_index_files(font_directory, text_set_directory, extensions=[".ttf", ".otf", ".ttc"]):
    """
    build the index files with respect to font files stored in font_directory directory

    Example:
    font_directory = "fonts"
    text_set_directory = "../alphabets/fine"
    """
    # collect the list of font paths under the font_directory directory
    font_paths = []
    for path in sorted(glob.glob(os.path.join(font_directory, "fonts", "**", "*"), recursive=True)):
        if os.path.splitext(path)[1] in extensions:
            font_paths.append(path)
    
    # make sure index directory exists
    font_index_dir = os.path.join(font_directory, "index")
    if not os.path.exists(font_index_dir):
        os.makedirs(font_index_dir)

    # for each text set file (.txt), get the list of compatible fonts, save it to disk (index file)
    for text_set_file_path in sorted(glob.glob(os.path.join(text_set_directory, "**", "*.txt"), recursive=True)):
        filtered_font_paths = filter_fonts(font_paths, text_set_file_path, extensions=extensions)
        with open(os.path.join(font_index_dir, 
                               os.path.splitext(os.path.basename(text_set_file_path))[0] + ".txt"), "w") as f:
            f.write("\n".join(sorted([os.path.basename(xx) for xx in filtered_font_paths])))


def is_unwanted_NotoCJK(font_file, unwanted_list):
    if font_file.startswith("Noto"):
        for unwanted in unwanted_list:
            if unwanted in font_file:
                return True
    else:
        return False 


def rebuild_cjk_index_file_for_NotoCJK(basename, path, keywords, unwanted_list):
    for keyword in keywords:
        if keyword in basename: 
            with open(path, "r") as f:
                font_files = f.read().split("\n")
            new_font_files = []
            for font_file in font_files:
                if not is_unwanted_NotoCJK(font_file, unwanted_list):
                    new_font_files.append(font_file)
            with open(path, "w") as f:
                f.write("\n".join(new_font_files))


def configure_NotoCJK_fonts(font_directory):
    chinese_keywords = ["chinese"]
    korean_keywords = ["korean"]
    japanese_keywords = ["hiragana", "katakana"]

    chinese_unwanted_list = ["CJKjp", "CJKkr", "CJKtc"]
    korean_unwanted_list = ["CJKjp", "CJKsc", "CJKtc"]
    japanese_unwanted_list = ["CJKsc", "CJKkr", "CJKtc"]
    
    for path in glob.glob(os.path.join(font_directory, "index", "**", "*.txt"), recursive=True):
        basename = os.path.basename(path)
        rebuild_cjk_index_file_for_NotoCJK(basename, path, chinese_keywords, chinese_unwanted_list)
        rebuild_cjk_index_file_for_NotoCJK(basename, path, korean_keywords, korean_unwanted_list)
        rebuild_cjk_index_file_for_NotoCJK(basename, path, japanese_keywords, japanese_unwanted_list)


def delete_unused_fonts(font_directory, extensions):
    used = set()
    for path in glob.glob(os.path.join(font_directory, "index", "**", "*.txt"), recursive=True):
        with open(path, "r") as f:
            used.update(f.read().split("\n"))

    font_directory = os.path.join(font_directory, "fonts")
    for file_name in os.listdir(font_directory):
        if os.path.splitext(file_name)[1] in extensions:
            if os.path.basename(file_name) not in used: 
                os.remove(os.path.join(font_directory, file_name))


def generate_one_font_description(path):
    try:
        face = Face(path)
    except Exception:
        print(path)
        raise

    basename = os.path.basename(path) 
    suffix = os.path.splitext(path)[1][1:]
    family_name = face.family_name.decode("utf-8")
    style_name = face.style_name.decode("utf-8")
    postscript_name = face.postscript_name or "N/A"
    if not isinstance(postscript_name, str):
        postscript_name = postscript_name.decode("utf-8") 
    num_faces = face.num_faces
    num_glyphs = face.num_glyphs 
    has_multiple_masters = face.has_multiple_masters
    variation_axes_count = 0 
    variable_font_weight = False 
    min_font_weight = "N/A"
    max_font_weight = "N/A"
    if has_multiple_masters:
        ttfont = fontTools.ttLib.TTFont(path) 
        for axis in ttfont["fvar"].axes:
            variation_axes_count += 1 
            if axis.axisTag == "wght":
                try:
                    face.set_var_design_coords(((axis.minValue + axis.maxValue) // 2,))
                except Exception:
                    pass 
                else:
                    variable_font_weight = True
                    min_font_weight = axis.minValue
                    max_font_weight = axis.maxValue 
                finally:
                    face = Face(path) 
    try:
        charmap_encoding_name = face.charmap.encoding_name 
    except Exception:
        charmap_encoding_name = "N/A"
    has_horizontal = face.has_horizontal
    has_vertical = face.has_vertical
    has_kerning = face.has_kerning
    has_fixed_sizes = face.has_fixed_sizes
    num_fixed_sizes = face.num_fixed_sizes
    is_fixed_width = face.is_fixed_width
    is_scalable = face.is_scalable
    has_glyph_names = face.has_glyph_names
    units_per_EM = face.units_per_EM
    face_format = face.get_format().decode("utf-8")
    is_sfnt = face.is_sfnt
    sfnt_name_count = face.sfnt_name_count
    is_tricky = face.is_tricky 

    return basename, suffix, family_name, style_name, postscript_name, num_faces, num_glyphs, \
           has_multiple_masters, variation_axes_count, variable_font_weight, min_font_weight, \
           max_font_weight, charmap_encoding_name, has_horizontal, has_vertical, has_kerning, \
           has_fixed_sizes, num_fixed_sizes, is_fixed_width, is_scalable, has_glyph_names, \
           units_per_EM, face_format, is_sfnt, sfnt_name_count, is_tricky 


def generate_font_description(font_directory, extensions=[".ttf", ".otf"], metadata_dir_name="metadata"):
    df = []
    for path in sorted(glob.glob(os.path.join(font_directory, "fonts", "**", "*"), recursive=True)):
        if os.path.splitext(path)[1] not in extensions:
            continue
        df.append(generate_one_font_description(path))
    columns = ["font_file", "suffix", "family_name", "style_name", "postscript_name", 
               "num_faces", "num_glyphs", "has_multiple_masters", "variation_axes_count", 
               "variable_font_weight", "min_font_weight", "max_font_weight", 
               "charmap_encoding_name", "has_horizontal", "has_vertical", "has_kerning", 
               "has_fixed_sizes", "num_fixed_sizes", "is_fixed_width", "is_scalable", 
               "has_glyph_names", "units_per_EM", "face_format", "is_sfnt", 
               "sfnt_name_count", "is_tricky"] 
    df = pd.DataFrame(df, columns=columns)

    metadata_path = os.path.join(font_directory, metadata_dir_name)
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)
    df.to_csv(os.path.join(metadata_path, "font_description.csv"), sep="\t", encoding="utf-8")


def generate_variable_weight_font_index(font_directory, prefix="variable_weight_", metadata_dir_name="metadata"):
    variable_weight = pd.read_csv(os.path.join(font_directory, metadata_dir_name, "font_description.csv"), 
                                  sep="\t", encoding="utf-8")
    variable_weight = variable_weight.loc[variable_weight["variable_font_weight"], "font_file"].tolist()
    index_path = os.path.join(font_directory, "index") 
    for path in sorted(glob.glob(os.path.join(font_directory, "index", "**", "*.txt"), recursive=True)):
        basename = os.path.basename(path)
        if basename.startswith(prefix):
            continue
        text_set_name = os.path.splitext(basename)[0] 
        with open(path, "r") as f:
            candidate_fonts = f.read().split("\n")
        valid_fonts = []
        for candidate in candidate_fonts:
            if candidate in variable_weight:
                valid_fonts.append(candidate)
        if len(valid_fonts) == 0:
            continue 
        if not os.path.exists(index_path):
            os.makedirs(index_path)
        with open(os.path.join(index_path, prefix + text_set_name + ".txt"), "w") as f:
            f.write("\n".join(sorted(valid_fonts))) 


_corner_names = {"latin": ["basic_latin_uppercase", "basic_latin_lowercase", "latin1", 
                           "latin_extended_A", "latin_extended_B", 
                           "IPA_letters", "IPA_supplementary"], 
                 "japanese": ["hiragana", "katakana"],
                 "misc": ["common_punctuations_symbols", "ascii_digits", "mathematical_operators"]}


def get_coarse(fine_name):
    if fine_name.startswith("n_ko"):
        return "n_ko"
    for k, v in _corner_names.items():
        if fine_name in v:
            return k 
    return fine_name.split("_")[0]

def build_coarse_index_files_from_fine_index_files(font_index_dir):
    df = []
    for i, path in enumerate(sorted(glob.glob(os.path.join(font_index_dir, "*.txt")), 
                             key=str.casefold)):
        name_ = os.path.splitext(os.path.basename(path))[0]
        if name_.startswith("variable_weight_"):
            continue
        df.append(name_)
    df = pd.DataFrame(df, columns=["fine"])
    df["coarse"] = df["fine"].apply(get_coarse)
    df = df[["coarse", "fine"]]

    relation_dict = collections.defaultdict(list)
    for i, row in df.iterrows():
        relation_dict[row["coarse"]].append(row["fine"])

    for coarse_name, list_fine_name in relation_dict.items():
        # coarse_name: str. list_fine_name: list of str.
        fonts = []
        if len(list_fine_name) == 1:
            continue
        for charset in list_fine_name:
            with open(os.path.join(font_index_dir, "{}.txt".format(charset)), "r") as f:
                fonts.append([xx for xx in f.read().split("\n") if len(xx) != 0])
        # remove duplicates and sort
        fonts = sorted(list(set(fonts[0]).intersection(*fonts[1:])))
        with open(os.path.join(font_index_dir, "{}.txt".format(coarse_name)), "w") as f:
            f.write("\n".join(fonts))


def generate_representative_font_index(font_directory):
    df = pd.read_csv("representative_font.csv")
    for alphabet_name in sorted(list(set(df.loc[:, "alphabet"]))):
        with open(os.path.join(font_directory, "index", "{}.txt".format(alphabet_name)), "r") as f:
            fonts = [xx for xx in f.read().split("\n") if len(xx) != 0]
        representative_font = df.loc[df["alphabet"] == alphabet_name, "representative_font"].iloc[0]
        with open(os.path.join(font_directory, "index", 
            "REPRESENTATIVE_{}.txt".format(alphabet_name)), "w") as f:
            f.write(representative_font)
        others = []
        for font in fonts:
            if font != representative_font:
                others.append(font)
        with open(os.path.join(font_directory, "index", 
            "OTHERS_{}.txt".format(alphabet_name)), "w") as f:
            f.write("\n".join(others))


if __name__ == "__main__":
    
    # Argument parsing
    args = parse_arguments()

    metadata_dir_name = "metadata"

    count_available_fonts(args.dir, metadata_dir_name)
    
















