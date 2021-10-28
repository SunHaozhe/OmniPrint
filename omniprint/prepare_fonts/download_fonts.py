import os 
import glob
import argparse
import shutil
import urllib
import datetime
import time
import multiprocessing 
import itertools
import zipfile
import pickle
import pandas as pd
from string import digits as string_digits
from string import punctuation as string_punctuation
from collections import defaultdict
import tqdm
import wget # https://pypi.org/project/wget/
import pypinyin # https://github.com/mozillazg/python-pinyin


def download_func(input_tuple):
    url, id_, tmp_save_dir = input_tuple 
    tmp_save_dir = os.path.join(tmp_save_dir, str(id_))
    if not os.path.exists(tmp_save_dir):
        os.makedirs(tmp_save_dir)
    try: 
        wget.download(url, out=tmp_save_dir, bar=False)
    except Exception as e:
        exception_type = str(type(e)).split("'")[1]
        exception_ = str(e)
        shutil.rmtree(tmp_save_dir) 
    else:
        exception_type = "N/A"
        exception_ = "N/A"
    current_utc_time = str(datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
    return current_utc_time, url, exception_type, exception_


def solve_filename_encoding_error(current_dir):
    """
    solve encoding error of unzipped font file names 
    """
    for path in glob.glob(os.path.join(current_dir, "**", "*"), recursive=True):
        if os.path.splitext(path)[1] in [".ttf", ".otf", ".ttc", ".TTF", ".OTF", ".TTC"]:
            new_file_name = os.path.basename(path).encode("cp437").decode("utf-8")
            if new_file_name != os.path.basename(path):
                os.rename(path, os.path.join(os.path.dirname(path), new_file_name))


def rename_lower_case(current_dir):
    for path in glob.glob(os.path.join(current_dir, "**", "*.TTF"), recursive=True):
        os.rename(path, os.path.splitext(path)[0] + ".ttf")

    for path in glob.glob(os.path.join(current_dir, "**", "*.OTF"), recursive=True):
        os.rename(path, os.path.splitext(path)[0] + ".otf")
        
    for path in glob.glob(os.path.join(current_dir, "**", "*.TTC"), recursive=True):
        os.rename(path, os.path.splitext(path)[0] + ".ttc")


def clean_corrupted_font_names(current_dir, extensions=[".ttf", ".otf", ".ttc"]):
    for path in glob.glob(os.path.join(current_dir, "**", "*"), recursive=True):
        if os.path.splitext(path)[1] not in extensions:
            continue
        file_name, ext_ = os.path.splitext(os.path.basename(path)) 
        dirname = os.path.dirname(path)
        new_file_name = file_name.strip().lstrip(string_digits).lstrip(string_punctuation)
        if new_file_name != file_name:
            os.rename(path, os.path.join(dirname, new_file_name + ext_))


def convert_all_cjk_file_names(current_dir, cjk_file_names_df, extensions=[".ttf", ".otf", ".ttc"]):
    """
    convert all file names containing CJK characters into pinyin (latin) form
    """
    for extension in ["*" + xx for xx in extensions]:
        for path in glob.glob(os.path.join(current_dir, "**", extension), recursive=True):
            basename = os.path.basename(path)
            pinyin_list = pypinyin.lazy_pinyin(basename)
            if basename != "".join(pinyin_list):
                pinyin_name = "".join([xx.capitalize() for xx in pinyin_list])
                dirname = os.path.dirname(path)
                idx_ = current_dir.split(os.sep)[-1]
                cjk_file_names_df.append((idx_, basename, pinyin_name, dirname))
                os.rename(path, os.path.join(dirname, pinyin_name))



if __name__ == "__main__":
    t0 = time.time()

    # read command line arguments
    parser = argparse.ArgumentParser(description="This script automatically downloads (and unzips) fonts") 
    parser.add_argument("-p", "--nb_processes", type=int, default=None)
    parser.add_argument("-t", "--wait_before_start", type=int, default=5, 
                        help="Number of seconds to wait before start. Default is 5 seconds")
    parser.add_argument("-i", "--include_ttc", action="store_true", default=False)
    args = parser.parse_args() 

    # determine the number of processes to use
    if args.nb_processes is None:
        args.nb_processes = multiprocessing.cpu_count() 

    print("Using {} CPU cores.".format(args.nb_processes))

    # print some message for the user
    print("Previously downloaded fonts will be deleted when the process starts.")

    if args.wait_before_start > 0:
        print("The process will start in {} seconds.".format(args.wait_before_start))
        # countdown
        for i in reversed(range(1, args.wait_before_start + 1)):
            print(i)
            time.sleep(1)

    print("The process starts.")
    print("This can take a while. Please be patient...")


    # clean tmp files resulted from interrupted process
    for path in glob.glob("*.tmp"):
        os.remove(path)

    # make temporary_save_directory directory
    tmp_save_dir = "temporary_save_directory"
    if os.path.exists(tmp_save_dir):
        shutil.rmtree(tmp_save_dir) 
    os.makedirs(tmp_save_dir)

    # read predefined URL list
    with open("predefined_url_list.txt", "r") as f:
        URLs = f.read().split("\n")

    # download fonts 
    with multiprocessing.Pool(args.nb_processes) as pool:
        imap_it = list(tqdm.tqdm(pool.imap(download_func, 
                                           zip(URLs, range(len(URLs)), 
                                           itertools.repeat(tmp_save_dir))), 
                       total=len(URLs))) 
        df = []
        for current_utc_time, url, exception_type, exception_ in imap_it:
            df.append((url, current_utc_time, exception_type, exception_))

    # make metadata 
    df = pd.DataFrame(df, columns=["URL", "download_UTC_time", "exception_type", "exception"])

    flags = [False] * df.shape[0]
    for name_ in os.listdir(tmp_save_dir):
        if os.path.isdir(os.path.join(tmp_save_dir, name_)):
            flags[int(name_)] = True 
    df["successful_download"] = flags 


    extensions = [".ttf", ".otf"]
    if args.include_ttc:
        extensions += [".ttc"]

    font2url_id = defaultdict(list)
    url_id2font = defaultdict(list)

    cjk_file_names_df = []
    for i, flag in enumerate(flags):
        if flag:
            current_dir = os.path.join(tmp_save_dir, str(i))
            # unzip
            for path in glob.glob(os.path.join(current_dir, "**", "*.zip"), recursive=True):
                try:
                    with zipfile.ZipFile(path, "r") as f:
                        f.extractall(os.path.dirname(path))
                except zipfile.BadZipFile:
                    pass 
                finally:
                    os.remove(path) 
            # rename files 
            solve_filename_encoding_error(current_dir) 
            rename_lower_case(current_dir) 
            clean_corrupted_font_names(current_dir, extensions=extensions)
            convert_all_cjk_file_names(current_dir, cjk_file_names_df, extensions=extensions)
            # build the two metadata dictionaries font2url_id and url_id2font
            for ext_ in ["*" + xx for xx in extensions]:
                for path in glob.glob(os.path.join(current_dir, "**", ext_), recursive=True):
                    font2url_id[os.path.basename(path)].append(i) 
                    url_id2font[i].append(os.path.basename(path))
    cjk_file_names_df = pd.DataFrame(cjk_file_names_df, columns=["URL_id", "original_name", 
                                                                 "font_file", "directory"])
    cjk_file_names_df.to_csv("cjk_file_name_table.csv", sep="\t", encoding="utf-8")


    df["URL_id"] = df.index
    df = df[["URL_id", "URL", "successful_download", "download_UTC_time", "exception_type", "exception"]]
    log_file_name = "log_download.csv"
    df.to_csv(log_file_name, sep="\t", encoding="utf-8")

    for key in font2url_id.keys():
        font2url_id[key] = sorted(font2url_id[key])
    for key in url_id2font.keys():
        url_id2font[key] = sorted(url_id2font[key])

    with open("font2url_id.pkl", "wb") as f:
        pickle.dump(font2url_id, f)
    with open("url_id2font.pkl", "wb") as f:
        pickle.dump(url_id2font, f)

    if len(flags) != sum(flags):
        print("{}/{} URLs failed.".format(len(flags) - sum(flags), len(flags)))
        print("Check log_download.csv for more details.")
    else:
        print("All of the {} URLs succeeded.".format(len(flags)))

    print("Done in {:.3f} s.".format(time.time() - t0))



