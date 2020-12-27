import time 
from dataset_formatter import * 
import argparse


t0 = time.time()

# read command line arguments
parser = argparse.ArgumentParser() 
parser.add_argument("-n", "--dataset_name", type=str, default="hello_world_classification_dataset")
parser.add_argument("-r", "--raw_dataset_path", type=str, default="../out/20201222_232645_378905")
parser.add_argument("-l", "--label_name", type=str, default="unicode_code_point")
parser.add_argument("-ir", "--is_regression", action="store_true", default=False)
args = parser.parse_args() 


AutoML_format(dataset_name=args.dataset_name, 
			  raw_dataset_path=args.raw_dataset_path, 
			  label_name=args.label_name, 
			  is_regression=args.is_regression)


print("Done in {:.4f} s.".format(time.time() - t0))


