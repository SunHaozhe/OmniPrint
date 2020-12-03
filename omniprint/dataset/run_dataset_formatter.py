import time 
from dataset_formatter import * 
import argparse


t0 = time.time()

# read command line arguments
parser = argparse.ArgumentParser(description="This script automatically downloads (and unzips) fonts") 
parser.add_argument("-n", "--dataset_name", type=str, default="hello_world_classification_dataset")
parser.add_argument("-r", "--raw_dataset_path", type=str, default="../out/20201203_110652_111286")
parser.add_argument("-l", "--label_name", type=str, default="unicode_code_point")
args = parser.parse_args() 


formatter = AutoMLformat(dataset_name=args.dataset_name, 
						 raw_dataset_path=args.raw_dataset_path, 
						 label_name=args.label_name)
formatter.save()

print("Done in {:.4f} s.".format(time.time() - t0))


