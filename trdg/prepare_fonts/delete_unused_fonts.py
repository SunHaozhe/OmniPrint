import time
import argparse
import os
import glob


t0 = time.time()

# read command line arguments
parser = argparse.ArgumentParser(description="Delete unused fonts in the generated font directory") 
parser.add_argument("-i", "--include_ttc", action="store_true", default=False)
args = parser.parse_args() 


font_directory = "fonts"
if not os.path.exists(font_directory):
	raise Exception("Font directory not found.")

extensions = [".ttf", ".otf"]
if args.include_ttc:
	extensions += [".ttc"]

print("Deleting unused fonts in the generated font directory.")

used = set()
for path in glob.glob(os.path.join(font_directory, "index", "*.txt")):
	with open(path, "r") as f:
		used.update(f.read().split("\n"))

for file_name in os.listdir(font_directory):
	if os.path.splitext(file_name)[1] in extensions:
		if os.path.basename(file_name) not in used: 
			os.remove(os.path.join(font_directory, file_name))

print("Done in {:.3f} s.".format(time.time() - t0)) 






