import argparse
import os, errno
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import random 
import string
import sys
import numpy as np 
import datetime
import pandas as pd 

import tqdm
from omniprint.string_generator import (
	create_strings_from_dict,
	create_strings_from_file,
	create_strings_from_wikipedia,
	create_strings_randomly,
)
from omniprint.utils import load_dict, load_fonts 
from omniprint.utils import add_txt_extension, generate_label_dataframe 
from omniprint.data_generator import TextDataGenerator
import multiprocessing



def parse_margins(x):
	x = x.split(",")
	if len(x) == 1:
		return [float(x[0])] * 4
	return [float(el) for el in x]


def parse_color(x):
	x = x.split(",")
	if len(x) == 1:
		return [int(x[0])] * 3
	return [int(el) for el in x]


def parse_affine_perspective_transformations(x):
	x = x.split(",")
	if len(x) == 1:
		return [float(x[0])] * 6
	return [float(el) for el in x]


def parse_linear_transform(x):
	x = x.split(",")
	if len(x) in [4, 5, 9]:
		return [float(el) for el in x]
	else:
		raise Exception("The length of --linear_transform must be 4, 5 or 9.")


def parse_perspective_transform(x):
	x = x.split(",")
	if len(x) == 8:
		return [float(el) for el in x]
	else:
		raise Exception("The length of --perspective_transform must be 8.")

def parse_morphological_image_processing_iteration(x):
	x = x.split(",") 
	assert len(x) in [2, 3]
	x_0 = int(x[0])
	assert x_0 >= 1 
	x_1 = int(x[1])
	assert x_1 >= 1 
	if len(x) == 2:
		return x_0, x_1, None 
	else:
		return x_0, x_1, x[2] 

def parse_morphological_image_processing(x):
	x = x.split(",") 
	assert len(x) in [1, 2]
	x_0 = int(x[0])
	assert x_0 >= 1 
	if len(x) == 1:
		return x_0, None 
	else:
		return x_0, x[1] 


def parse_arguments():
	"""
		Parse the command line arguments of the program.
	"""

	parser = argparse.ArgumentParser(
		description="Generate synthetic text data for text recognition."
	)
	parser.add_argument("--output_dir", type=str, nargs="?", help="The output directory", default="out")
	parser.add_argument(
		"-c",
		"--count",
		type=int,
		nargs="?",
		help="The number of images to be created.",
		default=10
	)
	parser.add_argument(
		"-s",
		"--size",
		type=int,
		nargs="?",
		help="Define the height of the produced images. If the option " +\
		"--ensure_square_layout is activated, then this will also be the " +\
		"width of the produced images, otherwise the width will be determined " +\
		"by both the length of the text and the height.",
		default=32,
	)
	parser.add_argument(
		"-p",
		"--nb_processes",
		type=int,
		nargs="?",
		help="Define the number of processes to use for image generation. " +\
		"If not provided, this equals to the number of CPU cores", 
		default=None,
	)
	parser.add_argument(
		"-e",
		"--extension",
		type=str,
		nargs="?",
		help="Define the extension to save the image with",
		default="png",
	)
	parser.add_argument(
		"-bl",
		"--blur",
		type=int,
		nargs="?",
		help="Apply gaussian blur to the resulting sample. Should be " +\
		"an integer defining the blur radius, 0 by default.",
		default=None,
	)
	parser.add_argument(
		"-rbl",
		"--random_blur",
		action="store_true",
		help="When set, the blur radius will be randomized between 0 and -bl.",
		default=False,
	)
	parser.add_argument(
		"-b",
		"--background",
		type=int,
		nargs="?",
		help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Image",
		default=0,
	)
	parser.add_argument(
		"-om",
		"--output_mask",
		action="store_true",
		help="Define if the generator will return masks for the text",
		default=False,
	)
	parser.add_argument(
		"-m",
		"--margins",
		type=parse_margins,
		nargs="?",
		help="Define the margins (percentage) around the text when rendered. Each element should be a float",
		default=[0, 0, 0, 0],
	)
	parser.add_argument(
		"-ft", "--font", type=str, nargs="?", help="Define font to be used"
	)
	parser.add_argument(
		"-fd",
		"--font_dir",
		type=str,
		nargs="?",
		help="Define a font directory to be used",
	)
	parser.add_argument(
		"-fidx",
		"--font_index",
		type=str,
		nargs="?",
		help="Define the font index file to be used, an example is fonts{}latin.txt".format(os.sep), 
	)
	parser.add_argument(
		"-id",
		"--image_dir",
		type=str,
		nargs="?",
		help="Define an image directory to use when background is set to image",
		default=os.path.join(os.path.split(os.path.realpath(__file__))[0], "images"),
	)
	parser.add_argument(
		"-dt", "--dict", type=str, nargs="?", help="Define the dictionary to be used",
		default="alphabets/fine/basic_latin_lowercase"
	)
	parser.add_argument(
		"-fwt",
		"--font_weight",
		type=float, 
		nargs="?",
		help="Define the width of the strokes",
		default=400,
	)
	parser.add_argument(
		"-rfwt",
		"--random_font_weight", 
		action="store_true",
		help="Use random font weight (stroke width).", 
		default=False
	)
	parser.add_argument(
		"-stf",
		"--stroke_fill",
		type=parse_color, 
		nargs="?",
		help="Define the color of the strokes",
		default=None,  
	)
	parser.add_argument(
		"-rstf",
		"--random_stroke_fill", 
		action="store_true",
		help="Use random color to fill strokes.", 
		default=False
	)
	parser.add_argument(
		"-im",
		"--image_mode",
		type=str,
		nargs="?",
		help="Define the image mode to be used. RGB is default, L means 8-bit grayscale images, 1 means 1-bit binary images stored with one pixel per byte, etc.",
		default="RGB",
	)
	parser.add_argument(
		"-rsd",
		"--random_seed",
		type=int,
		help="Random seed",
		default=None,
	)
	parser.add_argument(
		"-esl",
		"--ensure_square_layout",
		action="store_true", 
		help="Whether the width should be the same as the height",
		default=False,
	)
	parser.add_argument(
		"-otlwd",
		"--outline_width",
		type=int,
		help="Width of stroke outline. Not yet implemented",
		default=None,
	)
	parser.add_argument(
		"-fsz",
		"--font_size",
		type=int,
		help="Font size in point",
		default=192,
	)
	parser.add_argument(
		"-lt",
		"--linear_transform",
		type=parse_linear_transform,
		nargs="?",
		help="The parameter for linear transform. The length must be 4, 5 or 9. " +\
		"Length 4 corresponds low level parameterization, which means a, b, d, e, this " +\
		"is the most customizable parameterization. Length 5 and length 9 correspond to " +\
		"high level parameterization. Length 5 means rotation, shear_x, shear_y, " +\
		"scale_x, scale_y. Length 9 means rotation, shear_x, shear_y, " +\
		"scale_x, scale_y, alpha_, beta_, gamma_, delta_. If this parameter is set, " +\
		"other linear transform parameters like rotation, shear_x, etc. will be ignored",
		default=None,
	)
	parser.add_argument(
		"-rtn",
		"--rotation",
		type=float,
		nargs="?",
		help="Define rotation angle (in degree) of the generated text. " +\
		"Used only when --linear_transform is not set",
		default=None,
	)
	parser.add_argument(
		"-rrtn",
		"--random_rotation",
		action="store_true",
		help="Uniformly sample the value of rotation, the parameter --rotation needs to be set. " +\
		"The range is [-abs(rotation), abs(rotation)]", 
		default=False,
	)
	parser.add_argument(
		"-shrx",
		"--shear_x",
		type=float,
		nargs="?",
		help="High level linear transform parameter, horizontal shear. " +\
		"Used only when --linear_transform is not set",
		default=None,
	)
	parser.add_argument(
		"-rshrx",
		"--random_shear_x",
		action="store_true",
		help="Uniformly sample the value of shear_x, the parameter --shear_x needs to be set. " +\
		"The range is [-abs(shear_x), abs(shear_x)]", 
		default=False,
	)
	parser.add_argument(
		"-shry",
		"--shear_y",
		type=float,
		nargs="?",
		help="High level linear transform parameter, vertical shear. " +\
		"Used only when --linear_transform is not set",
		default=None,
	)
	parser.add_argument(
		"-rshry",
		"--random_shear_y",
		action="store_true",
		help="Uniformly sample the value of shear_y, the parameter --shear_y needs to be set. " +\
		"The range is [-abs(shear_y), abs(shear_y)]", 
		default=False,
	)
	parser.add_argument(
		"-sclx",
		"--scale_x",
		type=float,
		nargs="?",
		help="High level linear transform parameter, horizontal scale. " +\
		"Used only when --linear_transform is not set",
		default=None,
	)
	parser.add_argument(
		"-rsclx",
		"--random_scale_x",
		action="store_true",
		help="Uniformly sample the value of scale_x, the parameter --scale_x needs to be set. " +\
		"The range is [-abs(scale_x), abs(scale_x)]", 
		default=False,
	)
	parser.add_argument(
		"-scly",
		"--scale_y",
		type=float,
		nargs="?",
		help="High level linear transform parameter, vertical scale. " +\
		"Used only when --linear_transform is not set",
		default=None,
	)
	parser.add_argument(
		"-rscly",
		"--random_scale_y",
		action="store_true",
		help="Uniformly sample the value of scale_y, the parameter --scale_y needs to be set. " +\
		"The range is [-abs(scale_y), abs(scale_y)]", 
		default=False,
	)
	parser.add_argument(
		"-alpha",
		"--alpha",
		type=float,
		nargs="?",
		help="Customizable high level linear transform parameter, top left element in the 2x2 matrix. " +\
		"Used only when --linear_transform is not set",
		default=None,
	)
	parser.add_argument(
		"-ralpha",
		"--random_alpha",
		action="store_true",
		help="Uniformly sample the value of alpha, the parameter --alpha needs to be set. " +\
		"The range is [-abs(alpha), abs(alpha)]", 
		default=False,
	)
	parser.add_argument(
		"-beta",
		"--beta",
		type=float,
		nargs="?",
		help="Customizable high level linear transform parameter, top right element in the 2x2 matrix. " +\
		"Used only when --linear_transform is not set",
		default=None,
	)
	parser.add_argument(
		"-rbeta",
		"--random_beta",
		action="store_true",
		help="Uniformly sample the value of beta, the parameter --beta needs to be set. " +\
		"The range is [-abs(beta), abs(beta)]", 
		default=False,
	)
	parser.add_argument(
		"-gamma",
		"--gamma",
		type=float,
		nargs="?",
		help="Customizable high level linear transform parameter, bottom left element in the 2x2 matrix. " +\
		"Used only when --linear_transform is not set",
		default=None,
	)
	parser.add_argument(
		"-rgamma",
		"--random_gamma",
		action="store_true",
		help="Uniformly sample the value of gamma, the parameter --gamma needs to be set. " +\
		"The range is [-abs(gamma), abs(gamma)]", 
		default=False,
	)
	parser.add_argument(
		"-delta",
		"--delta",
		type=float,
		nargs="?",
		help="Customizable high level linear transform parameter, bottom right element in the 2x2 matrix. " +\
		"Used only when --linear_transform is not set",
		default=None,
	)
	parser.add_argument(
		"-rdelta",
		"--random_delta",
		action="store_true",
		help="Uniformly sample the value of delta, the parameter --delta needs to be set. " +\
		"The range is [-abs(delta), abs(delta)]", 
		default=False,
	)
	parser.add_argument(
		"-rtslnx",
		"--random_translation_x",
		action="store_true",
		help="Uniformly sample the value of horizontal translation. " +\
		"This will have no effect if horizontal margins are 0", 
		default=False,
	)
	parser.add_argument(
		"-rtslny",
		"--random_translation_y",
		action="store_true",
		help="Uniformly sample the value of vertical translation. " +\
		"This will have no effect if vertical margins are 0", 
		default=False,
	)
	parser.add_argument(
		"-pt",
		"--perspective_transform",
		type=parse_perspective_transform,
		nargs="?",
		help="Given the coordinates of the four corners of the first quadrilateral " +\
		"and the coordinates of the four corners of the second quadrilateral, " +\
		"perform the perspective transform that maps a new point in the first " +\
		"quadrilateral onto the appropriate position on the second quadrilateral. " +\
		"Perspective transformation simulates different angle of view. " +\
		"Enter 8 real numbers (float) which correspond to the 4 corner points (2D coordinates) " +\
		"of the target quadrilateral, these 4 corner points which be respectively " +\
		"mapped to [[0, 0], [1, 0], [0, 1], [1, 1]] in the source quadrilateral. " +\
		"[0, 0] is the top left corner, [1, 0] is the top left corner, [0, 1] is " +\
		"the bottom left corner, [1, 1] is the bottom right corner." +\
		"These coordinates have been normalized to unit square [0, 1]^2. Thus, " +\
		"the entered corner points should match the order of magnitude and must be convex. " +\
		"For example, 0,0,1,0,0,1,1,1 will produce identity transform. " +\
		"This option will have no effect if --random_perspective_transform is set. " +\
		"This option sometimes will cut off the periphery of the text, causing noisy " +\
		"data. ",
		default=None
	)
	parser.add_argument(
		"-rpt",
		"--random_perspective_transform",
		type=float,
		nargs="?",
		help="Randomly generate a convex quadrilateral which will be mapped to the normalized unit square, " +\
		"the value of each axis is independently sampled from the gaussian distribution, " +\
		"the standard deviation of the gaussian distribution is given by --random_perspective_transform. " +\
		"If this option is present but not followed by a command-line argument, the standard deviation " +\
		"0.05 will be used by default. ",
		default=None,
		const=0.05
	)
	parser.add_argument(
		"-gpr",
		"--gaussian_prior_resizing",
		type=float,
		help="If not None, apply Gaussian filter to smooth image prior to resizing, " +\
		"the argument of this parameter needs to be a float, which will be used as the " +\
		"standard deviation of Gaussian filter. Default is None, which means Gaussian " +\
		"filter is not used before resizing. ",
		default=None
	)
	parser.add_argument(
		"-morphero",
		"--morph_erosion",
		type=parse_morphological_image_processing_iteration,
		help="Morphological image processing - erosion. The argument must be a tuple " +\
		"separated by comma without space, the first element is the kernel " +\
		"size, the second element is the number of iterations. For example, 3,2 means " +\
		"kernel_size=3x3, iterations=2. 3,2,ellipse (3,2,cross) means using " +\
		"elliptical (cross-shaped) kernel respectively. If the third argument is not given, " +\
		"the default kernel shape will be rectangle. ",
		default=None
	)
	parser.add_argument(
		"-rmorphero",
		"--random_morph_erosion",
		action="store_true",
		help="Uniformly sample the value of morphological erosion, the parameter " +\
		"--morph_erosion needs to be set. " +\
		"The range is [1, kernel_size] ([1, iterations]) " +\
		"kernel_shape is randomly chosen among [rectangle, ellipse, cross].",  
		default=False
	)
	parser.add_argument(
		"-morphdil",
		"--morph_dilation",
		type=parse_morphological_image_processing_iteration,
		help="Morphological image processing - dilation. The argument must be a tuple " +\
		"separated by comma without space, the first element is the kernel " +\
		"size, the second element is the number of iterations. For example, 3,2 means " +\
		"kernel_size=3x3, iterations=2. 3,2,ellipse (3,2,cross) means using " +\
		"elliptical (cross-shaped) kernel respectively. If the third argument is not given, " +\
		"the default kernel shape will be rectangle. ",
		default=None
	)
	parser.add_argument(
		"-rmorphdil",
		"--random_morph_dilation",
		action="store_true",
		help="Uniformly sample the value of morphological dilation, the parameter " +\
		"--morph_dilation needs to be set. " +\
		"The range is [1, kernel_size] ([1, iterations]) " +\
		"kernel_shape is randomly chosen among [rectangle, ellipse, cross].", 
		default=False
	)
	parser.add_argument(
		"-morphope",
		"--morhp_opening",
		type=parse_morphological_image_processing,
		help="Morphological image processing - opening. The argument must be a tuple " +\
		"separated by comma without space, the first element is the kernel " +\
		"size, the second element is the kernel shape. For example, 3 means " +\
		"kernel_size=3x3. 3,ellipse (3,cross) means using " +\
		"elliptical (cross-shaped) kernel respectively. If the second argument is not given, " +\
		"the default kernel shape will be rectangle. ",
		default=None
	)
	parser.add_argument(
		"-rmorphope",
		"--random_morph_opening",
		action="store_true",
		help="Uniformly sample the value of morphological opening, the parameter " +\
		"--morph_opening needs to be set. " +\
		"The range is [1, kernel_size] " +\
		"kernel_shape is randomly chosen among [rectangle, ellipse, cross].",  
		default=False
	)
	parser.add_argument(
		"-morphclo",
		"--morhp_closing",
		type=parse_morphological_image_processing,
		help="Morphological image processing - closing. The argument must be a tuple " +\
		"separated by comma without space, the first element is the kernel " +\
		"size, the second element is the kernel shape. For example, 3 means " +\
		"kernel_size=3x3. 3,ellipse (3,cross) means using " +\
		"elliptical (cross-shaped) kernel respectively. If the second argument is not given, " +\
		"the default kernel shape will be rectangle. ",
		default=None
	)
	parser.add_argument(
		"-rmorphclo",
		"--random_morph_closing",
		action="store_true",
		help="Uniformly sample the value of morphological closing, the parameter " +\
		"--morph_closing needs to be set. " +\
		"The range is [1, kernel_size] " +\
		"kernel_shape is randomly chosen among [rectangle, ellipse, cross].",  
		default=False
	)
	parser.add_argument(
		"-morphgra",
		"--morhp_gradient",
		type=parse_morphological_image_processing,
		help="Morphological image processing - gradient. The argument must be a tuple " +\
		"separated by comma without space, the first element is the kernel " +\
		"size, the second element is the kernel shape. For example, 3 means " +\
		"kernel_size=3x3. 3,ellipse (3,cross) means using " +\
		"elliptical (cross-shaped) kernel respectively. If the second argument is not given, " +\
		"the default kernel shape will be rectangle. ",
		default=None
	)
	parser.add_argument(
		"-rmorphgra",
		"--random_morph_gradient",
		action="store_true",
		help="Uniformly sample the value of morphological gradient, the parameter " +\
		"--morph_gradient needs to be set. " +\
		"The range is [1, kernel_size] " +\
		"kernel_shape is randomly chosen among [rectangle, ellipse, cross].",  
		default=False
	)
	parser.add_argument(
		"-morphtoph",
		"--morhp_tophat",
		type=parse_morphological_image_processing,
		help="Morphological image processing - Top Hat. The argument must be a tuple " +\
		"separated by comma without space, the first element is the kernel " +\
		"size, the second element is the kernel shape. For example, 3 means " +\
		"kernel_size=3x3. 3,ellipse (3,cross) means using " +\
		"elliptical (cross-shaped) kernel respectively. If the second argument is not given, " +\
		"the default kernel shape will be rectangle. ",
		default=None
	)
	parser.add_argument(
		"-rmorphtoph",
		"--random_morph_tophat",
		action="store_true",
		help="Uniformly sample the value of morphological tophat, the parameter " +\
		"--morph_tophat needs to be set. " +\
		"The range is [1, kernel_size] " +\
		"kernel_shape is randomly chosen among [rectangle, ellipse, cross].",  
		default=False
	)
	parser.add_argument(
		"-morphblah",
		"--morhp_blackhat",
		type=parse_morphological_image_processing,
		help="Morphological image processing - Black Hat. The argument must be a tuple " +\
		"separated by comma without space, the first element is the kernel " +\
		"size, the second element is the kernel shape. For example, 3 means " +\
		"kernel_size=3x3. 3,ellipse (3,cross) means using " +\
		"elliptical (cross-shaped) kernel respectively. If the second argument is not given, " +\
		"the default kernel shape will be rectangle. ",
		default=None
	)
	parser.add_argument(
		"-rmorphblah",
		"--random_morph_blackhat",
		action="store_true",
		help="Uniformly sample the value of morphological blackhat, the parameter " +\
		"--morph_blackhat needs to be set. " +\
		"The range is [1, kernel_size] " +\
		"kernel_shape is randomly chosen among [rectangle, ellipse, cross].", 
		default=False
	)
	return parser.parse_args()


def main():
	"""
		Description: Main function
	"""

	# Argument parsing
	args = parse_arguments()

	if args.random_seed is not None:
		random.seed(args.random_seed)
		np.random.seed(2 * args.random_seed + 1)


	# Creating word list
	if args.dict:
		lang_dict = []
		args.dict = add_txt_extension(args.dict)
		if os.path.isfile(args.dict):
			with open(args.dict, "r", encoding="utf8", errors="ignore") as d:
				lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
		else:
			sys.exit("Cannot open dict")
	else:
		raise Exception("Please set the option --dict") 

	# Creating font (path) list
	if args.font_index:
		font_index = args.font_index.split(os.sep)
		if len(font_index) == 1: # only the text set file
			font_index_dir = "fonts"
			font_index_file = font_index[0]
		elif len(font_index) == 2:
			font_index_dir, font_index_file = font_index
		elif len(font_index) > 2:
			font_index_dir = os.sep.join(font_index[:-1])
			font_index_file = font_index[-1]
		else:
			raise Exception("Wrong --font_index format, a correct example fonts{}latin.txt".format(os.sep)) 
		font_index_file = add_txt_extension(font_index_file)  
		with open(os.path.join(font_index_dir, "index", font_index_file), "r") as f:
			fonts = [os.path.join(font_index_dir, "fonts", p) for p in f.read().split("\n")] 
	elif args.font_dir:
		fonts = []
		for p in glob.glob(os.path.join(args.font_dir, "*.ttf")):
			fonts.append(p) 
		for p in glob.glob(os.path.join(args.font_dir, "*.otf")):
			fonts.append(p) 
	elif args.font:
		if os.path.isfile(args.font):
			fonts = [args.font]
		else:
			sys.exit("Cannot open font")
	else:
		font_index_dir = "fonts"
		font_index_file = os.path.basename(args.dict) 
		font_index_file = add_txt_extension(font_index_file)  
		with open(os.path.join(font_index_dir, "index", font_index_file), "r") as f:
			fonts = [os.path.join(font_index_dir, "fonts", p) for p in f.read().split("\n")] 

	# Creating synthetic sentences (or word)
	strings = create_strings_from_dict(1, False, args.count, lang_dict)
	string_count = len(strings) 
	
	# determine fonts for each image 
	sampled_fonts = [fonts[random.randrange(0, len(fonts))] for _ in range(0, string_count)]

	# determine the number of processes to use 
	nb_processes = args.nb_processes
	if nb_processes is None:
		nb_processes = multiprocessing.cpu_count() 
	print("Using {} processes.".format(nb_processes))

	# create the dictionary of args 
	args_dict = vars(args)

	# use UTC time as the id of the generated dataset 
	args_dict["dataset_id"] = str(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"))
	dataset_dir = os.path.join(args.output_dir, args_dict["dataset_id"])
	# create data subdirectory
	output_data_dir = os.path.join(dataset_dir, "data")
	if not os.path.exists(output_data_dir):
		os.makedirs(output_data_dir)
	# create label subdirectory
	output_label_dir = os.path.join(dataset_dir, "label")
	if not os.path.exists(output_label_dir):
		os.makedirs(output_label_dir)
	args_dict["output_data_dir"] = output_data_dir

	# generate images using multiprocessing 
	labels = []
	with multiprocessing.Pool(nb_processes) as pool:
		imap_it = list(tqdm.tqdm(pool.imap(TextDataGenerator.generate_from_tuple, 
										   zip([i for i in range(0, string_count)], 
												strings, 
												sampled_fonts, 
												[args_dict] * string_count, 
												[False] * string_count)), 
								total=args.count))
	
	# collect labels from different processes 
	for label in imap_it:
		labels.append(label)

	# create the label file 
	external_dataframes = []
	if args.font_index:
		path = os.path.join(font_index_dir, "metadata", "font_description.csv")
		df = pd.read_csv(path, sep="\t", encoding="utf-8")
		df = df.loc[:, ["font_file", "family_name", "style_name", "postscript_name", 
						"variable_font_weight", "min_font_weight", "max_font_weight"]]
		external_dataframes.append(df)
	if len(external_dataframes) == 0:
		external_dataframes = None 
	generate_label_dataframe(labels, external_dataframes, save_path=output_label_dir)



if __name__ == "__main__":
	main()
