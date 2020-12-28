"""
Helper module to run "run.py"
"""


class Parameters:
	count = 1000
	#count = 5
	size = 32 
	ensure_square_layout = True
	#font_size = 12
	rotation = 30 
	random_rotation = True 
	shear_x = 0.5
	random_shear_x = True  
	image_mode = "L" # -im 
	#margins = "0,0,0,0"
	margins = "0.1,0.1,0.1,0.1"
	background = 1 # -b (by default, 0: Gaussian Noise)
	#language = "latin" # -l
	#dict_ = "alphabets/fine/chinese_group1" # --dict (alphabet)
	dict_ = "alphabets/fine/russian" # --dict (alphabet)
	#font = "fonts/fonts/Osaka.ttf"
	font_index = "fonts/russian" # -fidx
	#font_index = "fonts/variable_weight_russian" # -fidx
	#random_font_weight = True 
	#dict_ = "alphabets/fine/basic_latin_lowercase" # --dict (alphabet)
	#font_index = "prepare_fonts/fonts/basic_latin_lowercase" # -fidx
	#nb_processes = 1 
	#random_translation_x = True 
	#random_translation_y = True 
	#linear_transform = "1,0,0,1" 
	#random_seed = 42
	#perspective_transform = "0,0,1,0,0,1,1,1"
	#perspective_transform = "0.3,0.3,0.7,0,0,1,1,1"
	#perspective_transform = "0.3,0,0.99,0,0,0.99,1,0.9"
	#random_perspective_transform = 0.05   
	#gaussian_prior_resizing = 1 
	#stroke_fill = "255,0,0"
	#random_stroke_fill = True 
	#morph_erosion = "3,5"
	#random_morph_erosion = True 



if __name__ == "__main__":
	import subprocess

	ignore_list = ["__module__", "__dict__", "__weakref__", "__doc__"]

	cmd = "python3 run.py"
	
	parameters = vars(Parameters) 
	for key, value in parameters.items():
		if key in ignore_list:
			continue 
		if len(key) >= 2 and key[-1] == "_" and key[-2] != "_":
			key = key[:-1]
		if isinstance(value, bool) and value:
			cmd += " --{}".format(key)
		else:
			cmd += " --{} {}".format(key, str(value))

	# run the command 
	subprocess.call(cmd.split())


	




