"""
Helper module to run "run.py"
"""


class Parameters:
	count = 10
	width = 32
	random_skew = True # -rk
	skew_angle = 30 # -k
	image_mode = "L" # -im 
	margins = "0,0,0,0"
	fit = True
	background = 1 # -b
	distorsion = 1 # -d
	language = "cn" # -l
	dict_ = "alphabets/fine/chinese_group1" # --dict
	font_index = "prepare_fonts/fonts/chinese_group1" # -fidx









if __name__ == "__main__":
	import subprocess

	ignore_list = ["__module__", "__dict__", "__weakref__", "__doc__"]

	tmp = vars(Parameters) 
	parameters = {}
	for key, value in tmp.items():
		if key in ignore_list:
			continue 
		if len(key) >= 2 and key[-1] == "_" and key[-2] != "_":
			key = key[:-1]
		if isinstance(value, bool):
				parameters[key] = value
		else:
			parameters[key] = str(value)
	
	cmd = "python3 run.py"
	for key, value in parameters.items():
		if isinstance(value, bool) and value:
			cmd += " --{}".format(key)
		else:
			cmd += " --{} {}".format(key, value)
	
	# run the command 
	subprocess.call(cmd.split())


	




