import os 
import glob 
import numpy as np 
import pandas as pd 
from PIL import Image


def AutoML_format(dataset_name, raw_dataset_path, label_name, image_format="png", 
				 is_regression=False, output_dir="datasets"):
	"""
	https://github.com/codalab/chalab/wiki/Help:-Wizard-%E2%80%90-Challenge-%E2%80%90-Data

	Currently, only regression and multiclass classification 
	tasks are fully supported. 

	raw_dataset_path:
		For example, out/20201203_161302_297288 
	image_format:
		png, jpg, etc. 
	""" 
	if not os.path.exists(raw_dataset_path):
		raise Exception("Raw dataset not found, please check the path.")

	output_dir = os.path.join(output_dir, dataset_name)
	if not os.path.exists(output_dir):
	    os.makedirs(output_dir) 

	# DataName.data
	data_matrix = []
	for path in sorted(glob.glob(os.path.join(raw_dataset_path, 
											  "data", "*.{}".format(image_format)))):
		img = np.array(Image.open(path)).ravel()
		data_matrix.append(img)
	data_matrix = np.vstack(data_matrix) 

	data_matrix_str = []
	for row in range(data_matrix.shape[0]):
		data_matrix_str.append(" ".join([str(x) for x in data_matrix[row, :]]))
	data_matrix_str = "\n".join(data_matrix_str)
	with open(os.path.join(output_dir, "{}.data".format(dataset_name)), "w") as f:
		f.write(data_matrix_str)

	# DataName_feat.name (optional header)
	feature_name = ["value_{}".format(i) for i in range(data_matrix.shape[1])]
	feature_name_str = "\n".join(feature_name) 
	with open(os.path.join(output_dir, "{}_feat.name".format(dataset_name)), "w") as f:
		f.write(feature_name_str)

	# DataName_feat.type
	feature_type = ["Numerical" for i in range(data_matrix.shape[1])]
	feature_type_str = "\n".join(feature_type) 
	with open(os.path.join(output_dir, "{}_feat.type".format(dataset_name)), "w") as f:
		f.write(feature_type_str)

	# DataName.solution 
	df = pd.read_csv(os.path.join(raw_dataset_path, "label", "raw_labels.csv"), 
					 sep="\t", encoding="utf-8")
	solution_vector = df.loc[:, label_name].tolist()
	if not is_regression:
		# classification task 
		label_names = sorted(list(set(solution_vector)))
		label_name_dict = {}
		for i, label_name in enumerate(label_names):
			label_name_dict[label_name] = i 
		new_solution_vector = []
		for x in solution_vector:
			new_solution_vector.append(label_name_dict[x])
		solution_vector = new_solution_vector
		# DataName_label.name 
		with open(os.path.join(output_dir, "{}_label.name".format(dataset_name)), "w") as f:
			f.write("\n".join([str(x) for x in label_names]))

		solution_vector_one_hot = np.zeros((len(solution_vector), len(label_names)), dtype=int)
		for i, x in enumerate(solution_vector):
			solution_vector_one_hot[i, x] = 1
		solution_vector_str = []
		for i in range(solution_vector_one_hot.shape[0]):
			solution_vector_str.append(" ".join([str(x) for x in solution_vector_one_hot[i, :]]))
		solution_vector_str = "\n".join(solution_vector_str)
	else:
		solution_vector_str = "\n".join([str(x) for x in solution_vector])

	with open(os.path.join(output_dir, "{}.solution".format(dataset_name)), "w") as f:
		f.write(solution_vector_str) 
	
	# DataName_public.info (optional public documentation) 
	public_info = []
	public_info.append("usage='{}'".format(_get_official_name(dataset_name)))
	public_info.append("name='{}'".format(dataset_name))
	if is_regression:
		public_info.append("task='regression'")
		public_info.append("target_type='Numerical'")
		public_info.append("metric='r2_metric'")
		public_info.append("target_num=1")
		public_info.append("label_num=NA")
	else:
		public_info.append("task='multiclass.classification'")
		public_info.append("target_type='Binary'")
		public_info.append("metric='auc_metric'")
		public_info.append("target_num={}".format(len(label_names)))
		public_info.append("label_num={}".format(len(label_names)))
	public_info.append("feat_type='Numerical'")
	public_info.append("feat_num='{}'".format(data_matrix.shape[1]))
	public_info.append("is_sparse=0")
	public_info.append("time_budget=500") # maximum of this version of Chalab
	with open(os.path.join(output_dir, "{}_public.info".format(dataset_name)), "w") as f:
		f.write("\n".join(public_info)) 
	




def _get_official_name(name):
	res = []
	for x in name.split("_"):
		if len(x) >= 2:
			res.append(x[0].capitalize() + x[1:])
		else:
			res.append(x)
	return " ".join(res) 



