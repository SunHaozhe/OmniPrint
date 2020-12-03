import os 
import glob 
import numpy as np 
import pandas as pd 
from PIL import Image


class AutoMLformat:
	"""
	https://github.com/codalab/chalab/wiki/Help:-Wizard-%E2%80%90-Challenge-%E2%80%90-Data

	raw_dataset_path:
		For example, out/20201203_161302_297288 
	image_format:
		png, jpg, etc. 
	"""
	def __init__(self, dataset_name, raw_dataset_path, label_name, image_format="png"):
		if not os.path.exists(raw_dataset_path):
			raise Exception("Raw dataset not found, please check the path.")
		self.dataset_name = dataset_name
		self.raw_dataset_path = raw_dataset_path 
		self.label_name = label_name 
		self.image_format = image_format 

		self._build_data_matrix() 
		self._build_solution_vector()

		
	def _build_data_matrix(self):
		self.data_matrix = []
		for path in sorted(glob.glob(os.path.join(self.raw_dataset_path, 
												  "data", "*.{}".format(self.image_format)))):
			img = np.array(Image.open(path)).ravel()
			self.data_matrix.append(img)
		self.data_matrix = np.vstack(self.data_matrix) 

	def _build_solution_vector(self):
		df = pd.read_csv(os.path.join(self.raw_dataset_path, "label", "raw_labels.csv"), 
						 sep="\t", encoding="utf-8")
		self.solution_vector = df.loc[:, self.label_name].tolist()

	def save(self, output_dir="datasets"):
		output_dir = os.path.join(output_dir, self.dataset_name)
		if not os.path.exists(output_dir):
		    os.makedirs(output_dir) 

		self._save_data_matrix(output_dir)
		self._save_solution_vector(output_dir)
		

	def _save_data_matrix(self, output_dir):
		data_matrix_str = []
		for row in range(self.data_matrix.shape[0]):
			data_matrix_str.append(" ".join([str(x) for x in self.data_matrix[row, :]]))
		data_matrix_str = "\n".join(data_matrix_str)
		with open(os.path.join(output_dir, "{}.data".format(self.dataset_name)), "w") as f:
			f.write(data_matrix_str)

	def _save_solution_vector(self, output_dir):
		solution_vector_str = "\n".join([str(x) for x in self.solution_vector])
		with open(os.path.join(output_dir, "{}.solution".format(self.dataset_name)), "w") as f:
			f.write(solution_vector_str) 








