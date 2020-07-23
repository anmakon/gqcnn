import numpy as np
import csv
import os
import argparse

# Script to add noise or depth modifications to the datasets.
# You can also create sub-datasets by adding neither noise, nor depth
# modifications. Selection can be done randomly, randomly with excluding
# training data, manually or from csv files. 

class Modification():
	def __init__(self,selection,Cornell=False):

		# "Preallocate" variables
		self.image_arr = []
		self.pose_arr = []
		self.file_arr = []
		self.metric_arr = []
		self.noise_arr = []
		self.depth_arr = []
		self.noise = False
		self.depth = False
		self.counter = 0

		# Set paths
		if Cornell ==True:
			self.export_path = "./data/training/Subset_datasets/Cornell_"
			self.data_path = "./data/training/Cornell/tensors/"
			self.csv_dir = "./data/training/csv_files/"
			split = "./data/training/Cornell/splits/image_wise/train_indices.npz"

		else:
			self.export_path = "./data/training/Subset_datasets/DexNet_"
			self.data_path = "./data/training/dexnet_2_tensor/tensors/"
			self.csv_dir = "./data/training/csv_files/"
			split = "./data/training/dexnet_2_tensor/splits/image_wise/train_indices.npz"
		self.split = np.load(split)['arr_0']

		self.images_per_file = 500
		self.num_images = 500
		self.ratio_pos = 0.5

		# Set selection type
		self.random = False
		self.manual = False
		self.csv = False
		self.filter_training = True

		if selection == 'random' or selection == 'Random':
			self.random = True
		elif selection == 'manual' or selection == 'Manual':
			self.manual = True
		elif selection == 'csv':
			self.csv = True

		else:
			raise ValueError("No selection type chosen.")


	def _save_files(self,counter):
		count_string =  ("{0:05d}").format(counter)
		np.savez(self.export_path+"depth_ims_tf_table_"+count_string,self.image_arr)
		np.savez(self.export_path+"hand_poses_"+count_string,self.pose_arr)
		np.savez(self.export_path+"robust_ferrari_canny_"+count_string,self.metric_arr)
		np.savez(self.export_path+"files_"+count_string,self.file_arr)

		self.metric_arr = []
		self.image_arr = []
		self.pose_arr = []
		self.file_arr = []

		if self.noise:
			np.savez(self.export_path+"noise_and_tilting_"+count_string,self.noise_arr)
			self.noise_arr = []
		if self.depth:
			np.savez(self.export_path+"depth_info_"+count_string,self.depth_arr)
			self.depth_arr = []
		return None

	def _skip_grasp(self,robustness):
		if len(self.metric_arr) >= self.images_per_file*self.ratio_pos:
			if robustness >= 0.002:
				return True
		else:
			if robustness < 0.002:
				return True
		return False

	def _get_artificial_depth(self,depth_table,value):
		table = depth_table.max()
		maximum = depth_table.min()
		depth = table - (table-maximum)*value
		return depth

	def _read_csv_file(self):
		filenumber = []
		array = []
		print("Possible files: ")
		[print(name) for name in os.listdir(self.csv_dir)]
		filename = input("Choose the csv file: ")
		if '.csv' in filename:
			path = self.csv_dir+filename
		else:
			path = self.csv_dir+filename+'.csv'
		if not os.path.isfile(path):
			raise ValueError(path+ " is not a csv file. Check path")
		with open(path,newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter = ',')
			for row in reader:
				row = [int(string) for string in row]
				if row[0] == row[2]:
					images = 1+row[3]-row[1]
					filenumber.extend([row[0]]*images)
					array.extend(list(range(row[1],row[3]+1)))
				else:
					for sep_file in list(range(row[0],row[2])):
						if sep_file == row[0]:
							images = 1000-row[1]
							filenumber.extend([row[0]]*images)
							array.extend(list(range(row[1],1000)))
						elif sep_file == row[2]:
							images = row[3]
							filenumber.extend([row[3]]*images)
							array.extend(list(range(0,row[3])))
						else:
							images = 1000
							filenumber.extend([sep_file]*images)
							array.extend(list(range(0,1000)))
		print("Read in all the images")
		self.csv_object = filename.split('_')[-1].split('.')[0]
		return filenumber, array

	def modify_noise(self):
		self.noise = True
		self.export_path += 'Noise/'
		self._choose_file()
		return None

	def modify_depth(self):
		self.depth = True
		self.export_path += 'Depth/'
		self._choose_file()
		return None

	def no_modification(self):
		self.export_path += '/'
		self._choose_file()
		return None

	def _choose_file(self):
		if self.csv:
			tensors,arrays = self._read_csv_file()
			self.export_path = self.export_path[0:-1]+'_'+self.csv_object+'/'
		if not os.path.exists(self.export_path):
			os.mkdir(self.export_path)
		print("Save files to ",self.export_path)
		while True:
			if self.csv:
				print(len(tensors)," images for saving")
				for cnt, tensor in enumerate(tensors):
					# open and save each image
					array = arrays[cnt]
					self._add_modification(tensor,array)
				self._save_files(self.counter)
				print("Saved final file")
				return None
			if self.manual:
				tensor = int(input("Input the file number: "))
				array = int(input("Input the array position: "))
			if self.random:
				if 'dexnet' in self.data_path:
					array = np.random.randint(low=0,high=999)
					tensor = np.random.randint(low=0,high=6728)
				elif 'Cornell' in self.data_path:
					array = np.random.randint(low=0,high=499)
					tensor = np.random.randint(low=0,high=15)
				else:
					raise ValueError("Neither Cornell, nor dexnet in pathway. Abort")
					break
				if self.filter_training and tensor*1000+array in self.split:
					continue
			filenumber = ("{0:05d}").format(tensor)
			metrics = np.load(self.data_path+"robust_ferrari_canny_"+filenumber+".npz")['arr_0'][array]
			if self.random and self._skip_grasp(metrics):
				continue
			# Store grasp, add modifications
			self._add_modification(tensor,array)
			if self.manual and input("Save file?")=='y':
				self._save_files(self.counter)
				return None	
			if self.random and self.num_images <= self.counter*self.images_per_file:
				if len(self.metric_arr) > 0:
					self._save_files(self.counter)
					print("Saved final file")
				return None

	def _add_modification(self,tensor,array):
		filenumber = ("{0:05d}").format(tensor)	
		depth_ims = np.load(self.data_path+"depth_ims_tf_table_"+filenumber+".npz")['arr_0'][array]
		pose = np.load(self.data_path+"hand_poses_"+filenumber+".npz")['arr_0'][array]
		metrics = np.load(self.data_path+"robust_ferrari_canny_"+filenumber+".npz")['arr_0'][array]
		files = [tensor,array]
		if self.noise:
			for std in [0,0.0011]:
				self.image_arr.append(np.random.normal(scale=std,size=(32,32,1))+depth_ims)
				self.noise_arr.append([std,0])
				self.pose_arr.append(pose)
				self.metric_arr.append(metrics)
				self.file_arr.append(files)
		elif self.depth:
			self.image_arr.append(depth_ims)
			self.metric_arr.append(metrics)
			self.file_arr.append(files)
			self.pose_arr.append(pose.copy())
			self.depth_arr.append(-1)
			for relation in [0,0.5,1.0]:
				relative_depth = self._get_artificial_depth(depth_ims,relation)
				pose[2] = relative_depth	
				self.image_arr.append(depth_ims)
				self.metric_arr.append(metrics)
				self.file_arr.append(files)
				self.pose_arr.append(pose.copy())
				self.depth_arr.append(relation)
		else:
			self.image_arr.append(depth_ims)
			self.pose_arr.append(pose)
			self.metric_arr.append(metrics)
			self.file_arr.append(files)

		if len(self.metric_arr) >= self.images_per_file:
			self._save_files(self.counter)
			print("Saved file #",self.counter)
			self.counter+=1
		return None 

if __name__ == "__main__":
	parser= argparse.ArgumentParser(description="Add artificial depth or noise to DexNet datasets")
	parser.add_argument("--noise",
				type = bool,
				default = False,
				help = "Add noise to the images")
	parser.add_argument("--depth",
				type = bool,
				default = False,
				help = "Add artificial depth to the images")
	parser.add_argument("--selection",
				type = str,
				default = 'random',
				help = "Selection process. 'random', 'manual' or 'csv' possible.")
	parser.add_argument("--Cornell",
				type = bool,
				default = False,
				help = "Take Cornell dataset. Default is DexNet-2.0")
	args = parser.parse_args()
	selection = args.selection
	cornell = args.Cornell
	modifier = Modification(selection,cornell)
	if args.noise:
		modifier.modify_noise()
	elif args.depth:
		modifier.modify_depth()
	else:
		modifier.no_modification()


