import numpy as np
import csv
import os
import argparse
from PIL import Image

# Script to add grasp perturbation to images. Selection mode can be chosen as manual,
# random, random with filtering the training grasps and from csv_file.
# For random selction, the ratio of positive and negative grasps can be adjusted.

class Grasp_perturbation():
	def __init__(self,dset,selection,tensor=None,array=None,perturb=1):
		self.image_arr = []
		self.pose_arr = []
		self.file_arr = []
		self.metric_arr = []
		self.perturb_arr = []

		if dset == 'dexnet_2_tensor':
			out = 'DexNet'
		else:
			out = 'Cornell'

		self.images_per_file = 500
		self.num_grasps = 1000
		self.ratio_pos = 1
		
		self.perturbation = perturb  # 0 - rotation; 1 - translation x; 2 - translation y
		if self.perturbation == 0 :
			mode = 'Rotation'
		elif self.perturbation == 1:
			mode = 'Translation'
		elif self.perturbation == 2:
			mode = 'Translationy'
		self.perturb_step = 1 
		if self.perturbation == 1 or self.perturbation == 2:
			self.steps = 10
		elif self.perturbation == 0:
			self.steps = 60

		self.output_path = "./data/training/Subset_datasets/"+out+"_"+mode+"Perturb/"
		self.data_path = "./data/training/"+dset+"/tensors/"		
		split = "./data/training/"+dset+"/splits/image_wise/train_indices.npz"
		self.split = np.load(split)['arr_0']
		
		self.filter_training = False 

		self.manual_input = False 
		self.random = False
		self.csv_input = False

		if tensor is not None and array is not None:
			self.manual_input= True
			self.output_path = "./data/training/Subset_datasets/"+out+"_SinglePerturb/"
			self.main(tensor,array)
			return None
		elif selection == 'random':
			self.random = True
		elif selection == 'manual':
			self.manual_input = True
		elif selection == 'csv':
			self.csv_input = True
		self.main()

	def save_rest(self,pose,metric,files):
		self.pose_arr.append(pose)
		self.metric_arr.append(metric)
		self.file_arr.append(files)
		return None

	def save_files(self,counter):
		if not os.path.exists(self.output_path):
			os.mkdir(self.output_path)
		count_string =  ("{0:05d}").format(counter)
		np.savez(self.output_path+"depth_ims_tf_table_"+count_string,self.image_arr)
		np.savez(self.output_path+"hand_poses_"+count_string,self.pose_arr)
		np.savez(self.output_path+"robust_ferrari_canny_"+count_string,self.metric_arr)
		np.savez(self.output_path+"grasp_perturbations_"+count_string,self.perturb_arr)
		np.savez(self.output_path+"files_"+count_string,self.file_arr)
		self.image_arr = []
		self.pose_arr = []
		self.metric_arr = []
		self.perturb_arr = []
		self.file_arr = []
		print("Saved file ",count_string)
		return None

	def skip_image(self,robustness):
		if len(self.metric_arr) >= self.images_per_file*self.ratio_pos:
			if robustness >= 0.002:
				return True
		else:
			if robustness < 0.002:
				return True
		return False

	def list_of_images(self):
		filenumber = []
		array = []
		path = input("Give the path of the input file")
		# Adjust the path names for saving for the objects.
		split1 = path.split(separator="/")
		csv_object = split1[-1].split(separator="_")[-1][0:-4]
		save_path = self.output_path.split(separator="_")
		self.output_path = save_path[0:-2]+csv_object+save_path[-1]
		# Read the csv file
		with open(path+'.csv',newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter = ',')
			for row in reader:
				row = [int(string) for string in row]
				if row[0] == row[2]:
					images = row[3]-row[1]
					filenumber.extend([row[0]]*images)
					array.extend(list(range(row[1],row[3])))
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
		return filenumber, array

	def _add_rotation(self,depth_tf,deg):
		table = np.amax(depth_tf[:,:,0])
		im = Image.fromarray(depth_tf[:,:,0]).resize((200,200))
		new_im = im.rotate(deg,fillcolor=table).resize((32,32))
		depth_tf_rot = [[[point] for point in row] for row in np.asarray(new_im)]
		return depth_tf_rot

	def _add_translation(self,depth_tf,trans,y=False):
		table = np.amax(depth_tf[:,:,0])
		im = Image.fromarray(depth_tf[:,:,0]).resize((200,200))
		if not y:
			new_im = im.transform(im.size,method = Image.AFFINE,data=(1,0,trans,0,1,0),fillcolor=table).resize((32,32))
		else:
			new_im = im.transform(im.size,method = Image.AFFINE,data=(1,0,0,0,1,trans),fillcolor=table).resize((32,32))
		depth_tf_trans = [[[point] for point in row] for row in np.asarray(new_im)]
		return depth_tf_trans

	def main(self,given_tensor=None,given_array=None):
		tensor = 0
		filenumber = ("{0:05d}").format(tensor)
		array = 0
		counter = 0
		grasp_counter = 0
		std= 0.0
		x = "n"
		while x != 's':
			if not self.csv_input:
				if self.manual_input and x == 'n':
					if given_tensor is not None and given_array is not None:
						tensor = given_tensor
						array = given_array
					else:
						tensor = int(input("Input the file number: "))
						array = int(input("Input the array position: "))
					filenumber = ("{0:05d}").format(tensor)
				if self.random:
					if 'dexnet' in self.data_path:
						array = np.random.randint(low=0,high=999)
						tensor = np.random.randint(low=0,high=6728)
					elif 'Cornell' in self.data_path:
						array = np.random.randint(low=0,high=499)
						tensor = np.random.randint(low=0,high=15)
					else:
						print("Neither Cornell, nor dexnet in pathway.")
						break
					if self.filter_training and tensor*1000+array in self.split:
						continue
					filenumber = ("{0:05d}").format(tensor)							
				metrics = np.load(self.data_path+"robust_ferrari_canny_"+filenumber+".npz")['arr_0'][array]
				if self.random and self.skip_image(metrics):
					continue
				grasp_counter += 1
				depth_ims = np.load(self.data_path+"depth_ims_tf_table_"+filenumber+".npz")['arr_0'][array]
				pose = np.load(self.data_path+"hand_poses_"+filenumber+".npz")['arr_0'][array]
				files = [tensor,array]
				for i in range(0,self.steps+1,self.perturb_step):
					var = i - self.steps/2
					if self.perturbation == 0:
						new_depth_ims = self._add_rotation(depth_ims,var)
						perturb = [var,0,0]
					elif self.perturbation == 1:
						new_depth_ims = self._add_translation(depth_ims, var)
						perturb = [0,var,0]
					elif self.perturbation == 2:
						new_depth_ims = self._add_translation(depth_ims, var,y=True)
						perturb = [0,0,var]
					self.image_arr.append(new_depth_ims)
					self.perturb_arr.append(perturb)
					self.save_rest(pose,metrics,files)
				if self.manual_input and 'Single' in self.output_path:
					x = 's'
				elif self.manual_input:
					x = input("Press n for next image, s for saving: ")
				if len(self.perturb_arr) >= self.images_per_file:
					self.save_files(counter)
					counter +=1
				if grasp_counter == self.num_grasps:
					x = 's'
			else:			
				tensors, arrays = self.list_of_images()
				print(len(tensors)," images for saving")
				for cnt, tensor in enumerate(tensors):
					# open and save each image
					filenumber = ("{0:05d}").format(tensor)
					array = arrays[cnt]
					depth_ims = np.load(self.data_path+"depth_ims_tf_table_"+filenumber+".npz")['arr_0'][array]
					pose = np.load(self.data_path+"hand_poses_"+filenumber+".npz")['arr_0'][array]
					metrics = np.load(self.data_path+"robust_ferrari_canny_"+filenumber+".npz")['arr_0'][array]
					files = [tensor,array]
					for i in range(0,self.steps+1,self.perturb_step):
						var = i - self.steps/2
						if self.perturbation == 0:
							new_depth_ims = self._add_rotation(depth_ims,var)
							perturb = [var,0,0]
						elif self.perturbation == 1:
							new_depth_ims = self._add_translation(depth_ims, var)
							perturb = [0,var,0]
						elif self.perturbation == 2:
							new_depth_ims = self._add_translation(depth_ims, var,y=True)
							perturb = [0,0,var]
						self.image_arr.append(new_depth_ims)
						self.perturb_arr.append(perturb)
						self.save_rest(pose,metrics,files)
					if cnt % self.images_per_file == self.images_per_file-1:
						self.save_files(counter)
						counter +=1
						print("Saved file #",counter)
				self.save_files(counter)
				print("Saved final file")
				return None
		self.save_files(counter)
		return None

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dset",
				type = str,
				default = 'Cornell',
				help = "Dataset to take grasp from")
	parser.add_argument("--selection",
				type = str,
				default = 'manual',
				help="How are the grasps selected?")
	parser.add_argument("--file",
				type=int,
				default=None,
				help="file number of single grasp")
	parser.add_argument("--array",
				type=int,
				default=None,
				help="array number of single grasp")
	parser.add_argument("--type",
				type = str,
				default = 'translation',
				help ="Translation or rotation or translationy as perturbation")
	args = parser.parse_args()
	selection = args.selection
	tensor = args.file
	array = args.array
	dset = args.dset
	perturb = args.type
	if perturb == 'translation' or perturb == 'Translation':
		direction = 1
	elif perturb == 'rotation' or perturb == 'Rotation':
		direction = 0
	elif perturb == 'translationy' or perturb == 'Translationy' or perturb == 'translation y':
		direction = 2
	if dset == 'DexNet' or dset == 'dexnet' or dset == 'Dexnet':
		dset = 'dexnet_2_tensor'
	if tensor is not None and array is not None:
		Grasp_perturbation(dset,selection,tensor,array,direction)
	else:
		Grasp_perturbation(dset,selection)
