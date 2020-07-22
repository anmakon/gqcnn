#!/usr/bin/python

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import csv
import os
import argparse

class Conversion:
	""" Class to convert data from the Cornell format (rgb images, point clouds and positive/negative
	grasping rectangulars) to the DexNet format (32x32x1 depth images, hand poses and the robust epsilon metric).
	"""


	def __init__(self,export_path):
		self.visual_mode = False #To visualise the images/grasping rectangulars
		self.single_image_mode = False
		self.adjust_width = False
		self.creating_original = False
		self.camera_variation = True
		self.export_path = export_path 

		if not os.path.exists(os.path.abspath(self.export_path)):
			os.mkdir(self.export_path)
		for single_file in os.listdir(self.export_path): #Clean export directory from data.
			if ".npz" in single_file:
				os.remove(self.export_path+single_file)

		self.data_path = "./data/training/Cornell/original/"

		self.crop_size = 150
		resize_factor = 3 #Resizing factor 3 in DexNet real experiments
		self.resize_size = int(self.crop_size*2/resize_factor)
		if self.camera_variation is False:
			self.camera_height = 0.70 # Artificial camera height in Cornell
		else:
			self.camera_height = [0.65,0.75]

		# Variables to construct the dexnet dataset

		self.grasp_counter = 0 # Counter for dexnet dataset reconstruction (image_label)
		self.hand_poses = [] # Array for dexnet dataset reconstruction (hand_pose) 
		self.depth_ims_tf = [] # Array for dexnet dataset reconstruction (depth_ims_tf)
		self.rob = [] # Array for dexnet dataset reconstruction (robust_ferrari_canny)
		self.export_counter = 0 # Counter for dexnet dataset reconstruction (00000,00001,00002,...)
		self.pose_labels = [] # Array for dexnet dataset reconstruction (pose_label)
		self.positive_grasps = 0 # Counter for Dataset balance
		self.negative_grasps = 0 # Counter for dataset balance
		self.pose = 0 # Counter for dexnet dataset reconstruction (pose_labels)
		self.object_labels = [] # Array for dexnet dataset reconstruction (object_labels)
		self.image_labels = [] # Array for dexnet dataset reconstruction (image_labels)
		self.label_mapping = pd.read_csv(self.data_path+"z.txt",sep=" ", header=None,usecols=[i for i in range(2)]).drop_duplicates().to_numpy()

	def create_orig(self,num):
		"""
		Creating an depth image and a binary image from the pointcloud.
		Saving it for usage to plan grasps.
		"""
		self.creating_original = True
		filenum = ("{0:04d}").format(num)
		# image size: (480,640)
		crop_area = (160,120,480,360)
		# Get depth image, crop and save
		image=self._read_image('pcd'+filenum+'.txt').crop(crop_area)
		depth_im_tf = [[[point] for point in row] for row in np.asarray(image)]
		np.save(self.export_path+'depth_0.npy',depth_im_tf)
		# Get RGB image
		rgb_image = Image.open(self.data_path+'pcd'+filenum+'r.png')
		thresh = 200
		fn = lambda x: 250 if x > thresh else 0
		binary_image = rgb_image.crop(crop_area).convert('L').point(fn,mode='1')
		binary_image.save(self.export_path+'binary_0.png')


	def convert_all(self):
		# Iterate through all files in directory
		for any_file in os.listdir(self.data_path):
			if not 'cneg' in any_file and not 'cpos' in any_file and not 'z.' in any_file and any_file.endswith(".txt"):
				image = self._read_image(any_file) # Read in pointclouds
				file_num = any_file[-8:-4]
				self._open_grasps(image,file_num,'pos') # Convert & save positive grasps
				self._open_grasps(image,file_num,'neg') # Convert & save negative grasps
				self.pose += 1
		# Export the last batch of data
		if self.grasp_counter%1000 != 0:
			print("export last data")
			self._export_data()
		# Output the dataset balance
		print("Amount of positive grasps:",self.positive_grasps)
		print("Amount of negative grasps:",self.negative_grasps)
		print("Percentage of positive grasps:",self.positive_grasps/(self.positive_grasps+self.negative_grasps))

	def convert_one_image(self,num):
		self.single_image_mode = True
		filenum = ("{0:04d}").format(num)
		self.export_path += filenum+'_'
		filename = "pcd"+filenum+".txt"
		image = self._read_image(filename)
		self._open_grasps(image,filenum,'pos')
		self._open_grasps(image,filenum,'neg')
			
	def _read_image(self,filename):
		# read in picture, conversion 1D->2D according to Cornell website.
		point_cloud = pd.read_csv(self.data_path+filename,sep=" ",header=None, skiprows=10).to_numpy()
		if self.camera_variation:
			camera_height = np.random.random()*(self.camera_height[1]-self.camera_height[0])+self.camera_height[0]
		else:
			camera_height = self.camera_height
		if self.creating_original:
			depth_im = np.full((480,640),camera_height)
		else:
			depth_im = np.full((530,640),camera_height)
		for point in point_cloud:
			index = point[4]
			row = int(index//640) # Given by ReadMe of Cornell data!
			col = int(index%640)
			# Cornell - coordinates given in [mm] and from base cs of robot
			# Dexnet - coordinates given in [m] and from camera cs
			# --> conversion = x-Cornell/1000
			depth_im[row][col] = camera_height-(point[2]/1000)
			if self.visual_mode:
				depth_im[row][col] = point[2]
		im = Image.fromarray(depth_im)
		return im

	def _open_grasps(self,im,num,mode):
		try:
			if mode == 'pos':
				grasps = pd.read_csv(self.data_path+"pcd"+num+"cpos.txt",sep=" ",header=None).to_numpy()
				robust = 0.003
			elif mode == 'neg':
				grasps = pd.read_csv(self.data_path+"pcd"+num+"cneg.txt",sep=" ",header=None).to_numpy()
				robust = 0.001
		except:
			filename = "pcd"+num+"c"+mode+".txt"
			print("Couldn't open file:",filename)
			return None
		obj_index = np.where(self.label_mapping[:,0] == int(num))
		object_id = self.label_mapping[obj_index[0][0]][1]
		self._calc_grasps(im,grasps,robust,object_id,mode)
		return None		

	def _calc_grasps(self,im,grasps,robustness,object_id,mode):
		# Calculate the grasp centres and crop+rotate+resize depth image
		for i in range(0,len(grasps)//4):
			x = [coord[0] for coord in grasps[i*4:i*4+4]]
			y = [coord[1] for coord in grasps[i*4:i*4+4]]
			if np.isnan(x).any() or np.isnan(y).any():
				print("NaN in grasp. Skipping position.")
				continue
			x_dist = x[1]-x[0]
			y_dist = y[1]-y[0]
			angle = np.rad2deg(np.arctan2(y_dist,x_dist)) # angle in grasp-axis --> image axis in rad
			gripper_width = self._calc_gripper_width(x_dist,y_dist)
			x_cent = (min(x)+max(x))/2
			y_cent = (min(y)+max(y))/2
			crop_area = (x_cent-self.crop_size,y_cent-self.crop_size,x_cent+self.crop_size,y_cent+self.crop_size)
#			if (y_cent + (80/np.sin(np.deg2rad(angle)))) >= 480: #Test for removing grasps that are too close to the border of the image.
#				continue
			# Append lists for saving.
			grasp_depth, im, new_gripper_width, err= self._crop_and_safe(im.copy(),crop_area,angle,x,y,gripper_width) 
			if err:
				continue
			self.rob.append(robustness)
			self.hand_poses.append([x_cent,y_cent,grasp_depth,angle,x_cent,y_cent,new_gripper_width])
			self.object_labels.append(object_id)
			self.pose_labels.append(self.pose)
			self.image_labels.append(self.grasp_counter)
			self.grasp_counter +=1
			if mode == 'neg':
				self.negative_grasps += 1
			elif mode == 'pos':
				self.positive_grasps += 1
			if self.grasp_counter%100==0:
				print('Grasp #:',self.grasp_counter)

			# Export data to .npz files
			if self.grasp_counter%500==0:
				self._export_data()
				self.export_counter+=1	
		if self.visual_mode:
			return im
		return None 

	def _calc_gripper_width(self,x_dist,y_dist):
		if not self.adjust_width:
			factor = self.resize_size/(self.crop_size*2)
			if x_dist == 0:
				gripper_width = int(np.abs(y_dist)*factor)
			elif y_dist == 0:
				gripper_width = int(np.abs(x_dist)*factor)
			else:
				gripper_width = int(np.sqrt(x_dist**2+y_dist**2)*factor)
		else:
			# Width adjustment to have similar gripper width in all images.
			if x_dist == 0:
				gripper_width =np.abs(y_dist)
			elif y_dist == 0:
				gripper_width = np.abs(x_dist)
			else:
				gripper_width = np.sqrt(x_dist**2+y_dist**2)
		return gripper_width

	def _crop_and_safe(self,im,crop_area,angle,x,y,gripper_width):
		if self.adjust_width:
			scale_factor = 13/gripper_width
			res_im = im.crop(crop_area).rotate(angle).resize((int(300*scale_factor),int(300*scale_factor)))
			width,height = res_im.size
			res_im = res_im.crop((width/2-16,height/2-16,width/2+16,width/2+16))
			gripper_width = 13
		else:
			resizing = (self.resize_size,self.resize_size)
			final_crop = (self.resize_size/2-16,self.resize_size/2-16,self.resize_size/2+16,self.resize_size/2+16)
			res_im = im.crop(crop_area)
			if self.single_image_mode and self.grasp_counter == 0:
				self._save_single_image(np.asarray(res_im))
			res_im = res_im.rotate(angle).resize(resizing).crop(final_crop)
		zero_catcher = np.asarray(res_im)
			
		if self.visual_mode:
			cropped = res_im.resize((200,200))
			im = self._visualize_grasp(im.convert('RGB'),x,y,cropped)
		depth_im_tf = [[[point] for point in row] for row in np.asarray(res_im)]
		if self.single_image_mode:
			self._save_single_image(np.asarray(res_im))
		self.depth_ims_tf.append(depth_im_tf)
		grasp_depth = 0.4*np.max(depth_im_tf)+0.6*np.min(depth_im_tf) #ToDo - refine grasp depth. Might be an issue with the minimum due to those zeros!
		return grasp_depth, im, gripper_width, False

	def _save_single_image(self,depth):
		add = ''
		for sublist in depth:
			if add == '' and any(point < 0.1 for point in sublist):
				add = 'small'
				print("found zeros")
		if add =='':
			return None
		with open(self.export_path+("{0:03d}").format(self.export_counter)+add+".csv",'w',newline='') as csvfile:
			writer = csv.writer(csvfile,delimiter=',')
			for row in depth:
				writer.writerow(row)
		self.export_counter += 1
		return None

	def visualize_image(self,nbr):
		self.visual_mode = True
		if nbr is None:
			nbr = int(input("Input the file number: "))
		num = '{:04d}'.format(nbr)
		obj_index = np.where(self.label_mapping[:,0] == int(num))
		object_id = self.label_mapping[obj_index[0][0]][1]
		test_file = "pcd"+num+".txt"
		image = self._read_image(test_file)

		grasps = pd.read_csv(self.data_path+"pcd"+num+"cpos.txt",sep=" ",header=None).to_numpy()
		self.color = "green"
		im = self._calc_grasps(image,grasps,0.003,object_id)
		im.save(self.export_path+'positive.png')

		grasps = pd.read_csv(self.data_path+"pcd"+num+"cneg.txt",sep=" ",header=None).to_numpy()
		self.color = "red"
		im = self._calc_grasps(image,grasps,0,object_id)
		im.save(self.export_path+'negative.png')
		image.convert('RGB').save(self.export_path+'original.png')
		return None

	def _visualize_grasp(self,original,x,y,cropped):
		#original.show()
		draw = ImageDraw.Draw(original)
		draw.polygon([(x[0],y[0]),(x[1],y[1]),(x[2],y[2]),(x[3],y[3])],outline=self.color)
#		original.show()
#		cropped.show()
#		input("continue")
		return original 

	def _export_data(self):
		#here we will export the data
		numstr = '{:05d}'.format(self.export_counter)
		print('Export file',numstr)
		np.savez(self.export_path+"depth_ims_tf_table_"+numstr,self.depth_ims_tf)
		self.depth_ims_tf = []
		np.savez(self.export_path+"hand_poses_"+numstr,self.hand_poses)
		self.hand_poses = []
		np.savez(self.export_path+"object_labels_"+numstr,self.object_labels)
		self.object_labels = []
		np.savez(self.export_path+"pose_labels_"+numstr,self.pose_labels)
		self.pose_labels = []
		np.savez(self.export_path+"image_labels_"+numstr,self.image_labels)
		self.image_labels = []
		np.savez(self.export_path+"robust_ferrari_canny_"+numstr,self.rob)
		self.rob = []
		return None

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=("Convert original Cornell data to DexNet 2.0 format."))
	parser.add_argument("--create_vis",
				type = bool,
				default = False,
				help = "Create a visualisation of the grasping rectangles in Cornell.")
	parser.add_argument("--Cornell_num",
				type = int,
				default = None,
				help = "Input to visualise/convert one single file.")
	parser.add_argument("--create_orig",
				type = bool,
				default = None,
				help = "Create original data of whole image to plan grasps")
	parser.add_argument("--export_path",
				type = str,
				default = None,
				help = "Path to export the data.")
	args = parser.parse_args()
	create_vis = args.create_vis
	create_orig = args.create_orig
	cornell_num = args.Cornell_num
	export_path = args.export_path

	if export_path is None and not create_vis and cornell_num is not None and create_orig is None:
		export_path = "./data/training/Subset_datasets/Cornell/"
	elif export_path is None and create_vis:
		export_path = "../../Desktop/Cornell_presentation/"
	elif export_path is None and create_orig:
		export_path =  "./data/training/Grasp_plan_data/"
	elif export_path is None:
		export_path =  "./data/training/Cornell/tensors/"

	convert = Conversion(export_path)
	if create_vis:
		convert.visualize_image(cornell_num)
	elif create_orig:
		convert.create_orig(cornell_num)
	elif cornell_num is not None:
		convert.convert_one_image(cornell_num)
	else:
		convert.convert_all()
