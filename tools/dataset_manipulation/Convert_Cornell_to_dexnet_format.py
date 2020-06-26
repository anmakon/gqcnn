#!/usr/bin/python

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os

class Conversion:
	def __init__(self):
		self.testing = False 
		self.export_path = "./data/training/Cornell/tensors/"
		self.data_path = "./data/training/Cornell/original/"
		if self.testing == True:
			self.export_path = "/home/annako/Desktop/Cornell_presentation/"
		if not os.path.exists(os.path.abspath(self.export_path)):
			os.mkdir(self.export_path)
		self.crop_size = 150
		self.camera_height = 0.70 # Artificial camera height in Cornell

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
		self.label_mapping = None # Array for constructing object_labels 

	def start_converting(self):
		self.label_mapping = pd.read_csv("z.txt",sep=" ", header=None,usecols=[i for i in range(2)]).drop_duplicates().to_numpy()
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
	
	def _read_image(self,filename):
		# read in picture, conversion 1D->2D according to Cornell website.
		point_cloud = pd.read_csv(filename,sep=" ",header=None, skiprows=10).to_numpy()
#		depth_im = np.full((480,640),self.camera_height)
		depth_im = np.zeros((480,640))
		for point in point_cloud:
			index = point[4]
			row = int(index//640) # Given by ReadMe of Cornell data!
			col = int(index%640)
			# Cornell - coordinates given in [mm] and from base cs of robot
			# Dexnet - coordinates given in [m] and from camera cs
			# --> conversion = x-Cornell/1000
			depth_im[row][col] = self.camera_height-(point[2]/1000)
#			depth_im[row][col] = point[2]
#			depth_im[row][col] = 0.7 - point[2]/1000
		im = Image.fromarray(depth_im)
		return im

	def _open_grasps(self,im,num,mode):
		try:
			if mode == 'pos':
				grasps = pd.read_csv(self.data_path+"pcd"+num+"cpos.txt",sep=" ",header=None).to_numpy()
				self.positive_grasps += len(grasps)/4
				robust = 0.003
			elif mode == 'neg':
				grasps = pd.read_csv(self.data_path+"pcd"+num+"cneg.txt",sep=" ",header=None).to_numpy()
				self.negative_grasps += len(grasps)/4
				robust = 0.001
		except:
			filename = "pcd"+num+"c"+mode+".txt"
			print("Couldn't open file:",filename)
			return None
		obj_index = np.where(self.label_mapping[:,0] == int(num))
		object_id = self.label_mapping[obj_index[0][0]][1]
		self._calc_grasps(im,grasps,robust,object_id)
		return None		

	def _calc_grasps(self,im,grasps,robustness,object_id):
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

			# Append lists for saving.
			grasp_depth, im, new_gripper_width= self._crop_and_safe(im,crop_area,angle,x,y,gripper_width) 
			self.rob.append(robustness)
			self.hand_poses.append([x_cent,y_cent,grasp_depth,angle,x_cent,y_cent,new_gripper_width])
			self.object_labels.append(object_id)
			self.pose_labels.append(self.pose)
			self.image_labels.append(self.grasp_counter)
			self.grasp_counter +=1
			if self.grasp_counter%100==0:
				print('Grasp #:',self.grasp_counter)

			# Export data to .npz files
			if self.grasp_counter%500==0:
				self._export_data()
				self.export_counter+=1	
		if self.testing:
			return im
		return None 

	def _calc_gripper_width(self,x_dist,y_dist):
#		if x_dist == 0:
#			gripper_width = int(np.abs(y_dist)/5) # 1/5 because of resizing from 300x300 --> 60x60
#		elif y_dist == 0:
#			gripper_width = int(np.abs(x_dist)/5)
#		else:
#			gripper_width = int(np.sqrt(x_dist**2+y_dist**2)/5)
		
		# Width adjustment

		if x_dist == 0:
			gripper_width =np.abs(y_dist)
		elif y_dist == 0:
			gripper_width = np.abs(x_dist)
		else:
			gripper_width = np.sqrt(x_dist**2+y_dist**2)
		

		return gripper_width

	def _crop_and_safe(self,im,crop_area,angle,x,y,gripper_width):
		scale_factor = 13/gripper_width
		res_im = im.crop(crop_area).rotate(angle+180).resize((int(300*scale_factor),int(300*scale_factor)))
		width,height = res_im.size
		res_im = res_im.crop((width/2-16,height/2-16,width/2+16,width/2+16))

#		bin_im = rgb_im.crop(crop_area).rotate(angle+180).resize((60,60)).crop((14,14,46,46)).convert(mode="1")
		if self.testing == True:
			cropped = res_im.resize((200,200))
			im = self._visualize_grasp(im.convert('RGB'),x,y,cropped)
		depth_im_tf = [[[point] for point in row] for row in np.asarray(res_im)]
		self.depth_ims_tf.append(depth_im_tf)
		grasp_depth = 0.4*np.max(depth_im_tf)+0.6*np.min(depth_im_tf)
		return grasp_depth, im, 13

	def _visualize_image(self):
		self.testing = True
		self.label_mapping = pd.read_csv(self.data_path+"z.txt",sep=" ", header=None,usecols=[i for i in range(2)]).drop_duplicates().to_numpy()
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
		cropped.show()
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
	convert = Conversion()
#	convert._visualize_image()
	convert.start_converting()
