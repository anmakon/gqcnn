import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import csv

# Script to show images in the DexNet dataset.

class Show_DexNet_Data():
	def __init__(self,cornell):
		self.array_pointer = 0
		self.filenum = 0
		self.csv_files = []
	
		if cornell:
			self.data_path = "./data/training/Cornell/tensors/"
			self.savepath = './data/training/csv_files/Cornell_'
		else:
			self.data_path = "./data/training/dexnet_2_tensor/tensors/"
			self.savepath = './data/training/csv_files/DexNet_'
		
		self.saving_option = False
		if self.saving_option:
			self.savepath = "./analysis/csv_files/"

	def _scale(self,X,x_min=0,x_max=255):
		X_flattend = X.flatten()
		_scaled = np.interp(X_flattend,(0.6,0.75),(x_min,x_max)) # X_flattend.min() X_flattend.max()
		integ = _scaled.astype(np.uint8())
		integ.resize((32,32))	
		return integ

	def _browse_object_for_label(self,lab,obj_id):
		while True:
			if lab == self._get_label():
				return True
			else:
				self._increment()
				if obj_id != self._get_obj_id():
					return False # Return False if no further grasp of object with desired label.

	def _increment(self):
		if self.array_pointer == 999:
			self.array_pointer =0
			self.filenum += 1
		else:
			self.array_pointer += 1
		return None

	def _get_label(self):
		robust = np.load(self.data_path+'robust_ferrari_canny_'+self._gen_filepointer()+'.npz')['arr_0'][self.array_pointer]
		if robust >= 0.002:
			return 1	
		else:
			return 0

	def _get_obj_id(self):
		return np.load(self.data_path+'object_labels_'+self._gen_filepointer()+'.npz')['arr_0'][self.array_pointer]

	def _get_img_id(self):
		return np.load(self.data_path+'image_labels_'+self._gen_filepointer()+'.npz')['arr_0'][self.array_pointer]

	def create_csvs_grasps(self):
		x = 'n'
		while x == 'n':
			self.filenum = int(input("\nInput the start filenumber: "))
			self.array_pointer= int(input("Input the start array pointer: "))
			self._visualise()
			x = input("Correct? Click n for choosing again: ")
		print("Browsing for label.")
		label = int(input("1 for positive grasp, 0 for negative grasp:"))
		obj = self._get_obj_id()
		x = 'n'
		while x=='n':
			same_obj = self._browse_object_for_label(label,obj)
			if same_obj == False:
				print("Skip image")
				x = 'x'
			else:
				print("\nFilenumber:",self._gen_filepointer())
				print("Array: ",self.array_pointer)
				print("Label: ",self._get_label())
				print("Object ID: ",self._get_obj_id())
				self._visualise()
				x = input("Good? Click n for next image, s for saving, y for continuing, x to restart: ")
			if x == 'n':
				self._increment()
		if x == 'y':
			self.csv_files.append([self.filenum,self.array_pointer])
			print(self.csv_files)
		if x == 's':
			self.csv_files.append([self.filenum,self.array_pointer])
			self._save_csv_files()
			return None
		self.create_csvs_grasps()

	def main(self):
		while True:
			x = input("Press o for next object, i for next image, f for inputting file and c for closing: ")	
		#	input("Continue")
			if x == 'o':
				self._get_next_object(self.get_obj_id)
			elif x== 'i':
				self._get_next_image(self.get_img_id)
			elif x== 'c':
				break
			elif x=='f':	
				self.filenum = int(input("Input the start filenumber: "))
				self.array_pointer= int(input("Input the start array pointer: "))
			print("Object label is ", self._get_obj_id)
			self._visualise()
		return None

	def _get_next_object(self,prev_object_label):
		object_labels = np.load(self.data_path+'object_labels_'+self._gen_filepointer()+'.npz')['arr_0']
		if len(np.argwhere(object_labels > prev_object_label))>0:
			self.array_pointer = np.where(object_labels > prev_object_label)[0][0]
		else:
			self.filenum += 1
			self.get_next_object(prev_object_label)
		return None

	def _get_next_image(self,prev_image_label):
		image_labels = np.load(self.data_path+'image_labels_'+self._gen_filepointer()+'.npz')['arr_0']
		if image_labels[-1]==prev_image_label:
			self.filenum += 1
			self.array_pointer = 0
			self.get_next_image(prev_image_label)
		else:
			self.array_pointer = np.where(image_labels >= prev_image_label+1)[0][0]
		return None

	def _gen_filepointer(self):
		filepointer = '{:05d}'.format(self.filenum)
		return filepointer

	def _visualise(self):
		print("File number:",self._gen_filepointer())
		print("Array position:",self.array_pointer)

		depth_im_table = np.load(self.data_path+'depth_ims_tf_table_'+self._gen_filepointer()+'.npz')['arr_0'][self.array_pointer]
		depth_array = self._scale(depth_im_table[:,:,0],0,255)
		depth_im = Image.fromarray(depth_array,mode='L').resize((300,300))
		depth_im.show()
		if self.saving_option:
			saving = input("Save this image? Press y for saving, any other key for not saving: ")
			if saving == 'y' or saving == 'yes':
				self._save_depth_hist(depth_im_table,depth_im)
		return None

	def _save_csv_files(self):
		filestring = input("Save csv file as: ")
		with open(self.savepath+filestring+".csv",'w',newline='') as csvfile:
			writer = csv.writer(csvfile,delimiter= ',')
			for row in self.csv_files:
				writer.writerow(row)
		return None

	def _save_depth_hist(self,depths,depth_im):
		object_des = input("What type of object is this? ")
		with open(self.savepath+'DexNet_'+object_des+'_depth_'+self._gen_filepointer()+'_'+str(array_pointer)+'.csv','w',newline='') as csvfile:
			writer = csv.writer(csvfile,delimiter = ',')
			writer.writerow(['Depth_ims_tf_table_'+self._gen_filepointer()+'.npz'])
			writer.writerow(['Array_pointer: '+str(self.array_pointer)])
			writer.writerow(['Object type: '+object_des])
			table = np.squeeze(depths)
			for row in table:
				writer.writerow(row)
		depth_im.save(self.savepath+'DexNet_'+object_des+'_'+self._gen_filepointer()+'_'+str(self.array_pointer)+'.png')
		return None
	

if __name__=="__main__":
	parser = argparse.ArgumentParser(description="Show dataset images")
	parser.add_argument("dset",
				type=str,
				help="dataset to show images from")
	parser.add_argument("--browse",
				type = bool,
				default = False,
				help = "Bool to browse a dataset for labels")	
	args = parser.parse_args()
	if args.dset == 'cornell' or args.dset == 'Cornell':
		cornell = True #Cornell dataset chosen
	else:
		cornell = False #DexNet dataset chosen
	Presenter = Show_DexNet_Data(cornell)
	if args.browse:
		Presenter.create_csvs_grasps()
	else:
		Presenter.main()
