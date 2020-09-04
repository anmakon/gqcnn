import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv

# Show images of Cornell dataset. Has the option to export 
# csv_files with pointers of objects in Cornell.

class Show_Images():

	def __init__(self):
		self.data_path = "./data/training/Cornell/tensors/"
		self.create_object_csv = True # If true, object numbers can be searched in the dataset and a csv_file with pointers can be exported. If False, images can be visualised. 
		self.obj_type = None
		self.obj_num = None
		self.tensor = None
		self.main()

	def _scale(self, X, x_min=0.55, x_max=0.75):
		X_flattend = X.flatten()
		scaled = np.interp(X_flattend,(x_min,x_max),(0,255)) # X_flattend.min(),X_flattend.max()
		integ = scaled.astype(np.uint8)
		integ.resize((32,32))
		return integ

	def main(self):
		match = []
		cnt = 0
		label_mapping = pd.read_csv(self.data_path+"../original/z.txt",sep=" ", header=None,usecols=[0,1,2]).drop_duplicates().to_numpy()
		object_labels = np.array(self._load_object_labels()) 
		fileindex_saving = []
		x = 'o'
		while True:
			if x == 'o':
				print()
				self.obj_num = int(input("Input the object number: "))
				obj_name = label_mapping[np.where(label_mapping[:,1] == self.obj_num),2][0]
				match = np.where(self.obj_num == object_labels)
				print(len(match[0]),"images of: ",obj_name[0])
				cnt = 0
			elif x== 'i' and not self.create_object_csv:
				if cnt >= len(match[0]):
					print("Viewed all images of that object id. Restart at first image.")
					cnt =0
				else:
					cnt += 1
			elif x== 'c' and not self.create_object_csv:
				break
			elif x== 's' and self.create_object_csv:
				self._save_fileindex_csv(fileindex_saving)
				return None
			if not self.create_object_csv:
				filepointer,array_pointer = self._get_fileindex(match,cnt)
				self.obj_type = self.obj_name[0]
				self._visualise(filepointer,array_pointer)
				x = input("Press o for next object, i for next image and c for closing: ")	
			else:
				first_file,first_array = self._get_fileindex(match,0)
				current_file, current_array = self._get_fileindex(match,0)
				prev_match = match[0][0]
				for current_match in match[0][1:]:
					if current_match == prev_match+1:
						current_file, current_array = self._get_fileindex(current_match)
					else:
						fileindex_saving.append([first_file,first_array,current_file,current_array])
						first_file,first_array = self._get_fileindex(current_match)
					prev_match = current_match
				fileindex_saving.append([first_file,first_array,current_file,current_array])
				x = input("Press o for next object, s for saving: ")
		return None

	def _save_fileindex_csv(self, content):
		pathname = input("Name for the csv file: ")
		savepath = "./data/training/csv_files/"+pathname+".csv"
		with open(savepath,'w',newline='') as csvfile:
			writer = csv.writer(csvfile,delimiter = ',')
			for row in content:
				writer.writerow(row)
		return None

	def _get_fileindex(self,match,cnt=None):
		if cnt is None:
			position=match
		else:
			position = match[0][cnt]
		self.tensor = position // 500
		array_pointer = position % 500
		filepointer = self._gen_filepointer()
		return filepointer, array_pointer

	def _load_object_labels(self):
		object_labels = []
		for counter in range(0,17):
			self.tensor = counter
			filepointer = self._gen_filepointer()
			labels = np.load(self.data_path+'object_labels_'+filepointer+'.npz')['arr_0']
			[object_labels.append(label) for label in labels]
		return object_labels

	def _gen_filepointer(self):
		filepointer = '{:05d}'.format(self.tensor)
		return filepointer

	def _visualise(self,filepointer,array_pointer):
		print("File number:",filepointer)
		print("Array position:",array_pointer)
		depth_im_table = np.load(self.data_path+'depth_ims_tf_table_'+filepointer+'.npz')['arr_0'][array_pointer]
		org_depth_im_table = np.load(self.data_path+'orig_depth_ims_tf_table_'+filepointer+'.npz')['arr_0'][array_pointer]
		depth_array = self._scale(depth_im_table[:,:,0])
		org_depth_array = self._scale(org_depth_im_table[:,:,0],x_min=0,x_max=150)
		depth_im = Image.fromarray(depth_array).resize((300,300))
		depth_im.show()
		org_depth_im = Image.fromarray(org_depth_array).resize((300,300))
		org.depth_im.show()
		return None
		
	
if __name__=="__main__":
	Show_Images()
