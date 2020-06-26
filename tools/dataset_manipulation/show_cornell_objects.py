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
		self.testing = True # If true, object numbers can be searched in the dataset and a csv_file with pointers can be exported. For creating sub-datasets!
		self.obj_type = None
		self.obj_num = None
		self.tensor = None
		self.main()

	def scale(self, X, x_min=0.55, x_max=0.75):
		X_flattend = X.flatten()
		scaled = np.interp(X_flattend,(X_flattend.min(),X_flattend.max()),(0,255)) # X_flattend.min(),X_flattend.max()
		integ = scaled.astype(np.uint8)
		integ.resize((32,32))
		return integ

	def main(self):
		match = []
		cnt = 0
		label_mapping = pd.read_csv(self.data_path+"z.txt",sep=" ", header=None,usecols=[i for i in [0,1,2]]).drop_duplicates().to_numpy()
		object_labels = np.array(self.load_object_labels()) 
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
			elif x== 'i' and not self.testing:
				if cnt >= len(match[0]):
					print("Viewed all images of that object id. Restart at first image.")
					cnt =0
				else:
					cnt += 1
			elif x== 'c' and not self.testing:
				break
			elif x== 's' and self.testing:
				self.save_fileindex_csv(fileindex_saving)
				return None
			if not self.testing:
				filepointer,array_pointer = self.get_fileindex(match,cnt)
				self.obj_type = self.obj_name[0]
				self.visualize(filepointer,array_pointer)
				x = input("Press o for next object, i for next image and c for closing: ")	
			else:
				first_file,first_array = self.get_fileindex(match,0)
				for counter, _ in enumerate(match[0]):
					last_file, last_array = self.get_fileindex(match,counter)
					if last_file != first_file and last_array != first_array+1:
						fileindex_saving.append([first_file,first_array,previous_file,previous_array])
						first_file,first_array = last_file,last_array
					previous_file, previous_array = self.get_fileindex(match,counter)
				fileindex_saving.append([first_file,first_array,last_file,last_array])
				x = input("Press o for next object, s for saving: ")
		return None

	def save_fileindex_csv(self, content):
		pathname = input("Name for the csv file: ")
		savepath = "/home/annako/Documents/gqcnn/data/training/Subset_datasets/csv_files/"+pathname+".csv"
		with open(savepath,'w',newline='') as csvfile:
			writer = csv.writer(csvfile,delimiter = ',')
			for row in content:
				writer.writerow(row)
		return None

	def get_fileindex(self,match,cnt):
		position = match[0][cnt]
		self.tensor = position // 500
		array_pointer = position % 500
		filepointer = self.gen_filepointer()
		return filepointer, array_pointer

	def load_object_labels(self):
		object_labels = []
		for counter in range(0,17):
			self.tensor = counter
			filepointer = self.gen_filepointer()
			labels = np.load(self.data_path+'object_labels_'+filepointer+'.npz')['arr_0']
			[object_labels.append(label) for label in labels]
		return object_labels

	def gen_filepointer(self):
		filepointer = '{:05d}'.format(self.tensor)
		return filepointer

	def visualize(self,filepointer,array_pointer):
		print("File number:",filepointer)
		print("Array position:",array_pointer)
		depth_im_table = np.load(self.data_path+'depth_ims_tf_table_'+filepointer+'.npz')['arr_0'][array_pointer]
		org_depth_im_table = np.load(self.data_path+'orig_depth_ims_tf_table_'+filepointer+'.npz')['arr_0'][array_pointer]
		depth_array = self.scale(depth_im_table[:,:,0])
		org_depth_array = self.scale(org_depth_im_table[:,:,0],x_min=0,x_max=150)
		depth_im = Image.fromarray(depth_array).resize((300,300))
		depth_im.show()
		org_depth_im = Image.fromarray(org_depth_array).resize((300,300))
	#	saving = input("Press y to save the image: ")
		saving = 'n'
		if saving =='y' or saving =='yes':
			savepath= '../../Desktop/Depths/CSV_files/Cornell_'
			with open(savepath+'Adjusted_'+str(self.obj_type)+'_depth_'+str(self.object_num)+'.csv','w',newline='') as csvfile:
				writer = csv.writer(csvfile,delimiter = ',')
				writer.writerow(['Depth_ims_tf_table_'+filepointer+'.npz'])
				writer.writerow(['Array_pointer: '+str(array_pointer)])
				writer.writerow(['Object type: ' +str(self.obj_type)])
				table = np.squeeze(depth_im_table)
				for row in table:
					writer.writerow(row)
			with open(savepath+'Original_'+str(self.obj_type)+'_depth_'+str(self.object_num)+'.csv','w',newline='') as csvfile:
				writer = csv.writer(csvfile,delimiter = ',')
				writer.writerow(['Depth_ims_tf_table_'+filepointer+'.npz'])
				writer.writerow(['Array_pointer: '+str(array_pointer)])
				writer.writerow(['Object type: ' +str(self.obj_type)])
				org_table = np.squeeze(org_depth_im_table)
				for row in org_table:
					writer.writerow(row)
			savepath= '../../Desktop/Depths/Cornell_'
			binwidth = 0.0001
			data = np.resize(depth_im_table,(1024,))
			org_data = np.resize(org_depth_im_table,(1024,))
			plt.hist(data,bins=np.arange(min(data),max(data)+binwidth,binwidth))
			plt.xlabel('Depth [m]')
			plt.ylim((0,35))
			Grasp_depth = depth_im_table[16,16]
			plt.title('Depths in Cornell point cloud. ID: '+str(object_num)+'. '+str(self.obj_type))
			plt.text(Grasp_depth,5,'Grasp-depth')
			plt.savefig(savepath+'Adjusted_'+str(self.obj_type)+'_histogram_'+str(object_num)+'.png')
			plt.close()
	
			binwidth = 0.1
			plt.hist(org_data,bins=np.arange(min(org_data),max(org_data)+binwidth,binwidth))
			plt.xlabel('Depth [mm]')
			plt.ylim((0,35))
			Grasp_depth = org_depth_im_table[16,16]
			plt.title('Depths in Cornell point cloud. ID: '+str(object_num)+'. '+str(self.obj_type))
			plt.text(Grasp_depth,5,'Grasp-depth')
			plt.savefig(savepath+'Original_'+str(self.obj_type)+'_histogram_'+str(object_num)+'.png')
			plt.close()
			depth_im.save(savepath+'Adjusted_'+str(self.obj_type)+'_'+str(object_num)+'.png')
			org_depth_im.save(savepath+'Original_'+str(self.obj_type)+'_'+str(object_num)+'.png')
		return None
		
	
if __name__=="__main__":
	Show_Images()
