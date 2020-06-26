import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv

# Script to show images in the DexNet dataset.

def scale(X,x_min,x_max):
	X_flattend = X.flatten()
	scaled = np.interp(X_flattend,(X_flattend.min(),X_flattend.max()),(x_min,x_max)) # X_flattend.min() X_flattend.max()
	integ = scaled.astype(np.uint8())
	integ.resize((32,32))	
	return integ

def main():
	data_path = "./data/training/dexnet_2_tensor/tensors/"
	cnt = int(input("Input the start filenumber: "))
	array_pointer= int(input("Input the start array pointer: "))
	filepointer = gen_filepointer(cnt)
	object_label = np.load(data_path+'object_labels_'+filepointer+'.npz')['arr_0'][array_pointer]
	print("Object label is ", object_label)
	visualize(filepointer,array_pointer)
	x = input("Press o for next object, i for next image, f for inputting file and c for closing: ")	
	
	while True:
		x = input("Press o for next object, i for next image, f for inputting file and c for closing: ")	
	#	input("Continue")
		if x == 'o':
			object_label = np.load(data_path+'object_labels_'+filepointer+'.npz')['arr_0'][array_pointer]
			cnt,array_pointer = get_next_object(cnt,object_label,data_path)
		elif x== 'i':
			image_label = np.load(data_path+'image_labels_'+filepointer+'.npz')['arr_0'][array_pointer]
			cnt,array_pointer = get_next_image(cnt,image_label,data_path)
		elif x== 'c':
			break
		elif x=='f':	
			cnt = int(input("Input the start filenumber: "))
			array_pointer= int(input("Input the start array pointer: "))
			object_label = np.load(data_path+'object_labels_'+filepointer+'.npz')['arr_0'][array_pointer]
		print("Object label is ", object_label)
		filepointer = gen_filepointer(cnt)
		visualize(filepointer,array_pointer,data_path)
	return None

def get_next_object(cnt,prev_object_label,data_path):
	filepointer = gen_filepointer(cnt)
	object_labels = np.load(data_path+'object_labels_'+filepointer+'.npz')['arr_0']
	if len(np.argwhere(object_labels > prev_object_label))>0:
		array_pointer = np.where(object_labels > prev_object_label)[0][0]
	else:
		cnt += 1
		cnt, array_pointer = get_next_object(cnt,prev_object_label)
	return cnt,array_pointer

def get_next_image(cnt,prev_image_label,data_path):
	filepointer = gen_filepointer(cnt)
	image_labels = np.load(data_path+'image_labels_'+filepointer+'.npz')['arr_0']
	if image_labels[-1]==prev_image_label:
		cnt += 1
		array_pointer = 0
		cnt,array_pointer = get_next_image(cnt,prev_image_label,data_path)
	else:
		array_pointer = np.where(image_labels >= prev_image_label+1)[0][0]
	return cnt,array_pointer

def gen_filepointer(cnt):
	filepointer = '{:05d}'.format(cnt)
	return filepointer

def visualize(filepointer,array_pointer,data_path):
	print("File number:",filepointer)
	print("Array position:",array_pointer)
	depth_im_table = np.load(data_path+'depth_ims_tf_table_'+filepointer+'.npz')['arr_0'][array_pointer]
	depth_array = scale(depth_im_table[:,:,0],0,255)
	depth_im = Image.fromarray(depth_array,mode='L').resize((300,300))
	depth_im.show()
#	saving = input("Save this image? Press y for saving, any other key for not saving: ")
	saving = 'n'
	if saving == 'y' or saving == 'yes':	
		object_des = input("What type of object is this? ")
		savepath = '../../Desktop/Depths/'
		with open(savepath+'CSV_files/DexNet_'+object_des+'_depth_'+filepointer+'_'+str(array_pointer)+'.csv','w',newline='') as csvfile:
			writer = csv.writer(csvfile,delimiter = ',')
			writer.writerow(['Depth_ims_tf_table_'+filepointer+'.npz'])
			writer.writerow(['Array_pointer: '+str(array_pointer)])
			writer.writerow(['Object type: '+object_des])
			table = np.squeeze(depth_im_table)
			for row in table:
				writer.writerow(row)
		data = np.resize(depth_im_table,(1024,))
		binwidth = 0.0001
		plt.hist(data,bins=np.arange(min(data),max(data)+binwidth,binwidth))
		plt.xlabel('Depth [m]')
		plt.ylim((0,35))
		plt.title('Depths in dexnet depth image. File: '+filepointer+'_'+str(array_pointer))
		Grasp_depth = depth_im_table[16,16]
		plt.text(Grasp_depth,10,'Grasp-depth')
		plt.savefig(savepath+'DexNet_'+object_des+'_histogram_'+filepointer+'_'+str(array_pointer)+'.png')
		plt.close()
		depth_im.save(savepath+'DexNet_'+object_des+'_'+filepointer+'_'+str(array_pointer)+'.png')
	return None
	

if __name__=="__main__":
	main()
