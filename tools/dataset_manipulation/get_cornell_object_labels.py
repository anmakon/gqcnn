import numpy as np
import pandas as pd
import argparse

class Get_Object_Label():
	def __init__(self,file_num,array_pos):
		self.data_path = "./data/training/Cornell/tensors/"
		self.file_string = ("{0:05d}").format(file_num)
		self.array_pos = array_pos
		self._main()

	def _main(self):
		labels = self._load_object_labels()
		object_label = np.load(self.data_path+'object_labels_'+self.file_string+'.npz')['arr_0'][self.array_pos]
		image = labels[np.where(labels[:,1] == object_label),0]
		print("Images: ",image)
		return True
		
	def _load_object_labels(self):
		labels = pd.read_csv(self.data_path+"z.txt",sep=" ", header=None,usecols=[i for i in [0,1,2]]).drop_duplicates().to_numpy()
		return labels

#object_labels = np.load('object_labels_'+filecnt+'.npz')
#pose_labels = np.load('pose_labels_'+filecnt+'.npz')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=("Get the object and pose label for a Cornell grasp"))
	parser.add_argument("file_num",
				type = int,
				default = None,
				help="File number of grasp")
	parser.add_argument("array_pos",
				type = int,
				default = None,
				help="Array position of grasp")
	args = parser.parse_args()
	Get_Object_Label(args.file_num,args.array_pos)
