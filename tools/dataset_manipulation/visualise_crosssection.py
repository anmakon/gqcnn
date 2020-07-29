import numpy as np
import argparse
import os

from PIL import Image
import matplotlib.pyplot as plt


class CrossSection():
	def __init__(self,tensor,array,Cornell,DexNet):
		self.tensor = tensor
		self.array = array
	
		self.output_path = "./analysis/CrossSection/"
		if not os.path.exists(self.output_path):
			os.mkdir(self.output_path)
		if Cornell and not DexNet:
			self.data_path = "./data/training/Cornell/tensors/"
			self.output_path += "Cornell_"
			self.dset = 'Cornell'
		elif DexNet and not Cornell:
			self.data_path = "./data/training/dexnet_2_tensor/tensors/"
			self.output_path += "DexNet_"
			self.dset = 'DexNet'
		else:
			raise KeyError("No destinct dataset chosen.")
		
		self._main()

	def _main(self):
		self._read_data()
		x_values,y_values = self._get_crosssection()
		self._plot_data(x_values,y_values)
		self._plot_depthimage()

	def _plot_depthimage(self):
		im = Image.fromarray(self._scale(self.depth_image[:,:,0])).convert('RGB')
		im = im.resize((300,300))
		im.save(self.output_path+("{0:05d}").format(self.tensor)+'_%d.png'%self.array)

	def _scale(self,X):
		X_flattend = X.flatten()
		scaled = np.interp(X_flattend,(0.6,0.75),(0,255))
		integ = scaled.astype(np.uint8)
		integ.resize((32,32))
		return integ	

	def _plot_data(self,x_values,y_values):
		plt.plot(x_values)
		plt.title("Cross section x, tensor %d array %d %s " %(self.tensor,self.array,self.dset))
		plt.ylabel("Depth [m]")
		plt.xlabel("Pixel [m]")
		plt.ylim((0.6,0.75))
		plt.savefig(self.output_path+("{0:05d}").format(self.tensor)+'_%d_xsection'%self.array)

		plt.close()
		plt.plot(y_values)
		plt.title("Cross section y, tensor %d array %d %s " %(self.tensor,self.array,self.dset))
		plt.ylabel("Depth [m]")
		plt.xlabel("Pixel [m]")
		plt.ylim((0.6,0.75))
		plt.savefig(self.output_path+("{0:05d}").format(self.tensor)+'_%d_ysection'%self.array)

	def _get_crosssection(self):
		"""Get the crosssection of the depth image ^= the depth values of the middle row
		and the middle column of the depth image"""
		x_values = []
		y_values = []
		for row_count, row in enumerate(self.depth_image):
			for column_count,value in enumerate(row):
				if row_count == len(row)/2:
					x_values.append(value[0])
				if column_count == len(row)/2:
					y_values.append(value[0])
		return x_values,y_values

	def _read_data(self):
		tensor_format = ("{0:05d}").format(self.tensor)
		self.depth_image = np.load(self.data_path+"depth_ims_tf_table_"+tensor_format+".npz")['arr_0'][self.array]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=("Visualise the cross section of depth images"))
	parser.add_argument("tensor",
				type = int,
				default = None,
				help = "Tensor for depth image.")
	parser.add_argument("array",
				type = int,
				default = None,
				help = "Array position for depth image.")
	parser.add_argument("--Cornell",
				type = bool,
				default = False,
				help = "Using the Cornell dataset.")
	parser.add_argument("--DexNet",
				type = bool,
				default = False,
				help = "Using the DexNet dataset.")
	args = parser.parse_args()
	tensor = args.tensor
	array = args.array
	DexNet = args.DexNet
	Cornell = args.Cornell

	CrossSection(tensor,array,Cornell,DexNet)
	
