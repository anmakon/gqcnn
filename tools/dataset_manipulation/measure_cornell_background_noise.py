import numpy as np
from PIL import Image
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.patches as patches
import csv

# This is a script to measure the noise in Cornell background.
# Interactive mode possible where you click on the depth image
# to select the corners of the object and therefore what does
# not count for the background.

class NoiseMeasurement():
	def __init__(self):
		self.data_path = "./data/training/Cornell/tensors/"
		self.file = 0
		self.testing = False # set to false to go through images in steps of 10. Set to true to manually select images
		self.cnt = 0
		self.binwidth = 0.0001
		self.array = 0
		self.save_path = "../../Desktop/Cornell_noise/"
		self._main()

	def _scale(self,X, x_min=0, x_max=255):
		X_flattend = X.flatten()
		scaled = np.interp(X_flattend,(0.55,0.75),(x_min,x_max)) #X_flattend.min() and X_flattend.max()
		integ = scaled.astype(np.uint8)
		integ.resize((32,32))
		return integ
	
	def _main(self):
		x = 'f'
		all_mean = [] 
		all_std = []
		while True:
			if x == 'f':
				print()
				if not self.testing:
					if self.array <490:
						self.array += 10
					else:
						self.array = 0
						self.file += 1
				else:
					try:
						self.file = int(input("Input the file number: "))
						self.array =int(input("Input the array position: "))
					except:
						print("Input not an integer")
						continue
				print("File number: ",self.file)
				print("Array number: ",self.array)
				new_mean,new_std = self._measure_noise()
				if new_mean is not None:
					if input("Don't add noise to collection? ") != 'n':
						all_mean.append(new_mean)
						all_std.append(new_std)
						self.cnt += 1
			elif x == 's':
				self._save(all_mean,all_std)
				break
			if self.array %250 == 0:
				x = input("Press f for next file and s for saving: ")	
		return None

	def _gen_filepointer(self):
		filepointer = '{:05d}'.format(self.file)
		return filepointer

	def _fix_tilt(self,depth_table):
		u_l = depth_table[0:2,0:2].mean()
		u_r = depth_table[29:31,0:2].mean()
		l_l = depth_table[0:2,29:31].mean()
		horizontal = (u_r-u_l)/30 # [m/pixel]
		vertical = (l_l-u_l)/30 # [m/pixel]
		adjusted_depth = np.zeros((32,32))
		for x_cnt, row in enumerate(depth_table):
			for y_cnt, point in enumerate(row):
				adjusted_depth[x_cnt,y_cnt] = point - x_cnt*horizontal - y_cnt*vertical
		return adjusted_depth
	
	def _onclick(self,event):
		ix,iy = event.xdata, event.ydata
		if isinstance(ix,float) or isinstance(ix,int):	
			self.coords.append((ix,iy))
		if len(self.coords) >= 2:
			self.fig.canvas.mpl_disconnect(self.cid)
			plt.close()
		return None 
	
	def _measure_noise(self):
		filepointer = self._gen_filepointer()
		try:
			depth_im_table = np.load(self.data_path+'depth_ims_tf_table_'+filepointer+'.npz')['arr_0'][self.array]
		except:
			print("File not available.")
			return None,None
		self.coords = []
		orig_data = depth_im_table[:,:,0]
		data = self._fix_tilt(orig_data)
		im = Image.fromarray(self._scale(data))
		self.fig,ax = plt.subplots(1)
		ax.imshow(im,cmap=col.Colormap('Greens'))
		self.cid = self.fig.canvas.mpl_connect('button_press_event',self._onclick)
		plt.show()
		try:
			x_1 = self.coords[0][0]
			x_2 = self.coords[1][0]
			y_1 = self.coords[0][1]
			y_2 = self.coords[1][1]
		except:
			print("Not enough values to get rectangle")
			return None,None
		if self.testing:
			fig,ax = plt.subplots(1)
			ax.imshow(im,cmap=col.Colormap('Greens'))
			rect = patches.Rectangle((x_1,y_1),x_2-x_1,y_2-y_1)
			ax.add_patch(rect)
			plt.show()
		removed = np.array([])
		removed_without_tilt = np.array([])
		for x_cnt, row in enumerate(data):
			for y_cnt, point in enumerate(row):
				if x_1 <= y_cnt <= x_2 and y_1 <= x_cnt <= y_2:
					continue
				else:
					removed = np.append(removed,point)
					removed_without_tilt = np.append(removed_without_tilt,orig_data[x_cnt,y_cnt])
		print("Picture background std:",removed.std())
		return [removed.mean(),removed_without_tilt.mean()], [removed.std(),removed_without_tilt.std()]

	def _save(self,mean_noise,std_noise):
		with open(self.save_path+'mean_noise.csv','w',newline='') as csvfile:
			writer = csv.writer(csvfile,delimiter = ',')
			writer.writerow(['Mean noise in '+str(self.cnt)+' images'])
			writer.writerow(['Mean noise corrected tilt: '+str(mean([point[0] for point in mean_noise]))])
			writer.writerow(['Mean noise uncorrected tilt: '+str(mean([point[1] for point in mean_noise]))])
			writer.writerow(['Corrected tilt','Uncorrected tilt'])
			for row in mean_noise:
				writer.writerow(row)
		with open(self.save_path+'std_noise.csv','w',newline='') as csvfile:
			writer = csv.writer(csvfile,delimiter = ',')
			writer.writerow(['Std of noise in '+str(self.cnt)+' images'])
			writer.writerow(['Std noise corrected tilt: '+str(mean([point[0] for point in std_noise]))])
			writer.writerow(['Std noise uncorrected tilt: '+str(mean([point[1] for point in std_noise]))])
			writer.writerow(['Corrected tilt','Uncorrected tilt'])
			for row in std_noise:
				writer.writerow(row)
		return None

if __name__=="__main__":
	noise = NoiseMeasurement()
