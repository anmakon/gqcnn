import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mplimg

class Browse_Planned_Grasps:
	def __init__(self):
		self.data_path = './analysis/execution_real_grasps/'
		self.save_path = './analysis/Grasps_planed_by_GQCNN_trained_on_'
		self._browse_images()
	
	def _onclick(self,event):
		print(event.key)
		self.key = event.key
		self.fig.canvas.mpl_disconnect(self.click)
		plt.close()
		return None	

	def _browse_images(self):
		for root,files,dirs in os.walk(self.data_path):
			DexNet_images = [image for image in dirs if 'DexNet' in image]
			Cornell_images = [image for image in dirs if not 'DexNet' in image]

			print("Model trained on DexNet")
			print(len(DexNet_images)," images")
			good_grasps, all_grasps,images = self._browse_dset(DexNet_images,root)
			self._save_csv(self.save_path+'DexNet.csv',images,all_grasps,good_grasps)

			print("Model trained on Cornell")
			print(len(Cornell_images)," images")
			good_grasps, all_grasps,images = self._browse_dset(Cornell_images,root)
			self._save_csv(self.save_path+'Cornell.csv',images,all_grasps,good_grasps)

	def _browse_dset(self,all_images,root):
		images= []
		good_grasps = 0
		cnt = 0
		for image in all_images:
			self.fig,ax = plt.subplots(1,figsize = (8,8))
			img = mplimg.imread(os.path.join(root,image))
			ax.imshow(img)
			plt.title(image)
			self.click = self.fig.canvas.mpl_connect('key_press_event',self._onclick)
			plt.show()
			if self.key == 'y':
				# This is a good grasp
				good_grasps += 1
				images.append(image.split('_')[2])
			elif self.key == 'n':
				# This is a bad grasp
				print("whooooza")
			elif self.key == 'x':
				# Abort the process
				print("Abort")
				break
			cnt += 1
			if cnt %100 == 0:
				print(cnt)
		return good_grasps,cnt,images

	def _save_csv(self,save_path,images,all_grasps,good_grasps):
		with open(save_path,'w',newline='') as csvfile:
			writer = csv.writer(csvfile,delimiter = ',')
			writer.writerow(['All predicted grasps: '+str(all_grasps)])
			writer.writerow(['Successful predicted grasps: '+str(good_grasps)])
			for row in images:
				print(row)
				writer.writerow(row)
		return None

if __name__ =='__main__':
	Browse_Planned_Grasps()
