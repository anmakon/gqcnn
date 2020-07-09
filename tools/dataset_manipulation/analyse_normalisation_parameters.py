import numpy as np
import matplotlib.pyplot as plt
import os
import csv

class Analyse_Normalisation():
	def __init__(self):
		cornell_path = "./data/training/Cornell/"
		self.cornell = True
		self.output_path = "./analysis/Normalisation/"
		if not os.path.exists(self.output_path):
			os.mkdir(self.output_path)
		dexnet_path = "./data/training/dexnet_2_tensor/"
		self.num_random_files = 10000
		self._compute_mean_and_std(cornell_path)
		self.cornell = False
#		self.robust = 0.002
		self._compute_mean_and_std(dexnet_path)
		return None

	def _compute_mean_and_std(self,path):
		self.im_mean = 0
		self.im_std = 0
		self.depths = []
		self.pos_depths = []
		self.neg_depths = []
		self.mean_pos = 0
		self.mean_neg = 0
		self.std_pos = 0
		self.std_neg = 0
		num_pos = 0
		num_neg = 0
		num_summed = 0
		outliers = []
		index = self._read_training_indices(path)
		num_random_files = min(self.num_random_files,len(index))
		random_index = np.random.choice(index,
							size=num_random_files,
							replace=False)
		# ToDo: make this dependent on the config file
		if self.cornell:
			random_file_indices = [[pos//500,pos%500] for pos in random_index]
		else:
			random_file_indices = [[pos//1000,pos%1000] for pos in random_index]

		for tensor,array in random_file_indices:
			im_data,lab = self._load_training_data(path,int(tensor),int(array))
			num = im_data.shape[0]*im_data.shape[1]*im_data.shape[2]
			self.im_mean += np.sum(im_data)
			if lab == 1:
				self.mean_pos += np.sum(im_data)
				num_pos += num
			else:
				self.mean_neg += np.sum(im_data)
				num_neg += num
			num_summed += num
		self.im_mean = self.im_mean / num_summed
		self.mean_pos = self.mean_pos / num_pos
		self.mean_neg = self.mean_neg / num_neg
		print("Mean is:",self.im_mean)
		for tensor,array in random_file_indices:
			im_data, lab = self._load_training_data(path,int(tensor),int(array))
			self.im_std += np.sum((im_data - self.im_mean)**2)
			data = im_data.flatten().tolist()
			if lab == 1:
				self.std_pos += np.sum((im_data-self.mean_pos)**2)
				self.pos_depths.extend(data)
			else:
				self.std_neg += np.sum((im_data-self.mean_neg)**2)
				self.neg_depths.extend(data)
			if min(data)< 0.2:
				print("Found depth image with outlier.")
				print("Dataset:",path.split('/')[-1])
				print("Tensor: ", tensor, "; Array: ",array)
				outliers.append([tensor,array,tensor,array])
			self.depths.extend(data)
		self.im_std = np.sqrt(self.im_std / num_summed)	
		self.std_pos = np.sqrt(self.std_pos / num_pos)
		self.std_neg = np.sqrt(self.std_neg / num_neg)
		print("Std is:",self.im_std)
		self._plt_histograms()
		if len(outliers) >= 1:
			self._save_outlier_pointers(outliers)
	
	def _save_outlier_pointers(self,outliers):
		export_path = "./data/training/csv_files"
		if self.cornell:
			export_path += "/Cornell_outliers.csv"
		else:
			export_path += "/DexNet_outliers.csv"
		with open(export_path,'w',newline='') as csvfile:
			writer = csv.writer(csvfile,delimiter = ',')
			for row in outliers:
				writer.writerow(row)
		return None

	def _plt_histograms(self):
		self._plt_single_hist(self.im_mean,self.im_std,self.depths,"")
		self._plt_single_hist(self.mean_pos,self.std_pos,self.pos_depths,"_positive")
		self._plt_single_hist(self.mean_neg,self.std_neg,self.neg_depths,"_negative")


	def _plt_single_hist(self,mean,std,depths,mode):
		if self.cornell:
			data = 'Cornell'
			unit = '[m]'
		else:
			data= 'DexNet'
			unit = '[units]'
		path = self.output_path + "Depths_in_"+data+mode
		
		binwidth = 0.001
		n,bins,_ =plt.hist(depths,bins = np.arange(min(depths),max(depths)+binwidth,binwidth))
		plt.title("Depth values in "+data+" " + mode[1:])
		plt.xlabel("Depth from Camera "+unit)
		plt.vlines(mean,0,max(n),colors='r')
		plt.text(bins[1],max(n)*7/8,"Mean: "+str(round(mean,3)))
		plt.vlines(mean-std,0,max(n),colors='r')
		plt.vlines(mean+std,0,max(n),colors='r')	
		plt.text(bins[1],max(n)*6/8,"Std: "+str(round(std,3)))
		plt.savefig(path+"_all")	
		plt.text(mean-2.5*std,max(n)*7/8,"Mean: "+str(round(mean,3)))
		plt.text(mean-2.5*std,max(n)*6/8,"Std: "+str(round(std,3)))
		plt.xlim((mean-3*std,mean+3*std))
		plt.savefig(path)	
		plt.close()
		return None


	def _load_training_data(self,path,tensor,array):
		robust = 0.002
		depth_path = path +"tensors/depth_ims_tf_table_"
		label_path = path +"tensors/robust_ferrari_canny_"
		robustness = np.load(label_path+("{0:05d}").format(tensor)+".npz")['arr_0'][array]
		if robustness >= robust:
			label = 1
		else:
			label = 0
		depth = np.load(depth_path+("{0:05d}").format(tensor)+".npz")['arr_0'][array]
		return depth,label

	def _read_training_indices(self,path):
		path += "splits/image_wise/train_indices.npz"
		return np.load(path)['arr_0'] 
	
if __name__ == "__main__":
	Analyse_Normalisation()
