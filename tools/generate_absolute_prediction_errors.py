import numpy as np
import csv

import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os

class Absolute_Prediction_Error():
	def __init__(self):
		return None

	def create_errors(self,data_dir,output_dir):
		# Read in the data
		train_labels, train_predictions,val_labels,val_predictions = self._read_data(data_dir)
		all_labels = np.concatenate([train_labels,val_labels])
		all_predictions = np.concatenate([train_predictions,val_predictions])

		#Calculate the prediction errors
		self.train_error_pos,self.train_error_neg = self._calculate_prediction_errors(train_predictions,train_labels)
		self.val_error_pos,self.val_error_neg = self._calculate_prediction_errors(val_predictions,val_labels)
		self.all_error_pos,self.all_error_neg = self._calculate_prediction_errors(all_predictions,all_labels)
		
		#Plot and save the histograms
		self._calculate_accuracy()
		self._plot_errors(output_dir)

		return None

	def _calculate_accuracy(self):
		# Calculate validation accuracy, all accuracy and accuracy on successful and unsuccessful grasps
		# Validation
		
		val_correct = len(np.where(self.val_error_pos<=0.5)[0])+len(np.where(self.val_error_neg<=0.5)[0])
		val_images = len(self.val_error_pos)+len(self.val_error_neg)
		print("Validation accuracy is: ",val_correct/val_images*100,"%")
	
		all_correct = len(np.where(self.all_error_pos<=0.5)[0])+len(np.where(self.all_error_neg<=0.5)[0])
		all_images = len(self.all_error_pos)+len(self.all_error_neg)
		print("Accuracy is: ",all_correct/all_images*100,"%")
		
		pos_correct = len(np.where(self.all_error_pos<=0.5)[0])
		pos_images = len(self.all_error_pos)
		print("Accuracy on successful images is: ",pos_correct/pos_images*100,"%")

		neg_correct = len(np.where(self.all_error_neg<=0.5)[0])
		neg_images = len(self.all_error_neg)
		print("Accuracy on unsuccessful images is: ",neg_correct/neg_images*100,"%")


		return None

	def _plot_errors(self,output_dir):
		binwidth = 0.02
		plt.rc('axes',edgecolor='w')
		plt.rc('text',color='w')
		plt.rc(('xtick','ytick'),c='w')

		plt.hist(self.train_error_pos,bins=np.arange(0,1+binwidth,binwidth),color=(0.616,0.773,0.730))
		plt.xlabel("Absolute Prediction Error",color='w',fontsize=14)
		plt.title("Error on successful training grasps",fontsize=18)
		plt.savefig(output_dir+"err_pos_train",transparent=True)
		plt.close()

		plt.hist(self.train_error_neg,bins=np.arange(0,1+binwidth,binwidth),color=(0.616,0.773,0.730))
		plt.xlabel("Absolute Prediction Error",color='w',fontsize=14)
		plt.title("Error on unsuccessful training grasps",fontsize=18)
		plt.savefig(output_dir+"err_neg_train",transparent=True)
		plt.close()

		plt.hist(self.val_error_pos,bins=np.arange(0,1+binwidth,binwidth),color=(0.616,0.773,0.730))
		plt.xlabel("Absolute Prediction Error",color='w',fontsize=14)
		plt.title("Error on successful validation grasps",fontsize=18)
		plt.savefig(output_dir+"err_pos_val",transparent=True)
		plt.close()

		plt.hist(self.val_error_neg,bins=np.arange(0,1+binwidth,binwidth),color=(0.616,0.773,0.730))
		plt.xlabel("Absolute Prediction Error",color='w',fontsize=14)
		plt.title("Error on unsuccessful validation grasps",fontsize=18)
		plt.savefig(output_dir+"err_neg_val",transparent=True)
		plt.close()

		plt.hist(self.all_error_pos,bins=np.arange(0,1+binwidth,binwidth),color=(0.616,0.773,0.730))
		plt.xlabel("Absolute Prediction Error",color='w',fontsize=14)
		plt.title("Error on successful grasps",fontsize=18)
		plt.savefig(output_dir+"err_pos_all",transparent=True)
		plt.close()

		plt.hist(self.all_error_neg,bins=np.arange(0,1+binwidth,binwidth),color=(0.616,0.773,0.730))
		plt.xlabel("Absolute Prediction Error",color='w',fontsize=14)
		plt.title("Error on unsuccessful grasps",fontsize=18)
		plt.savefig(output_dir+"err_neg_all",transparent=True)
		plt.close()

	def _calculate_prediction_errors(self,predictions,labels):
		pos_ind = np.where(labels==1)
		neg_ind = np.where(labels==0)
		pos_prediction_errors = np.abs(labels[pos_ind] - predictions[pos_ind])
		neg_prediction_errors = np.abs(labels[neg_ind] - predictions[neg_ind])
		
		return pos_prediction_errors,neg_prediction_errors

	def _read_data(self,data_dir):
		train_labels = np.load(data_dir+"/train_result.cres/labels.npz")['arr_0']
		train_predictions = np.load(data_dir+"/train_result.cres/predictions.npz")['arr_0']
		val_labels = np.load(data_dir+"/val_result.cres/labels.npz")['arr_0']
		val_predictions = np.load(data_dir+"/val_result.cres/predictions.npz")['arr_0']
		return train_labels, train_predictions, val_labels, val_predictions
		

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=("Create CSV files for the absolute prediction errors"))
	parser.add_argument("data_dir",
				type=str,
				default=None,
				help="path to where the data is stored")
	parser.add_argument("--output_dir",
				type=str,
				default=None,
				help="path to save the csv files")
	
	args = parser.parse_args()
	data_dir = args.data_dir
	output_dir = args.output_dir

	if output_dir is None:
		output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
									"../analysis/Abs_pred_error/")

	# Turn relative paths absolute.
	if not os.path.isabs(output_dir):
		output_dir = os.path.join(os.getcwd(), output_dir)

        # Make the output dir.
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)


	saver = Absolute_Prediction_Error()
	saver.create_errors(data_dir,output_dir)




