import numpy as np
import matplotlib.pyplot as plt

from PIL import Image,ImageDraw,ImageFont
import json
from autolab_core import YamlConfig,Logger,BinaryClassificationResult,Point

import os
import argparse

from gqcnn.model import get_gqcnn_model
from gqcnn.grasping import Grasp2D

# This is a script to conduct a detailed analysis of a Cornell- or Dexnet subset.
# The influence of noise or different grasping depths on the GQCNN can be analysed

class GQCNN_Analyse():

	def __init__(self, verbose=True, plot_backend="pdf"):

		self.metric_thresh = 0.002    # Change metric threshold here if needed!
		self.verbose = verbose
		plt.switch_backend(plot_backend)
		self.num_images = 300		# Amount of images for plotting. Set to None if you want to plot all images

	def scale(self,X,x_min=0,x_max=255):
		X_flattend = X.flatten()
		scaled = np.interp(X_flattend,(X_flattend.min(),X_flattend.max()),(x_min,x_max)) # X_flattend.min() X_flattend.max()
		integ = scaled.astype(np.uint8)
		integ.resize((32,32))
		return integ

	def _plot_grasp(self,image_arr,width,results,j,noise_arr=None,depth_arr=None,perturb_arr=None):
		# Creating images with the grasps.
		# Adding text for noise/depth investigations
		# Visualising the grasp

		data = self.scale(image_arr[:,:,0])
		image = Image.fromarray(data)
		font = ImageFont.truetype(font="/home/annako/Desktop/arial_narrow_7.ttf",size=20)
		draw = ImageDraw.Draw(image)
		draw.line([16-width/2,16,16+width/2,16],fill=128) # Grasp line
		draw.line([16-width/2,13,16-width/2,19],fill=128) # Vertical lines for end of grasp
		draw.line([16+width/2,13,16+width/2,19],fill=128)
		image = image.resize((300,300))
		# Add prediction and label
		draw2 = ImageDraw.Draw(image)
		draw2.text((3,3), "Pred: %.3f; Label: %.1f" % (results.pred_probs[j],results.labels[j]),fill=50,font=font)
		if noise_arr is not None:
			draw2.text((3,18), "Added noise: %.4f; Added tilting: %.3f" % (noise_arr[j,0],noise_arr[j,1]),fill=50,font=font)
		if depth_arr is not None:
			if depth_arr[j] == -1:
				draw2.text((3,18), "Original depth",fill=50,font=font)
			else:
				draw2.text((3,18), "Realtive depth %.2f" % depth_arr[j],fill=50,font=font)
		if perturb_arr is not None:
			draw2.text((3,18), "Grasp rotation: %.1f degree" % perturb_arr[j],fill=50,font=font)
		return image

	def _plot_histograms(self,predictions,labels,savestring, output_dir):

		pos_errors,neg_errors = self._calculate_prediction_errors(predictions,labels)
		binwidth = 0.02

		plt.hist(pos_errors,bins=np.arange(0,1+binwidth,binwidth))
		plt.xlabel("Absolute Prediction Error",fontsize=14)
		plt.title("Error on successful grasps",fontsize=18)
		plt.savefig(output_dir+"/err_pos_"+savestring+".png")
		plt.close()		

		plt.hist(neg_errors,bins=np.arange(0,1+binwidth,binwidth))
		plt.xlabel("Absolute Prediction Error",fontsize=14)
		plt.title("Error on unsuccessful grasps",fontsize=18)
		plt.savefig(output_dir+"/err_neg_"+savestring+".png")
		plt.close()
		
#		plt.rc('axes',edgecolor='w')
#		plt.rc('text',color='w')
#		plt.rc(('xtick','ytick'),c='w')
#
#		plt.hist(pos_errors,bins=np.arange(0,1+binwidth,binwidth),color=(0.616,0.773,0.730))
#		plt.xlabel("Absolute Prediction Error",color='w',fontsize=14)
#		plt.title("Error on successful grasps",fontsize=18)
#		plt.savefig(output_dir+"/err_pos_"+savestring+".png",transparent=True)
#		plt.close()		
#
#		plt.hist(neg_errors,bins=np.arange(0,1+binwidth,binwidth),color=(0.616,0.773,0.730))
#		plt.xlabel("Absolute Prediction Error",color='w',fontsize=14)
#		plt.title("Error on unsuccessful grasps",fontsize=18)
#		plt.savefig(output_dir+"/err_neg_"+savestring+".png",transparent=True)
#		plt.close()

	def _calculate_prediction_errors(self,predictions,labels):
		pos_ind = np.where(labels==1)
		neg_ind = np.where(labels==0)
		pos_prediction_errors = np.abs(labels[pos_ind] - predictions[pos_ind])
		neg_prediction_errors = np.abs(labels[neg_ind] - predictions[neg_ind])
		return pos_prediction_errors, neg_prediction_errors
		
	def _plot_grasp_perturbations(self,degrees,accuracies,output_dir):
		plt.plot(degrees,accuracies,'-x',linewidth=2,markersize=10)
		plt.grid(color=(0.686,0.667,0.667),linestyle='--')
		plt.title('Classification accuracy for DexNet \n with grasp rotations',fontsize=16)
		plt.xlabel("Grasp rotation perturbation [deg]",fontsize=12)
		plt.ylabel("Classification accuracy [%]",fontsize=12)
		plt.ylim((0,102))
		plt.savefig(output_dir+"/Grasp_rotation_accuracy.png")
		plt.close()
	
	def run_analysis(self, model_dir,output_dir,data_dir,noise_analysis,depth_analysis,perturb_analysis):

		# Determine model name
		model_name = ""
		model_root = model_dir
		while model_name == "" and model_root != "":
			model_root, model_name = os.path.split(model_root)
		
		# Store Noise and Depth investigation in their corresponding directories
		if noise_analysis:
			output_dir = os.path.join(output_dir, "Noise_Comparison/")
		if depth_analysis:
			output_dir = os.path.join(output_dir, "Depth_Comparison/")
		if perturb_analysis:
			output_dir = os.path.join(output_dir, "Perturbation_Analysis/")

		# Set up logger.
		self.logger = Logger.get_logger(self.__class__.__name__,
						log_file=os.path.join(
							output_dir, "analysis.log"),
						silence=(not self.verbose),
						global_log_file=self.verbose)

		self.logger.info("Analyzing model %s" % (model_name))
		self.logger.info("Saving output to %s" % (output_dir))

		# Run predictions
		result = self._run_prediction(model_dir, output_dir,data_dir,noise_analysis,depth_analysis,perturb_analysis) 
		

	def _run_prediction(self,model_dir,model_output_dir,data_dir,noise_analysis,depth_analysis,perturb_analysis):
		"""Predict the outcome of the file for a single model."""

		# Read in model config.
		model_config_filename = os.path.join(model_dir,"config.json")
		with open(model_config_filename) as data_file:
			model_config = json.load(data_file)

		# Load model.
		self.logger.info("Loading model %s" % (model_dir))
		log_file = None
		for handler in self.logger.handlers:
			if isinstance(handler, logging.FileHandler):
				log_file = handler.baseFilename
		gqcnn = get_gqcnn_model(verbose=self.verbose).load(
			model_dir, verbose=self.verbose, log_file=log_file)
		gqcnn.open_session()
		gripper_mode = gqcnn.gripper_mode
		angular_bins = gqcnn.angular_bins

		# Load data
		if noise_analysis:
			image_arr,pose_arr,labels,width_arr,file_arr,noise_arr = self._read_data(data_dir,noise=True)
		elif depth_analysis:
			image_arr,pose_arr,labels,width_arr,file_arr,depth_arr = self._read_data(data_dir,depth=True)
		elif perturb_analysis:
			image_arr,pose_arr,labels,width_arr,file_arr,perturb_arr = self._read_data(data_dir,perturb=True)
		else:
			image_arr,pose_arr,labels,width_arr,file_arr = self._read_data(data_dir)
		# Predict outcomes
		predictions = gqcnn.predict(image_arr,pose_arr)
		gqcnn.close_session()
		results = BinaryClassificationResult(predictions[:,1],labels)

		# Log the results
		if noise_analysis:
			# Analyse the error rates in regard to the noise levels of the images
			noise_levels = np.unique(noise_arr)
			for current_noise in noise_levels:
				pred = predictions[noise_arr[:,0]==current_noise]
				lab = labels[noise_arr[:,0]==current_noise]
				res = BinaryClassificationResult(pred[:,1],lab)
				self._plot_histograms(pred[:,1],lab,str(current_noise),model_output_dir)
				self.logger.info("Noise: %.4f Model %s error rate: %.3f" %
					(current_noise, model_dir, res.error_rate))
				self.logger.info("Noise: %.4f Model %s loss: %.3f" %
					(current_noise, model_dir, res.cross_entropy_loss))
		elif depth_analysis:
			# Analyse the error rates in regard to the grasping depth in the images
			depth_levels = np.unique(depth_arr)
			for current_depth in depth_levels:
				if current_depth == -1:
					depth_mode = 'original'
				else:
					depth_mode = 'relative %.2f' % (current_depth)
				pred = predictions[depth_arr==current_depth]
				lab = labels[depth_arr==current_depth]
				res = BinaryClassificationResult(pred[:,1],lab)
				self._plot_histograms(pred[:,1],lab,depth_mode,model_output_dir)
				self.logger.info("Depth %s Model %s error rate: %.3f" %
					(depth_mode, model_dir, res.error_rate))
				self.logger.info("Depth: %s Model %s loss: %.3f" %
					(depth_mode, model_dir, res.cross_entropy_loss))
		elif perturb_analysis:
			# Analyse the error rates in regard to the grasping perturb in the images
			perturb_levels = np.unique(perturb_arr)
			accuracies = []
			for current_perturb in perturb_levels:
				perturb_mode = 'rotation %.0f deg' % (current_perturb)
				pred = predictions[perturb_arr==current_perturb]
				lab = labels[perturb_arr==current_perturb]
				res = BinaryClassificationResult(pred[:,1],lab)
				self._plot_histograms(pred[:,1],lab,'rotation_%.0f_deg'%(current_perturb),model_output_dir)
				self.logger.info("Grasp %s Model %s error rate: %.3f" %
					(perturb_mode, model_dir, res.error_rate))
				accuracies.append(100-res.error_rate)
				self.logger.info("Grasp %s Model %s loss: %.3f" %
					(perturb_mode, model_dir, res.cross_entropy_loss))
			self._plot_grasp_perturbations(perturb_levels,accuracies,model_output_dir)
		else:
			self._plot_histograms(predictions[:,1],labels,'',model_output_dir)
			self.logger.info("Model %s error rate: %.3f" %
				(model_dir, results.error_rate))
			self.logger.info("Model %s loss: %.3f" %
				(model_dir, results.cross_entropy_loss))

		cnt = 0 # Counter for grouping the same images with different noise/depth levels
		if self.num_images == None or self.num_images > len(width_arr):
			self.num_images = len(width_arr)
		for j in range(0,self.num_images):
			try:
				if file_arr[j][1] != file_arr[j-1][1]:
					cnt = 0
				else:
					cnt += 1
			except:
				print("Could not access file_arr. Does it exist?")
				cnt += 1
			if noise_analysis:
				image = self._plot_grasp(image_arr[j],width_arr[j],results,j,noise_arr=noise_arr)
			elif depth_analysis:
				image = self._plot_grasp(image_arr[j],width_arr[j],results,j,depth_arr=depth_arr)
			elif perturb_analysis:
				image = self._plot_grasp(image_arr[j],width_arr[j],results,j,perturb_arr=perturb_arr)
			else:
				image = self._plot_grasp(image_arr[j],width_arr[j],results,j)
			try:
				image.save(os.path.join(model_output_dir,"%05d_%03d_example_%03d.png" % (file_arr[j][0],file_arr[j][1],cnt)))
			except:
				print("Saving image did not work. Maybe due to non-excisting file_arr")
				image.save(os.path.join(model_output_dir,"Example_%03d.png" % (cnt)))
		return results


	def _read_data(self,data_dir, noise=False,depth=False,perturb=False):
		# Read in the data from the given directory.
		# Appends all .npz file into the same array.
		# Warning: This might be unsuitable for too many images!
		# If the dataset is too big, think about adjusting this to 
		# predicting on bunch of images at a time.

		read_file_arr = True

		files = os.listdir(data_dir)
		image_arr = np.empty((32,32,1)) 
		metric_arr = np.empty([])
		pose_arr = np.empty([])
		width_arr = np.empty([])
		file_arr = np.empty([])
		noise_arr = np.empty([])
		depth_arr = np.empty([])
		labels = []
		numbers = [string[-9:-4] for string in files if '.npz' in string]
		counter = len(list(set(numbers)))
		filenumber = ("{0:05d}").format(0)
		# Read in first file
		image_arr= np.load(data_dir+"depth_ims_tf_table_"+filenumber+".npz")['arr_0']
		poses = np.load(data_dir+"hand_poses_"+filenumber+".npz")['arr_0']
		metric_arr = np.load(data_dir+"robust_ferrari_canny_"+filenumber+".npz")['arr_0']
		pose_arr = poses[:,2:3]
		width_arr = poses[:,-1]
		try:
			file_arr = np.load(data_dir+"files_"+filenumber+".npz")['arr_0']
		except:
			print("Could not load files_00000.npz")
			print("Will ignore files_XXXXX.npz files")
			read_file_arr = False
		label = 1* (metric_arr > self.metric_thresh)
		labels = label.astype(np.uint8)
		for i in range(1,counter):
			# Loop through the rest of the files
			filenumber = ("{0:05d}").format(i)
			try:
				image_arr= np.concatenate((image_arr,np.load(data_dir+"depth_ims_tf_table_"+filenumber+".npz")['arr_0']))
				poses = np.load(data_dir+"hand_poses_"+filenumber+".npz")['arr_0']
				metrics = np.load(data_dir+"robust_ferrari_canny_"+filenumber+".npz")['arr_0']
				if read_file_arr:
					file_arr = np.concatenate((file_arr,np.load(data_dir+"files_"+filenumber+".npz")['arr_0']))
			except:
				print("Could not open file with ",filenumber)
				print("Continue.")
				continue
			metric_arr = np.concatenate((metric_arr,metrics))
			pose_arr = np.concatenate((pose_arr,poses[:,2:3]))
			width_arr = np.concatenate((width_arr,poses[:,-1]))
			label = 1* (metrics > self.metric_thresh)
			labels = np.append(labels,label.astype(np.uint8))
		if noise:
			# Add the noise levels, if analysing noise
			noise_arr = np.load(data_dir+"noise_and_tilting_00000.npz")['arr_0']
			for i in range(1,counter):
				filenumber = ("{0:05d}").format(i)
				try:
					noise_arr = np.concatenate((noise_arr,np.load(data_dir+"noise_and_tilting_"+filenumber+".npz")['arr_0']))
				except:
					print("Could not open noise file with filenumber: ",filenumber)
					print("Continue")
					continue
			return image_arr,pose_arr,labels,width_arr,file_arr,noise_arr
		if depth:
			# Add the depth levels, if analysing depth
			depth_arr = np.load(data_dir+"depth_info_00000.npz")['arr_0']
			for i in range(1,counter):
				filenumber = ("{0:05d}").format(i)
				try:
					depth_arr = np.concatenate((depth_arr,np.load(data_dir+"depth_info_"+filenumber+".npz")['arr_0']))
				except:
					print("Could not open depth file with filenumber",filenumber)
					print("Continue.")
					continue
			#print("Shape pose_arr: ",pose_arr.shape)
			#print("Shape image_arr: ",image_arr.shape)
			return image_arr,pose_arr,labels,width_arr,file_arr,depth_arr
		if perturb:
			# Add the perturb levels, if analysing perturb
			perturb_arr = np.load(data_dir+"grasp_perturbations_00000.npz")['arr_0']
			for i in range(1,counter):
				filenumber = ("{0:05d}").format(i)
				try:
					perturb_arr = np.concatenate((perturb_arr,np.load(data_dir+"grasp_perturbations_"+filenumber+".npz")['arr_0']))
				except:
					print("Could not open perturb file with filenumber",filenumber)
					print("Continue.")
					continue
			#print("Shape pose_arr: ",pose_arr.shape)
			#print("Shape image_arr: ",image_arr.shape)
			return image_arr,pose_arr,labels,width_arr,file_arr,perturb_arr
		return image_arr,pose_arr,labels,width_arr,file_arr


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=("Analyse a GQCNN with Tensorflow on single data"))
	parser.add_argument("model_name",
				type=str,
				default=None,
				help="name of model to analyse")
	parser.add_argument("data_dir",
				type=str,
				default=None,
				help="path to where the data is stored")
	parser.add_argument("--output_dir",
				type=str,
				default=None,
				help="path to save the analysis")
	parser.add_argument("--analysis",
				type=str,
				default=None,
				help="Should there be a special analysis? Can be depth or noise.")
	
	args = parser.parse_args()
	model_name = args.model_name
	output_dir = args.output_dir
	data_dir = args.data_dir
	analysis_type = args.analysis

	 # Create model dir.
	model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
								"../models")
	model_dir = os.path.join(model_dir, model_name)

	if output_dir is None:
		output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
									"../analysis/SingleFiles")
	
	# Set the noise and depth analysis

	if analysis_type == 'noise' or analysis_type == 'Noise':
		noise_analysis = True
		depth_analysis = False
		perturb_analysis = False
	elif analysis_type == 'depth' or analysis_type == 'Depth':
		noise_analysis = False
		depth_analysis = True
		perturb_analysis = False
	elif analysis_type == 'perturbation' or analysis_type == 'Perturbation':
		noise_analysis = False
		depth_analysis = False
		perturb_analysis = True
	else:
		noise_analysis = False
		depth_analysis = False
		perturb_analysis = False

	# Turn relative paths absolute.
	if not os.path.isabs(output_dir):
		output_dir = os.path.join(os.getcwd(), output_dir)

	# Make the output dir.
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	# Initalise analyser and run analysis.
	analyser = GQCNN_Analyse()
	analyser.run_analysis(model_dir,output_dir,data_dir,noise_analysis,depth_analysis,perturb_analysis)
