import numpy as np
import matplotlib.pyplot as plt

from PIL import Image,ImageDraw
import json
from autolab_core import YamlConfig,Logger,BinaryClassificationResult,Point
from visualization import Visualizer2D as vis2d

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
		self.num_images = 100		# Amount of images for plotting

	def scale(self,X,x_min=0,x_max=255):
		X_flattend = X.flatten()
		scaled = np.interp(X_flattend,(X_flattend.min(),X_flattend.max()),(x_min,x_max)) # X_flattend.min() X_flattend.max()
		integ = scaled.astype(np.uint8)
		integ.resize((32,32))
		return integ

	def _plot_grasp(self,image_arr,width,results,j,noise_arr=None,depth_arr=None):
		# Creating images with the grasps.
		# Adding text for noise/depth investigations
		# Visualising the grasp

		data = self.scale(image_arr[:,:,0])
		image = Image.fromarray(data)
		draw = ImageDraw.Draw(image)
		draw.line([16-width/2,16,16+width/2,16],fill=128) # Grasp line
		draw.line([16-width/2,13,16-width/2,19],fill=128) # Vertical lines for end of grasp
		draw.line([16+width/2,13,16+width/2,19],fill=128)
		image = image.resize((300,300))
		# Add prediction and label
		draw2 = ImageDraw.Draw(image)
		draw2.text((3,3), "Pred: %.3f; Label: %.1f" % (results.pred_probs[j],results.labels[j]),fill=50)
		if noise_arr is not None:
			draw2.text((3,12), "Added noise: %.4f; Added tilting: %.3f" % (noise_arr[j,0],noise_arr[j,1]),fill=50)
		if depth_arr is not None:
			if depth_arr[j] == 0:
				draw2.text((3,12), "Original depth",fill=50)
			else:
				draw2.text((3,12), "Realtive depth",fill=50)
		return image

	def run_analysis(self, model_dir,output_dir,data_dir,noise_analysis,depth_analysis):

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

		# Set up logger.
		self.logger = Logger.get_logger(self.__class__.__name__,
						log_file=os.path.join(
							output_dir, "analysis.log"),
						silence=(not self.verbose),
						global_log_file=self.verbose)

		self.logger.info("Analyzing model %s" % (model_name))
		self.logger.info("Saving output to %s" % (output_dir))

		# Run predictions
		result = self._run_prediction(model_dir, output_dir,data_dir,noise_analysis,depth_analysis) 
		

	def _run_prediction(self,model_dir,model_output_dir,data_dir,noise_analysis,depth_analysis):
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
				self.logger.info("Noise: %.4f Model %s error rate: %.3f" %
					(current_noise, model_dir, res.error_rate))
				self.logger.info("Noise: %.4f Model %s loss: %.3f" %
					(current_noise, model_dir, res.cross_entropy_loss))
		elif depth_analysis:
			# Analyse the error rates in regard to the grasping depth in the images
			depth_levels = np.unique(depth_arr)
			for current_depth in depth_levels:
				if current_depth == 0:
					depth_mode = 'original'
				else:
					depth_mode = 'relative '+ ("{0:02d}").format(current_depth)
				pred = predictions[depth_arr==current_depth]
				lab = labels[depth_arr==current_depth]
				res = BinaryClassificationResult(pred[:,1],lab)
				self.logger.info("Depth %s Model %s error rate: %.3f" %
					(depth_mode, model_dir, res.error_rate))
				self.logger.info("Depth: %s Model %s loss: %.3f" %
					(depth_mode, model_dir, res.cross_entropy_loss))
		else:
			self.logger.info("Model %s error rate: %.3f" %
				(model_dir, results.error_rate))
			self.logger.info("Model %s loss: %.3f" %
				(model_dir, results.cross_entropy_loss))

		cnt = 0 # Counter for grouping the same images with different noise/depth levels
		for j in range(0,self.num_images):
			if file_arr[j][1] != file_arr[j-1][1]:
				cnt = 0
			else:
				cnt += 1
			if noise_analysis:
				image = self._plot_grasp(image_arr[j],width_arr[j],results,j,noise_arr=noise_arr)
			elif depth_analysis:
				image = self._plot_grasp(image_arr[j],width_arr[j],results,j,depth_arr=depth_arr)
			else:
				image = self._plot_grasp(image_arr[j],width_arr[j],results,j)
			image.save(os.path.join(model_output_dir,"%05d_%03d_example_%03d.png" % (file_arr[j][0],file_arr[j][1],cnt)))
		return results


	def _read_data(self,data_dir, noise=False,depth=False):
		# Read in the data from the given directory.
		# Appends all .npz file into the same array.
		# Warning: This might be unsuitable for too many images!
		# If the dataset is too big, think about adjusting this to 
		# predicting on bunch of images at a time.
		files = os.listdir(data_dir)
		image_arr = np.empty((32,32,1)) 
		metric_arr = np.empty([])
		pose_arr = np.empty([])
		width_arr = np.empty([])
		file_arr = np.empty([])
		noise_arr = np.empty([])
		depth_arr = np.empty([])
		labels = []
		numbers = [string[-9:-4] for string in files]
		counter = len(list(set(numbers)))-2
		filenumber = ("{0:05d}").format(0)
		# Read in first file
		image_arr= np.load(data_dir+"depth_ims_tf_table_"+filenumber+".npz")['arr_0']
		poses = np.load(data_dir+"hand_poses_"+filenumber+".npz")['arr_0']
		metric_arr = np.load(data_dir+"robust_ferrari_canny_"+filenumber+".npz")['arr_0']
		pose_arr = poses[:,2:3]
		width_arr = poses[:,-1]
		file_arr = np.load(data_dir+"files_"+filenumber+".npz")['arr_0']
		label = 1* (metric_arr > self.metric_thresh)
		labels = label.astype(np.uint8)
		for i in range(1,counter):
			# Loop through the rest of the files
			filenumber = ("{0:05d}").format(i)
			try:
				image_arr= np.concatenate((image_arr,np.load(data_dir+"depth_ims_tf_table_"+filenumber+".npz")['arr_0']))
				poses = np.load(data_dir+"hand_poses_"+filenumber+".npz")['arr_0']
				metrics = np.load(data_dir+"robust_ferrari_canny_"+filenumber+".npz")['arr_0']
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
			return image_arr,pose_arr,labels,width_arr,file_arr,depth_arr
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
	if analysis_type is None:
		noise_analysis = False
		depth_analysis = False
	elif analysis_type == 'noise':
		noise_analysis = True
		depth_analysis = False
	elif analysis_type == 'depth':
		noise_analysis = False
		depth_analysis = True

	# Turn relative paths absolute.
	if not os.path.isabs(output_dir):
		output_dir = os.path.join(os.getcwd(), output_dir)

	# Make the output dir.
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	# Initalise analyser and run analysis.
	analyser = GQCNN_Analyse()
	analyser.run_analysis(model_dir,output_dir,data_dir,noise_analysis,depth_analysis)
