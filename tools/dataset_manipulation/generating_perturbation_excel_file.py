import numpy as np
import xlsxwriter as xlsx
import csv
import os
import pandas as pd


# This is a file to generate an excel file showing the perturbation analysis.
# 

class Generate_Excel():
	def __init__(self):
		self.data_path = "./analysis/SingleFiles/"
		self.export_path = "./analysis/SingleFiles/"
		self.csv_Cornell = "./data/training/csv_files/Cornell_SingleFiles.csv"
		self.csv_DexNet = "./data/training/csv_files/DexNet_SingleFiles.csv"
		self.rgb_path = "./data/training/Cornell/"
		self.model = ''
		self.data = ''
		self.extract_data_path = ''
		self.current_worksheet = None		
		self.labels = pd.read_csv("./data/training/Cornell/original/z.txt",sep =" ", header = None, usecols=[i for i in [0,1]]).drop_duplicates().to_numpy()

		self._main()


	def _main(self):
		self._generate_file()
		
		#Cornell data on DexNet model		
		self.data = 'Cornell'
		datapoints = self._read_csv(self.csv_Cornell)
		self.model = 'DexNet'
		self._select_current_worksheet()
		self.extract_data_path = self._get_extraction_path()
		self._folderwalk(datapoints)
		
		#Cornell data on Cornell model

		self.model = 'Cornell'
		self.extract_data_path = self._get_extraction_path()
		self._folderwalk(datapoints)
			
		# DeXNet data on DexNet model		
		self.data = 'DexNet'
		datapoints = self._read_csv(self.csv_DexNet)
		self.model = 'DexNet'
		self._select_current_worksheet()
		self.extract_data_path = self._get_extraction_path()
		self._folderwalk(datapoints)

		self.workbook.close()

	def _folderwalk(self,datapoints):
		for root, dirs, files in os.walk(self.extract_data_path):
			for folder in dirs:
				path = os.path.join(root,folder)
				splits = folder.split('_')
				index = datapoints.index([int(splits[0]),int(splits[1])])
				if splits[2] == 'rotation':
					self._save_rotation(path,index)
					if self._check_prediction_changes(path) and self.model=='DexNet':
						self.current_worksheet.write(index+1,3,'Yes',self.cell_format)
				elif splits[2] == 'translation':
					if self.data == 'Cornell' and self.model =='Cornell':
						self._save_rgb_images(datapoints,index,splits)
					self._save_translation(path,index,splits)
					if self._check_prediction_changes(path) and self.model=='DexNet':
						self.current_worksheet.write(index+1,4,'x - Yes \n',self.cell_format)
				elif splits[2] == 'translationy':
					self._save_translationy(path,index)
					if self._check_prediction_changes(path) and self.model=='DexNet':
						self.current_worksheet.write(index+1,4,'y - Yes \n',self.cell_format)

	def _save_rgb_images(self,datapoints,index,splits):
		row = index + 1

		filestring = ("{0:05d}").format(int(splits[0]))
		object_label = np.load(self.rgb_path+"tensors/object_labels_"+filestring+".npz")['arr_0'][int(splits[1])]
		image = self.labels[np.where(self.labels[:,1]==object_label),0]
		for i in range(0,len(image[0])):
			image_pointer = ("{0:04d}").format(int(image[0][i]))
			image_path = self.rgb_path+"original/pcd"+image_pointer+'r.png'
			if os.path.isfile(image_path):
				self.current_worksheet.insert_image(row,6,image_path,{'x_scale':0.3,'y_scale':0.3})
				return None

	def _check_prediction_changes(self,path):
		with open(path+'/analysis.log') as f:
			f = f.readlines()
			
		for line in f:
			if 'Correct predictions:' in line:
				correct = int(line.split(" ")[-1])
			if 'False predictions:' in line:
				false = int(line.split(" ")[-1])
		try:
			if correct != 0 and false != 0:
				return True
		except:
			raise ValueError ("Couldn't find predictions in ",path)
		
		return False

	def _save_rotation(self,path,index):
		row = index + 1
		if self.model == 'DexNet':
			column = 7 
		elif self.model == 'Cornell':
			column = 10
		else:
			raise ValueError ("Neither Cornell, nor DexNet as a model")
		self.current_worksheet.insert_image(row,column,path+'/Grasp_rotation_err.png',{'y_scale':0.4,'x_scale':0.4})

	def _save_translation(self,path,index,point):
		row = index + 1
		if self.model == 'DexNet':
			namestring = ("{0:05d}").format(int(point[0]))+'_'+("{0:03d}").format(int(point[1]))+'_example'
			self.current_worksheet.write(row,0,point[0])
			self.current_worksheet.write(row,1,point[1])
			self.current_worksheet.insert_image(row,5,path+'/'+namestring+'_006.png',{'y_scale':0.4,'x_scale':0.4,'y_offset':25,'x_offset':3})
			column = 8 
		elif self.model == 'Cornell':
			column = 11
		else:
			raise ValueError ("Neither Cornell, nor DexNet as a model")
		self.current_worksheet.insert_image(row,column,path+'/Grasp_translation_err.png',{'y_scale':0.4,'x_scale':0.4})

	def _save_translationy(self,path,index):
		row = index + 1
		if self.model == 'DexNet':
			column = 9 
		elif self.model == 'Cornell':
			column = 12
		else:
			raise ValueError ("Neither Cornell, nor DexNet as a model")
		self.current_worksheet.insert_image(row,column,path+'/Grasp_translationy_err.png',{'y_scale':0.4,'x_scale':0.4})
	def _select_current_worksheet(self):
		if self.data == 'Cornell':
			self.current_worksheet = self.cornell_worksheet
		elif self.data == 'DexNet':
			self.current_worksheet = self.dexnet_worksheet
		else:
			raise ValueError ("Neither 'Cornell' nor 'DexNet' selected as data")
		return None

	def _get_extraction_path(self):
		if self.data != self.model:
			path = self.data_path+self.data+'_on_'+self.model+'_Single_Analysis'
		else:
			path = self.data_path+self.data+'_Single_Analysis'
		return path
			

	def _read_csv(self,path):
		csv_file = []
		with open(path,newline='') as csvfile:
			reader = csv.reader(csvfile,delimiter=',')
			for row in reader:
				csv_file.append([int(val) for val in row])
		return csv_file

	def _generate_file(self):
		self.workbook = xlsx.Workbook(self.export_path+"Evaluation_SingleFiles.xlsx")
	
		heading_format = self.workbook.add_format({'italic': True,'font_size':14,'align':'center','valign':'center','text_wrap':True})
		self.cell_format = self.workbook.add_format({'italic': True,'font_size':12,'align':'center','valign':'center','text_wrap':True})

		worksheet = self.workbook.add_worksheet("Cornell data")
		worksheet.write(0,0, "File",heading_format)
		worksheet.write(0,1,"Array",heading_format)
		worksheet.write(0,2,"Opinion on grasp quality",heading_format)
		worksheet.write(0,3,"Change prediction with rotation?",heading_format)
		worksheet.write(0,4,"Change prediction with translation?",heading_format)
		worksheet.write(0,5,"Depth image \n Green - DexNet \n Blue - Cornell",heading_format)
		worksheet.write(0,6,"Object",heading_format)
		worksheet.write(0,7,"Cornell data on model trained on DexNet - Absolute prediction error vs. rotation",heading_format)
		worksheet.write(0,8,"Cornell data on model trained on DexNet - Absolute prediction error vs. translation (x-axis)",heading_format)
		worksheet.write(0,9,"Cornell data on model trained on DexNet - Absolute prediction error vs. translation (y-axis)",heading_format)
		worksheet.write(0,10,"Cornell data on model trained on Cornell - Absolute prediction error vs. rotation",heading_format)
		worksheet.write(0,11,"Cornell data on model trained on Cornell - Absolute prediction error vs. translation (x-axis)",heading_format)
		worksheet.write(0,12,"Cornell data on model trained on Cornell - Absolute prediction error vs. translation (y-axis)",heading_format)
		self.cornell_worksheet = self._set_worksheet_size(worksheet)

		worksheet = self.workbook.add_worksheet("DexNet data")
		worksheet.write(0,0, "File",heading_format)
		worksheet.write(0,1,"Array",heading_format)
		worksheet.write(0,2,"Opinion on grasp quality",heading_format)
		worksheet.write(0,3,"Change prediction with rotation?",heading_format)
		worksheet.write(0,4,"Change prediction with translation?",heading_format)
		worksheet.write(0,5,"Depth image",heading_format)
		worksheet.write(0,7,"DexNet data on model trained on DexNet - Absolute prediction error vs. rotation",heading_format)
		worksheet.write(0,8,"DexNet data on model trained on DexNet - Absolute prediction error vs. translation (x-axis)",heading_format)
		worksheet.write(0,9,"DexNet data on model trained on DexNet - Absolute prediction error vs. translation (y-axis)",heading_format)
		self.dexnet_worksheet = self._set_worksheet_size(worksheet)

	def _set_worksheet_size(self,worksheet):
		worksheet.set_column(0,1,4.7)
		worksheet.set_column(2,4,9.8)
		worksheet.set_column(5,5,17)
		worksheet.set_column(6,12,35)
		worksheet.set_row(0,90)
		for i in range(1,100):
			worksheet.set_row(i,140)
		return worksheet

if __name__ == "__main__":

	Generate_Excel()
