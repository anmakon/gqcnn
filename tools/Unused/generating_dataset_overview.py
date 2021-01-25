import numpy as np
import xlsxwriter as xlsx
import csv
import os
import pandas as pd


# This is a file to generate an excel file showing the perturbation analysis.
# 

class Generate_Excel():
	def __init__(self):
		self.data_path = "./analysis/Dset_Overview/"
		self.export_path = "./analysis/Dset_Overview/"
		self.csv_Cornell = "./data/training/csv_files/Cornell_GB_Single.csv"
		self.csv_DexNet = "./data/training/csv_files/DexNet_GB_Single.csv"
		self.model = ''
		self.data = ''
		self.extract_data_path = ''
		self.worksheet = None		

		self._main()


	def _main(self):
		self._generate_file()
		
		#Cornell data

		self.data = 'Cornell'
		datapoints = self._read_csv(self.csv_Cornell)
		self._add_objects_to_worksheet()
		self._get_extraction_path()
		self._folderwalk(datapoints)
			
		# DexNet data
		self.data = 'DexNet'
		datapoints = self._read_csv(self.csv_DexNet)
		self._get_extraction_path()
		self._folderwalk(datapoints)

		self.workbook.close()

	def _folderwalk(self,datapoints):
		for root, dirs, files in os.walk(self.cornell_model_path):
			for folder in dirs:
				self.model = 'Cornell'
				path = os.path.join(root,folder)
				splits = folder.split('_')
				index = datapoints.index([int(splits[0]),int(splits[1])])
				if self.data == 'DexNet':
					row = 3+int(index/2)*2 # Modify to have alternative Cornell/DexNet data order
				else:
					row = 2+int(index/2)*2
				prediction,ground_truth = self._check_prediction(path)
				self._save_prediction(path,row,splits,prediction,ground_truth)
		for root, dirs, files in os.walk(self.dexnet_model_path):
			for folder in dirs:
				self.model = 'DexNet'
				path = os.path.join(root,folder)
				splits = folder.split('_')
				index = datapoints.index([int(splits[0]),int(splits[1])])
				if self.data == 'DexNet':
					row = 3+int(index/2)*2
				else:
					row = 2+int(index/2)*2

				prediction,ground_truth = self._check_prediction(path)
				self._save_prediction(path,row,splits,prediction,ground_truth)
				

	def _check_prediction(self,path):
		with open(path+'/analysis.log') as f:
			f = f.readlines()
		ground_truth = 'N/A'
		pred = 0.0
		for line in f:
			if 'in x' in line and '0 pixel' in line and 'prediction' in line:
				pred = float(line.split(" ")[-1])
			if 'positive grasps' in line and 'negative grasps' in line:
				if line.split(" ")[-6] == '1':
					ground_truth = 'positive'
				elif line.split(" ")[-3] == '1':
					ground_truth = 'negative'
		return pred, ground_truth

	def _add_objects_to_worksheet(self):
		objects = ['Mug','Mug','Banana','Banana','Apple','Apple']
		obj_ID = ['94','537','229','596','221','898']
		for i,obj in enumerate(objects):
			self.worksheet.write(i+2,0,obj,self.cell_format)
			self.worksheet.write(i+2,1,obj_ID[i],self.cell_format)
		return None

	def _save_prediction(self,path,row,point,prediction,ground_truth):
		if ground_truth == 'positive':
			start = 3
		else:
			start = 6
		if self.model == 'DexNet':
			column = start+2
			if self.data =='DexNet':
				self.worksheet.write(row,2,'DexNet',self.cell_format)
			else:
				self.worksheet.write(row,2,'Cornell',self.cell_format)
		elif self.model == 'Cornell':
			namestring = 'Depth_image.png'
			self.worksheet.insert_image(row,start,path+'/'+namestring,{'y_scale':0.4,'x_scale':0.4,'y_offset':5,'x_offset':5})
			column = start+1
		else:
			raise ValueError ("Neither Cornell, nor DexNet as a model")
		if (prediction >= 0.5 and ground_truth =='positive') or (prediction<0.5 and ground_truth =='negative'):
			self.worksheet.write(row,column,prediction,self.cell_format)
		else:
			self.worksheet.write(row,column,prediction,self.cell_format_red)

	def _get_extraction_path(self):
		if self.data == 'Cornell':
			self.cornell_model_path = self.data_path+self.data+'_Single_Analysis'
			self.dexnet_model_path = self.data_path+self.data+'_on_DexNet_Single_Analysis'
		else:
			self.dexnet_model_path = self.data_path+self.data+'_Single_Analysis'
			self.cornell_model_path = self.data_path+self.data+'_on_Cornell_Single_Analysis'

		return None

	def _read_csv(self,path):
		csv_file = []
		with open(path,newline='') as csvfile:
			reader = csv.reader(csvfile,delimiter=',')
			for row in reader:
				csv_file.append([int(val) for val in row])
		return csv_file

	def _generate_file(self):
		self.workbook = xlsx.Workbook(self.export_path+"Dataset_Overview.xlsx")
	
		heading_format = self.workbook.add_format({'italic': True,'font_size':14,'align':'center','valign':'center','text_wrap':True})
		self.cell_format = self.workbook.add_format({'italic': True,'font_size':12,'align':'center','valign':'center','text_wrap':True})
		self.cell_format_red = self.workbook.add_format({'italic': True,'font_color':'red','font_size':12,'align':'center','valign':'center','text_wrap':True})

		worksheet = self.workbook.add_worksheet("Dataset_Overview")
		
		worksheet.write(1,0,"Object",heading_format)
		worksheet.write(1,1,"ID",heading_format)
		worksheet.write(1,2,"Dset",heading_format)

		worksheet.write(1,3,"Depth image",heading_format)
		worksheet.write(1,4,"GQCNN-Cornell prediction",heading_format)
		worksheet.write(1,5,"GQCNN-DexNet prediction",heading_format)
		worksheet.write(1,6,"Depth image",heading_format)
		worksheet.write(1,7,"GQCNN-Cornell prediction",heading_format)
		worksheet.write(1,8,"GQCNN-DexNet prediction",heading_format)

		worksheet.merge_range(0,3,0,5,"Ground truth positive",heading_format)
		worksheet.merge_range(0,6,0,8,"Ground truth negative",heading_format)
		self.worksheet = self._set_worksheet_size(worksheet)

	def _set_worksheet_size(self,worksheet):
		worksheet.set_column(0,2,10)
		worksheet.set_column(3,3,18)
		worksheet.set_column(4,5,15)
		worksheet.set_column(6,6,18)
		worksheet.set_column(7,8,15)
		worksheet.set_row(0,20)
		worksheet.set_row(1,50)
		for i in range(2,100):
			worksheet.set_row(i,100)
		return worksheet

if __name__ == "__main__":

	Generate_Excel()
