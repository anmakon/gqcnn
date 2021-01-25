#!/usr/bin/python

import numpy as np
import pandas as pd
from PIL import Image
import csv


class Conversion:
    def __init__(self):
        self.data_path = "./data/training/Cornell/original/"
        self.export_path = "/home/annako/Desktop/"
        self.camera_height = 0.70
        self.filename = ""

    def _start_converting(self):
        while True:
            num = int(input("Input the file number: "))
            self.filename = self.data_path + 'pcd' + '{:04d}'.format(num) + '.txt'
            depth_table, org_table = self._read_image()
            self._create_csv(depth_table, org_table)

    def _create_csv(self, depth_table, org_table):
        with open(self.export_path + 'CSV_files_Original_' + self.filename[-8:-4] + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Filename: ' + self.filename])
            for row in org_table:
                writer.writerow(row)
        with open(self.export_path + 'CSV_files_Adjusted_' + self.filename[-8:-4] + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Filename: ' + self.filename])
            for row in depth_table:
                writer.writerow(row)
        return None

    def _read_image(self):
        point_cloud = pd.read_csv(self.filename, sep=" ", header=None, skiprows=10).to_numpy()
        depth_table = np.full((480, 640), self.camera_height)
        org_table = np.zeros((480, 640))
        for point in point_cloud:
            index = point[4]
            row = int(index // 640)
            col = int(index % 640)
            if point[2] < 0:
                depth_table[row][col] = self.camera_height
            else:
                depth_table[row][col] = self.camera_height - (point[2] / 1000)
            org_table[row][col] = point[2]
        return depth_table, org_table


if __name__ == "__main__":
    convert = Conversion()
    convert._start_converting()
