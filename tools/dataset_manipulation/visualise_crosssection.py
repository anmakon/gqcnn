import numpy as np
import argparse
import os

from PIL import Image
import matplotlib.pyplot as plt
from tools.path_settings import DATA_PATH, EXPER_PATH


class CrossSection:
    def __init__(self, tensor, array, Cornell, DexNet, single):
        self.single = single
        if single:
            self.output_path = EXPER_PATH + "/Single_Analysis/"
        else:
            self.output_path = EXPER_PATH + "/CrossSection/"
        self.tensor = tensor
        self.array = array

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if single:
            self.data_path = "./data/training/Subset_datasets/Cornell_SinglePerturb/"
            self.dset = 'Cornell'
        elif Cornell and not DexNet:
            self.data_path = "./data/training/Cornell/tensors/"
            self.output_path += "Cornell_"
            self.dset = 'Cornell'
        elif DexNet and not Cornell:
            self.data_path = DATA_PATH + "dexnet_2_tensor/tensors/"
            self.output_path += "DexNet_"
            self.dset = 'DexNet'
        else:
            raise KeyError("No distinct dataset chosen.")

        self._main()

    def _main(self):
        self._read_data()
        x_values, y_values = self._get_crosssection()
        self._plot_data(x_values, y_values)
        if not self.single:
            self._plot_depthimage()

    def _plot_depthimage(self):
        im = Image.fromarray(self._scale(self.depth_image[:, :, 0])).convert('RGB')
        im = im.resize((300, 300))
        im.save(self.output_path + "{0:05d}".format(self.tensor) + '_%d.png' % self.array)

    def _scale(self, X):
        X_flattend = X.flatten()
        scaled = np.interp(X_flattend, (0.6, 0.75), (0, 255))
        integ = scaled.astype(np.uint8)
        integ.resize((32, 32))
        return integ

    def _plot_data(self, x_values, y_values):
        plt.plot(x_values)
        plt.title("Cross section x, tensor %d array %d %s " % (self.tensor, self.array, self.dset))
        plt.ylabel("Depth [m]")
        plt.xlabel("Pixel [m]")
        plt.hlines(self.grasp_depth, 16 - 6, 16 + 6, color='r')
        plt.vlines(16 - 6, self.grasp_depth - 0.01, self.grasp_depth + 0.01, color='r')
        plt.vlines(16 + 6, self.grasp_depth - 0.01, self.grasp_depth + 0.01, color='r')
        plt.ylim((0.6, 0.75))
        plt.savefig(self.output_path + "{0:05d}".format(self.tensor) + '_%d_xsection' % self.array)
        plt.close()

        if not self.single:
            plt.plot(y_values)
            plt.title("Cross section y, tensor %d array %d %s " % (self.tensor, self.array, self.dset))
            plt.ylabel("Depth [m]")
            plt.xlabel("Pixel [m]")
            plt.ylim((0.6, 0.75))
            plt.savefig(self.output_path + "{0:05d}".format(self.tensor) + '_%d_ysection' % self.array)
            plt.close()

    def _get_crosssection(self):
        """Get the crosssection of the depth image ^= the depth values of the middle row
        and the middle column of the depth image"""
        x_values = []
        y_values = []
        for row_count, row in enumerate(self.depth_image):
            for column_count, value in enumerate(row):
                if row_count == len(row) / 2:
                    x_values.append(value[0])
                if column_count == len(row) / 2:
                    y_values.append(value[0])
        return x_values, y_values

    def _read_data(self):
        if self.single:
            tensor = 0
            array = 0
        else:
            tensor = self.tensor
            array = self.array
        tensor_format = "{0:05d}".format(tensor)
        self.depth_image = np.load(self.data_path + "depth_ims_tf_table_" + tensor_format + ".npz")['arr_0'][array]
        hand_pose = np.load(self.data_path + "hand_poses_" + tensor_format + ".npz")['arr_0'][array]
        self.grasp_depth = hand_pose[2]
        self.grasp_width = hand_pose[6]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise the cross section of depth images")
    parser.add_argument("tensor",
                        type=int,
                        default=None,
                        help="Tensor for depth image.")
    parser.add_argument("array",
                        type=int,
                        default=None,
                        help="Array position for depth image.")
    parser.add_argument("--single",
                        type=bool,
                        default=False,
                        help="Visualise cross section of single analysis.")
    parser.add_argument("--Cornell",
                        type=bool,
                        default=False,
                        help="Using the Cornell dataset.")
    parser.add_argument("--DexNet",
                        type=bool,
                        default=False,
                        help="Using the DexNet dataset.")
    args = parser.parse_args()
    tensor = args.tensor
    array = args.array
    single = args.single
    DexNet = args.DexNet
    Cornell = args.Cornell

    CrossSection(tensor, array, Cornell, DexNet, single)
