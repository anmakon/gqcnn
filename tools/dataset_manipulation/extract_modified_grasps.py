import numpy as np
import csv
import os
import argparse
from tools.path_settings import DATA_PATH, EXPER_PATH


# Script to add noise or depth modifications to the datasets.
# You can also create sub-datasets by adding neither noise, nor depth
# modifications. Selection can be done randomly, randomly with excluding
# training data, manually or from csv files. 


class Modification():
    def __init__(self, selection, Cornell=False, counter=None, path=None, tensor=None, array=None):

        # "Preallocate" variables
        self.image_arr = []
        self.pose_arr = []
        self.file_arr = []
        self.metric_arr = []
        self.noise_arr = []
        self.depth_arr = []
        self.object_arr = []
        self.identifier_arr = []
        self.noise = False
        self.depth = False
        if counter is None:
            self.counter = 0
        else:
            self.counter = counter

        # Set paths
        if Cornell:
            self.identifier = 0
            self.export_path = "./data/training/Subset_datasets/Cornell_"
            self.data_path = "./data/training/Cornell/tensors/"
            split = "./data/training/Cornell/splits/image_wise/val_indices.npz"
            self.num_images = 3000

        else:
            self.identifier = 1
            self.export_path = DATA_PATH + "DexNet_Subsets/"
            self.data_path = DATA_PATH + "dexnet_2_tensor/tensors/"
            split = DATA_PATH + "dexnet_2_tensor/splits/image_wise/val_indices.npz"
            self.num_images = 500

        if path is not None:
            self.export_path = DATA_PATH + "DexNet_Subsets/" + path

        self.csv_dir = DATA_PATH + "/PerfectPredictions/data.csv"
        self.split = np.load(split)['arr_0']

        self.images_per_file = 500
        self.ratio_pos = 1.0

        # Set selection type
        self.random = False
        self.manual = False
        self.csv = False
        self.txt = False
        self.filter_training = True

        self.tensor = None
        self.array = None

        if selection == 'random' or selection == 'Random':
            self.random = True
        elif selection == 'manual' or selection == 'Manual':
            self.manual = True
            if tensor is not None and array is not None:
                self.tensor = tensor
                self.array = array
        elif selection == 'csv':
            self.csv = True
        elif selection == 'txt':
            self.txt = True
        else:
            raise ValueError("No selection type chosen.")

    def _save_files(self, counter):
        count_string = "{0:05d}".format(counter)
        np.savez(self.export_path + "depth_ims_tf_table_" + count_string, self.image_arr)
        np.savez(self.export_path + "hand_poses_" + count_string, self.pose_arr)
        np.savez(self.export_path + "robust_ferrari_canny_" + count_string, self.metric_arr)
        np.savez(self.export_path + "files_" + count_string, self.file_arr)
        np.savez(self.export_path + "object_labels_" + count_string, self.object_arr)
        np.savez(self.export_path + "identifier_" + count_string, self.identifier_arr)

        self.metric_arr = []
        self.image_arr = []
        self.pose_arr = []
        self.file_arr = []
        self.object_arr = []
        self.identifier_arr = []

        if self.noise:
            np.savez(self.export_path + "noise_and_tilting_" + count_string, self.noise_arr)
            self.noise_arr = []
        if self.depth:
            np.savez(self.export_path + "depth_info_" + count_string, self.depth_arr)
            self.depth_arr = []
        return None

    def _skip_grasp(self, robustness):
        if len(self.metric_arr) >= self.images_per_file * self.ratio_pos:
            if robustness >= 0.002:
                return True
        else:
            if robustness < 0.002:
                return True
        return False

    def _get_artificial_depth(self, depth_table, value):
        table = depth_table.max()
        closest_point = depth_table.min()
        depth = table - (table - closest_point) * value
        return depth

    def _read_txt_file(self):
        filenumber = []
        array = []
        filename = DATA_PATH + 'reprojections/generated_val_indices.txt'
        with open(filename, 'r') as txtfile:
            for row in txtfile:
                data = row.split(',')
                if 'Tensor' in data:
                    continue
                filenumber.append(int(data[0]))
                array.append(int(data[1]))
        return filenumber, array

    def _read_csv_file(self):
        filenumber = []
        array = []
        print("Possible files: ")
        [print(name) for name in os.listdir(self.csv_dir)]
        filename = input("Choose the csv file: ")
        if '.csv' in filename:
            path = self.csv_dir + filename
        else:
            path = self.csv_dir + filename + '.csv'
        if not os.path.isfile(path):
            raise ValueError(path + " is not a csv file. Check path")
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                row = [int(string) for string in row]
                if row[0] == row[2]:
                    images = 1 + row[3] - row[1]
                    filenumber.extend([row[0]] * images)
                    array.extend(list(range(row[1], row[3] + 1)))
                else:
                    for sep_file in list(range(row[0], row[2])):
                        if sep_file == row[0]:
                            images = self.num_images - row[1]
                            filenumber.extend([row[0]] * images)
                            array.extend(list(range(row[1], self.num_images)))
                        elif sep_file == row[2]:
                            images = row[3]
                            filenumber.extend([row[3]] * images)
                            array.extend(list(range(0, row[3])))
                        else:
                            filenumber.extend([sep_file] * self.num_images)
                            array.extend(list(range(0, self.num_images)))
        print("Read in all the images")
        self.csv_object = filename.split('_')[-1].split('.')[0]
        return filenumber, array

    def modify_noise(self):
        self.noise = True
        self.export_path += 'Noise/'
        self._choose_file()
        return None

    def modify_depth(self):
        self.depth = True
        self.export_path += 'Depth/'
        self._choose_file()
        return None

    def no_modification(self):
        self.export_path += '/'
        self._choose_file()
        return self.counter

    def _choose_file(self):
        if self.csv:
            tensors, arrays = self._read_csv_file()
            self.export_path = self.export_path[0:-1] + self.csv_object + '/'
        elif self.txt:
            tensors, arrays = self._read_txt_file()
        if not os.path.exists(self.export_path):
            os.mkdir(self.export_path)
        print("Save files to ", self.export_path)
        while True:
            if self.csv or self.txt:
                print(len(tensors), " images for saving")
                for cnt, tensor in enumerate(tensors):
                    # open and save each image
                    array = arrays[cnt]
                    self._add_modification(tensor, array)
                self._save_files(self.counter)
                print("Saved final file")
                return None
            if self.manual:
                if self.tensor is not None and self.array is not None:
                    tensor = self.tensor
                    array = self.array
                else:
                    tensor = int(input("Input the file number: "))
                    array = int(input("Input the array position: "))
            if self.random:
                if 'dexnet' in self.data_path:
                    array = np.random.randint(low=0, high=999)
                    tensor = np.random.randint(low=0, high=6728)
                elif 'Cornell' in self.data_path:
                    array = np.random.randint(low=0, high=499)
                    tensor = np.random.randint(low=0, high=15)
                else:
                    raise ValueError("Neither Cornell, nor dexnet in pathway. Abort")
                    break
                if self.filter_training and tensor * 1000 + array not in self.split:
                    continue
            filenumber = "{0:05d}".format(tensor)
            metrics = np.load(self.data_path + "robust_ferrari_canny_" + filenumber + ".npz")['arr_0'][array]
            if self.random and self._skip_grasp(metrics):
                continue
            # Store grasp, add modifications
            self._add_modification(tensor, array)
            if self.manual:
                if self.tensor is not None and self.array is not None:
                    self._save_files(self.counter)
                    return None
                elif input("Save file?") == 'y':
                    self._save_files(self.counter)
                    return None
            if self.random and self.num_images <= self.counter * self.images_per_file:
                if len(self.metric_arr) > 0:
                    self._save_files(self.counter)
                    print("Saved final file")
                return None

    def _add_modification(self, tensor, array):
        filenumber = "{0:05d}".format(tensor)
        try:
            depth_ims = np.load(self.data_path + "depth_ims_tf_table_" + filenumber + ".npz")['arr_0'][array]
        except:
            raise IndexError("Not available with filenumber %s, array %d" % (filenumber, array))
        pose = np.load(self.data_path + "hand_poses_" + filenumber + ".npz")['arr_0'][array]
        metrics = np.load(self.data_path + "robust_ferrari_canny_" + filenumber + ".npz")['arr_0'][array]
        object_label = np.load(self.data_path + "object_labels_" + filenumber + ".npz")['arr_0'][array]
        files = [tensor, array]
        if self.noise:
            for std in [0, 0.0011]:
                self.image_arr.append(np.random.normal(scale=std, size=(32, 32, 1)) + depth_ims)
                self.noise_arr.append([std, 0])
                self.object_arr.append(object_label)
                self.pose_arr.append(pose)
                self.metric_arr.append(metrics)
                self.file_arr.append(files)
                self.identifier_arr.append(self.identifier)
        elif self.depth:
            self.image_arr.append(depth_ims)
            self.metric_arr.append(metrics)
            self.file_arr.append(files)
            self.object_arr.append(object_label)
            self.pose_arr.append(pose.copy())
            self.depth_arr.append(-1)
            self.identifier_arr.append(self.identifier)
            for relation in [0, 0.5, 1.0]:
                relative_depth = self._get_artificial_depth(depth_ims, relation)
                pose[2] = relative_depth
                self.image_arr.append(depth_ims)
                self.metric_arr.append(metrics)
                self.file_arr.append(files)
                self.pose_arr.append(pose.copy())
                self.depth_arr.append(relation)
                self.identifier_arr.append(self.identifier)
        else:
            self.image_arr.append(depth_ims)
            self.pose_arr.append(pose)
            self.metric_arr.append(metrics)
            self.object_arr.append(object_label)
            self.file_arr.append(files)
            self.identifier_arr.append(self.identifier)

        if len(self.metric_arr) >= self.images_per_file:
            self._save_files(self.counter)
            print("Saved file #", self.counter)
            self.counter += 1
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add artificial depth or noise to DexNet datasets")
    parser.add_argument("--noise",
                        type=bool,
                        default=False,
                        help="Add noise to the images")
    parser.add_argument("--depth",
                        type=bool,
                        default=False,
                        help="Add artificial depth to the images")
    parser.add_argument("--selection",
                        type=str,
                        default='random',
                        help="Selection process. 'random', 'manual', 'txt' or 'csv' possible.")
    parser.add_argument("--mixture",
                        type=bool,
                        default=False,
                        help="Take Cornell and DexNet-2.0 dataset")
    parser.add_argument("--file",
                        type=int,
                        default=None,
                        help="File to take for manual selection")
    parser.add_argument("--array",
                        type=int,
                        default=None,
                        help="Array to take for manual selection")
    parser.add_argument("--Cornell",
                        type=bool,
                        default=False,
                        help="Take Cornell dataset. Default is DexNet-2.0")
    args = parser.parse_args()
    selection = args.selection
    cornell = args.Cornell
    mixture = args.mixture
    tensor = args.file
    array = args.array
    if mixture:
        # Get mixture of DexNet and Cornell data
        modifier = Modification(selection, True, path="Both")
        counter = modifier.no_modification()
        modifier = Modification(selection, False, counter=counter, path="Both")
        modifier.no_modification()
    else:
        modifier = Modification(selection, cornell, tensor=tensor, array=array)
        if args.noise:
            modifier.modify_noise()
        elif args.depth:
            modifier.modify_depth()
        else:
            modifier.no_modification()
