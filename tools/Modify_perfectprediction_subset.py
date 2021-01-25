import numpy as np
from PIL import Image
import os
from path_settings import DATA_PATH, EXPER_PATH

data_dir = DATA_PATH + "reprojections/20210120/tensors/"
output_dir = EXPER_PATH + "DexNet_Subsets/PerfectPredictions_modified/"


def scale(data, xmin, xmax, fwd=True):
    """Scale a depth image in order to visualise it as a greyscale
    image. Fixed scaling from [0.6,0.75] to [0,255]."""
    data_fl = data.flatten()
    if fwd:
        scaled = np.interp(data_fl, (xmin, xmax), (0, 255))
        result = scaled.astype(np.uint8)
        result.resize((32, 32))
    else:
        scaled = np.interp(data_fl, (0, 255), (xmin, xmax))
        result = scaled.astype(np.float)
        result.resize((32, 32))
    return result


if __name__ == "__main__":
    file_pointers = np.load(data_dir + "files_00000.npz")["arr_0"]
    hand_poses = []
    identifier = []
    object_labels = []
    metrics = []
    new_images = []
    pointers = []
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for i, pointer in enumerate(file_pointers):
        print("%s - %s" % (i, pointer))
    m = int(input("Select number for file to choose: "))
    new_pointer = file_pointers[m]
    images = np.load(data_dir + "depth_ims_tf_table_00000.npz")['arr_0']
    im_data = images[m]
    hand_pose = np.load(data_dir + "hand_poses_00000.npz")["arr_0"][m]
    # id = np.load(data_dir + "identifier_00000.npz")["arr_0"][m]
    object_label = np.load(data_dir + "obj_labels_00000.npz")["arr_0"][m]
    metric = np.load(data_dir + "robust_ferrari_canny_00000.npz")["arr_0"][m]
    minimum = np.amin(im_data)
    maximum = np.amax(im_data)
    scaled_im = scale(im_data, minimum, maximum)
    i = 0
    while True:
        Image.fromarray(scaled_im).save(output_dir + "image.png")
        input()
        new_im = Image.open(output_dir + "image.png")
        new_im_data = scale(np.asarray(new_im), minimum, maximum, fwd=False).reshape((32, 32, 1))
        new_images.append(new_im_data)
        hand_poses.append(hand_pose)
        # identifier.append(id)
        object_labels.append(object_label)
        metrics.append(metric)
        pointers.append([new_pointer[0], "%03d".format(i)])
        i += 1
        if input("Next modification [y/n]?") != 'y':
            break
    np.savez(output_dir + "depth_ims_tf_table_00000", new_images)
    np.savez(output_dir + "hand_poses_00000", hand_poses)
    # np.savez(output_dir + "identifier_00000", identifier)
    np.savez(output_dir + "object_labels_00000", object_labels)
    np.savez(output_dir + "robust_ferrari_canny_00000", metrics)
    np.savez(output_dir + "files_00000", pointers)
