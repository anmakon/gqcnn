import numpy as np
import perception
from perception import(BinaryImage,CameraIntrinsics,ColorImage,DepthImage,RgbdImage)

Cornell = True
if Cornell:
	array = 72
	dset = 'Cornell'
	tensor = '00000'
else:
	array = 650
	dset = 'dexnet_2_tensor'
	tensor = '02536'


filename = "/media/psf/Home/Documents/Grasping_Research/docker/pcb_"+dset+"_"+tensor+"_"+str(array)+".pcd"
depth_data = np.load("./data/training/"+dset+"/tensors/depth_ims_tf_table_"+tensor+".npz")['arr_0'][array]
camera_intr = CameraIntrinsics.load("./data/calib/primesense/primesense.intr")
#camera_intr = CameraIntrinsics.load("./data/calib/phoxi/phoxi.intr")
depth_im = DepthImage(depth_data,frame=camera_intr.frame)

# Crop and resize camera intrinsics as in DexNet dataset generation ln 439 in tool/generate_gqcnn_dataset.py

camera_intr_scale = 32.0/96.0
cropped_camera_intr = camera_intr.crop(32,32,16,16)

final_camera_intr = cropped_camera_intr.resize(camera_intr_scale)
final_camera_intr.cx = 16
final_camera_intr.cy = 16
print(final_camera_intr.cx)

PC = final_camera_intr.deproject(depth_im)
print(min(PC.data[2]))
print(max(PC.data[2]))
data = []
for cnt,x in enumerate(PC.data[0]):
	data.append([PC.data[0,cnt],PC.data[1,cnt],PC.data[2,cnt]])
pcd_header = '''# .PCD v.7 - Point Cloud Data file format
	FIELDS x y z
	SIZE 4 4 4
	TYPE F F F
	COUNT 1 1 1
	WIDTH 1024
	HEIGHT 1
	VIEWPOINT 0 0 0 1 0 0 0
	POINTS 1024
	DATA ascii
	'''
with open(filename, 'w') as f:
	f.write(pcd_header)
	np.savetxt(f,data,'%f %f %f')
