import numpy as np
import matplotlib.pyplot as plt

# 
# Generate histograms that show the grasp height in Cornell 
# Histograms are saved in the current directory.
#

data_path = "./data/training/dexnet_2_tensor/tensors/"
save_path = "../../Desktop/DexNet_Graspdepth/"
grasp_dist_table = []
grasp_dist_max = []
grasp_dist_cent = []
grasp_dist_relative = []

for i in range(0,6700):
	filecnt = ('{0:05d}').format(i)
	hand_pose = np.load(data_path+'hand_poses_'+filecnt+'.npz')['arr_0']
	depth_im_table = np.load(data_path+'depth_ims_tf_table_'+filecnt+'.npz')['arr_0']
	for j in range(0,len(depth_im_table)):
		grasp_table = np.max(depth_im_table[j])
		grasp_max = np.min(depth_im_table[j])
		grasp_depth = hand_pose[j][2]
		grasp_cent = depth_im_table[j][16,16][0]
		grasp_dist_table.append(grasp_table-grasp_depth)
		grasp_dist_max.append(grasp_depth-grasp_max)
		grasp_dist_cent.append(grasp_cent-grasp_depth)
		if (grasp_table-grasp_max) >0.001:
			grasp_dist_relative.append((grasp_table-grasp_depth)/(grasp_table-grasp_max))

print("Start plotting")
print("Max relative: ",max(grasp_dist_relative))
print("Min relative: ",min(grasp_dist_relative))
ymax = 285000
plt.hist(grasp_dist_relative,range=(0,1.3),bins=60)
plt.xlim(0,1.3)
plt.ylim(0,ymax)
plt.xlabel("(Max(image)-grasp_depth)/(Max(image)-Min(image))")
plt.title("Relative distance table to grasp height in DexNet")
plt.savefig(save_path+"Grasp_dist_relative.png")
plt.close()
binwidth=0.001
plt.hist(grasp_dist_table,range=(0,0.06),bins=np.arange(min(grasp_dist_table),max(grasp_dist_table)+binwidth,binwidth))
plt.xlim(0,0.06)
plt.ylim(0,ymax)
plt.xlabel("Max(image)-grasp_depth")
plt.title("Distance table to grasp height in DexNet")
plt.savefig(save_path+"Grasp_dist_table.png")
plt.close()
plt.hist(grasp_dist_max,range=(-0.01,0.05),bins=np.arange(min(grasp_dist_max),max(grasp_dist_max)+binwidth,binwidth))
plt.xlim(-0.01,0.05)
plt.ylim(0,ymax)
plt.xlabel("Grasp_depth-Min(image)")
plt.title("Distance grasp height to maximum height in DexNet")
plt.savefig(save_path+"Grasp_dist_maximum.png")
plt.close()
plt.hist(grasp_dist_cent,range=(-0.04,0.02),bins=np.arange(min(grasp_dist_cent),max(grasp_dist_cent)+binwidth,binwidth))
plt.xlim(-0.04,0.02)
plt.ylim(0,ymax)
plt.xlabel("Depth(image_center)-grasp_depth")
plt.title("Distance grasp height to grasp center height in DexNet")
plt.savefig(save_path+"Grasp_dist_center.png")

