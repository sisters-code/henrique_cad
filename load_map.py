import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def normalize(map):
    rescaled_map = map - np.amin(map)
    rescaled_map = rescaled_map / np.amax(map)
    return rescaled_map

np_file = 'valid_mask_list.npy'
mask_list = np.load(np_file)

n = mask_list.shape[0]
pos_map = np.zeros((224, 224))
neg_map = np.zeros((224, 224))
for i in range(n):
    mask = mask_list[i,:,:]
    pos_map[np.where(mask == 1)] += 1
    neg_map[np.where(mask == -1)] += 1
pos_map = normalize(pos_map)
neg_map = normalize(neg_map)
gray_img = np.uint8(255 * pos_map)
pos_heatmap = cv2.applyColorMap(np.uint8(255 * pos_map), cv2.COLORMAP_JET)
neg_heatmap = cv2.applyColorMap(np.uint8(255 * neg_map), cv2.COLORMAP_JET)
# plt.imsave( args.face_part_names[0] +'train_pos_map.jpg', pos_heatmap)
# plt.imsave( args.face_part_names[0] +'train_neg_map.jpg', neg_heatmap)

dir = 'lime'
os.makedirs(dir, exist_ok=True)
cv2.imwrite(os.path.join(dir, np_file.split('_')[0] +'_pos_map.jpg'), pos_heatmap)
cv2.imwrite(os.path.join(dir, np_file.split('_')[0] +'_neg_map.jpg'), neg_heatmap)
# plt.imsave( os.path.join(dir, np_file.split('_')[0] +'_pos_map.jpg'), pos_heatmap)
# plt.imsave( os.path.join(dir, np_file.split('_')[0] +'_neg_map.jpg'), neg_heatmap)