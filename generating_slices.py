import numpy as np
from skimage.transform import resize
# Taking slices from 3D volume
def scalenorm(data):
   if abs(np.amin(data)) > np.amax(data):
       max_range_value = abs(np.amin(data))
       min_range_value = np.amin(data)
   else:
       max_range_value = np.amax(data)
       min_range_value = -1.0 * np.amax(data)
   return 2.0 * data / (max_range_value - min_range_value)

def inline_slicing(s, gt, flag):
   slice_size = (512, 512)
   input_imgs = np.zeros((s.shape[0],) + slice_size, dtype="float32")
   gt_imgs = np.zeros((gt.shape[0],) + slice_size, dtype="int64")
   labels = np.array([flag] * s.shape[0], dtype="int64")
   isinline = np.zeros((s.shape[0],), dtype="int64")  # 0 for inline
   slice_index = np.zeros((s.shape[0],), dtype="int64")
   for i in range(s.shape[0]):
       slice_gt_i = resize(gt[i, :, :].transpose(), (512, 512))
       slice_s_i = resize(s[i, :, :].transpose(), (512, 512))
       input_imgs[i] = (slice_s_i + 1) / 2.0
       gt_imgs[i] = slice_gt_i
       slice_index[i] = i
   input_imgs = input_imgs.reshape(s.shape[0], 512, 512, 1).astype("float32")
   gt_imgs = gt_imgs.reshape(gt.shape[0], 512, 512, 1).astype("int64")
   return input_imgs, gt_imgs, labels, isinline, slice_index

def crossline_slicing(s, gt, flag):
   slice_size = (512, 512)
   input_imgs = np.zeros((s.shape[1],) + slice_size, dtype="float32")
   gt_imgs = np.zeros((gt.shape[1],) + slice_size, dtype="int64")
   labels = np.array([flag] * s.shape[1], dtype="int64")
   iscrossline = np.ones((s.shape[1],), dtype="int64")  # 1 for crossline
   slice_index = np.zeros((s.shape[1],), dtype="int64")
   for i in range(s.shape[1]):
       slice_gt_i = resize(gt[:, i, :].transpose(), (512, 512))
       slice_s_i = resize(s[:, i, :].transpose(), (512, 512))
       input_imgs[i] = (slice_s_i + 1) / 2.0
       gt_imgs[i] = slice_gt_i
       slice_index[i] = i
   input_imgs = input_imgs.reshape(s.shape[1], 512, 512, 1).astype("float32")
   gt_imgs = gt_imgs.reshape(gt.shape[1], 512, 512, 1).astype("int64")
   return input_imgs, gt_imgs, labels, iscrossline, slice_index

def both_inline_crossline(s, gt, flag):
   in_imgs, in_gt, in_labels, in_flag, in_idx = inline_slicing(s, gt, flag)
   cr_imgs, cr_gt, cr_labels, cr_flag, cr_idx = crossline_slicing(s, gt, flag)
   train_input_imgs = np.concatenate((in_imgs, cr_imgs), axis=0)
   train_gt_imgs = np.concatenate((in_gt, cr_gt), axis=0)
   train_labels = np.concatenate((in_labels, cr_labels), axis=0)
   in_or_cross = np.concatenate((in_flag, cr_flag), axis=0)
   slice_index = np.concatenate((in_idx, cr_idx), axis=0)
   return train_input_imgs, train_gt_imgs, train_labels, in_or_cross, slice_index