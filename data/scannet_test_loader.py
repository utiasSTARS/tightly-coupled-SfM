#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
import cv2
import re
import csv
import glob
import random
import pickle
from PIL import Image
from liegroups import SE3
from scipy import interpolate


class ScanNetLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_path, test_split_path, num_frames=3, transform_img=None):
        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.orig_height = 968
        self.orig_width = 1296
        self.height = 256
        self.width = 448
        self.transform_img = transform_img

        self.test_frames = np.loadtxt(test_split_path, dtype=np.unicode_)
        self.test_data = []

        for i in range(0, len(self.test_frames), 4): #4 lines per test sequence (2 imgs, 2 depth)
            test_frame_1 = str(self.test_frames[i]).split('/')
            test_frame_2 = str(self.test_frames[i+1]).split('/')
            scan = test_frame_1[3]

            imageid_1 = int(re.findall(r'frame-(.+?).color.jpg', test_frame_1[-1])[0])
            imageid_2 = int(re.findall(r'frame-(.+?).color.jpg', test_frame_2[-1])[0])            
            self.test_data.append((scan, imageid_1, imageid_2))

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx, s=8):

        # for (scanid, imageid_1, imageid_2) in test_data:

        scanid, imageid_1, imageid_2 = self.test_data[idx]
        scandir = os.path.join(self.dataset_path, scanid)
        num_frames = len(os.listdir(os.path.join(scandir, 'pose'))) #this num_frames refers to the length of the scanid

        images = []

        id_list = [imageid_1, imageid_2]

        for i in range(0,self.num_frames-2): #add to the two original images until we have num_frames images
            other_id = max(imageid_1 - (1+i)*s,0) ## grab frames before target image, since we already have a future image with imageid2
            
            ### if imageid_1 is the first frame in the sequence, we should sample in the other direction
            if other_id == imageid_1:
                other_id = imageid_2 + (1+i)*s
        
            id_list.append(other_id)
        
        ### load poses before changing id_list to string with leading zeros
        poses = []
        for id in id_list:
            pose = np.loadtxt(os.path.join(scandir, 'pose', '%d.txt'%id), delimiter=' ')
            poses.append(pose)
        poses = np.array(poses)

        # pose2 = np.loadtxt(os.path.join(scandir, 'pose', '%d.txt'%imageid_2), delimiter=' ')
        # pose1 = np.linalg.inv(pose1)
        # pose2 = np.linalg.inv(pose2)
        # pose_gt = np.dot(pose2, np.linalg.inv(pose1))            
        
        
        id_list = [str(id).zfill(6) for id in id_list]
        
        ### load images, resize, and then update intrinsics
        imgs = []
        img_filenames = []
        for id in id_list:
            image_file = os.path.join(scandir, 'frame-{}.color.jpg'.format(id))
            img, zoom_x, zoom_y, orig_img_width, orig_img_height = self.load_image(image_file)
            imgs.append(img)
            img_filenames.append(image_file)

        target_idx = 0 #first image is the one we're concerned with (that's the depth map we're evaluating, and using pose change from 0 to 1)
        source_idx_list = [i for i in range(1,self.num_frames)]

        ## depth ground truth only needed for the first frame of the sample
        
        depth_file = os.path.join(scandir, 'frame-{}.depth.pgm'.format(id_list[0]))
        depth_gt = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        depth_gt = (depth_gt/1000.0).astype(np.float32)
        depth_gt = depth_gt.astype(np.float32)
        
        # depths = []
        # for id in id_list:
        #     depth_file = os.path.join(scandir, 'frame-{}.depth.pgm'.format(id))
        #     depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        #     depth = (depth/1000.0).astype(np.float32)
        #     depth = depth.astype(np.float32)
        #     depth = self.fill_depth(depth)
        #     depths.append(depth.reshape((1,480, 640)))

        # depths = np.concatenate(depths)
        

        K = os.path.join(scandir, 'intrinsic/intrinsic_color.txt')
        K = np.loadtxt(K, delimiter=' ')
        K[0]*=zoom_x
        K[1]*=zoom_y
        
        K = K[0:3, 0:3]
        intrinsics = K.reshape(1,3,3).repeat(3,axis=0)

        ### format stuff the way we need for our standard dataloader with transforms:
        lie_alg = []
        transformed_lie_alg = []
        for i in range(0,len(source_idx_list)):
            lie_alg.append(list(self.compute_target(poses, target_idx, source_idx_list[i])))    
            transformed_lie_alg.append(list(self.compute_target(poses, target_idx, source_idx_list[i])))   

        if self.transform_img != None:
            orig, transformed = self.transform_img((imgs, intrinsics, lie_alg), (imgs, intrinsics, transformed_lie_alg)) 
            orig_imgs, orig_intrinsics, orig_lie_alg = orig
            transformed_imgs, transformed_intrinsics, transformed_lie_alg = transformed  
        else:
            orig_imgs, orig_intrinsics, orig_lie_alg = imgs, intrinsics, lie_alg
            transformed_imgs, transformed_intrinsics, transformed_lie_alg = imgs, intrinsics, lie_alg
      
        flow_imgs_fwd, flow_imgs_fwd_list = [], []
        flow_imgs_back, flow_imgs_back_list = [], []

        # target_im = {'color_left': orig_imgs[target_idx], 'color_aug_left': transformed_imgs[target_idx], 'depth': depths, 'depth_gt': depth_gt, 'filenames':img_filenames}
        target_im = {'color_left': orig_imgs[target_idx], 'color_aug_left': transformed_imgs[target_idx], 'filenames':img_filenames, 'depth_gt': depth_gt}
        source_imgs = {'color_left': [orig_imgs[i] for i in source_idx_list], 'color_aug_left': [transformed_imgs[i] for i in source_idx_list] }
        intrinsics = {'color_left': orig_intrinsics, 'color_aug_left': transformed_intrinsics}
        lie_alg = {'color': orig_lie_alg, 'color_aug': transformed_lie_alg}        

        
        return target_im, source_imgs, lie_alg, intrinsics, (flow_imgs_fwd, flow_imgs_back)
  
    
    def fill_depth(self, depth):
        x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                           np.arange(depth.shape[0]).astype("float32"))
        xx = x[depth > 0]
        yy = y[depth > 0]
        zz = depth[depth > 0]
    
        grid = interpolate.griddata((xx, yy), zz.ravel(),
                                    (x, y), method='nearest')
        return grid
    

    def load_image(self, img_file):
        img_height = self.height 
        img_width = self.width 

        img = np.array(Image.open(img_file))
        orig_img_height = img.shape[0]
        orig_img_width = img.shape[1]

        zoom_y = img_height/orig_img_height
        zoom_x = img_width/orig_img_width

    #    img = np.array(Image.fromarray(img).crop([425, 65, 801, 305]))
        img = np.array(Image.fromarray(img).resize((img_width, img_height), resample = Image.ANTIALIAS))
        return img, zoom_x, zoom_y, orig_img_width, orig_img_height    
    
    def compute_target(self, poses, target_idx, source_idx):
        T2 = SE3.from_matrix(poses[target_idx,:,:],normalize=True).inv()
        T1 = SE3.from_matrix(poses[source_idx,:,:],normalize=True)
        dT_gt = T2.dot(T1) #pose change from source to a target (for reconstructing source from target)
        gt_lie_alg = dT_gt.log()
        
        vo_lie_alg = np.copy(gt_lie_alg)
        gt_correction = np.zeros(gt_lie_alg.shape)
        dt = 0

        return gt_lie_alg, vo_lie_alg, gt_correction, dt

# dset = ScanNetLoader('imgs')
# target_im, source_imgs, lie_alg, intrinsics, (flow_imgs_fwd, flow_imgs_back) = dset.__getitem__(0)
# print(lie_alg['color'][0][0])
# print(lie_alg['color'][1][0])