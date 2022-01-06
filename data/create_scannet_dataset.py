### based on DeepV2D loader https://github.com/princeton-vl/DeepV2D

import numpy as np
import scipy
import imageio
import os
import concurrent.futures
from PIL import Image
import argparse
from liegroups import SE3
import re
import pickle
import cv2
import glob

parser = argparse.ArgumentParser(description='')
parser.add_argument("--source_dir", type=str, default='/path/to/ScanNet/imgs')
parser.add_argument("--target_dir", type=str, default='/desired/path/to/ScanNet-preprocessed')
parser.add_argument("--frame_skip", type=int, default=6)
parser.add_argument("--img_height", type=int, default=256)
parser.add_argument("--img_width", type=int, default=448)
parser.add_argument("--remove_static", action='store_true', default=False)
args = parser.parse_args()


from scipy import interpolate

def load_depth(depth_file):
    img_height = args.img_height
    img_width = args.img_width 
    
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    depth = (depth/1000.0).astype(np.float32)
    depth = depth.astype(np.float32)
    
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]

    depth_filled = interpolate.griddata((xx, yy), zz.ravel(),
                                (x, y), method='nearest')

    resized_depth = cv2.resize(depth_filled, dsize=(img_width, img_height))
    return resized_depth

def load_image(img_file):
    img_height = args.img_height
    img_width = args.img_width 
    img = np.array(Image.open(img_file))
    orig_img_height = img.shape[0]
    orig_img_width = img.shape[1]
    zoom_y = img_height/orig_img_height
    zoom_x = img_width/orig_img_width
#    img = np.array(Image.fromarray(img).crop([425, 65, 801, 305]))
    img = np.array(Image.fromarray(img).resize((img_width, img_height), resample = Image.ANTIALIAS))
    return img, zoom_x, zoom_y, orig_img_width, orig_img_height

def load_scan(source_dir, scan):

    scan_path = os.path.join(source_dir, scan)
    # print(scan_path)
    imfiles = glob.glob(scan_path +'/**pose.txt')
    ixs = [im.replace('frame-','').replace('.pose.txt','') for im in imfiles]
    # print(imfiles[0])
    ixs = [os.path.basename(im) for im in ixs]
    ixs = sorted([int(im[:-1].lstrip('0')+im[-1]) for im in ixs])

    images = []
    img_list = sorted(glob.glob(scan_path+'/**color.jpg'))
    for i in ixs[::args.frame_skip]:
        images.append(img_list[i])

    # print(images)
    poses = []
    nan_idx_list = []
    for num, i in enumerate(ixs[::args.frame_skip]):
        posefile = os.path.join(scan_path, 'pose', '%d.txt' % i)
        pose = np.loadtxt(posefile, delimiter=' ').astype(np.float32) 
        if pose[0][0] == 'nan' or np.any(np.isnan(np.array(pose))) or pose[0][0]==-np.inf:
            print('found nan at {}'.format(num))
            nan_idx_list.append(num) 
        #poses.append(np.linalg.inv(pose)) # convert c2w->w2c
        poses.append(pose)

    depths = []
    depth_list = sorted(glob.glob(scan_path+'/**depth.pgm'))
    for i in ixs[::args.frame_skip]:
        depthfile = depth_list[i]
        depths.append(depthfile)

    color_intrinsics = np.loadtxt(os.path.join(scan_path, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')
    depth_intrinsics = np.loadtxt(os.path.join(scan_path, 'intrinsic', 'intrinsic_depth.txt'), delimiter=' ')

    datum = images, depths, poses, color_intrinsics, depth_intrinsics

    nan_idx_list.reverse()
    for idx in nan_idx_list:
        del(depths[idx])
        del(poses[idx])
        del(images[idx])
        print('deleting nan from', idx)
        
    print('any nan remaining?',np.any(np.isnan(np.array(poses))))
    print('any -inf remaining?', np.any(np.array(poses) == -np.inf))

    print(len(images), len(depths), len(poses), len(color_intrinsics), len(depth_intrinsics))
    return datum

for scan in sorted(os.listdir(args.source_dir)):
    scanid = int(re.findall(r'scene(.+?)_', scan)[0])
    
    if scanid > 660: ##only process data within training set (leaving test data separate)
        continue
    print(scan) 

    seq_info = {}
    filenames = []
    target_idx_list, source_idx_list = [], []
    seq_target_dir = '{}/med_res/{}'.format(args.target_dir, scan)
    os.makedirs(seq_target_dir, exist_ok=True)
    os.makedirs(os.path.join(seq_target_dir, 'color'), exist_ok=True)
    os.makedirs(os.path.join(seq_target_dir, 'depth'), exist_ok=True)
    
    images, depths, poses, color_intrinsics, depth_intrinsics = load_scan(args.source_dir, scan)
    gt_poses = np.array(poses)

    seq_info['intrinsics_left'] = np.array(color_intrinsics)[0:3, 0:3].reshape((1,3,3)).repeat(len(images),0)
    i = 0
    # with concurrent.futures.ProcessPoolExecutor() as executor: 
    #     for filename, output in zip(images, executor.map(load_image, images)):
    #         img, zoomx, zoomy, orig_img_width, orig_img_height = output
    #         # print(filename)
    #         filename_split = filename.split(args.source_dir)[1]
    #         new_filename = '{}/med_res/{}'.format(args.target_dir, filename_split.replace('/frame-','/color/frame-'))
    #         imageio.imwrite(new_filename, img)
    #         seq_info['intrinsics_left'][i,0] *= zoomx
    #         seq_info['intrinsics_left'][i,1] *= zoomy
    #         images[i] = np.array(new_filename)
    #         i+=1
    #     i = 0
        
    # seq_info['cam_02'] = np.array(images)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for filename, output in zip(depths, executor.map(load_depth, depths)):
            depth_img = output
            
            filename_split = filename.split(args.source_dir)[1]
            new_filename = '{}/med_res/{}'.format(args.target_dir, filename_split.replace('/frame-','/depth/frame-'))
            new_filename = new_filename.replace('.pgm', '.npz')
            np.savez_compressed(new_filename, depth_img)
    
    

    seq_info['sparse_gt_pose'] = gt_poses
    seq_info['sparse_vo'] = seq_info['sparse_gt_pose'] #use gt as a placeholder for now
    seq_info['ts'] = np.arange(0, len(gt_poses))
    
    print(seq_info['ts'].shape, seq_info['cam_02'].shape, seq_info['sparse_gt_pose'].shape, seq_info['intrinsics_left'].shape)
    

    with open('{}/med_res/{}/sequence_data'.format(args.target_dir, scan) + '.pkl', 'wb') as f:
        pickle.dump(seq_info, f, pickle.HIGHEST_PROTOCOL)
