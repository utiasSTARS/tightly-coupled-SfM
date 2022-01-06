import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils.learning_helpers import save_obj, load_obj, disp_to_depth, load_ckp
from data.kitti_loader import process_sample_batch, process_sample
from data.scannet_test_loader import ScanNetLoader
from utils.learning_helpers import *
from utils.custom_transforms import *
from models.depth_models import depth_model
from models.pose_models import pose_model
from train_mono import solve_pose, solve_pose_iteratively
import os
import glob
import cv2
from liegroups.torch import SE3
from scannet_eval_utils import *


if __name__=='__main__':
    path_to_ws = '/path/to/tightly-coupled-SfM/' ##update this
    path_to_dset = '/path/to/ScanNet/imgs' ##update this
    model_dir = path_to_ws + 'results/scannet-4-iter'
    post_process = True #use the standard post-processing that flips images, recomputes depth, and merges with unflipped depth
    save_pred_disps = False
    load_pred_disps = False
    iterations = 8    
    
    
    test_split_path = '{}data/splits/scannet_test_img_list.txt'.format(path_to_ws)
    output_path = '{}/paper_plots_and_data/saved_scannet_predictions'.format(path_to_ws)

    config = load_obj('{}/config'.format(model_dir))
    config['minibatch'] = 6
    config['load_pretrained'] = True
    device = config['device']

    ### Load Models
    depth_model = depth_model(config).to(device)
    pose_model = pose_model(config).to(device)
    pose_model, depth_model, _, _, _ = load_ckp(model_dir, True, pose_model, depth_model, None)
    pose_model.train(False).eval()
    depth_model.train(False).eval()    

    ### Load data
    test_dset = ScanNetLoader(path_to_dset, test_split_path, transform_img=get_data_transforms(config)['test'])
    test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=8)

    ### initialize lists for storing gt and predictions
    depth_list = []
    gt_depths = []
    pred_disps = []
    pred_poses_fwd = []
    pred_poses_inv = []
    pred_poses_comb = []
    gt_poses = []

    ### Iterate through all test images, get predictions and load ground truth
    with torch.no_grad():
        if load_pred_disps == False:
            for k, data in enumerate(test_dset_loaders):
                target_img, _, _, _, _  = data
                target_depth_gt = target_img['depth_gt'].numpy()
                # depths_gt = target_img['depth'].type(torch.FloatTensor).to(device)
                # depths_gt = [depths_gt[:,d].unsqueeze(1) for d in range(0,depths_gt.shape[1])]
                
                target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, target_img_aug, \
                    source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = process_sample_batch(data, config)
                gt_lie_alg = gt_lie_alg_list[0].cpu()
                B = target_img.shape[0]
                
                imgs = torch.cat([target_img] + [source_img_list[0]],0)
                
                if post_process == True:
                    # Post-processed results require each image to have two forward passes
                    imgs = torch.cat((imgs, torch.flip(imgs, [3])), 0) 
                
                disparities = depth_model(imgs, epoch=50)
                disparities, depths = disp_to_depth(disparities[0], config['min_depth'], config['max_depth'])
                disparities = disparities.cpu()[:, 0].numpy()
                if post_process == True:
                    N = disparities.shape[0] // 2
                    disparities = batch_post_process_disparity(disparities[:N], disparities[N:, :, ::-1])    
                    # intrinsics = intrinsics[:N]

                disparities = torch.from_numpy(disparities).unsqueeze(1).type(torch.FloatTensor).to(device)

                target_disparities = disparities[0:B]
                source_disp_1 = disparities[B:(2*B)]
                disparities = [target_disparities, source_disp_1]         
                depths = [1.0/disp for disp in disparities]    

                if iterations==1:
                    poses, poses_inv = solve_pose(pose_model, target_img, [source_img_list[0]], flow_imgs)
                else:
                    poses, poses_inv = solve_pose_iteratively(iterations, depths, pose_model, target_img, [source_img_list[0]], intrinsics)
                fwd_pose_vec, inv_pose_vec = poses[0].clone(), poses_inv[0].clone()                
                
                target_depth = 30*depths[0][:,0].cpu().numpy()
                fwd_pose_vec[:,0:3] = 30*fwd_pose_vec[:,0:3]
                inv_pose_vec[:,0:3] = 30*inv_pose_vec[:,0:3]
                depth_list.append(target_depth)
                pred_disps.append(disparities[0][:,0].cpu().numpy())
                pred_poses_fwd.append(SE3.exp(fwd_pose_vec.cpu()).as_matrix().numpy())
                pred_poses_inv.append(SE3.exp(-inv_pose_vec.cpu()).as_matrix().numpy())
                pred_poses_comb.append(SE3.exp( (fwd_pose_vec.cpu() - inv_pose_vec.cpu())/2 ).as_matrix().numpy())
                
                gt_depths.append(target_depth_gt)
                gt_poses.append(SE3.exp(gt_lie_alg).as_matrix().numpy())
                

            depth_list = np.concatenate(depth_list)
            pred_disps = np.concatenate(pred_disps)
            pred_poses_fwd = np.concatenate(pred_poses_fwd)
            pred_poses_inv = np.concatenate(pred_poses_inv)
            pred_poses_comb = np.concatenate(pred_poses_comb)
            gt_depths = np.concatenate(gt_depths)
            gt_poses = np.concatenate(gt_poses)

            if save_pred_disps:
                print("-> Saving predicted disparities to ", output_path)
                np.save('{}/disps_scannet_split.npy'.format(output_path), pred_disps) ##annoying filename to be compatible with opt script
                np.save('{}/gt_depths.npy'.format(output_path), gt_depths) ## only save this here, not in opt script
                np.save('{}/gt_pose.npy'.format(output_path),gt_poses) ## only save this here, not in opt script
                np.save('{}/pred_pose_fwd.npy'.format(output_path),pred_poses_fwd)
                np.save('{}/pred_pose_inv.npy'.format(output_path),pred_poses_inv)
                np.save('{}/pred_pose_comb.npy'.format(output_path),pred_poses_comb)
                
        else:
            # Load predictions from file
            print("-> Loading predictions")
            pred_disps = np.load('{}/disps_scannet_split.npy'.format(output_path))
            gt_depths = np.load('{}/gt_depths.npy'.format(output_path))
            gt_poses = np.load('{}/gt_pose.npy'.format(output_path))
            pred_poses_fwd = np.load('{}/pred_pose_fwd.npy'.format(output_path))
            pred_poses_inv = np.load('{}/pred_pose_inv.npy'.format(output_path))
            pred_poses_comb = np.load('{}/pred_pose_comb.npy'.format(output_path))
            
            from liegroups import SE3
            pred_poses_comb = np.array([SE3.exp(p).as_matrix() for p in pred_poses_comb])
            pred_poses_fwd = np.array([SE3.exp(p).as_matrix() for p in pred_poses_fwd])
            pred_poses_inv = np.array([SE3.exp(p).as_matrix() for p in pred_poses_inv])


    pose_results = {}
    depth_results = {}
    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        gt_pose = gt_poses[i]
        
        pred_pose = pred_poses_comb[i]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 30 / pred_disp
        
        scalar = compute_scaling_factor(gt_depth, pred_depth)
        pred_depth = scalar * pred_depth
        
        depth_metrics = compute_depth_errors(gt_depth, pred_depth)
        pose_metrics = compute_pose_errors(gt_pose, pred_pose)

        if i == 0:
            for pkey in pose_metrics:
                pose_results[pkey] = []
            for dkey in depth_metrics:
                depth_results[dkey] = []

        for pkey in pose_metrics:
            pose_results[pkey].append(pose_metrics[pkey])

        for dkey in depth_metrics:
            depth_results[dkey].append(depth_metrics[dkey])


    ### aggregate metrics
    for pkey in pose_results:
        pose_results[pkey] = np.mean(pose_results[pkey])

    for dkey in depth_results:
        depth_results[dkey] = np.mean(depth_results[dkey])

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "Rot (deg)", "Tr (deg)", "Tr(cm)"))
    print(("{:10.4f} & "*len(depth_results)).format(*depth_results.values()), ("{:16.4f} & "*len(pose_results)).format(*pose_results.values()))

