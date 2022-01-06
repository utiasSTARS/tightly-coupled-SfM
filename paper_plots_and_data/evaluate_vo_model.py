import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import validate
from train_mono import solve_pose, solve_pose_iteratively
from data.kitti_loader import process_sample_batch, process_sample
from models.dnet_layers import ScaleRecovery
import models.pose_models as pose_models
import models.depth_models as depth_models
from utils.learning_helpers import save_obj, load_obj, disp_to_depth, batch_post_process_disparity, load_ckp
from utils.custom_transforms import *
from vis import *
import os
from validate import compute_trajectory as tt
import glob

path_to_ws = '/path/to/tightly-coupled-SfM/' ##update this
path_to_dset_downsized = '/path/to/KITTI-odometry/preprocessed/'
model_dir = path_to_ws + 'results/kitti-odometry-4-iter'

load_from_mat = False #Make True to load paper results rather than recomputing
dnet_rescaling = True
gt_scaling = True ### scale translations to align with ground truth (note that both rescaling methods can be true)

frame_skip = 1 ## greater than 1 to increase image stride to simulate larger motion
new_iteration_num = None # None for same  as training

seq_list =['09_02', '10_02']
plot_axis=[0,2]
cam_height = 1.65
    
results_dir = model_dir + '/results/odometry/'
os.makedirs(results_dir, exist_ok=True)
logger = validate.ResultsLogger('{}/metrics.csv'.format(results_dir))
for seq in seq_list:
    print('sequence: {}'.format(seq))
    config = load_obj('{}/config'.format(model_dir))
    config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/' #
    
    config['load_stereo'] = False
    config['augment_motion'] = False
    config['augment_backwards'] = False
    config['img_per_sample']=2
    config['test_seq'] = [seq]
    config['minibatch'] = 1
    config['load_pretrained'] = True
    config['data_format'] = 'odometry'
    config['correction_rate'] = frame_skip
    
    if new_iteration_num is not None:
        config['iterations'] = new_iteration_num

    device=config['device']
        
    ### dataset and model loading    
    from data.kitti_loader_stereo import KittiLoaderPytorch
    data_list={'test': [seq]}
    test_dset = KittiLoaderPytorch(config, data_list, mode='test', transform_img=get_data_transforms(config)['test'])
    test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=6)
    if load_from_mat == False:
        depth_model = depth_models.depth_model(config).to(device)
        pose_model = pose_models.pose_model(config).to(device)
        pose_model, depth_model, _, _, _ = load_ckp(model_dir, True, pose_model, depth_model, None)
        pose_model.train(False).eval()
        depth_model.train(False).eval()    

    if dnet_rescaling == True:
        dgc = ScaleRecovery(config['minibatch'], 192, 640).to(device) 
 
    if load_from_mat == False:
        depth_model, pose_model = depth_model.train(False).eval(), pose_model.train(False).eval()  # Set model to evaluate mode
        fwd_pose_list1, inv_pose_list1 = [], []
        fwd_pose_list2, inv_pose_list2 = [], []
        vo_pose_list = []
        gt_list = []
        dnet_scale_factor_list = []
        
        with torch.no_grad():
            for k, data in enumerate(test_dset_loaders):
                target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, target_img_aug, \
                    source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = process_sample_batch(data, config)
                pose_results = {'source1': {}, 'source2': {} }

                batch_size = target_img.shape[0]
                imgs = torch.cat([target_img, source_img_list[0]],0)
                 
                disparities = depth_model(imgs, epoch=50)
                disparities, depths = disp_to_depth(disparities[0], config['min_depth'], config['max_depth'])

                target_disparities = disparities[0:batch_size]
                source_disp_1 = disparities[batch_size:(2*batch_size)]
                disparities = [target_disparities, source_disp_1]         
                depths = [1.0/disp for disp in disparities]

                if config['iterations']==1:
                    poses, poses_inv = solve_pose(pose_model, target_img, source_img_list, flow_imgs)                
                else:
                    poses, poses_inv = solve_pose_iteratively(config['iterations'], depths, pose_model, target_img, source_img_list, intrinsics)
                
                
                fwd_pose_vec1, inv_pose_vec1 = poses[0].clone(), poses_inv[0].clone()

                depth = 30*depths[0]
                fwd_pose_vec1[:,0:3] = 30*fwd_pose_vec1[:,0:3]
                inv_pose_vec1[:,0:3] = 30*inv_pose_vec1[:,0:3]
 
                if dnet_rescaling == True:
                    dnet_scale_factor = dgc(depth, intrinsics, cam_height)
                    dnet_scale_factor_list.append(dnet_scale_factor.cpu().numpy())

                fwd_pose_list1.append(fwd_pose_vec1.cpu().detach().numpy())
                inv_pose_list1.append(inv_pose_vec1.cpu().detach().numpy())
                gt_list.append(gt_lie_alg_list[0].cpu().numpy())
        

            fwd_pose_list1 = np.concatenate(fwd_pose_list1)
            inv_pose_list1 = np.concatenate(inv_pose_list1)
            gt_list = np.concatenate(gt_list)

            if dnet_rescaling == True:
                dnet_scale_factor_list = np.concatenate(dnet_scale_factor_list).reshape((-1,1))

        data = {'seq': config['test_seq'][0],
                'config': config,
                'fwd_pose_vec1': fwd_pose_list1,
                'inv_pose_vec1': inv_pose_list1,
                'gt_pose_vec': gt_list,
                'dnet_scale_factor': dnet_scale_factor_list,    
        }
        save_obj(data, '{}/{}_results'.format(results_dir, config['test_seq'][0]))

    else:
        data = load_obj('{}/{}_results'.format(results_dir, config['test_seq'][0]))

    gt_pose_vec = data['gt_pose_vec']
    unscaled_pose_vec = (data['fwd_pose_vec1'] - data['inv_pose_vec1'])/2
  
    if gt_scaling == True:
        gt_scale = np.mean(np.linalg.norm(gt_pose_vec[:,0:3],axis=1))/np.mean(np.linalg.norm(unscaled_pose_vec[:,0:3],axis=1))
        scaled_pose_vec_gt = np.array(unscaled_pose_vec)
        scaled_pose_vec_gt[:,0:3] = gt_scale*scaled_pose_vec_gt[:,0:3]
    
    if dnet_rescaling == True:
        print('ground plane scale factor (dnet): {}'.format(np.mean(data['dnet_scale_factor'])))
        print('ground plane std. dev. scale factor (dnet): {}'.format(np.std(data['dnet_scale_factor'])))
        scaled_pose_vec_dnet = np.array(unscaled_pose_vec)
        scaled_pose_vec_dnet[:,0:3] = scaled_pose_vec_dnet[:,0:3]*np.repeat(data['dnet_scale_factor'],3,axis=1)
        
        if gt_scaling == True:
            gt_scale = np.mean(np.linalg.norm(gt_pose_vec[:,0:3],axis=1))/np.mean(np.linalg.norm(scaled_pose_vec_dnet[:,0:3],axis=1))
            scaled_pose_vec_dnet[:,0:3] = gt_scale*scaled_pose_vec_dnet[:,0:3]

       
    ## Compute Trajectories
    gt_traj = test_dset_loaders.dataset.raw_gt_trials[0]
    orig_est, gt, errors, cum_dist = tt(unscaled_pose_vec,gt_traj,method='unscaled', compute_seg_err=True)
    logger.log(seq, 'unscaled', errors[0], errors[1], errors[2], errors[3])

    if dnet_rescaling == True:
        scaled_est_dnet, _, errors, _ = tt(scaled_pose_vec_dnet,gt_traj, method='scaled (dnet)', compute_seg_err=True)
        logger.log(seq, 'dnet scaled', errors[0], errors[1], errors[2], errors[3])
    
    if gt_scaling == True:    
        scaled_est_gt, _, errors, _ = tt(scaled_pose_vec_gt,gt_traj, method='scaled (gt)', compute_seg_err=True)
        logger.log(seq, 'gt scaled', errors[0], errors[1], errors[2], errors[3])    
    logger.log('', '', '', '', '', '')
    
    
    ## Plot trajectories
    plt.figure()
    plt.grid()
    plt.plot(gt[:,plot_axis[0],3], gt[:,plot_axis[1],3], linewidth=1.5, color='black', label='gt')
    plt.plot(orig_est[:,plot_axis[0],3],orig_est[:,plot_axis[1],3], linewidth=1.5, linestyle='--', label='est')
    if dnet_rescaling == True:
        plt.plot(scaled_est_dnet[:,plot_axis[0],3],scaled_est_dnet[:,plot_axis[1],3], linewidth=1.5, linestyle='--', label='rescaled est (DNet)')
    if gt_scaling == True:
        plt.plot(scaled_est_gt[:,plot_axis[0],3],scaled_est_gt[:,plot_axis[1],3], linewidth=1.5, linestyle='--', label='rescaled est')
    plt.legend()
    plt.title('Topdown (XY) Trajectory Seq. {}'.format(seq.replace('_','-')))
    plt.savefig('{}/seq-{}-topdown-scaled.png'.format(results_dir, seq))
