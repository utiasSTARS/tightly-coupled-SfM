import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils.learning_helpers import save_obj, load_obj, load_ckp
import time 
from utils.custom_transforms import get_data_transforms
import os
from vis import *
from data.kitti_loader import process_sample, process_sample_batch
from models.depth_w_access import depth_model
from models.pose_models import pose_model

from optimizer import DepthOptimizer
from validate import compute_trajectory as tt

torch.cuda.set_device(1)
if __name__ == "__main__":
    ### general data loading options
    path_to_ws = '/path/to/tight-coupled-SfM/' ##update this
    data_format = 'odometry' ## eigen or odometry or scannet
    load_from_mat = False
    
    if data_format == 'odometry':
        from data.kitti_loader_stereo import KittiLoaderPytorch
        path_to_dset_downsized = '/path/to/KITTI-odometry-downsized-stereo/' ##update this
        saved_model_dir = 'results/kitti-odometry-4-iter'
        seq_list = ['09_02','10_02']
        plot_axis = [0,2]
        save_predictions = False ## only save disparity predictions if evaluating on the eigen split
        cam_height=1.65
        num_source_imgs = 2
        scaling_mode = 'unscaled' ## use DNet rescaling to align with metric ground truth.
        extra_egomotion_iters = 0
        predictions_save_path = ''
    
    if data_format == 'eigen':
        from data.kitti_loader_eigen import KittiLoaderPytorch
        path_to_dset_downsized = '/path/to/KITTI-eigen-split/' ##update this
        saved_model_dir = 'results/kitti-eigen-4-iter'
        seq_list = ['eigen'] #eigen or eigen_benchmark
        plot_axis = [0,2]
        save_predictions = True 
        cam_height=1.65
        num_source_imgs = 2
        scaling_mode = 'unscaled' ## use DNet rescaling to align with metric ground truth.
        extra_egomotion_iters = 0
        predictions_save_path = '{}/paper_plots_and_data/saved_eigen_predictions'.format(path_to_ws)
    
    if data_format == 'scannet':
        from data.scannet_test_loader import ScanNetLoader
        path_to_dset_downsized = '/path/to/ScanNet/imgs' ##update this
        test_split_path = '{}data/splits/scannet_test_img_list.txt'.format(path_to_ws)
        saved_model_dir = 'results/scannet-4-iter'
        seq_list = ['scannet']
        plot_axis = [0,2]
        save_predictions = True 
        cam_height=1
        num_source_imgs = 1
        scaling_mode = 'scaled'
        extra_egomotion_iters = 4 ## use more to account for the 6-dof motion
        predictions_save_path = '{}/paper_plots_and_data/saved_scannet_predictions'.format(path_to_ws)
        
    
    ### Optimization Options
    options= {'epochs':20,
        'optimizer': 'adam',
        'optimize_depth_weights_bottleneck_beyond': False,
        'optimize_depth_weights_all': False,
        'optimize_depth_encoder': True,
        'optimize_pose_weights_all': False,
        'optimize_depth_pred': False, 
        'optimize_pose_pred': False, ##NOT IMPLEMENTED
        'optimize_depth_bottleneck_values': False, 
        'optimize_depth_skip_connections': False, ##NOT IMPLEMENTED                    
        'diff_img_argmin': True,
        'automasking': True,
        'mode': scaling_mode, # use unscaled if you want to perform online rescaling w camera height and ground plane (KITTI)
        'l_depth_consist': True,
        'l_depth_consist_weight': 0.15,
        'l_depth_init': True, ## optimized depth shouldn't be too different from starting depth
        'l_depth_init_weight': 0.1,
        'l_inverse_reconstruction': True,
        'l_smooth': False,
        'l_smooth_weight': 2,
        'l_scale_factor': False, ## NOT IMPLEMENTED
        'l_pose_consist': False,
        'avg_final_epochs': 5, #averages the poses for final N epochs of optimization (leave as 1 for no averaging)
        'num_source_imgs': num_source_imgs, ## of images we project to the (middle) target image
        'extra_egomotion_iterations': extra_egomotion_iters, #0 for num. of iterations used during training
        'minibatch': 6,  #perform optimization over N independent minibatches
        'lr':2e-4, #2e-4
        'plotting':False, #don't plot for sequential mode
        'save_predictions': save_predictions, 
        'predictions_save_path': predictions_save_path,
    }

    dir = path_to_ws + saved_model_dir 
    options['results_dir'] = dir + '/results/depth_opt_sequential/'
    os.makedirs(options['results_dir'], exist_ok=True)
    config = load_obj('{}/config'.format(dir))
    config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/' #   

         ## Ensure that config file is set up properly (turn off options that may have been used during training)
    config['skip'] = options['num_source_imgs']
    config['img_per_sample'] = options['num_source_imgs'] + 1
    config['load_stereo'] = False
    config['augment_motion'] = False
    config['augment_backwards'] = False
    config['minibatch'] = options['minibatch']
    config['load_pretrained'] = True
    config['data_format'] = data_format
    config['camera_height'] = cam_height
    config['correction_rate'] = 1
    config['iterations'] = config['iterations'] + options['extra_egomotion_iterations']
    device = config['device']

    ''' Load Models'''   
    depth_model = depth_model(config).to(device)
    pose_model = pose_model(config).to(device)
    pose_model, depth_model, _, _, _ = load_ckp(path_to_ws+saved_model_dir, True, pose_model, depth_model, None)
    depth_model.train(False).eval()
    pose_model.train(False).eval()


    ### various freezing/unfreezing of weights for the desired optimization mode
    if options['optimize_depth_weights_bottleneck_beyond'] == True:
        for param in depth_model.parameters():#.encoder.parameters():
            param.requires_grad = False

        for param in depth_model.feature_convs.parameters():
            param.requires_grad = True

        for param in depth_model.depth_upconvs.parameters():
            param.requires_grad = True

        for param in depth_model.iconvs.parameters():
            param.requires_grad = True
            
        for param in depth_model.predict_disps.parameters():
            param.requires_grad = True

    if options['optimize_depth_weights_all'] == True:
        ## unfreeze all depth weights
        for param in depth_model.parameters():#.encoder.parameters():
            param.requires_grad = True
            
    if options['optimize_pose_weights_all'] == True:
        for param in pose_model.parameters():
            param.requires_grad = True

    if options['optimize_depth_encoder'] == True:
        ## unfreeze all depth weights
        for param in depth_model.encoder.parameters():#.encoder.parameters():
            param.requires_grad = True

    ### perform the optimization over all frames, for all sequences in the list.
    for seq in seq_list:
        depth_optimizer = DepthOptimizer(options, config, pose_model, depth_model, seq)
        seqs = {'test': [seq]}
        if load_from_mat==False:
            print('warning - optimization across the whole sequence will take some time to complete')
            fwd_pose_list, fwd_pose_list_opt = [], []
            inv_pose_list, inv_pose_list_opt = [], []   
            gt_list = []
            pred_disps = []
            pred_poses_fwd = []
            pred_poses_inv = []
            pred_poses_comb = []
            
            print('sequence: {}'.format(seq))

            if data_format == 'odometry':    
                test_dset = KittiLoaderPytorch(config, seqs, mode='test', transform_img=get_data_transforms(config)['test'])
                print('dset length', len(test_dset.left_cam_filenames[0]))

            if data_format == 'eigen':
                test_dset = KittiLoaderPytorch(config, None, mode=seq, transform_img=get_data_transforms(config)['test'])
            
            if data_format == 'scannet':
                test_dset = ScanNetLoader(path_to_dset_downsized, test_split_path, num_frames=options['num_source_imgs']+1, transform_img=get_data_transforms(config)['test'])
                
            test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=8)

            t_init=time.time()
            
            for img_idx, data in enumerate(test_dset_loaders):
                # print(img_idx)
        
                optimized_results = depth_optimizer.optimize_window(img_idx, data)
                
                poses = optimized_results['poses_opt']
                poses_inv = optimized_results['poses_inv_opt']
                poses_init = optimized_results['poses_init']
                poses_inv_init = optimized_results['poses_inv_init']
                gt_poses = optimized_results['gt_poses']

                scale_factor = optimized_results['scale_factor']
                scale_factor_init = optimized_results['scale_factor_init']    
                minibatch_size = int(poses.shape[0]/options['num_source_imgs'])

                ### reorganizing poses to correspond to 'poses' being forward motion, 'poses_inv' being backward motion
                def rearrange_tensor(tensor, minibatch_size, num_source_imgs):
                    return torch.cat([tensor[n*minibatch_size:(n+1)*minibatch_size].unsqueeze(1) for n in range(0, num_source_imgs)],1)
                
                poses_init = rearrange_tensor(poses_init, minibatch_size, options['num_source_imgs'])
                poses = rearrange_tensor(poses, minibatch_size, options['num_source_imgs'])
                poses_inv_init = rearrange_tensor(poses_inv_init, minibatch_size, options['num_source_imgs'])
                poses_inv = rearrange_tensor(poses_inv, minibatch_size, options['num_source_imgs'])
                gt_poses = rearrange_tensor(gt_poses, minibatch_size, options['num_source_imgs'])

                if options['save_predictions']==True:
                    pred_disp = optimized_results['disp_opt']  
                    pred_disps.append(pred_disp)
                    pred_poses_fwd.append(poses[:,0].numpy()) ## only save first pose, not the second source image that was added
                    pred_poses_inv.append(-poses_inv[:,0].numpy())
                    pred_poses_comb.append((poses[:,0].numpy() - poses_inv[:,0].numpy())/2)

                for b in range(0,poses_init.shape[0]):
                    ## rescaling    
                    poses_init[b,:,0:3] = 30*scale_factor_init.item()*poses_init[b,:,0:3]
                    poses[b,:,0:3] = 30*scale_factor.item()*poses[b,:,0:3]
                    poses_inv_init[b,:,0:3] = 30*scale_factor_init.item()*poses_inv_init[b,:,0:3]
                    poses_inv[b,:,0:3] = 30*scale_factor.item()*poses_inv[b,:,0:3]                           
                    
                        ### fwd pose list is all of the poses going from in 'fwd' direction.. source1-->target, and target-->source2    
                        ### these values have the same sign for forward translation... both negative when vehicle moves forwards 
                    fwd_pose_list.append(poses_init[b,0].unsqueeze(0).numpy())
                    fwd_pose_list_opt.append(poses[b,0].unsqueeze(0).numpy())
                    gt_list.append(gt_poses[b,0].unsqueeze(0).numpy())
                    
                    if options['num_source_imgs'] ==2:
                        fwd_pose_list.append(poses_inv_init[b,1].unsqueeze(0).numpy())
                        fwd_pose_list_opt.append(poses_inv[b,1].unsqueeze(0).numpy())
                        gt_list.append(-gt_poses[b,1].unsqueeze(0).numpy())

                        ### fwd pose list is all of the poses going from in 'fwd' direction.. target-->source1, and source2-->target    
                        ### these values have the same sign for forward translation... both positive when vehicle moves forwards                     
                    if options['num_source_imgs'] ==2:
                        inv_pose_list.append(-poses_init[b,1].unsqueeze(0).numpy()) 
                        inv_pose_list_opt.append(-poses[b,1].unsqueeze(0).numpy())
                    
                    inv_pose_list.append(-poses_inv_init[b,0].unsqueeze(0).numpy())
                    inv_pose_list_opt.append(-poses_inv[b,0].unsqueeze(0).numpy())                


            fwd_pose_vec = np.array(fwd_pose_list)[:,0]
            fwd_pose_vec_opt = np.array(fwd_pose_list_opt)[:,0]
            inv_pose_vec = np.array(inv_pose_list)[:,0]
            inv_pose_vec_opt = np.array(inv_pose_list_opt)[:,0]
            gt_pose_vec = np.array(gt_list)[:,0]
            

            if options['save_predictions']==True:
                pred_disps = np.concatenate(pred_disps)
                pred_poses_fwd = np.concatenate(pred_poses_fwd)
                pred_poses_inv = np.concatenate(pred_poses_inv)
                pred_poses_comb = np.concatenate(pred_poses_comb)
                print("-> Saving predicted disparities to ", options['predictions_save_path'])
                print("run evaluate_depth_eigen.py from within paper_plots_and_data to evaluate depth.")
                np.save('{}/disps_{}_split.npy'.format(options['predictions_save_path'], seq_list[0]), pred_disps)
                np.save('{}/pred_pose_fwd.npy'.format(options['predictions_save_path']),pred_poses_fwd)
                np.save('{}/pred_pose_inv.npy'.format(options['predictions_save_path']),pred_poses_inv)
                np.save('{}/pred_pose_comb.npy'.format(options['predictions_save_path']),pred_poses_comb)

            data = {'seq': seq,
                    'config': config,
                    'options': options,
                    'fwd_pose_vec': fwd_pose_vec,
                    'fwd_pose_vec_opt': fwd_pose_vec_opt,
                    'inv_pose_vec': inv_pose_vec,
                    'inv_pose_vec_opt': inv_pose_vec_opt,            
                    'gt_pose_vec': gt_pose_vec,

            }
            save_obj(data, '{}/{}_results_minibatch_{}'.format(options['results_dir'], seq, options['minibatch']))

        else:
            data = load_obj('{}/{}_results_minibatch_{}_paper_result'.format(options['results_dir'], seq, options['minibatch']))
            fwd_pose_vec = data['fwd_pose_vec']
            fwd_pose_vec_opt = data['fwd_pose_vec_opt']
            inv_pose_vec = data['inv_pose_vec']
            inv_pose_vec_opt = data['inv_pose_vec_opt']    
            gt_pose_vec = data['gt_pose_vec']
    
        if data_format == 'odometry':
            test_dset = KittiLoaderPytorch(config, seqs, mode='test', transform_img=get_data_transforms(config)['test'])
            test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=6)

            ## Compute Trajectories
            gt_traj = test_dset_loaders.dataset.raw_gt_trials[0]
            comb_pose_vec = (inv_pose_vec + fwd_pose_vec)/2
            scale_factor = np.mean(np.linalg.norm(gt_pose_vec[:,0:3],axis=1))/np.mean(np.linalg.norm(comb_pose_vec[:,0:3],axis=1))
            comb_pose_vec[:,0:3] = scale_factor*comb_pose_vec[:,0:3]
            orig_est, gt, errors, cum_dist = tt(comb_pose_vec,gt_traj,method='original', compute_seg_err=True)

            comb_pose_vec_opt = (inv_pose_vec_opt + fwd_pose_vec_opt)/2
            scale_factor = np.mean(np.linalg.norm(gt_pose_vec[:,0:3],axis=1))/np.mean(np.linalg.norm(comb_pose_vec_opt[:,0:3],axis=1))
            comb_pose_vec_opt[:,0:3] = scale_factor*comb_pose_vec_opt[:,0:3]
            opt_est_comb, _, _, _ = tt(comb_pose_vec_opt,gt_traj,method='optimized', compute_seg_err=True)

            ## Plot trajectories
            plt.figure()
            plt.grid()
            plt.plot(gt[:,plot_axis[0],3], gt[:,plot_axis[1],3], linewidth=1.5, color='black', label='gt')
            plt.plot(orig_est[:,plot_axis[0],3],orig_est[:,plot_axis[1],3], linewidth=1.5, linestyle='--', label='orig')
            plt.plot(opt_est_comb[:,plot_axis[0],3],opt_est_comb[:,plot_axis[1],3], linewidth=1.5, linestyle='--', label='opt')
            plt.legend()
            plt.title('Topdown (XY) Trajectory Seq. {}'.format(config['test_seq'][0].replace('_','-')))
            plt.savefig('{}/seq-{}-topdown.png'.format(options['results_dir'], seq))


            fig = plotN([comb_pose_vec, comb_pose_vec_opt, gt_pose_vec], labels=['orig', 'opt', 'gt'])
            fig.suptitle('pose vecs')
            plt.savefig('{}/seq-{}-pose_vecs-test.png'.format(options['results_dir'], seq))
            plt.close(fig) 

        
