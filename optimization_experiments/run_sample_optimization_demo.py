import numpy as np
import torch
import sys
sys.path.append('../')
from utils.learning_helpers import save_obj, load_obj, load_ckp
from utils.custom_transforms import get_data_transforms
import os
import copy
from vis import *
from plot_loss_surface import generate_loss_surface
from data.kitti_loader import process_sample, process_sample_batch
from models.depth_w_access import depth_model
from models.pose_models import pose_model
from data.kitti_loader_stereo import KittiLoaderPytorch
from optimizer_for_cont_plot import DepthOptimizer

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings
    from https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

if __name__ == "__main__":      
    ### general data loading options
    path_to_ws = '/path/to/tightly-coupled-SfM/' ##update this
    path_to_dset_downsized = '/path/to/KITTI-odometry-preprocessed/' ##update this
    saved_model_dir = 'results/kitti-odometry-4-iter'
    img_idx_list = [220, 230, 240] ## these are the examples we provide, and can't be changed unless KITTI dataset is downloaded and preprocessed

    ## specify the optimization parameters
    options= {'epochs':20,
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
        'mode': 'unscaled', # use unscaled if you want to perform online rescaling w camera height and ground plane (KITTI)        
        'l_depth_consist': True,
        'l_depth_consist_weight': 0.15,
        'l_depth_init': True, ## optimized depth shouldn't be too different from starting depth
        'l_depth_init_weight': 0.1,
        'l_inverse_reconstruction': True,
        'l_smooth': False,
        'l_smooth_weight': 0.2,
        'l_pose_consist': False,
        'avg_final_epochs': 5, #False for no averaging
        'num_source_imgs': 2,
        'minibatch': 3, ## fixed to 1 for single img optimization
        'lr':2e-4,
        'extra_egomotion_iterations': 0, #0 for num. of iterations used during training
        'plotting':True,
        'save_disps': False,
        'disp_save_path': '',
    }

    dir = path_to_ws + saved_model_dir 
        ## Ensure that config file is set up properly (turn off options that may have been used during training)
    options['results_dir'] = dir + '/results/depth_opt_single/'
    os.makedirs(options['results_dir'], exist_ok=True)
    config = load_obj('{}/config'.format(dir))
    config['skip'] = options['num_source_imgs']
    config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/' #   
    config['img_per_sample'] = options['num_source_imgs'] + 1
    config['load_stereo'] = False
    config['augment_motion'] = False
    config['augment_backwards'] = False
    config['minibatch'] = options['minibatch']
    config['load_pretrained'] = True
    config['data_format'] = 'odometry'
    config['camera_height'] = 1.65
    config['correction_rate'] = 1
    config['iterations'] = config['iterations'] + options['extra_egomotion_iterations'] #0 for num. of iterations used during training
    device = config['device']
    seq='09'

    img_idx_list_mod = [int(idx/(options['minibatch']*config['skip'])) for idx in img_idx_list]
    
    ''' Load Models'''   
    depth_model = depth_model(config).to(device)
    pose_model = pose_model(config).to(device)
    pose_model, depth_model, _, _, _ = load_ckp(path_to_ws+saved_model_dir, True, pose_model, depth_model, None)
    depth_model.train(False).eval()
    pose_model.train(False).eval()

    ### initially freeze all weights
    for param in depth_model.parameters():#.encoder.parameters():
        param.requires_grad = False
    for param in pose_model.parameters():
        param.requires_grad = False
        
    if options['optimize_depth_weights_bottleneck_beyond'] == True:
        ## unfreeze depth decoder 
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

    if options['optimize_depth_encoder'] == True:
        ## unfreeze all depth weights
        for param in depth_model.encoder.parameters():#.encoder.parameters():
            param.requires_grad = True
            
    if options['optimize_pose_weights_all'] == True:
        for param in pose_model.parameters():
            param.requires_grad = True

    print('performing depth optimization - warning, generating the loss curves via grid search is time consuming, this may take a few minutes.')
    for data_name in ['seq_09_02_imgs_36', 'seq_09_02_imgs_38', 'seq_09_02_imgs_40']:

        data_name_full = '{}/{}/results/depth_opt_single/{}'.format(path_to_ws, saved_model_dir, data_name)
        data = load_obj(data_name_full)

        depth_optimizer = DepthOptimizer(options, config, pose_model, depth_model, seq)

        depth_optimizer.losses = []
        depth_optimizer.poses = []            
        os.makedirs(options['results_dir']+'/{}'.format(data_name), exist_ok=True)

            ## Optimization of depth network
        depth_optimizer.optimize_window(data_name, data)
        full_optimized_results = depth_optimizer.full_results
        
        for q in range(0,len(full_optimized_results)):
            optimized_results = full_optimized_results[q].copy()
        
            stacked_poses = optimized_results['stacked_poses_opt'].clone()
            stacked_poses_init = optimized_results['stacked_poses_init'].clone()
            poses = optimized_results['poses_opt'].clone()
            poses_inv = optimized_results['poses_inv_opt'].clone()
            poses_init = optimized_results['poses_init'].clone()
            poses_inv_init = optimized_results['poses_inv_init'].clone()
            gt_poses = optimized_results['gt_poses'].clone()
            depths_init = [d.clone() for d in optimized_results['depths_init']]
            depths = [d.clone() for d in optimized_results['depths_opt']]

            if options['mode'] == 'unscaled':
                scale_factor = optimized_results['scale_factor'][0].item()
                scale_factor_init = optimized_results['scale_factor_init'] [0].item()   
            if options['mode'] == 'scaled':
                ### scale factor assumed to be 1, and we rescale with how much the median target depth changes:
                scale_factor_init = 1
                scale_factor = torch.median(optimized_results['depths_init'][0]) / torch.median(optimized_results['depths_opt'][0])
                scale_factor = scale_factor[0].item()

                ## Apply our scaling factor of 30
            poses[:,0:3] = 30*poses[:,0:3]
            poses_inv[:,0:3] = 30*poses_inv[:,0:3]
            poses_init[:,0:3] = 30*poses_init[:,0:3]
            gt_poses[:,0:3] = gt_poses[:,0:3]
            
            poses_inv_init[:,0:3] = 30*poses_inv_init[:,0:3]   
            depths_init = [30*d for d in depths_init]
            depths = [30*d for d in depths]        
            
            if options['minibatch'] > 1:
                depths = [d[0:1] for d in depths]
                depths_init = [d[0:1] for d in depths_init]
            

                ## Generate our loss surface by varying translation and computing the loss at each pose
                ## The minimum should align with ground truth for more accurate depth maps.
            loss_surface_results = generate_loss_surface(data, depths, poses[0:1].to(config['device']), sample_trans=True)
            init_loss_surface_results = generate_loss_surface(data, depths_init, poses_init[0:1].to(config['device']), sample_trans=True)
            loss_surface_results_yaw = generate_loss_surface(data, depths, poses[0:1].to(config['device']), sample_yaw=True)
            init_loss_surface_results_yaw = generate_loss_surface(data, depths_init, poses_init[0:1].to(config['device']), sample_yaw=True)
            ### save the three items needed: x axis values, y axis values, and the colormap value
            
            if q == 0:
                xs = [scale_factor_init*(init_loss_surface_results['delta_list'] + poses_init[0,2].item() )]
                ys = [init_loss_surface_results['reconstruction_errors']]
                xs_yaw= [init_loss_surface_results_yaw['delta_list_yaw'] + poses_init[0,4].item()]
                ys_yaw = [init_loss_surface_results_yaw['reconstruction_errors_yaw']]
            else:
                xs.append(scale_factor*( loss_surface_results['delta_list'] + poses[0,2].item() ))
                ys.append(loss_surface_results['reconstruction_errors'])  
                xs_yaw.append(loss_surface_results_yaw['delta_list_yaw'] + poses[0,4].item() )
                ys_yaw.append(loss_surface_results_yaw['reconstruction_errors_yaw'])                      
        
        c = np.arange(len(full_optimized_results),-1, -1)
        xs.reverse()
        ys.reverse()
        xs_yaw.reverse()
        ys_yaw.reverse()               
        
            ### loss vs translation vs epoch curves
        fig, ax = plt.subplots()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.tight_layout()
        ax.tick_params(labelsize=24)
        ax.set_ylabel('Reconstruction Error', fontsize=22)
        ax.set_xlabel('Translation (m.)', fontsize=22)
        lc = multiline(xs[1::2], ys[1::2], c[1::2], cmap='RdYlGn', lw=2.5)
        axcb = fig.colorbar(lc)
        axcb.ax.tick_params(labelsize=24) 
        axcb.set_label('Epoch', fontsize=22)
        ax.plot(2*[gt_poses[0,2].item()], [loss_surface_results['best_error']-0.01, init_loss_surface_results['best_error']+0.01], color='black', label='Ground Truth Trans.',linewidth=3, linestyle='dotted')
        
        marker_list = [".", "*", "*", '1', '2', '3', '4', 4,5,6]
        for p in [0,2]:
            ax.scatter([scale_factor*30*stacked_poses[0][p, 2].item()],[loss_surface_results['best_error']], label='Opt. Pred. iter ' + str(p+1), marker=marker_list[p], s=100, color='darkolivegreen', edgecolor='black',zorder=10+p)
            ax.scatter([scale_factor_init*30*stacked_poses_init[0][p, 2].item()],[init_loss_surface_results['original_error']], label='Orig. Pred. iter ' + str(p+1), marker=marker_list[p], s=100, color='red', edgecolor='black', zorder=15+p)
        
        ax.set_title("PFT Visualization (Fwd. Trans. Axis)", fontsize=20,y=0.9)    

        ax.grid()
        plt.legend(fontsize=15)
        
        plt.subplots_adjust(bottom=0.18)
        plt.subplots_adjust(left = 0.18)
        plt.subplots_adjust(top=0.9)
        plt.subplots_adjust(right=0.9)
        plt.savefig('{}/continuous_error_curves-{}-iter-{}.png'.format(options['results_dir'],  data_name, config['iterations']))

            ### loss vs yaw vs epoch curves
        fig, ax = plt.subplots()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.tight_layout()
        ax.tick_params(labelsize=24)
        ax.set_ylabel('Reconstruction Error', fontsize=22)
        ax.set_xlabel('Yaw (rad)', fontsize=22)
        lc = multiline(xs_yaw, ys_yaw, c, cmap='RdYlGn', lw=2.5)
        axcb = fig.colorbar(lc)
        axcb.ax.tick_params(labelsize=24) 
        axcb.set_label('Epoch', fontsize=22)
        ax.plot(2*[gt_poses[0,4].item()], [loss_surface_results_yaw['best_error_yaw']-0.01, init_loss_surface_results_yaw['best_error_yaw']+0.01], color='black', label='Ground Truth Yaw',linewidth=3, linestyle='dotted')
        
        marker_list = [".", "*", "*", '1', '2', '3', '4', 4,5,6]
        for p in [0,2]:
            ax.scatter([stacked_poses[0][p, 4].item()],[loss_surface_results_yaw['best_error_yaw']], label='Opt. Pred. iter ' + str(p+1), marker=marker_list[p], s=100, color='darkolivegreen', edgecolor='black',zorder=10+p)
            ax.scatter([stacked_poses_init[0][p, 4].item()],[init_loss_surface_results_yaw['original_error']], label='Orig. Pred. iter ' + str(p+1), marker=marker_list[p], s=100, color='red', edgecolor='black', zorder=15+p)
        ax.grid()
        plt.legend(fontsize=15)
        
        plt.subplots_adjust(bottom=0.18)
        plt.subplots_adjust(left = 0.18)
        plt.subplots_adjust(top=0.9) #0.97
        plt.subplots_adjust(right=0.9)
        ax.set_title("PFT Visualization (Yaw Axis)", fontsize=20,y=0.9)    
        
        plt.savefig('{}/continuous_error_curves_yaw-{}-iter-{}.png'.format(options['results_dir'],  data_name, config['iterations']))
            
    print('results saved to {}'.format(options['results_dir']))