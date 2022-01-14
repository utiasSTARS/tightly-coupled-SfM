import numpy as np
import torch
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import validate
from train_mono import solve_pose, solve_pose_iteratively
from data.kitti_loader import process_sample_batch
from models.dnet_layers import ScaleRecovery
from utils.learning_helpers import save_obj, load_obj, disp_to_depth, batch_post_process_disparity
from utils.custom_transforms import *
from vis import *
import os
from validate import compute_trajectory as tt
import glob


path_to_ws = '/home/brandonwagstaff/tightly-coupled-SfM/' ##update this
# path_to_ws = '/home/brandon/Desktop/Projects/iterative-optimizable-vo/'s

load_from_mat = True #Make True to load paper results rather than recomputing
dnet_rescaling = True
post_process_depths = False


perturb_trans = False
perturb_yaw = True
trans_perturbation_list = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5] #in meters 
yaw_perturbation_list = [0, 0.1, 0.25, 0.5, 1, 3, 5]

seq = '09_02'

model_list = ['results/202103292232-kitti-unscaled-1-iter-med-res-less-data', 'results/202103270016-kitti-unscaled-2-iter-med-res-less-data', 'results/202103231952-kitti-unscaled-3-iter-med-res-less-data'] 
model_names = ['1-iter', '2-iter', '3-iter']

path_to_dset_downsized = '/media/datasets/KITTI-odometry-downsized-stereo/'
plot_axis=[0,2]

results_dir = path_to_ws + 'paper_plots_and_data/perturbation_exp_results/'
os.makedirs(results_dir, exist_ok=True)
logger_list = [] #where results are stored


if load_from_mat == False:
    for model_num, model in enumerate(model_list):
        dir = path_to_ws + model
        logger_list.append(validate.ResultsLogger('{}/{}_{}_trans_pert_{}-yaw_pert_{}-metrics.csv'.format(results_dir, seq, model_names[model_num],perturb_trans, perturb_yaw)))
        print('sequence: {}'.format(seq))
        cam_height = 1.65

        config = load_obj('{}/config'.format(dir))
        print(config)
        config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/' #
        
        config['load_stereo'] = False
        config['augment_motion'] = False
        config['augment_backwards'] = False
        config['img_per_sample']=2
        config['test_seq'] = [seq]
        config['minibatch'] = 1
        config['load_pretrained'] = True
        config['data_format'] = 'odometry'
        config['estimator'] = 'orbslam'

        device=config['device']
            
        ### dataset and model loading    
        from data.kitti_loader_stereo import KittiLoaderPytorch
        seqs = {'test': [seq]}
        test_dset = KittiLoaderPytorch(config, seqs, mode='test', transform_img=get_data_transforms(config)['test'])
        test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=6)

        import models.pose_models as pose_models
        import models.depth_models as models
        
        depth_model = models.depth_model(config).to(device)
        pose_model = pose_models.pose_model(config).to(device)
        pretrained_depth_path = glob.glob('{}/**depth**best-loss-val_seq-**-test_seq-{}**.pth'.format(dir, ''))[0]
        pretrained_pose_path = glob.glob('{}/**pose**best-loss-val_seq-**-test_seq-{}**.pth'.format(dir, ''))[0]
        depth_model.load_state_dict(torch.load(pretrained_depth_path))
        pose_model.load_state_dict(torch.load(pretrained_pose_path))
        pose_model.train(False).eval()
        depth_model.train(False).eval()    
          
            
        if dnet_rescaling == True:
            # dgc = ScaleRecovery(config['minibatch'], 128, 448).to(device) 
            dgc = ScaleRecovery(config['minibatch'], 192, 640).to(device) 
            # dgc = ScaleRecovery(config['minibatch'], 256, 832).to(device) 
        
        for p in range(0, len(trans_perturbation_list)):
            if perturb_trans==True:
                trans_pert = trans_perturbation_list[p]
            else:
                trans_pert = 0
            
            if perturb_yaw==True:
                yaw_pert = yaw_perturbation_list[p]
            else:
                yaw_pert = 0
            
            depth_model, pose_model = depth_model.train(False).eval(), pose_model.train(False).eval()  # Set model to evaluate mode
            fwd_pose_list1, inv_pose_list1 = [], []
            fwd_pose_list2, inv_pose_list2 = [], []
            gt_list = []
            learned_scale_factor_list = []
            dnet_scale_factor_list = []
            
            with torch.no_grad():
                for k, data in enumerate(test_dset_loaders):
                    target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, target_img_aug, \
                        source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = process_sample_batch(data, config)
                    pose_results = {'source1': {}, 'source2': {} }

                    batch_size = target_img.shape[0]
                    imgs = torch.cat([target_img, source_img_list[0]],0)
                    
                    if post_process_depths == True:
                        # Post-processed results require each image to have two forward passes
                        imgs = torch.cat((imgs, torch.flip(imgs, [3])), 0)    
                    
                    disparities = depth_model(imgs, epoch=50)
                    disparities, depths = disp_to_depth(disparities[0], config['min_depth'], config['max_depth'])
                    disparities = disparities.cpu()[:, 0].numpy()
                    if post_process_depths == True:
                        N = disparities.shape[0] // 2
                        disparities = batch_post_process_disparity(disparities[:N], disparities[N:, :, ::-1])    

                    disparities = torch.from_numpy(disparities).unsqueeze(1).type(torch.FloatTensor).to(device)

                    target_disparities = disparities[0:batch_size]
                    source_disp_1 = disparities[batch_size:(2*batch_size)]
                    # target_disparities = [disp[0:batch_size] for disp in disparities]
                    # source_disp_1 = [disp[batch_size:(2*batch_size)] for disp in disparities]

                    disparities = [target_disparities, source_disp_1]         
                    
                    # depths = [disp_to_depth(disp[0], config['min_depth'], config['max_depth'])[1] for disp in disparities] ####.detach()
                    depths = [1.0/disp for disp in disparities]

                    flow_imgs_fwd_list, flow_imgs_back_list = flow_imgs
                    
                    if config['iterations']==1:
                        poses, poses_inv = solve_pose(pose_model, target_img, source_img_list, flow_imgs)
                        for i in range(0, len(poses)):
                            poses[i][:,2] = poses[i][:,2] + np.random.uniform(-trans_pert, trans_pert)
                            poses[i][:,4] = poses[i][:,4] + np.random.uniform(-yaw_pert, yaw_pert)
                    
                    else:
                        poses, poses_inv = solve_pose_iteratively(config['iterations'], depths, pose_model, target_img, source_img_list, intrinsics, trans_pert = trans_pert, yaw_pert = yaw_pert)
                    
                    
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
                        'learned_scale_factor': learned_scale_factor_list,    
                        'dnet_scale_factor': dnet_scale_factor_list,    
                }

            gt_pose_vec = data['gt_pose_vec']
            gt_pose_vec[:,0:3] = gt_pose_vec[:,0:3]
            unscaled_pose_vec = data['fwd_pose_vec1']
            print('norm of translation: {}'.format(np.average(np.linalg.norm(unscaled_pose_vec[:,0:3],axis=1))))
            scaled_pose_vec_ransac = np.array(unscaled_pose_vec)
            
            if dnet_rescaling == True:
                print('ground plane scale factor (dnet): {}'.format(np.mean(data['dnet_scale_factor'])))
                print('ground plane std. dev. scale factor (dnet): {}'.format(np.std(data['dnet_scale_factor'])))
                scaled_pose_vec_dnet = np.array(unscaled_pose_vec)
                scaled_pose_vec_dnet[:,0:3] = scaled_pose_vec_dnet[:,0:3]*np.repeat(data['dnet_scale_factor'],3,axis=1)

            
            ## Compute Trajectories
            gt_traj = test_dset_loaders.dataset.raw_gt_trials[0]
            gt_traj[:,0:3,3] = gt_traj[:,0:3,3]
            # orig_est, gt, errors, cum_dist = tt(unscaled_pose_vec,gt_traj,method='unscaled')
            # logger.log(seq, 'unscaled', errors[0], errors[1], errors[2], errors[3])

            if dnet_rescaling == True:
                _, _, errors, _ = tt(scaled_pose_vec_dnet,gt_traj, method='scaled (dnet)',compute_seg_err=True)
                logger_list[-1].log(seq, 'trans_pert_{}_yaw_pert_{}'.format(trans_pert, yaw_pert), errors[0], errors[1], errors[2], errors[3])
            # logger_list[-1].log('', '', '', '', '', '')
            
    data = {'results': [l.results for l in logger_list]}
    save_obj(data, '{}/seq-{}-perturbation_results_trans_{}_yaw_{}'.format(results_dir, seq, perturb_trans, perturb_yaw))
    

data = load_obj('{}/seq-{}-perturbation_results_trans_{}_yaw_{}'.format(results_dir, seq, perturb_trans, perturb_yaw))
logger_list = data['results']
    
if perturb_trans == True:
    x_axis = trans_perturbation_list
    x_label = 'translation perturbation range (m)'

if perturb_yaw == True:
    x_axis = yaw_perturbation_list
    x_label = 'yaw perturbation range ($^o$)'

if perturb_trans == True and perturb_yaw == True:
    x_axis = range(0,len(trans_perturbation_list))
    x_label = 'trans + yaw perturbation magnitude'


print('Please run `generate_subplot.py` from within `perturbation_exp_results` subdirectory for plotting after generating results.')
    ### absolute plots
# plt.figure()
# plt.clf()
# for i in range(0,1000):
#     x=1
# plt.plot(x)

# plt.figure()
# plt.grid()
# #plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.subplots_adjust(left=0.15)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.tick_params(labelsize=18)
# print(logger_list)
# for idx, log in enumerate(logger_list):
#     r_mse_list = log['r_mse_list']  
#     print(r_mse_list)
#     plt.plot(x_axis, r_mse_list, label=model_names[idx].replace('-',' '))

# plt.legend(fontsize=15)
# # plt.title('Relative rotation Error Vs. Perturbation range')
# plt.ylabel('$r_{err}$ ($^o \slash 100$m)', fontsize=18)
# plt.xlabel(x_label, fontsize=18)
# plt.savefig('{}/seq-{}-r_err_vs_perturbation_trans_{}_yaw_{}.pdf'.format(results_dir, seq, perturb_trans, perturb_yaw))

   
# plt.figure()
# plt.grid()
# # plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.subplots_adjust(left=0.15)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.tick_params(labelsize=22)

# for idx, log in enumerate(logger_list):
#     t_mse_list = log['t_mse_list']   
#     plt.plot(x_axis, t_mse_list, linewidth=2, label=model_names[idx].replace('-',' '))

# plt.legend(fontsize=15)
# # plt.title('Relative Translation Error Vs. Perturbation range')
# plt.ylabel('$t_{err}$ (\%)', fontsize=18)
# plt.xlabel(x_label, fontsize=18)
# plt.savefig('{}/seq-{}-t_err_vs_perturbation_trans_{}_yaw_{}.pdf'.format(results_dir, seq, perturb_trans, perturb_yaw))

# plt.figure()
# plt.grid()
# #plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.subplots_adjust(left=0.15)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.tick_params(labelsize=18)
# for idx, log in enumerate(logger_list):
#     t_ate_list = log['t_ate_list']
#     plt.plot(x_axis, t_ate_list, label=model_names[idx].replace('-',' '))

# plt.legend(fontsize=15)
# # plt.title('Abs. Trans. Error Vs. Perturbation range')
# plt.ylabel('t-ATE (m)', fontsize=18)
# plt.xlabel(x_label, fontsize=18)
# plt.savefig('{}/seq-{}-t_ate_vs_perturbation_trans_{}_yaw_{}.pdf'.format(results_dir, seq, perturb_trans, perturb_yaw))

# plt.figure()
# plt.grid()
# #plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.subplots_adjust(left=0.15)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.tick_params(labelsize=18)
# for idx, log in enumerate(logger_list):
#     r_ate_list = log['r_ate_list']
#     plt.plot(x_axis, r_ate_list, label=model_names[idx].replace('-',' '))

# plt.legend(fontsize=15)
# # plt.title('Abs. Rot. Error Vs. Perturbation range')
# plt.ylabel('r-ATE ($^o$)', fontsize=18)
# plt.xlabel(x_label, fontsize=18)
# plt.savefig('{}/seq-{}-r_ate_vs_perturbation_trans_{}_yaw_{}.pdf'.format(results_dir, seq, perturb_trans, perturb_yaw))


#     ### relative plots
# plt.figure()
# plt.grid()
# #plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.tick_params(labelsize=18)
# for idx, log in enumerate(logger_list):
#     t_mse_list = log['t_mse_list']
#     t_mse_list = [t_mse/t_mse_list[0] for t_mse in t_mse_list]   
#     plt.plot(x_axis, t_mse_list, linewidth=2, label=model_names[idx].replace('-',' '))

# plt.legend(fontsize=15)
# # plt.title('Relative Translation Error Vs. Perturbation range')
# plt.ylabel('relative $t_{err}$ (\%) increase (unitless)', fontsize=18)
# plt.xlabel(x_label, fontsize=18)
# plt.savefig('{}/seq-{}-relative_t_err_vs_perturbation_trans_{}_yaw_{}.pdf'.format(results_dir, seq, perturb_trans, perturb_yaw))


# plt.figure()
# plt.grid()
# #plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.tick_params(labelsize=18)
# for idx, log in enumerate(logger_list):
#     r_mse_list = log['r_mse_list'] 
#     print(r_mse_list)
#     r_mse_list = [r_mse/r_mse_list[0] for r_mse in r_mse_list]  
#     plt.plot(x_axis, r_mse_list, label=model_names[idx].replace('-',' '))

# plt.legend(fontsize=15)
# # plt.title('Relative rotation Error Vs. Perturbation range')
# plt.ylabel('relative $r_{err}$ ($^o \slash 100$m) increase (unitless)', fontsize=18)
# plt.xlabel(x_label, fontsize=18)
# plt.savefig('{}/seq-{}-relative_r_err_vs_perturbation_trans_{}_yaw_{}.pdf'.format(results_dir, seq, perturb_trans, perturb_yaw))

# plt.figure()
# plt.grid()
# #plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.tick_params(labelsize=18)
# for idx, log in enumerate(logger_list):
#     t_ate_list = log['t_ate_list']
#     t_ate_list = [t_ate/t_ate_list[0] for t_ate in t_ate_list]  
#     plt.plot(x_axis, t_ate_list, label=model_names[idx].replace('-',' '))

# plt.legend(fontsize=15)
# # plt.title('relative Abs. Trans. Error Vs. Perturbation range')
# plt.ylabel('relative t-ATE (m) increase (unitless)', fontsize=18)
# plt.xlabel(x_label, fontsize=18)
# plt.savefig('{}/seq-{}-relative_t_ate_vs_perturbation_trans_{}_yaw_{}.pdf'.format(results_dir, seq, perturb_trans, perturb_yaw))

# plt.figure()
# plt.grid()
# #plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.tick_params(labelsize=18)
# for idx, log in enumerate(logger_list):
#     r_ate_list = log['r_ate_list']
#     r_ate_list = [r_ate/r_ate_list[0] for r_ate in r_ate_list]  
#     plt.plot(x_axis, r_ate_list, label=model_names[idx].replace('-',' '))

# plt.legend(fontsize=15)
# # plt.title('Abs. Rot. Error Vs. Perturbation range')
# plt.ylabel('relative r-ATE ($^o$) increase (unitless)', fontsize=18)
# plt.xlabel(x_label, fontsize=18)
# plt.savefig('{}/seq-{}-relative_r_ate_vs_perturbation_trans_{}_yaw_{}.pdf'.format(results_dir, seq, perturb_trans, perturb_yaw))
