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
from utils.learning_helpers import save_obj, load_obj, disp_to_depth, batch_post_process_disparity, load_ckp
from utils.custom_transforms import *
from vis import *
import os
from validate import compute_trajectory as tt
import glob


path_to_ws = 'path/to/tightly-coupled-SfM/' ##update this
path_to_dset_downsized = '/media/datasets/KITTI-odometry-downsized-stereo/'

load_from_mat = False #Make True to load paper results rather than recomputing
dnet_rescaling = True
post_process_depths = False
scaling_factor_list = [0.7,0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

new_iteration_num = None #None for same # as training
seq = '09_02'

model_list = ['results/kitti-odometry-1-iter-ablation', 
    'results/kitti-odometry-2-iter-ablation', 
    'results/kitti-odometry-3-iter-ablation'] 
model_names = ['1-iter', '2-iter', '3-iter']

plot_axis=[0,2]
results_dir = path_to_ws + 'paper_plots_and_data/depth_scaling_results'
os.makedirs(results_dir, exist_ok=True)
logger_list = [] #where results are stored
translation_norms = {}


if load_from_mat == False:
    print('warning - this script will take some time to run.')
    for model_num, model in enumerate(model_list):  
        dir = path_to_ws + model
        logger_list.append(validate.ResultsLogger('{}/{}_{}_depth_scaling_results.csv'.format(results_dir, seq, model_names[model_num])))
        print('sequence: {}'.format(seq))
        cam_height = 1.65

        config = load_obj('{}/config'.format(dir))
        config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/' #
        
        config['load_stereo'] = False
        config['augment_motion'] = False
        config['augment_backwards'] = False
        config['img_per_sample']=2
        config['test_seq'] = [seq]
        config['minibatch'] = 1
        config['load_pretrained'] = True
        config['data_format'] = 'odometry'
        config['correction_rate'] = 1
        
        if model_names[model_num] != '1-iter-(depth-input)':
            config['depth_input_to_posenet']=False

        if new_iteration_num is not None:
            config['iterations'] = new_iteration_num
        
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
        pose_model, depth_model, _, _, _ = load_ckp(dir, True, pose_model, depth_model, None)
        pose_model.train(False).eval()
        depth_model.train(False).eval()   
            
        if dnet_rescaling == True:
            # dgc = ScaleRecovery(config['minibatch'], 128, 448).to(device) 
            dgc = ScaleRecovery(config['minibatch'], 192, 640).to(device) 
            # dgc = ScaleRecovery(config['minibatch'], 256, 832).to(device) 
        
        translation_norm_list = []
        for p in range(0, len(scaling_factor_list)):
            scaling_factor = scaling_factor_list[p]
            
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

                    disparities = [target_disparities, source_disp_1]         
                    
                    depths = [scaling_factor*1.0/disp for disp in disparities]

                    flow_imgs_fwd_list, flow_imgs_back_list = flow_imgs
                    
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
                        'learned_scale_factor': learned_scale_factor_list,    
                        'dnet_scale_factor': dnet_scale_factor_list,    
                }

            gt_pose_vec = data['gt_pose_vec']
            gt_pose_vec[:,0:3] = gt_pose_vec[:,0:3]
            unscaled_pose_vec = data['fwd_pose_vec1']
            translation_norm = np.average(np.linalg.norm(unscaled_pose_vec[:,0:3],axis=1))
            print('norm of translation: {}'.format(translation_norm))
            translation_norm_list.append(translation_norm)
            
            scaled_pose_vec_ransac = np.array(unscaled_pose_vec)
            
            if dnet_rescaling == True:
                print('ground plane scale factor (dnet): {}'.format(np.mean(data['dnet_scale_factor'])))
                print('ground plane std. dev. scale factor (dnet): {}'.format(np.std(data['dnet_scale_factor'])))
                scaled_pose_vec_dnet = np.array(unscaled_pose_vec)
                scaled_pose_vec_dnet[:,0:3] = scaled_pose_vec_dnet[:,0:3]*np.repeat(data['dnet_scale_factor'],3,axis=1)

            
            ## Compute Trajectories
            gt_traj = test_dset_loaders.dataset.raw_gt_trials[0]
            gt_traj[:,0:3,3] = gt_traj[:,0:3,3]

            if dnet_rescaling == True:
                _, _, errors, _ = tt(scaled_pose_vec_dnet,gt_traj, method='scaled (dnet)', compute_seg_err=True)
                logger_list[-1].log(seq, 'dnet_depth_scale_{}'.format(scaling_factor), errors[0], errors[1], errors[2], errors[3])
         
            translation_norms[model_names[model_num]] = translation_norm_list
            

    data = {'results': [l.results for l in logger_list], 'trans_norms': translation_norms}
    save_obj(data, '{}/seq-{}-depth_scaling_results'.format(results_dir, seq))
    

data = load_obj('{}/seq-{}-depth_scaling_results'.format(results_dir, seq))
logger_list = data['results']
translation_norms = data['trans_norms']
    
### combined plot for paper

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tight_layout()
f, axarr = plt.subplots(2, sharex=True, sharey=False)
axarr[0].tick_params(labelsize=23)
axarr[1].tick_params(labelsize=23)

l0 = []
for idx, log in enumerate(logger_list):
    t_mse_list = log['t_mse_list']  
    l,=axarr[0].plot(scaling_factor_list, t_mse_list, linewidth=2, marker='o', label=model_names[idx].replace('--','/').replace('-',' '))
    l0.append(l)
    

l1=[]
for idx, name in enumerate(model_names):
    trans_norm = np.array(translation_norms[name])
    print('trans norm', trans_norm)
    trans_norm = trans_norm/trans_norm[3] ## normalize with norm at scale=1
    print('test')
    print('rel trans norm', trans_norm)
    l,=axarr[1].plot(scaling_factor_list, trans_norm, linewidth=2, marker='o', label=model_names[idx].replace('--','/').replace('-',' '))
    l1.append(l)
    
l,=axarr[1].plot(scaling_factor_list, scaling_factor_list, linewidth=2, linestyle='-', label='ideal')
l1.append(l)
   
axarr[0].set_ylabel('$t_{err}$ (\%)', fontsize=24)
axarr[1].set_ylabel('Rel. Trans. Norm', fontsize=24)
axarr[1].set_xlabel('Depth Scaling Factor', fontsize=24)

axarr[0].grid()        
axarr[1].grid()    

legend_labels = [m.replace('--','/').replace('-',' ') for m in model_names] + ['ideal']
print(legend_labels)
f.legend(l1, legend_labels, fontsize=18,loc="center right")

plt.subplots_adjust(hspace = 0.6)
plt.subplots_adjust(bottom=0.2)
plt.subplots_adjust(left=0.18)

f.suptitle('Depth Scaling Experiment', fontsize=25)
plt.savefig('{}/seq-{}-depth_scaling_combined.pdf'.format(results_dir, seq))

# plt.figure()
# plt.grid()
# plt.subplots_adjust(bottom=0.15)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.tick_params(labelsize=23)
# for idx, log in enumerate(logger_list):
#     t_mse_list = log['t_mse_list'] 
#     # t_mse_list = [t_mse/t_mse_list[0] for t_mse in t_mse_list]   
#     plt.plot(scaling_factor_list, t_mse_list, linewidth=3, label=model_names[idx].replace('-',' '))

# plt.legend(fontsize=17)
# # plt.title('Relative Translation Error Vs. Perturbation range')
# plt.ylabel('$t_{err}$ (\%)', fontsize=24)
# plt.xlabel('Depth Scaling Factor', fontsize=24)
# plt.savefig('{}/seq-{}-t_err_vs_depth_scale_factor.pdf'.format(results_dir, seq))

# plt.figure()
# plt.grid()
# #plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.tick_params(labelsize=23)
# for idx, log in enumerate(logger_list):
#     r_mse_list = log['r_mse_list'] 
#     # t_mse_list = [t_mse/t_mse_list[0] for t_mse in t_mse_list]   
#     plt.plot(scaling_factor_list, r_mse_list, linewidth=3, label=model_names[idx].replace('-',' '))

# plt.legend(fontsize=17)
# # plt.title('Relative Translation Error Vs. Perturbation range')
# plt.ylabel('$r_{err}$ ($^o \slash 100$m)', fontsize=24)
# plt.xlabel('Depth Scaling Factor', fontsize=24)
# plt.savefig('{}/seq-{}-r_err_vs_depth_scale_factor.pdf'.format(results_dir, seq))

# plt.figure()
# plt.grid()
# #plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.tick_params(labelsize=23)
# for idx, name in enumerate(model_names):
#     trans_norm = np.array(translation_norms[name])
#     trans_norm = trans_norm/trans_norm[3] ## normalize with norm at scale=1
#     # t_mse_list = [t_mse/t_mse_list[0] for t_mse in t_mse_list]   
#     plt.plot(scaling_factor_list, trans_norm, linewidth=3, label=model_names[idx].replace('-',' '))

# plt.plot(scaling_factor_list, scaling_factor_list, linewidth=2, linestyle='-', label='ideal')

# plt.legend(fontsize=19)
# # plt.title('Relative Translation Error Vs. Perturbation range')
# plt.ylabel('Relative Translation Norm', fontsize=24)
# plt.xlabel('Depth Scaling Factor', fontsize=24)
# plt.savefig('{}/seq-{}-relative_translation_norm.pdf'.format(results_dir, seq))
                      

