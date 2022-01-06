import sys
sys.path.append('../')
import torch
import numpy as np
from losses import get_smooth_loss, SSIM_Loss
import copy
from train_mono import solve_pose_iteratively
from utils.learning_helpers import disp_to_depth, batch_post_process_disparity
from models.dnet_layers import ScaleRecovery
from vis import *
from helpers import *

class DepthOptimizer():
    def __init__(self, options, config, pose_model, depth_model, seq):
        self.options = options
        self.config = config
        self.pose_model = pose_model.train(False).eval()
        self.depth_model = depth_model.train(False).eval()
        self.dgc = ScaleRecovery(config['minibatch'], 192, 640).to(config['device']) 
        self.ssim_loss = SSIM_Loss()
        self.losses = []
        self.poses = [] 
        self.seq = seq
        self.pose_avg_list = []
        self.disp_avg_list = []
        self.pose_inv_avg_list = []
        self.full_results = []

    def compute_optimization_loss(self, opt_step_num, img_idx, target_img, target_disparity, fwd_data, inv_data):
        '''
        computes the loss used for a single optimization step
        
        Inputs:
            opt_step_num:       The current optimization step - used for saving imgs at each step
            img_idx:            Img idx that we're optimizing - used for saving imgs        
            target_img:         Target img - size (B,3,H,W)     
            target_disparity:   disparity (inverse depth) prediction for target image - size (B,H,W)
            fwd_data:           outputs of solve_pose_iteratively() containing the error imgs, reconstructed imgs, and masks
                                    fwd data is reconstructing target img from each source img
            inv_data:           outputs of solve_pose_iteratively() containing the error imgs, reconstructed imgs, and masks
                                    inv data is reconstructing source imgs from the target img     
        Outputs:
            loss: scalar loss that is a sum of all of the individual loss terms
        '''
        batch_size = target_img.shape[0]
        loss=0
        if self.options['diff_img_argmin']==True:     
            diff_imgs = fwd_data['diff_img']
            
            diff_imgs_reshaped = [fwd_data['diff_img'][i*batch_size:(i+1)*batch_size] for i in range(0, self.options['num_source_imgs'])]
            diff_imgs_reshaped = torch.cat(diff_imgs_reshaped, 1).unsqueeze(2) ### batch_size, num_source_imgs, 3, H, W
            diff_img_min, idx = torch.min(diff_imgs_reshaped, 1)

            if self.options['plotting']==True:
                img_rec_comb = torch.zeros((batch_size,3,diff_img_min.shape[2],diff_img_min.shape[3]),device=diff_img_min.device)
                for b in range(0, batch_size):
                    for src_idx in range(0,self.options['num_source_imgs']):
                        img_idx_i = (idx[b]==src_idx).repeat(3,1,1) #3,192,640
                        img_rec_comb[b,img_idx_i] = fwd_data['img_rec'][b, img_idx_i]

            valid_mask_min = torch.cat([fwd_data['valid_mask'][i*batch_size:(i+1)*batch_size] for i in range(0, self.options['num_source_imgs'])],1)
            valid_mask_min = valid_mask_min.sum(1,keepdim=True).clamp(0,1)
            
            if self.options['automasking'] == True:
                auto_mask_error = [fwd_data['auto_mask_error'][i*batch_size:(i+1)*batch_size] for i in range(0, self.options['num_source_imgs'])]
                auto_mask_error = torch.cat(auto_mask_error, 1).unsqueeze(2)
                auto_mask_min, _ = torch.min(auto_mask_error,1)
                auto_mask = (diff_img_min < auto_mask_min).float()
                valid_mask_min= auto_mask * valid_mask_min
            loss += (diff_img_min*valid_mask_min*fwd_data['weight_mask'][0:batch_size]).sum(3).sum(2).sum(0) / valid_mask_min.sum(3).sum(2).sum(0)

        diff_imgs_masked = fwd_data['diff_img']*fwd_data['valid_mask']*fwd_data['weight_mask'] 
        if self.options['diff_img_argmin']==False:   
            loss += 0.25*diff_imgs_masked.sum() / fwd_data['valid_mask'].sum()

        diff_imgs_inv_masked = inv_data['diff_img']*inv_data['valid_mask']*inv_data['weight_mask']
        if self.options['l_inverse_reconstruction']==True:
            if self.options['automasking'] == True:
                diff_imgs_inv_masked = diff_imgs_inv_masked * inv_data['auto_mask']
                loss += 0.25*diff_imgs_inv_masked.sum() / (inv_data['valid_mask']*inv_data['auto_mask']).sum()
            else:
                loss += 0.25*diff_imgs_inv_masked.sum() / inv_data['valid_mask'].sum()

        if self.options['l_depth_consist']==True:
            loss += self.options['l_depth_consist_weight']*((-fwd_data['weight_mask']+1)).mean()
            if self.options['l_inverse_reconstruction']==True:
                loss +=self.options['l_depth_consist_weight']*((-inv_data['weight_mask']+1)).mean()

            ## ensure optimized depth is similar to initial depth
        if self.options['l_depth_init'] == True:
            loss += self.options['l_depth_init_weight']*self.ssim_loss(target_disparity, self.target_disparity.clone().detach()).mean()
                                        
        if self.options['l_smooth']==True:
            loss += self.options['l_smooth_weight']*get_smooth_loss(target_disparity, target_img)

        if self.options['l_pose_consist']==True:
            loss += 0.1*(fwd_data['poses'] + inv_data['poses']).abs().mean()

        if self.options['plotting'] == True:
            if opt_step_num == 0 or opt_step_num == self.options['epochs']-1:
                if self.options['diff_img_argmin']==True:
                    diff_imgs_plotting = torch.cat([diff_imgs, diff_img_min],0)
                    img_recs_plotting = torch.cat([fwd_data['img_rec'], img_rec_comb, target_img],0)
                    valid_masks_plotting = torch.cat([fwd_data['valid_mask'], valid_mask_min],0)

                    ### plot imgs for paper                    
                    for i in range(0, target_img.shape[0]):
                        plotting_imgs = torch.cat([target_img[i].unsqueeze(0), target_disparity[i].unsqueeze(0).repeat(1,3,1,1), diff_img_min[i].unsqueeze(0).repeat(1,3,1,1)],0)
                        plot_img_array(plotting_imgs.cpu().detach(), nrow=3, save_file = '{}/{}/combined_iter_{}_sample_{}.pdf'.format(self.options['results_dir'], img_idx, opt_step_num,i))


        # self.losses.append(loss.item())
        return loss

    def optimize_window(self, img_idx, data):
        self.full_results = []
        optimized_results = {}
        
        target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, \
            target_img_aug, source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = data

            ## First pass - get initial pose and depth estimates
        with torch.no_grad():
            batch_size = target_img.shape[0]
            imgs = torch.cat([target_img]+source_img_list,0)
            _, skips = self.depth_model(x=imgs, return_disp=False, epoch=50)
            disparities, _ = self.depth_model(x=None, skips=skips, epoch=50)
            
            skips = [skip.clone().detach() for skip in skips]
            fixed_skips = skips[0:-1]
            bottleneck_feat = skips[-1]

            target_disparity = disparities[0][0:batch_size]
            self.target_disparity = target_disparity.clone().detach()
            self.target_disparity_median = torch.median(self.target_disparity)
            source_disparities = [disparities[0][(i*batch_size):((i+1)*batch_size)] for i in range(1, len(source_img_list)+1)]
            disparities = [target_disparity] + source_disparities        
            depths = [disp_to_depth(disp, self.config['min_depth'], self.config['max_depth'])[1] for disp in disparities] ####.detach()

            poses, poses_inv, outputs = solve_pose_iteratively(self.config['iterations'], depths, self.pose_model, target_img, source_img_list, intrinsics, return_errors=True)

            h = 192
            w = 640  
            batch_size = target_img.shape[0]          
            optimized_results['poses_init'] = torch.cat(poses,0).cpu().detach()
            optimized_results['poses_inv_init'] = torch.cat(poses_inv,0).cpu().detach()
            optimized_results['gt_poses'] = torch.cat(gt_lie_alg_list,0).cpu().detach()
            optimized_results['gt_poses_inv'] = -torch.cat(gt_lie_alg_list,0).cpu().detach()
            optimized_results['depths_init'] = [d.clone() for d in depths]
            optimized_results['stacked_poses_init'] = outputs['fwd']['poses'].cpu().detach()
            optimized_results['stacked_poses_inv_init'] = outputs['inv']['poses'].cpu().detach()
            source_imgs = torch.cat(source_img_list,0)        
            

            ## only use deepcopy if we're optimizing network weights. otherwise just use our trained model
        params = []          
        if self.options['optimize_depth_weights_bottleneck_beyond']==True or self.options['optimize_depth_weights_all']==True:                 
            depth_model = copy.deepcopy(self.depth_model).train(False).eval()
            params.append({'params': depth_model.parameters(), 'lr': self.options['lr']})
        elif self.options['optimize_depth_encoder'] == True:
            depth_model = copy.deepcopy(self.depth_model).train(False).eval()
            params.append({'params': depth_model.encoder.parameters(),'lr': self.options['lr']})
        else:
            depth_model = self.depth_model.train(False).eval()
        
            ## optimize all depth network weights
        if self.options['optimize_pose_weights_all'] == True:
            pose_model = copy.deepcopy(self.pose_model).train(False).eval()
            params.append({'params': pose_model.parameters(), 'lr': self.options['lr']})
        else:
            pose_model = self.pose_model.train(False).eval()
            
            ## optimize depth predictions instead of network weights
        if self.options['optimize_depth_pred']==True:
            disparities = torch.cat(disparities,1)
            disparities = torch.nn.functional.interpolate(disparities, (int(disparities.shape[2]/4), int(disparities.shape[3]/4)), mode='bilinear').clone().detach()
            disparities = disparities.requires_grad_()  
            params.append({'params': disparities, 'lr': self.options['lr']})     
            
            ## Optimize the bottleneck values, keeping the weights fixed
        if self.options['optimize_depth_bottleneck_values']==True:

            bottleneck_feat = bottleneck_feat.clone().detach()
            bottleneck_feat = bottleneck_feat.requires_grad_()
            
            bottleneck_feat_2 = fixed_skips[-1].clone().detach()
            bottleneck_feat_2 = bottleneck_feat_2.requires_grad_()
            params.append({'params': bottleneck_feat, 'lr': self.options['lr']})
            params.append({'params': bottleneck_feat_2, 'lr': self.options['lr']})

        optimizer = torch.optim.Adam(params)

            ## optimize for specified number of epochs
        for i in range(0,self.options['epochs']):
            optimizer.zero_grad()

                ## recompute the depth and pose every iteration because these both change when updating depth weights
            if self.options['optimize_depth_weights_all'] == True or self.options['optimize_depth_encoder'] == True:
                disparities_list, _ = depth_model(x=imgs, skips=None, epoch=50) #recompute everything if optimizing all weights
            if self.options['optimize_depth_weights_bottleneck_beyond'] == True:
                disparities_list, _ = depth_model(x=None, skips=skips, epoch=50) #use initial skip connections for only optimizing decoder
                
            if self.options['optimize_depth_bottleneck_values'] == True:
                skips = fixed_skips[0:-1] + [bottleneck_feat_2] + [bottleneck_feat]
                disparities_list, _ = depth_model(x=None, skips=skips, epoch=50) #use initial skip connections for only optimizing decoder
                                

            if self.options['optimize_depth_pred'] == False: #use original disparities (that are being optimized)
                target_disparity = disparities_list[0][0:batch_size]
                source_disparities = [disparities_list[0][(i*batch_size):((i+1)*batch_size)] for i in range(1, len(source_img_list)+1)]
                disparities_list = [target_disparity] + source_disparities    
            else:
                disparities_upsample = torch.nn.functional.interpolate(disparities, (int(disparities.shape[2]*4), int(disparities.shape[3]*4)), mode='bilinear')
                disparities_list = [disparities_upsample[:,i:i+1] for i in range(0, disparities.shape[1])]  
                target_disparity = disparities_list[0]
                source_disparities = disparities_list[1:]

            depths = [disp_to_depth(disp, self.config['min_depth'], self.config['max_depth'])[1] for disp in disparities_list] 
            poses, poses_inv, outputs = solve_pose_iteratively(self.config['iterations'], depths, \
                pose_model, target_img, source_img_list, intrinsics, return_errors=True)
            

            poses = torch.cat(poses,0)
            poses_inv = torch.cat(poses_inv,0)

            target_depth = depths[0]
            source_disparities = torch.cat(source_disparities,0)

            if self.options['mode'] == 'unscaled':   ##only need to do this if we are performing online rescaling 
                scale_factor = self.dgc(target_depth.clone().detach(), intrinsics, self.config['camera_height']/30.)
            else:
                scale_factor = torch.FloatTensor([1])

            if i==0:
                optimized_results['scale_factor_init'] = scale_factor.clone()

            loss = self.compute_optimization_loss(i,img_idx, target_img, target_disparity, outputs['fwd'], outputs['inv'])


                ## only backprop up until the last epoch, since we don't recompute results after backproping
            if i < self.options['epochs']-1:            
                loss.backward()
                optimizer.step()  

                ### get disparity predictions by averaging normal and flipped images if we want to do depth eval
            pred_disp = get_disp_for_eigen(depth_model, target_img, self.config)
            self.disp_avg_list.append(pred_disp)    
            self.pose_avg_list.append(poses.cpu().detach())
            self.pose_inv_avg_list.append(poses_inv.cpu().detach())

        
        ### after optimization, average the last N predictions
            if i > self.options['avg_final_epochs']:
                optimized_results['poses_opt'] = avg_final_predictions(self.pose_avg_list, self.options['avg_final_epochs'])
                optimized_results['poses_inv_opt'] = avg_final_predictions(self.pose_inv_avg_list, self.options['avg_final_epochs'])
                optimized_results['disp_opt'] =  avg_final_predictions(self.disp_avg_list, self.options['avg_final_epochs'])
                
                self.pose_avg_list = []
                self.pose_inv_avg_list = [] #make sure to clear the list after each sample is optimized       
                self.disp_avg_list = []
            else:
                optimized_results['poses_opt'] = self.pose_avg_list[-1]
                optimized_results['poses_inv_opt'] = self.pose_inv_avg_list[-1]
                optimized_results['disp_opt'] = self.disp_avg_list[-1]
                
            optimized_results['depths_opt'] = depths     
            optimized_results['stacked_poses_opt'] = outputs['fwd']['poses']
            optimized_results['stacked_poses_inv_opt'] = (outputs['inv']['poses'])
            optimized_results['scale_factor'] = scale_factor
            self.full_results.append(optimized_results.copy())
        
        del depth_model
        del pose_model
        
        return optimized_results
