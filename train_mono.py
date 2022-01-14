import time
import torch
from utils.learning_helpers import *
from models.stn import *
from data.kitti_loader import process_sample_batch
from losses import SSIM_Loss

def compute_pose_consistency_loss(poses, poses_inv):
    pose_consistency_loss = 0

    for pose, pose_inv in zip(poses, poses_inv):
        t_s1 = pose[:,0:6]
        t_s1_inv = pose_inv[:,0:6]
        pose_consistency_loss += (t_s1 + t_s1_inv).abs()

    return pose_consistency_loss.mean()

def solve_pose(pose_model, target_img, source_img_list, flow_imgs):
    poses, poses_inv = [], []

    flow_imgs_fwd, flow_imgs_back = flow_imgs
    for source_img, flow_img_fwd, flow_img_back in zip(source_img_list, flow_imgs_fwd, flow_imgs_back):        
        if flow_imgs_fwd[0] != None:
            fwd_imgs = [target_img, source_img, flow_img_fwd] # get rid third img (the optical flow image) if not wanted
            back_imgs = [source_img, target_img, flow_img_back] 
        else:
            fwd_imgs = [target_img, source_img]
            back_imgs = [source_img, target_img]
        
        fwd_imgs = torch.cat(fwd_imgs,1)
        back_imgs = torch.cat(back_imgs,1)
        
        pose = pose_model(fwd_imgs)
        pose_inv = pose_model(back_imgs)

        poses.append(pose)
        poses_inv.append(pose_inv)
        
    return poses, poses_inv

def solve_pose_iteratively(num_iter, depths, pose_model, target_img, source_img_list, intrinsics, return_errors=False):
    num_source_imgs = len(source_img_list)
    batch_size = target_img.shape[0]
    split = num_source_imgs*batch_size
    depth, source_depths = depths[0], depths[1:] #separate disparity list into source and target disps

    outputs = {'fwd':{}, 'inv':{}}
    poses, poses_inv = [], []
    
    diff_imgs, weight_masks, img_recs, valid_masks = [], [], [], []
    diff_imgs_inv, weight_masks_inv, img_recs_inv, valid_masks_inv = [], [], [], []
    ssim_loss = SSIM_Loss()

    source_depths = torch.cat(source_depths,0)

    target_depths = depth.repeat(num_source_imgs,1,1,1)
    source_imgs = torch.cat(source_img_list,0) 
    intrinsics = intrinsics.repeat(2*num_source_imgs, 1,1)
    target_imgs = target_img.repeat(num_source_imgs,1,1,1)
    fwd_imgs = torch.cat([target_imgs, source_imgs],1)
    inv_imgs = torch.cat([source_imgs, target_imgs],1)
    imgs = torch.cat([fwd_imgs, inv_imgs],0)

    full_poses = pose_model(imgs)

    target_depth_full = torch.cat([target_depths, source_depths],0)
    source_depth_full = torch.cat([source_depths, target_depths],0)

    img_rec, valid_mask, projected_depth, computed_depth = inverse_warp2(imgs[:,3:6], target_depth_full, source_depth_full, -full_poses, intrinsics, 'zeros') #forwards
    
    stacked_poses = full_poses.clone().unsqueeze(1)

    for i in range(0,num_iter-1):
        new_imgs = imgs.clone()
        new_imgs[:,0:3] = new_imgs[:,0:3]*valid_mask
        new_imgs[:,3:6] = img_rec
        full_pose_corr = pose_model(new_imgs)
        full_poses = full_poses + full_pose_corr
        stacked_poses = torch.cat([stacked_poses,full_poses.clone().unsqueeze(1)],1)
        img_rec, valid_mask, projected_depth, computed_depth = inverse_warp2(imgs[:,3:6], target_depth_full, source_depth_full, -full_poses, intrinsics, 'zeros') #forwards

    if return_errors:
        
        auto_mask_error = (0.15*(imgs[:,0:3] - imgs[:,3:6]).abs().clamp(0, 1) + 0.85*ssim_loss(imgs[:,0:3], imgs[:,3:6]) ).mean(1,True)
 
        ## first 3 channels of imgs always correspond to the reconstruction target (target img is not always the reconstruction target)
        diff_imgs_full = (0.15*(img_rec - imgs[:,0:3].clone().detach()).abs().clamp(0, 1) + 0.85*ssim_loss(imgs[:,0:3].clone().detach(), img_rec) ).mean(1,True)
 
        auto_mask = (diff_imgs_full < auto_mask_error).float()
        
        diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)
        weight_masks = (1 - diff_depth)
        
        outputs['fwd'] = {'diff_img': diff_imgs_full[0:split], 'img_rec': img_rec[0:split], \
            'valid_mask':valid_mask[0:split], 'weight_mask':weight_masks[0:split], \
                'poses': stacked_poses[0:split], 'auto_mask_error': auto_mask_error[0:split], 'auto_mask':auto_mask[0:split]} 
        
        outputs['inv'] = {'diff_img': diff_imgs_full[split:], 'img_rec': img_rec[split:], \
            'valid_mask':valid_mask[split:], 'weight_mask':weight_masks[split:], \
                'poses': stacked_poses[split:], 'auto_mask_error': auto_mask_error[split:], 'auto_mask': auto_mask[split:]} 
         
        new_imgs = imgs.clone()
        new_imgs[:,0:3] = new_imgs[:,0:3]*valid_mask
        new_imgs[:,3:6] = img_rec
        outputs['comb'] = {'imgs': new_imgs, 'valid_mask':valid_mask}


    poses = stacked_poses[0:split].clone()
    poses_inv = stacked_poses[split:].clone()
    
    poses = [poses[batch_size*i:(batch_size*(i+1))] for i in range(0,num_source_imgs)]
    poses_inv = [poses_inv[batch_size*i:(batch_size*(i+1))] for i in range(0, num_source_imgs)]
    
    poses = [poses[i][:,-1] for i in range(0,num_source_imgs)]
    poses_inv = [poses_inv[i][:,-1] for i in range(0, num_source_imgs)]
            
    if return_errors:
        return poses, poses_inv, outputs
    else:
        return poses, poses_inv
    
## compute disparities in same batch    
def solve_disp(depth_model, target_img, source_img_list):   
    batch_size = target_img.shape[0]   
    imgs = torch.cat([target_img] + source_img_list,0)
    disparities = depth_model(imgs)
    target_disparities = [disp[0:batch_size] for disp in disparities]
    source_disp_1 = [disp[batch_size:(2*batch_size)] for disp in disparities]
    source_disp_2 = [disp[2*batch_size:(3*batch_size)] for disp in disparities]
    
    disparities = [target_disparities, source_disp_1, source_disp_2]
    return disparities    

class Trainer():
    def __init__(self, config, models, loss, optimizer):
        self.config = config
        self.device = config['device']
        self.depth_model = models[0]
        self.pose_model = models[1]
        self.optimizer = optimizer
        self.loss = loss


    def forward(self, dset, epoch, phase):
        dev = self.device
        start = time.time()
        if phase == 'train' and self.config['freeze_depthnet'] is False: self.depth_model.train(True)
        else:  
            self.depth_model.train(False)  
            self.depth_model.eval()
        if phase == 'train' and self.config['freeze_depthnet'] is False: self.pose_model.train(True)
        else:
            self.pose_model.train(False)
            self.pose_model.eval()

        dset_size = len(dset)
        running_loss = None         
            # Iterate over data.
        for batch_num, data in enumerate(dset):
            target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, target_img_aug, \
            source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = process_sample_batch(data, self.config)
                
            pose = []
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                disparities = solve_disp(self.depth_model, target_img_aug, source_img_aug_list)
                
                if disparities[0][0].median() <=0.0000001 and disparities[0][0].mean() <=0.00000001:
                    print("warning - depth est has failed")

                depths = [disp_to_depth(disp[0], self.config['min_depth'], self.config['max_depth'])[1] for disp in disparities]

                poses, poses_inv = solve_pose_iteratively(self.config['iterations'], depths, self.pose_model, target_img_aug, source_img_aug_list, intrinsics_aug)
              
                minibatch_loss=0  
                losses = self.loss(source_img_list, target_img, [poses, poses_inv], disparities, intrinsics_aug,epoch=epoch)
                                     
                                        
                if self.config['l_pose_consist']==True:
                    losses['l_pose_consist'] = self.config['l_pose_consist_weight']*compute_pose_consistency_loss(poses, poses_inv)
                    losses['total'] += losses['l_pose_consist']
                    
                minibatch_loss += losses['total']
                
                if running_loss is None:
                    running_loss = losses
                else:
                    for key, val in losses.items():
                        if val.item() != 0:
                            running_loss[key] += val.data
                                
                if phase == 'train':   
                    minibatch_loss.backward()
                    self.optimizer.step()
        
        print("{} epoch completed in {} seconds.".format(phase, timeSince(start)))  
        if epoch > 0:                     
            for key, val in running_loss.items():
                running_loss[key] = val.item()/float(batch_num)
            print('{} Loss: {:.6f}'.format(phase, running_loss['total']))
            return running_loss
        else:
            return None
