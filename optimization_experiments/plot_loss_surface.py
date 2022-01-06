import numpy as np
import torch
import sys
sys.path.append('../')

from helpers import compute_photometric_error
from vis import *
from losses import SSIM_Loss


def generate_loss_surface(data, depths, pose_vec, sample_trans=False, sample_yaw=False, plotting=False, results_dir=None):
    ssim_loss = SSIM_Loss()
    results = {}

    target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, \
        target_img_aug, source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = data
                
    source_img = source_img_list[0]
    
    if sample_trans == True:
        reconstruction_errors = []
        delta_list = torch.arange(-3*pose_vec[0,2].abs(),3*pose_vec[0,2].abs(), 6*pose_vec[0,2].abs()/50.)
        results['delta_list'] = delta_list.numpy()

    if sample_yaw ==True:
        reconstruction_errors_yaw = []
        delta_list_yaw = torch.arange(-0.02,0.02, 0.04/50)
        results['delta_list_yaw'] = delta_list_yaw.numpy()

    orig_errors = compute_photometric_error(target_img[0:1], source_img[0:1], depths[0], depths[1], pose_vec, intrinsics[0:1])
    diff_img = orig_errors['diff_img']*orig_errors['valid_mask']*orig_errors['weight_mask']
    original_error = diff_img.sum(3).sum(2) / orig_errors['valid_mask'].sum(3).sum(2)
    results['original_error'] = original_error.item()
    
    if sample_trans == True:
        best_error = original_error.clone()
        best_delta = torch.FloatTensor([0])
        best_pose_vec = pose_vec.clone()
        for num, delta in enumerate(delta_list):
            test_pose_vec = pose_vec.clone()
            test_pose_vec[:,2] += delta
            
            sample_errors = compute_photometric_error(target_img[0:1], source_img[0:1], depths[0], depths[1], test_pose_vec, intrinsics[0:1])
            diff_img = sample_errors['diff_img']*sample_errors['valid_mask']*sample_errors['weight_mask']
            sample_error = diff_img.sum(3).sum(2) / sample_errors['valid_mask'].sum(3).sum(2)
            mean_error = diff_img.sum(3).sum(2) / sample_errors['valid_mask'].sum(3).sum(2)
            reconstruction_errors.append(mean_error.cpu().detach().numpy())

            #compare each error image in batch with current best, and update 'best pose' if error is smallest:
            for d in range(0, mean_error.size(0)):
                if mean_error[d] < best_error[d]:
                    best_error[d] = mean_error[d].clone()
                    best_pose_vec[d] = test_pose_vec[d]
                    best_delta = delta.item()

        results['reconstruction_errors'] = np.array(reconstruction_errors).reshape(-1)
        results['best_trans_delta'] = best_delta
        results['best_pose_vec'] = best_pose_vec.cpu().numpy()
        results['best_error'] = best_error.item()  
        
    if sample_yaw == True:
        best_error = original_error.clone()
        best_yaw_delta = torch.FloatTensor([0])
        best_pose_vec = pose_vec.clone()        
        
        for num, yaw_delta in enumerate(delta_list_yaw):
            test_pose_vec = pose_vec.clone()
            test_pose_vec[:,4] += yaw_delta
            sample_errors = compute_photometric_error(target_img[0:1], source_img[0:1], depths[0], depths[1], test_pose_vec[0:1], intrinsics[0:1])
            diff_img = sample_errors['diff_img']*sample_errors['valid_mask']*sample_errors['weight_mask']
            sample_error = diff_img.sum(3).sum(2) / sample_errors['valid_mask'].sum(3).sum(2)
            mean_error = diff_img.sum(3).sum(2) / sample_errors['valid_mask'].sum(3).sum(2)
            reconstruction_errors_yaw.append(mean_error.cpu().detach().numpy())

            #compare each error image in batch with current best, and update 'best pose' if error is smallest:
            for d in range(0, mean_error.size(0)):
                if mean_error[d] < best_error[d]:
                    best_error[d] = mean_error[d].clone()
                    best_pose_vec[d] = test_pose_vec[d]
                    best_yaw_delta = yaw_delta.item()
        
        results['reconstruction_errors_yaw']=np.array(reconstruction_errors_yaw).reshape(-1)
        results['best_yaw_delta'] = best_yaw_delta
        results['best_pose_vec_yaw'] = best_pose_vec.cpu().numpy()
        results['best_error_yaw'] = best_error.item()

    return results
