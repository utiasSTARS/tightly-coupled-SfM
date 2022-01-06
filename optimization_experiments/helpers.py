import torch
import numpy as np
from losses import SSIM_Loss
from models.stn import *
import time
from utils.learning_helpers import disp_to_depth, batch_post_process_disparity

def compute_photometric_error(target_img, source_img, target_depth, source_depth, pose, intrinsics):

    ssim_loss = SSIM_Loss()
    img_rec, valid_mask, projected_depth, computed_depth = inverse_warp2(source_img, target_depth, source_depth, -pose, intrinsics, 'zeros') #forwards
    diff_img = ( 0.15*(img_rec - target_img.clone().detach()).abs().clamp(0, 1) + 0.85*ssim_loss(target_img.clone().detach(), img_rec) ).mean(1,True)
    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)
    weight_mask = (1 - diff_depth)
    diff_img = diff_img 
    
    auto_mask = ( 0.15*(source_img - target_img.clone().detach()).abs().clamp(0, 1) + 0.85*ssim_loss(target_img.clone().detach(), source_img) ).mean(1,True)
    auto_mask = (diff_img < auto_mask).float()
    valid_mask = auto_mask*valid_mask
    
    results = {'diff_img': diff_img, 'img_rec': img_rec, 'valid_mask':valid_mask, 'weight_mask':weight_mask, 'poses': pose} 

    return results

def avg_final_predictions(pred_list, num):
    pred_list = pred_list[-num:]
    pred_avg = torch.zeros(pred_list[0].shape)
    for i in range(0, len(pred_list)):
        pred_avg += pred_list[i]

    pred_avg = pred_avg/len(pred_list)

    return pred_avg

def get_disp_for_eigen(depth_model, target_img, config):
    with torch.no_grad():
                # Post-processed results require each image to have two forward passes
        target_img = torch.cat((target_img, torch.flip(target_img, [3])), 0)        
        
        disparities, _ = depth_model(target_img, epoch=50)
        
        disps, _ = disp_to_depth(disparities[0], config['min_depth'], config['max_depth'])
        
        pred_disp = disps.cpu().detach()[:, 0].numpy()
        
        # if post_process == True:
        N = pred_disp.shape[0] // 2
        pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1]) 
    return pred_disp       