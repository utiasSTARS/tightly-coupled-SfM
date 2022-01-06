import math
import torch
import time
from torch.optim import Optimizer
import numpy as np
from liegroups import SE3, SO3
import pickle
import shutil
import sys
sys.path.insert(0,'..')
from utils.custom_transforms import *

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir + '/checkpoint.pt'
    torch.save(state, f_path)
    print('checkpoint saved to {}'.format(f_path))
    if is_best:
        best_fpath = best_model_dir + '/best_model.pt'
        shutil.copyfile(f_path, best_fpath)
        print('best model saved to {}'.format(best_fpath))
        
def load_ckp(checkpoint_fpath, load_best, pose_model, depth_model, optimizer):
    if load_best:
        checkpoint_fpath = '{}/best_model/best_model.pt'.format(checkpoint_fpath)
        epoch = 1 #restarting training with the pretrained model, so epoch goes to 1
    else:
        checkpoint_fpath = '{}/checkpoint.pt'.format(checkpoint_fpath)
        
    checkpoint = torch.load(checkpoint_fpath)
    pose_model.load_state_dict(checkpoint['pose_state_dict'])
    depth_model.load_state_dict(checkpoint['depth_state_dict'])
    best_loss = checkpoint['best_val_loss']
    ### if not loading the best model, load the most recent one and update the optimizer as well
    if load_best==False:
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch'] +1
    else:    
        best_loss = 1e5 ## since we're not continuing training checkpoint, validation loss resets

    return pose_model, depth_model, optimizer, epoch, best_loss         

def save_state(state, filename='test.pth.tar'):
    torch.save(state, filename)
    
def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch == 70 or epoch == 71 or epoch == 72 or epoch == 73 or epoch == 74 or epoch == 75:
        print('LR is reduced by {}'.format(0.5))
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            param_group['lr'] = param_group['lr']*0.5
            
    if epoch !=0 and epoch%lr_decay_epoch==0:
        print('LR is reduced by {}'.format(0.5))
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            param_group['lr'] = param_group['lr']*0.5

    return optimizer   

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper. Taken from Monodepth2
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(depth, min_depth, max_depth):
    """Convert a depth prediction back into a disparity prediction
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    
    disp = 1/depth
    
    unscaled_disp = (disp- min_disp)/(max_disp - min_disp) 
    return unscaled_disp

###Moving average filter of size W
def moving_average(a, n) : #n must be odd)
    if n == 1:
        return a
    else:
        for i in range(a.shape[1]):
            if (n % 2) == 0:
                n -=1
            ret = np.cumsum(a[:,i], dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            a[:,i] = np.pad(ret[n - 1:-2] / n , int((n-1)/2+1), 'edge')
        return a
    
    import torch
    
def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def save_traj_to_txt(traj, folder, filename):
    ## traj is N x 4 x 4 (SE3)
    ## output folder
    
    with open('{}/{}.txt'.format(folder, filename), 'w') as f:
        for i in range(0, traj.shape[0]):
            t = traj[i,0:3,:].reshape(-1)
            f.write(" ".join(str(x) for x in t) + '\n')
    
    f.close()
            