import torch
import sys
sys.path.insert(0,'..')
from train_mono import Trainer
from validate import test_depth_and_reconstruction, test_trajectory

import models.pose_models as pose_models
import models.depth_models as depth_models
from utils.learning_helpers import *
from utils.custom_transforms import *
import losses
from vis import *
import numpy as np
import datetime
import time
from tensorboardX import SummaryWriter
import argparse
import torch.backends.cudnn as cudnn
import os
import glob
 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
parser = argparse.ArgumentParser(description='training arguments.')

'''System Options'''
parser.add_argument('--flow_type', type=str, default='none', help='classical, or none')  ##none used for this paper.
parser.add_argument('--num_scales', type=int, default=1) ## loss computed only at the highest resolution scale.
parser.add_argument('--img_resolution', type=str, default='med') # low (128x448) med (192 x640) or high (256 x 832) 
parser.add_argument('--img_per_sample', type=int, default=3) # one target image, and rest are source images - currently fixed at 3
parser.add_argument('--iterations', type=int, default=1) ## number of egomotion network iterations.

'''Training Arguments'''
parser.add_argument('--data_dir', type=str, default='/path/to/kitti/preprocessed')
parser.add_argument('--data_format', type=str, default='odometry') #odmetry or eigen or scannet
parser.add_argument('--date', type=str, default='0000000')
parser.add_argument('--train_seq', nargs='+', type=str, default=['00_02', '02_02'])
parser.add_argument('--val_seq', nargs='+',type=str, default=['05_02'])
parser.add_argument('--test_seq', nargs='+', type=str, default=['09_02'])
parser.add_argument('--augment_motion', action='store_true', default=False)
parser.add_argument('--minibatch', type=int, default=12) #60
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--lr', type=float, default=9e-4)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lr_decay_epoch', type=float, default=4)
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--max_depth', type=float, default=80/30) 
parser.add_argument('--min_depth', type=float, default=0.1) 

''' Losses'''
parser.add_argument('--l_reconstruction', action='store_true', default=True, help='use photometric reconstruction losses (l1, ssim)')
parser.add_argument('--l_ssim', action='store_true', default=True, help='without ssim, only use L1 error')
parser.add_argument('--l1_weight', type=float, default=0.15) #0.15
parser.add_argument('--l_ssim_weight', type=float, default=0.85) #0.85
parser.add_argument('--with_auto_mask', action='store_true', default=True, help='with the the mask for stationary points')

parser.add_argument('--l_pose_consist', action='store_true', default=True, help='ensure forward and backward pose predictions align')
parser.add_argument('--l_pose_consist_weight', type=float, default=5)
parser.add_argument('--l_inverse', action='store_true', default=True, help='reproject target image to source images as well')
parser.add_argument('--l_depth_consist', action='store_true', default=False, help='Depth consistency loss from https://arxiv.org/pdf/1908.10553.pdf')
parser.add_argument('--l_depth_consist_weight', type=float, default=0.14) 
parser.add_argument('--with_depth_mask', action='store_true', default=False, help='with the depth consistency mask for moving objects and occlusions or not')
parser.add_argument('--camera_height', type=float, default=1.70) #1.52 for oxford, 1.70 for KITTI
parser.add_argument('--l_smooth', action='store_true', default=True)
parser.add_argument('--l_smooth_weight', type=float, default=0.05) #0.15

'''
'results/initial-posenet-kitti-192x640' contains partially trained egomotion network weights
make sure to load this network if training a model from scratch.
This is not a required step, but is recommended to guarantee proper initialization of the networks.
Otherwise, just load the model we pretrained on oxford robotcar prior to training on KITTI or ScanNet 
'''
parser.add_argument('--load_from_checkpoint', action='store_true', default=False)
parser.add_argument('--load_best_model', action='store_true', default=True, help='load the best model and start at epoch 1, instead of resuming from the final checkpoint')
parser.add_argument('--pretrained_dir', type=str, default='results/oxford-pretrained-192x640')      
parser.add_argument('--pretrained_plane_dir', type=str, default='') 
        
args = parser.parse_args()
config={
    'num_frames': None,
    'skip':1,    ### if not one, we skip every 'skip' samples that are generated ({1,2}, {2,3}, {3,4} becomes {1,2}, {3,4})
    'correction_rate': 1, ### if not one, only perform corrections every 'correction_rate' frames (samples become {1,3},{3,5},{5,7} when 2)
    'freeze_posenet': False,
    'freeze_depthnet': False,
    }
for k in args.__dict__:
    config[k] = args.__dict__[k]

print('Starting training')

data_list={}
data_list['train'] = args.train_seq
data_list['val'] = args.val_seq
data_list['test'] = args.test_seq

args.data_dir = '{}/{}_res'.format(args.data_dir, config['img_resolution'])
config['data_dir'] = '{}/{}_res'.format(config['data_dir'], config['img_resolution'])

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id) ##ensure that our data augmentation is truly random
    
if args.data_format == 'odometry' or args.data_format == 'scannet': ## same dataloader for both.
    from data.kitti_loader_stereo import KittiLoaderPytorch #loads left and right images as separate sequences
    dsets = {x: KittiLoaderPytorch(config, data_list, mode=x, transform_img=get_data_transforms(config)[x], \
                                augment=config['augment_motion']) for x in ['train', 'val']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=config['minibatch'], shuffle=True, num_workers=8, worker_init_fn=worker_init_fn) for x in ['train', 'val']}

    val_dset = KittiLoaderPytorch(config, data_list, mode='val', transform_img=get_data_transforms(config)['val'])
    val_dset_loaders = torch.utils.data.DataLoader(val_dset, batch_size=config['minibatch'], shuffle=False, num_workers=8, worker_init_fn=worker_init_fn)

    test_dset = KittiLoaderPytorch(config, data_list, mode='test', transform_img=get_data_transforms(config)['test'])
    test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=8, worker_init_fn=worker_init_fn)

    eval_dsets = {'val': val_dset_loaders, 'test':test_dset_loaders}

if args.data_format == 'eigen':
    from data.kitti_loader_eigen import KittiLoaderPytorch
    dsets = {x: KittiLoaderPytorch(config, None, mode=x, transform_img=get_data_transforms(config)[x], \
                                augment=config['augment_motion'], skip=config['skip']) for x in ['train', 'val']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=config['minibatch'], shuffle=True, num_workers=8) for x in ['train', 'val']}

    val_dset = KittiLoaderPytorch(config, None, mode='val', transform_img=get_data_transforms(config)['val'])
    val_dset_loaders = torch.utils.data.DataLoader(val_dset, batch_size=config['minibatch'], shuffle=False, num_workers=8)

    eval_dsets = {'val': val_dset_loaders}    

def main():
    results = {}
    config['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    start = time.time()
    now= datetime.datetime.now()
    ts = '{}-{}-{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute)
    start_epoch = 0
    val_writer=None
    train_writer = None
    best_val_loss = 1e5


    ### initialize the models
    depth_model = depth_models.depth_model(config).to(config['device'])
    pose_model = pose_models.pose_model(config).to(config['device'])
    
        ### freeze weights if desired
    if config['freeze_depthnet']: print('Freezing depth network weights.')
    if config['freeze_posenet']: print('Freezing pose network weights.')
    for param in depth_model.parameters():
        param.requires_grad = not config['freeze_depthnet']         
    for param in pose_model.parameters():
        param.requires_grad = not config['freeze_posenet']  

        ### setup loss and optimizer
    params = [{'params': depth_model.parameters()},
                {'params': pose_model.parameters(), 'lr': 2*config['lr']}]
    loss = losses.Compute_Loss(config)
    optimizer = torch.optim.Adam(params, lr=config['lr'], weight_decay = config['wd']) #, amsgrad=True)
    
    
        ### load pretrained model or resume from a checkpoint
    if config['load_from_checkpoint'] == True or config['load_best_model'] == True:
        pose_model, depth_model, optimizer, start_epoch, best_val_loss = load_ckp(config['pretrained_dir'], config['load_best_model'], pose_model, depth_model, optimizer)
        print('loading checkpoint from {}, starting at epoch {}'.format(config['pretrained_dir'], start_epoch))
    epochs = range(start_epoch,config['num_epochs'])
    models = [depth_model, pose_model]

    trainer = Trainer(config, models, loss, optimizer)
    cudnn.benchmark = True

    for epoch in epochs:
        np.random.seed(epoch) ##update seed for ensuring randomness in data augmentation

        optimizer = exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=config['lr_decay_epoch']) ## reduce learning rate as training progresses  
        print("Epoch {}".format(epoch))
        train_losses = trainer.forward(dset_loaders['train'], epoch, 'train')
        with torch.no_grad():
            val_losses = trainer.forward(dset_loaders['val'], epoch, 'val')    

        if val_writer is None:
            val_writer = SummaryWriter(comment="tw-val-{}-test_seq-{}_val".format(args.val_seq[0], args.test_seq[0]))
        if train_writer is None:
            train_writer = SummaryWriter(comment="tw-val-{}-test_seq-{}_train".format(args.val_seq[0], args.test_seq[0]))
            
        if train_losses is not None and val_losses is not None:
            for key, value in train_losses.items():
                train_writer.add_scalar('{}'.format(key), value, epoch+1)
                val_writer.add_scalar('{}'.format(key), val_losses[key], epoch+1)

        for key, dset in eval_dsets.items():
            print("{} Set, Epoch {}".format(key, epoch))
        
            if epoch > 0: 
                ###plot images, depth map, explainability mask
                img_array, disparity, exp_mask, d = test_depth_and_reconstruction(device, models, dset, config, epoch=epoch)
                
                source_disp, reconstructed_disp, d_masks = d[0], d[1], d[2]
                img_array = plot_img_array(img_array)
                train_writer.add_image(key+'/imgs',img_array,epoch+1) 
                for i,d in enumerate(disparity):
                    train_writer.add_image(key+'/depth-{}/target-depth'.format(i), plot_disp(d), epoch+1)

                    ### For depth consistency
                    train_writer.add_image(key+'/depth-{}/source-depth'.format(i), plot_disp(source_disp[i]), epoch+1)
                    train_writer.add_image(key+'/depth-{}/reconstructed-depth'.format(i), plot_disp(reconstructed_disp[i]), epoch+1)
                   
                d_masks = plot_img_array(d_masks)
                train_writer.add_image(key+'/depth/masks', d_masks, epoch+1)

                exp_mask = plot_img_array(exp_mask)
                train_writer.add_image(key+'/exp_mask', exp_mask, epoch+1)
                                    
                ###evaluate trajectories 
                if args.data_format == 'odometry':   
                    est_lie_alg, gt_lie_alg, est_traj, gt_traj, errors = test_trajectory(config, device, models, dset, epoch)
              
                    errors = plot_6_by_1(est_lie_alg - gt_lie_alg, title='6x1 Errors')  
                    pose_vec_est = plot_6_by_1(est_lie_alg, title='Est')   
                    pose_vec_gt = plot_6_by_1(gt_lie_alg, title='GT')    
                    est_traj_img = plot_multi_traj(est_traj, 'Est.', gt_traj, 'GT', key+' Set')

                    train_writer.add_image(key+'/est_traj', est_traj_img, epoch+1)
                    train_writer.add_image(key+'/errors', errors, epoch+1)
                    train_writer.add_image(key+'/gt_lie_alg', pose_vec_gt, epoch+1)
                    train_writer.add_image(key+'/est_lie_alg', pose_vec_est, epoch+1)
    


                if args.save_results and key == 'val':   ##Save the best models
                    os.makedirs('results/{}/best_model'.format(config['date']), exist_ok=True)

                    checkpoint = {
                        'epoch': epoch,
                        'pose_state_dict': pose_model.state_dict(),
                        'depth_state_dict': depth_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss
                    }

                    if (val_losses['l_reconstruct_forward'] + val_losses['l_reconstruct_inverse']) < best_val_loss and epoch > 0: # and epoch > 2*(config['iterations']-1):
                        is_best = True
                        best_val_loss = (val_losses['l_reconstruct_forward'] + val_losses['l_reconstruct_inverse'])
                        print("Lowest validation loss (saving new best model)")   
                    else:
                        is_best = False

                    ### save our config file
                    save_obj(config, 'results/{}/config'.format(config['date']))
                    f = open("results/{}/config.txt".format(config['date']),"w")
                    f.write( str(config) )
                    f.close()
                    
                    ## save models and checkpoint
                    save_ckp(checkpoint, is_best, 'results/{}'.format(config['date']), 'results/{}/best_model'.format(config['date']))
    
    duration = timeSince(start)    
    print("Training complete (duration: {})".format(duration))
 
main()
