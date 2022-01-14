# Tightly Coupled SfM

Accompanying code for 'On the Coupling of Depth and Egomotion Networks for Self-Supervised Structure from Motion'

## Dependencies:
* numpy
* scipy
* [pytorch](https://pytorch.org/) 
* [liegroups](https://github.com/utiasSTARS/liegroups)
* [pyslam](https://github.com/utiasSTARS/pyslam)
* [tensorboardX](https://github.com/lanpa/tensorboardX) (optional)

# Datasets

We trained and tested on the KITTI dataset. Download the raw dataset (see http://www.cvlibs.net/datasets/kitti/raw_data.php). We provide a dataloader, but we first require that the data be preprocessed. To do so, run `create_kitti_odometry_data.py` within the `data` directory (be sure to specify the source and target directory). For training with the Eigen split, first run `create_kitti_eigen_data.py`. 

For ScanNet experiments, reproduction is more challenging due to the amount of data required. If interested, download the data (see https://github.com/ScanNet/ScanNet), and unpack the data (see https://github.com/ScanNet/ScanNet/tree/master/SensReader/c++) prior to running `create_scannet_dataset.py` to preprocess the data. 


# Paper Reproduction

Please download our trained models from https://drive.google.com/drive/folders/1Tkq3PSSwqMLGsgibbt23f2WhO_UUpvOu?usp=sharing and unpack it into the main directory. 

We provide three trained models: `kitti-odometry-4-iter` is our best odometry model, `kitti-odometry-4-iter` is our best model trained on the Eigen split, and `scannet-4-iter` is our model trained on ScanNet. We provide several scripts to evaluate these models.  In all of these scripts, the path to the main workspace, and to the downloaded datasets must be specified in order to run these scripts.

From within the `paper_plots_and_data` directory: 

`evaluate_vo_model.py` can be run to process the KITTI odometry results on sequences 09 and 10. Note that this evaluates the direct model output without inference-time optimization (PFT). *this will generate Table 5 row (4/4)*

`evaluate_depth_eigen.py` can be run in order to evaluate depth for the Eigen test set. Note that `export_gt_depth_kitti_eigen.py` must be run from the `data` directory to first process the ground-truth depths. *this will generate the table 5, 4-iteration, non-optimized result in the supplementary material*

`evaluate_error_scannet.py` can be run to evaluate depth and pose accuracy on ScanNet. *this will generate the '4/8 iter. (no PFT)' row in table 2*

From within the `optimization_experiments` directory:

run `run_sample_optimization_demo.py` to optimize over a minibatch, with data that we provide (note that no dataset must be downloaded to run this demo). This will generate results in `results/kitti-odometry-4-iter/results/depth_opt_single`. Included are the initial and optimized depth and error images, as well as the 1D loss surfaces (figure 3b in the manuscript)

For the full optimization experiments, run `run_sequential_optimization.py`. Note that this can be used to optimize over the odometry sequences (default), but the `data_format` variable can be switched to `eigen` or `scannet` to optimize the results for the test sequences. If `eigen` or `scannet` is used, the optimized data will be saved within `paper_plots_and_data`, and can be evaluted using `evaluate_depth_eigen.py` and `evaluate_error_scannet.py` (make sure to set `load_pred_disps=True` in these scripts to evaluate the optimized data). For the odometry mode, set `load_from_mat=True` to produce the optimized results without having to perform the actual optimization, and without having to download any datasets.

    * `odometry` format will generate our result in Table 3 *
    * `eigen` format will generate our result in Table 6 *
    * `scannet` format will generate our main result in Table 2 *

# Training

Two bash scripts are provided that will run the training experiments for KITTI and ScanNet. Ensure that the dataset directory is updated prior to running, and that your data has been preprocessed (see Datasets section)

`run_mono_exps_kitti.sh`
`run_scannet_exps.sh`

We provide the pretrained Oxford robotcar model that can be used to jumpstart the training. Note that starting with a pretrained model is important for us, because we evaluate the loss in the the forward direction (source->target) and the inverse direction (target->source) which requires the pose network to learn the difference between f(I_s,I_t) and f(I_t,I_s) which is not trivial. To ensure stability in the first epoch, using a pretrained model (that has been trained to distinguish forward poses from inverse poses) simplified our experiments. 

During training, to visualize some results during the training procedure, open a tensorboard from the main directory:

`tensorboard --logdir runs` 
