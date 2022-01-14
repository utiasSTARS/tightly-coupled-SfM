import sys
sys.path.append('../../')
from utils.learning_helpers import save_obj, load_obj
import matplotlib.pylab as plt

plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tight_layout()
plt.savefig('test.png')

seq='09_02'

trans_perturbation_list = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5] #in meters 
yaw_perturbation_list = [0, 0.1, 0.25, 0.5, 1, 3, 5]
model_names = ['1-iter', '2-iter','3-iter']


trans_perturb_file = 'seq-{}-perturbation_results_trans_True_yaw_False'.format(seq)
yaw_perturb_file = 'seq-{}-perturbation_results_trans_False_yaw_True'.format(seq)

trans_data = load_obj(trans_perturb_file)
yaw_data = load_obj(yaw_perturb_file)


trans_logger_list = trans_data['results']
print(len(trans_logger_list))
yaw_logger_list = yaw_data['results']
### plot both in subplots
plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tight_layout()
f, axarr = plt.subplots(2, sharex=False, sharey=False)
axarr[0].tick_params(labelsize=20)
axarr[1].tick_params(labelsize=20)

l0 = []
l1 = []
for idx, log in enumerate(trans_logger_list):
    t_mse_list = log['t_mse_list']  
    l,=axarr[0].plot(trans_perturbation_list, t_mse_list, linewidth=2, marker='o', label=model_names[idx].replace('--','/').replace('-',' '))
    l0.append(l)
    
    # r_mse_list = log['r_mse_list']
    # l,=axarr[1].plot(trans_perturbation_list, r_mse_list, linewidth=2, marker='o', label=model_names[idx].replace('--','/').replace('-',' '))
    # l1.append(l)    

l1=[]
for idx, log in enumerate(yaw_logger_list):
    t_mse_list = log['t_mse_list'] 
    l,=axarr[1].plot(yaw_perturbation_list, t_mse_list, linewidth=2, marker='o', label=model_names[idx].replace('--','/').replace('-',' '))
    l1.append(l)

l1.append(l)
   
axarr[0].set_ylabel('$t_{err}$ (\%)', fontsize=19)
axarr[1].set_ylabel('$t_{err}$ (\%)', fontsize=19)
axarr[0].set_xlabel('Trans. Perturb. Range (m)', fontsize=19)
axarr[1].set_xlabel('Yaw Perturb. Range ($^o$)', fontsize=19)

# axarr[0].set_title('First Epoch', fontsize=19)
# axarr[1].set_title('Final Epoch', fontsize=19)
axarr[0].grid()        
axarr[1].grid()    

legend_labels = [m.replace('--','/').replace('-',' ') for m in model_names]
print(legend_labels)
plt.legend(l1, legend_labels, fontsize=16)

plt.subplots_adjust(hspace = 0.8)
plt.subplots_adjust(bottom=0.2)
plt.subplots_adjust(left=0.15)

f.suptitle('Pose Perturbation Experiment', fontsize=27)

plt.savefig('seq-{}-perturbation_combined.pdf'.format(seq))
