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
frame_skip_list = [0,1,2]
iteration_list = [1,2,3,4,6]
seq = '09_02'
model_names = ['1--1-iter', '2--2-iter', '3--3-iter', '4--4-iter', '4--6-iter']

data = load_obj('seq-{}-frame_skip_results'.format(seq))
logger_list = data['results']


### plot both in subplots
plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tight_layout()
f, axarr = plt.subplots(2, sharex=True, sharey=False)
axarr[0].tick_params(labelsize=23)
axarr[1].tick_params(labelsize=23)

# # make the y ticks integers, not floats
# xint = []
# locs, labels = plt.xticks()
# for each in locs:
#     xint.append(int(each))
# plt.xticks(xint)


l0 = []
for idx, log in enumerate(logger_list):
    t_mse_list = log['t_mse_list']  
    l,=axarr[0].plot(frame_skip_list, t_mse_list, linewidth=2, marker='o', label=model_names[idx].replace('--','/').replace('-',' '))
    l0.append(l)

l1=[]
for idx, log in enumerate(logger_list):
    r_mse_list = log['r_mse_list']  
    l,=axarr[1].plot(frame_skip_list, r_mse_list, linewidth=2, marker='o', label=model_names[idx].replace('--','/').replace('-',' '))
    l1.append(l)

axarr[0].set_ylabel('$t_{err}$ (\%)', fontsize=24)
axarr[1].set_ylabel('$r_{err}$ ($^o \slash 100$m)', fontsize=24)
axarr[1].set_xlabel('\# Frame Skips', fontsize=24)

# axarr[0].set_title('First Epoch', fontsize=19)
# axarr[1].set_title('Final Epoch', fontsize=19)
axarr[0].grid()        
axarr[1].grid()    

legend_labels = [m.replace('--','/').replace('-',' ') for m in model_names]
print(legend_labels)
plt.legend(l1, legend_labels, fontsize=18)

plt.subplots_adjust(hspace = 0.6)
plt.subplots_adjust(bottom=0.2)
plt.subplots_adjust(left=0.15)



f.suptitle('Frame Skip Experiment', fontsize=25)
plt.savefig('seq-{}-frame_skip_combined.pdf'.format(seq))