import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


updates_tradaboostr2_vals_c1 = [13.61800435, 11.30848738, 9.63709887, 9.32638212, 9.19932445]
updates_tradaboostr2_vals_c2 = [11.02848196, 9.78700991, 9.05541820, 8.22941416, 8.24419890]
updates_tradaboostr2_vals_c3 = [9.86654808, 9.22546593, 8.54098844, 8.48342448, 8.29542260]

xlim_list = [2, 4, 6, 8, 10]
box = mlines.Line2D([], [], color = 'r', marker='s', linestyle='None',
                          markersize=10, label='Updated_two_stage_TrAdaBoost.R2')
plt.plot(xlim_list, updates_tradaboostr2_vals_c1, '-sk', color = 'r')
plt.plot(xlim_list, updates_tradaboostr2_vals_c2, '--sk', color = 'r')
plt.plot(xlim_list, updates_tradaboostr2_vals_c3, '-.sk', color = 'r')

# plt.plot(Adaboost_r2value, color = 'm')

# plt.xlim()
plt.xticks(np.arange(2, 12, 2.0))
plt.yticks(np.arange(7, 15, 1.0))
plt.xlabel('RMS Error', fontsize=16)
plt.ylabel('CV Folds', fontsize=16)
plt.legend(handles= [box])
plt.show()
