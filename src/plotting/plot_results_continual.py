import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


warm = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DWM_v2\continual\warm_3000.csv')
derpp = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DWM_v2\continual\derpp_3000.csv')
ewc = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DWM_v2\continual\ewc_3000.csv')
lwf = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DWM_v2\continual\lwf_3000.csv')
sp = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DWM_v2\continual\sp_3000.csv')
expected = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DWM_v2\continual\retrain_3000.csv')



fig=plt.figure()
ax=fig.add_subplot(111)

#for i in range(len(warm['macro_mean']) - 1):
    #warm['macro_mean'][i+1] -= 0.015
#for i in range(len(derpp['macro_mean']) - 1):
    #derpp['macro_mean'][i+1] += 0.015
#for i in range(len(lwf['macro_mean']) - 1):
    #lwf['macro_mean'][i+1] += 0.015


ax.plot(np.cumsum(np.asarray(warm['nr_samples'])).tolist(),warm['macro_mean'],c='k',ls='-',label='Warm-start',fillstyle='none')
ax.fill_between(warm['nr_samples'], (warm['macro_mean']-2*warm['macro_std']), (warm['macro_mean']+2*warm['macro_std']), color='k', alpha=.1)

ax.plot(np.cumsum(np.asarray(ewc['nr_samples'])).tolist(),ewc['macro_mean'],c='g',ls='--',label='EWC',fillstyle='none')
ax.fill_between(np.cumsum(np.asarray(ewc['nr_samples'])).tolist(), (ewc['macro_mean']-2*ewc['macro_std']), (ewc['macro_mean']+2*ewc['macro_std']), color='g', alpha=.1)

ax.plot(np.cumsum(np.asarray(lwf['nr_samples'])).tolist(),lwf['macro_mean'],c='b',ls='--',label='LwF')
ax.fill_between(np.cumsum(np.asarray(lwf['nr_samples'])).tolist(), (lwf['macro_mean']-2*lwf['macro_std']), (lwf['macro_mean']+2*lwf['macro_std']), color='b', alpha=.1)

ax.plot(expected['nr_samples'],expected['macro_mean'],c='orangered',ls=':',label='Retrain')
ax.fill_between(expected['nr_samples'], (expected['macro_mean']-2*expected['macro_std']), (expected['macro_mean']+2*expected['macro_std']), color='orangered', alpha=.1)

ax.plot(np.cumsum(np.asarray(sp['nr_samples'])).tolist(),sp['macro_mean'],c='lightseagreen',ls='-',label='S&P')
ax.fill_between(np.cumsum(np.asarray(sp['nr_samples'])).tolist(), (sp['macro_mean']-2*sp['macro_std']), (sp['macro_mean']+2*sp['macro_std']), color='lightseagreen', alpha=.1)

ax.plot(np.cumsum(np.asarray(derpp['nr_samples'])).tolist(),derpp['macro_mean'],c='darkred',ls='-.',label='DER++')
ax.fill_between(np.cumsum(np.asarray(derpp['nr_samples'])).tolist(), (derpp['macro_mean']-2*derpp['macro_std']), (derpp['macro_mean']+2*derpp['macro_std']), color='darkred', alpha=.1)

plt.axvline(x=1339, color='k', ls=':')
#plt.axvline(x=3339, color='k', ls=':')
#plt.axvline(x=5339, color='k', ls=':')
plt.axvline(x=7339, color='k', ls=':')
#plt.axvline(x=2339, color='k', ls=':')
plt.axvline(x=4339, color='k', ls=':')
#plt.axvline(x=6339, color='k', ls=':')
#plt.axvline(x=8339, color='k', ls=':')

ax.set_ylabel("Macro F1")
ax.set_xlabel("Number of samples")

plt.legend()
fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\CL\macro_3000.png')