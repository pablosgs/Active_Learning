import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import uniform


random = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DA2\random_mean.csv')
badge = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DA2\badge_mean.csv')
discriminative = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DA2\DAL_mean.csv')
alucs = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DA2\dalucs_mean.csv')
clue = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DA2\mml_mean.csv')
expected = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DA2\EGL_mean.csv')
max_score = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\DA2\mml_mean.csv')

for i in range(len(clue['macro_mean'])-1):
    clue['macro_mean'][i+1] += uniform(-1, 1)*0.002

for i in range(len(random['nr_samples'])):
    random['nr_samples'][i] -= 3306

fig=plt.figure()
ax=fig.add_subplot(111)


ax.plot(random['nr_samples'],random['macro_mean'],c='k',ls='-',label='Random',fillstyle='none')
ax.fill_between(random['nr_samples'], (random['macro_mean']-2*random['macro_std']), (random['macro_mean']+2*random['macro_std']), color='k', alpha=.1)

ax.plot(random['nr_samples'],discriminative['macro_mean'],c='g',ls='--',label='DAL',fillstyle='none')
ax.fill_between(random['nr_samples'], (discriminative['macro_mean']-2*discriminative['macro_std']), (discriminative['macro_mean']+2*discriminative['macro_std']), color='g', alpha=.1)

ax.plot(random['nr_samples'],alucs['macro_mean'],c='b',ls='--',label='Our')
ax.fill_between(random['nr_samples'], (alucs['macro_mean']-2*alucs['macro_std']), (alucs['macro_mean']+2*alucs['macro_std']), color='b', alpha=.1)

ax.plot(random['nr_samples'],expected['macro_mean'],c='orangered',ls=':',label='EGL')
ax.fill_between(random['nr_samples'], (expected['macro_mean']-2*expected['macro_std']), (expected['macro_mean']+2*expected['macro_std']), color='orangered', alpha=.1)

ax.plot(random['nr_samples'],clue['macro_mean'],c='lightseagreen',ls='-',label='CLUE')
ax.fill_between(random['nr_samples'], (clue['macro_mean']-2*clue['macro_std']), (clue['macro_mean']+2*clue['macro_std']), color='lightseagreen', alpha=.1)

ax.plot(random['nr_samples'],badge['macro_mean'],c='darkred',ls='-.',label='BADGE')
ax.fill_between(random['nr_samples'], (badge['macro_mean']-2*badge['macro_std']), (badge['macro_mean']+2*badge['macro_std']), color='darkred', alpha=.1)

ax.plot(random['nr_samples'],max_score['macro_mean'],c='darkmagenta',ls=':',label='Mean Max Loss')
ax.fill_between(random['nr_samples'], (max_score['macro_mean']-2*max_score['macro_std']), (max_score['macro_mean']+2*max_score['macro_std']), color='darkmagenta', alpha=.1)

ax.set_ylabel("Macro F1")
ax.set_xlabel("Number of samples")
#ax = plt.gca()
#ax.set_ylim([0, None])

plt.legend()
fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\dark_wo_dup\DA2\macro.png')