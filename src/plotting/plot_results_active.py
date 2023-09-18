import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import uniform


random = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\dwm_wo_duplicates\DA\random.csv')
badge = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\dwm_wo_duplicates\DA\badge.csv')
discriminative = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\dwm_wo_duplicates\DA\DAL.csv')
alucs = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\dwm_wo_duplicates\DA\dalucs.csv')
clue = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\dwm_wo_duplicates\DA\mean_max_loss.csv')
expected = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\dwm_wo_duplicates\DA\EGL.csv')
max_score = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\dwm_wo_duplicates\DA\mean_max_loss.csv')


fig=plt.figure()
ax=fig.add_subplot(111)


ax.plot(random['nr_samples'],random['micro_mean'],c='k',ls='-',label='Random',fillstyle='none')
ax.fill_between(random['nr_samples'], (random['micro_mean']-2*random['micro_std']), (random['micro_mean']+2*random['micro_std']), color='k', alpha=.1)

ax.plot(random['nr_samples'],discriminative['micro_mean'],c='g',ls='--',label='DAL',fillstyle='none')
ax.fill_between(random['nr_samples'], (discriminative['micro_mean']-2*discriminative['micro_std']), (discriminative['micro_mean']+2*discriminative['micro_std']), color='g', alpha=.1)

ax.plot(random['nr_samples'],alucs['micro_mean'],c='b',ls='--',label='Our')
ax.fill_between(random['nr_samples'], (alucs['micro_mean']-2*alucs['micro_std']), (alucs['micro_mean']+2*alucs['micro_std']), color='b', alpha=.1)

ax.plot(random['nr_samples'],expected['micro_mean'],c='orangered',ls=':',label='EGL')
ax.fill_between(random['nr_samples'], (expected['micro_mean']-2*expected['micro_std']), (expected['micro_mean']+2*expected['micro_std']), color='orangered', alpha=.1)

ax.plot(random['nr_samples'],clue['micro_mean'],c='lightseagreen',ls='-',label='CLUE')
ax.fill_between(random['nr_samples'], (clue['micro_mean']-2*clue['micro_std']), (clue['micro_mean']+2*clue['micro_std']), color='lightseagreen', alpha=.1)

ax.plot(random['nr_samples'],badge['micro_mean'],c='darkred',ls='-.',label='BADGE')
ax.fill_between(random['nr_samples'], (badge['micro_mean']-2*badge['micro_std']), (badge['micro_mean']+2*badge['micro_std']), color='darkred', alpha=.1)

ax.plot(random['nr_samples'],max_score['micro_mean'],c='darkmagenta',ls=':',label='Mean Max Loss')
ax.fill_between(random['nr_samples'], (max_score['micro_mean']-2*max_score['micro_std']), (max_score['micro_mean']+2*max_score['micro_std']), color='darkmagenta', alpha=.1)

ax.set_ylabel("Micro F1")
ax.set_xlabel("Number of samples")
ax = plt.gca()
ax.set_ylim([0.55, 0.9])

plt.legend()
fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\dark_wo_dup\DA\micro_1.png')