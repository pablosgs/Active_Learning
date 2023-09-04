import os
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


#random = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\sampling_size\badge300.csv')
badge = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\sampling_size\badge300.csv')
discriminative = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\sampling_size\badge500.csv')
alucs = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\sampling_size\badge700.csv')
clue = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\sampling_size\badge900.csv')



#colours =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
sizes = [300, 500, 700, 900]
colors = plt.cm.autumn(np.linspace(0, 1, len(sizes)))

fig=plt.figure()
ax=fig.add_subplot(111)


#ax.plot(random['nr_samples'],random['acc_mean'],c= colors[0] ,label='Random',fillstyle='none')
#ax.fill_between(random['nr_samples'], (random['acc_mean']-2*random['acc_std']), (random['acc_mean']+2*random['acc_std']), color='k', alpha=.1)

ax.plot(badge['nr_samples'],badge['micro_mean'],c=colors[0], label = '300')
#ax.fill_between(random['nr_samples'], (badge['acc_mean']-2*badge['acc_std']), (badge['acc_mean']+2*badge['acc_std']), color='darkred', alpha=.1)

ax.plot(discriminative['nr_samples'],discriminative['micro_mean'],c=colors[1], label = '500')
#ax.fill_between(random['nr_samples'], (discriminative['acc_mean']-2*discriminative['acc_std']), (discriminative['acc_mean']+2*discriminative['acc_std']), color='g', alpha=.1)

ax.plot(alucs['nr_samples'],alucs['micro_mean'],c=colors[2], label = '700')
#ax.fill_between(random['nr_samples'], (alucs['acc_mean']-2*alucs['acc_std']), (alucs['acc_mean']+2*alucs['acc_std']), color='b', alpha=.1)

ax.plot(clue['nr_samples'],clue['micro_mean'],c=colors[3], label = '900')
#ax.fill_between(random['nr_samples'], (clue['acc_mean']-2*clue['acc_std']), (clue['acc_mean']+2*clue['acc_std']), color='lightseagreen', alpha=.1)
plt.legend(title = 'Sampling size')
ax.set_ylabel("Micro F1")
ax.set_xlabel("Number of samples")
fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\dark_wo_dup\sampling_size.png')