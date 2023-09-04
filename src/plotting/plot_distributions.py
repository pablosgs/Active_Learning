import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  


labels = [
        'Cybercrime',
       'Drugs / Narcotics', 'Financial Crime', 'Goods and Services',
       'Sexual Abuse', 'Violent Crime']
rand_df = pd.read_csv(r'data\datasets\numbers_random.csv', on_bad_lines='warn')
rand_data1 = rand_df.iloc[0]
rand_data2 = rand_df.iloc[26]
badge_df = pd.read_csv(r'data\datasets\numbers_mml.csv', on_bad_lines='warn')
badge_data1 = badge_df.iloc[0]
badge_data2 = badge_df.iloc[26]

#badge_data2[0] += 50
#badge_data2[1] += 50
#badge_data2[2] -= 210
#badge_data2[3] += 50
#badge_data2[4] += 30
#badge_data2[5] += 30

fig = plt.figure()

f, axs = plt.subplots(1, 2, figsize=(12,10), sharey=True)

axs[0].bar(labels, rand_data1, label='Initial distribution', alpha=0.5, color='b')
axs[0].bar(labels, rand_data2, bottom=rand_data1, label='Final distribution', alpha=0.5, color='r')
plt.sca(axs[0])
axs[0].set_ylabel("Number of samples")
axs[0].set_xlabel("Label")
axs[0].set_title("Random",fontsize=18)
plt.setp(axs[0].get_xticklabels(), rotation=30, horizontalalignment='right')
plt.legend(loc='upper left')

axs[1].bar(labels, badge_data1, label='Initial distribution', alpha=0.5, color='b')
axs[1].bar(labels, badge_data2, bottom=rand_data1, label='Final distribution', alpha=0.5, color='r')
plt.sca(axs[1])
axs[1].set_ylabel("Number of samples")
axs[1].set_xlabel("Label")
axs[1].set_title("Mean Max Loss",fontsize=18)
plt.setp(axs[1].get_xticklabels(), rotation=30, horizontalalignment='right')
plt.legend(loc='upper left')
fig.align_labels(axs[1])



plt.suptitle('Number of instances samples by label')
plt.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\AL\dark\final_distributions_new_5.png')