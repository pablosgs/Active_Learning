import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


confident_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\confident_01.csv')
confident_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\confident_02.csv')
confident_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\confident_005.csv')

recall_y_confident = [confident_005['micro_mean'][1] - confident_005['micro_mean'][0], confident_01['micro_mean'][1] -confident_01['micro_mean'][0], confident_02['micro_mean'][1] - confident_02['micro_mean'][0]]
recall_e_confident = [confident_005['micro_std'][1], confident_01['micro_std'][1], confident_02['micro_std'][1]]
precision_y_confident = [confident_005['macro_mean'][1] - confident_005['macro_mean'][0], confident_01['macro_mean'][1] - confident_01['macro_mean'][0], confident_02['macro_mean'][1] -confident_02['macro_mean'][0]]
precision_e_confident = [confident_005['macro_std'][1], confident_01['macro_std'][1], confident_02['macro_std'][1]]
acc_y_confident = [confident_005['acc_mean'][1] - confident_005['acc_mean'][0], confident_01['acc_mean'][1] - confident_01['acc_mean'][0], confident_02['acc_mean'][1] -confident_02['acc_mean'][0]]
acc_e_confident = [confident_005['acc_std'][1], confident_01['acc_std'][1], confident_02['acc_std'][1]]
x = [5, 10 , 20]

retag_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\retag_02.csv')
retag_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\retag_01.csv')
retag_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\retag_005.csv')
recall_y_retag = [retag_005['micro_mean'][1] - retag_005['micro_mean'][0], retag_01['micro_mean'][1] - retag_01['micro_mean'][0], retag_02['micro_mean'][1] - retag_02['micro_mean'][0]]
recall_e_retag = [retag_005['micro_std'][1], retag_01['micro_std'][1], retag_02['micro_std'][1]]
precision_y_retag = [retag_005['macro_mean'][1] - retag_005['macro_mean'][0], retag_01['macro_mean'][1] - retag_01['macro_mean'][0], retag_02['macro_mean'][1] - retag_02['macro_mean'][0]]
precision_e_retag = [retag_005['macro_std'][1], retag_01['macro_std'][1], retag_02['macro_std'][1]]
acc_y_retag = [retag_005['acc_mean'][1] - retag_005['acc_mean'][0], retag_01['acc_mean'][1] - retag_01['acc_mean'][0], retag_02['acc_mean'][1] - retag_02['acc_mean'][0]]
acc_e_retag = [retag_005['acc_std'][1], retag_01['acc_std'][1], retag_02['acc_std'][1]]




datamap_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\datamap_02.csv')
datamap_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\datamap_01.csv')
datamap_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\datamap_005.csv')
recall_y_datamap = [datamap_005['micro_mean'][1] - datamap_005['micro_mean'][0], datamap_01['micro_mean'][1] - datamap_01['micro_mean'][0], datamap_02['micro_mean'][1] - datamap_02['micro_mean'][0]]
recall_e_datamap = [datamap_005['micro_std'][1], datamap_01['micro_std'][1], datamap_02['micro_std'][1]]
precision_y_datamap = [datamap_005['macro_mean'][1] - datamap_005['macro_mean'][0], datamap_01['macro_mean'][1] - datamap_01['macro_mean'][0], datamap_02['macro_mean'][1] - datamap_02['macro_mean'][0]]
precision_e_datamap = [datamap_005['macro_std'][1], datamap_01['macro_std'][1], datamap_02['macro_std'][1]]
acc_y_datamap = [datamap_005['acc_mean'][1] - datamap_005['acc_mean'][0], datamap_01['acc_mean'][1] - datamap_01['acc_mean'][0], datamap_02['acc_mean'][1] - datamap_02['acc_mean'][0]]
acc_e_datamap = [datamap_005['acc_std'][1], datamap_01['acc_std'][1], datamap_02['acc_std'][1]]


curriculum_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\curriculum_02.csv')
curriculum_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\curriculum_01.csv')
curriculum_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\curriculum_005.csv')
recall_y_curriculum = [curriculum_005['micro_mean'][1] - curriculum_005['micro_mean'][0], curriculum_01['micro_mean'][1] - curriculum_01['micro_mean'][0], curriculum_02['micro_mean'][1] - curriculum_02['micro_mean'][0]]
recall_e_curriculum = [curriculum_005['micro_std'][1], curriculum_01['micro_std'][1], curriculum_02['micro_std'][1]]
precision_y_curriculum = [curriculum_005['macro_mean'][1] - curriculum_005['macro_mean'][0], curriculum_01['macro_mean'][1] - curriculum_01['macro_mean'][0], curriculum_02['macro_mean'][1] - curriculum_02['macro_mean'][0]]
precision_e_curriculum = [curriculum_005['macro_std'][1], curriculum_01['macro_std'][1], curriculum_02['macro_std'][1]]
acc_y_curriculum = [curriculum_005['acc_mean'][1] - curriculum_005['acc_mean'][0], curriculum_01['acc_mean'][1] - curriculum_01['acc_mean'][0], curriculum_02['acc_mean'][1] - curriculum_02['acc_mean'][0]]
acc_e_curriculum = [curriculum_005['acc_std'][1], curriculum_01['acc_std'][1], curriculum_02['acc_std'][1]]


dist_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\dist_02.csv')
dist_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\dist_01.csv')
dist_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\dist_005.csv')
recall_y_dist = [dist_005['micro_mean'][1] - dist_005['micro_mean'][0], dist_01['micro_mean'][1] - dist_01['micro_mean'][0], dist_02['micro_mean'][1] - dist_02['micro_mean'][0]]
recall_e_dist = [curriculum_005['micro_std'][1], curriculum_01['micro_std'][1], curriculum_02['micro_std'][1]]
precision_y_dist = [dist_005['macro_mean'][1] - dist_005['macro_mean'][0], dist_01['macro_mean'][1] - dist_01['macro_mean'][0], dist_02['macro_mean'][1] - dist_02['macro_mean'][0]]
precision_e_dist = [curriculum_005['micro_std'][1], curriculum_01['micro_std'][1], curriculum_02['micro_std'][1]]
acc_y_dist = [dist_005['acc_mean'][1] - dist_005['acc_mean'][0], dist_01['acc_mean'][1] - dist_01['acc_mean'][0], dist_02['acc_mean'][1] - dist_02['acc_mean'][0]]
acc_e_dist = [curriculum_005['micro_std'][1], curriculum_01['micro_std'][1], curriculum_02['micro_std'][1]]


leitner_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\leitner_02.csv')
leitner_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\leitner_01.csv')
leitner_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\leitner_005.csv')
recall_y_leitner = [leitner_005['micro_mean'][1] - leitner_005['micro_mean'][0], leitner_01['micro_mean'][1] - leitner_01['micro_mean'][0], leitner_02['micro_mean'][1] - leitner_02['micro_mean'][0]]
recall_e_leitner = [leitner_005['micro_std'][1], leitner_01['micro_std'][1], leitner_02['micro_std'][1]]
precision_y_leitner = [leitner_005['macro_mean'][1] - leitner_005['macro_mean'][0], leitner_01['macro_mean'][1] - leitner_01['macro_mean'][0], leitner_02['macro_mean'][1] - leitner_02['macro_mean'][0]]
precision_e_leitner = [leitner_005['macro_std'][1], leitner_01['macro_std'][1], leitner_02['macro_std'][1]]
acc_y_leitner = [leitner_005['acc_mean'][1] - leitner_005['acc_mean'][0], leitner_01['acc_mean'][1] - leitner_01['acc_mean'][0], leitner_02['acc_mean'][1] - leitner_02['acc_mean'][0]]
acc_e_leitner = [leitner_005['acc_std'][1], leitner_01['acc_std'][1], leitner_02['acc_std'][1]]

knn_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\knn_entropy_02.csv')
knn_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\knn_entropy_01.csv')
knn_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\knn_entropy_005.csv')
recall_y_knn = [knn_005['micro_mean'][1] - knn_005['micro_mean'][0], knn_01['micro_mean'][1] - knn_01['micro_mean'][0], knn_02['micro_mean'][1] - knn_02['micro_mean'][0]]
recall_e_knn = [leitner_005['micro_std'][1], leitner_01['micro_std'][1], leitner_02['micro_std'][1]]
precision_y_knn = [knn_005['macro_mean'][1] - knn_005['macro_mean'][0], knn_01['macro_mean'][1] - knn_01['macro_mean'][0], knn_02['macro_mean'][1] - knn_02['macro_mean'][0]]
precision_e_knn = [leitner_005['micro_std'][1], leitner_01['micro_std'][1], leitner_02['micro_std'][1]]
acc_y_knn = [knn_005['acc_mean'][1] - knn_005['acc_mean'][0], knn_01['acc_mean'][1] - knn_01['acc_mean'][0], knn_02['acc_mean'][1] - knn_02['acc_mean'][0]]
acc_e_knn = [leitner_005['micro_std'][1], leitner_01['micro_std'][1], leitner_02['micro_std'][1]]


gold_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\gold_02.csv')
gold_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\gold_01.csv')
gold_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\gold_005.csv')
recall_y_gold = [gold_005['micro_mean'][1] - gold_005['micro_mean'][0], gold_01['micro_mean'][1] - gold_01['micro_mean'][0], gold_02['micro_mean'][1] - gold_02['micro_mean'][0]]
recall_e_gold = [gold_005['micro_std'][1], gold_01['micro_std'][1], gold_02['micro_std'][1]]
precision_y_gold = [gold_005['macro_mean'][1] - gold_005['macro_mean'][0], gold_01['macro_mean'][1] - gold_01['macro_mean'][0], gold_02['macro_mean'][1] - gold_02['macro_mean'][0]]
precision_e_gold = [gold_005['macro_std'][1], gold_01['macro_std'][1], gold_02['macro_std'][1]]
acc_y_gold = [gold_005['acc_mean'][1] - gold_005['acc_mean'][0], gold_01['acc_mean'][1] - gold_01['acc_mean'][0], gold_02['acc_mean'][1] - gold_02['acc_mean'][0]]
acc_e_gold = [gold_005['acc_std'][1], gold_01['acc_std'][1], gold_02['acc_std'][1]]


dled_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\dled_02.csv')
dled_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\dled_01.csv')
dled_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\robustness\dled_005.csv')
recall_y_dled = [dled_005['micro_mean'][1] - dled_005['micro_mean'][0], dled_01['micro_mean'][1] - dled_01['micro_mean'][0], dled_02['micro_mean'][1] - dled_02['micro_mean'][0]]
recall_e_dled = [gold_005['micro_std'][1], gold_01['micro_std'][1], gold_02['micro_std'][1]]
precision_y_dled = [dled_005['macro_mean'][1] - dled_005['macro_mean'][0], dled_01['macro_mean'][1] - dled_01['macro_mean'][0], dled_02['macro_mean'][1] - dled_02['macro_mean'][0]]
precision_e_dled = [gold_005['macro_std'][1], gold_01['macro_std'][1], gold_02['macro_std'][1]]
acc_y_dled = [dled_005['acc_mean'][1] - dled_005['acc_mean'][0], dled_01['acc_mean'][1] - dled_01['acc_mean'][0], dled_02['acc_mean'][1] - dled_02['acc_mean'][0]]
acc_e_dled = [gold_005['acc_std'][1], gold_01['acc_std'][1], gold_02['acc_std'][1]]


########### PLOT RECALL #############
fig=plt.figure()
ax=fig.add_subplot(111)


plt.errorbar(x, recall_y_confident, list(map(abs,recall_e_confident)), c='orange', fmt='o-', ecolor='orange', elinewidth=1, capsize=3, label='Confident')
plt.errorbar(x, recall_y_retag, list(map(abs,recall_e_retag)), c = 'darkblue', fmt='o-', ecolor='darkblue', elinewidth=1, capsize=3, label='Retag')

plt.errorbar(x, recall_y_datamap, list(map(abs,recall_e_datamap)), c='r', fmt='o-', ecolor='r', elinewidth=1, capsize=3, label='Datamap Confidence')
plt.errorbar(x, recall_y_curriculum, list(map(abs,recall_e_curriculum)), c = 'g', fmt='o-', ecolor='g', elinewidth=1, capsize=3, label='Curriculum Spotter')
plt.errorbar(x, recall_y_leitner, list(map(abs,recall_e_leitner)), c = 'palegreen', fmt='o-', ecolor='palegreen', elinewidth=1, capsize=3, label='Leitner Spotter')

plt.errorbar(x, recall_y_dist, list(map(abs,recall_e_dist)), c='lawngreen', fmt='o-', ecolor='lawngreen', elinewidth=1, capsize=3, label='Mean Distance')
plt.errorbar(x, recall_y_knn, list(map(abs,recall_e_knn)), c = 'cyan', fmt='o-', ecolor='cyan', elinewidth=1, capsize=3, label='KNN Entropy')
plt.errorbar(x, recall_y_gold, list(map(abs,recall_e_gold)), c = 'gold', fmt='o-', ecolor='gold', elinewidth=1, capsize=3, label='Gold standard')
plt.errorbar(x, recall_y_dled, list(map(abs,recall_e_dled)), c = 'slateblue', fmt='o-', ecolor='slateblue', elinewidth=1, capsize=3, label='DLED (Our)')
plt.axhline(y=0, color='k', linestyle='--')

plt.xlabel("Error Rate (%)")
plt.ylabel("Micro F1 improvement")
plt.legend()
plt.show()
plt.legend()

fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\ALC\robustness_micro.png')

########### PLOT precision #############
fig=plt.figure()
ax=fig.add_subplot(111)


plt.errorbar(x, precision_y_confident, list(map(abs,precision_e_confident)), c='orange', fmt='o-', ecolor='orange', elinewidth=1, capsize=3, label='Confident')
plt.errorbar(x, precision_y_retag, list(map(abs,precision_e_retag)), c = 'darkblue', fmt='o-', ecolor='darkblue', elinewidth=1, capsize=3, label='Retag')

plt.errorbar(x, precision_y_datamap, list(map(abs,precision_e_datamap)), c='r', fmt='o-', ecolor='r', elinewidth=1, capsize=3, label='Datamap Confidence')
plt.errorbar(x, precision_y_curriculum, list(map(abs,precision_e_curriculum)), c = 'g', fmt='o-', ecolor='g', elinewidth=1, capsize=3, label='Curriculum Spotter')
plt.errorbar(x, precision_y_leitner, list(map(abs,precision_e_leitner)), c = 'palegreen', fmt='o-', ecolor='palegreen', elinewidth=1, capsize=3, label='Leitner Spotter')


plt.errorbar(x, precision_y_dist, list(map(abs,precision_e_dist)), c='lawngreen', fmt='o-', ecolor='lawngreen', elinewidth=1, capsize=3, label='Mean Distance')
plt.errorbar(x, precision_y_knn, list(map(abs,precision_e_knn)), c = 'cyan', fmt='o-', ecolor='cyan', elinewidth=1, capsize=3, label='KNN Entropy')
plt.errorbar(x, precision_y_gold, list(map(abs,precision_e_gold)), c = 'gold', fmt='o-', ecolor='gold', elinewidth=1, capsize=3, label='Gold standard')
plt.errorbar(x, precision_y_dled, list(map(abs,precision_e_dled)), c = 'slateblue', fmt='o-', ecolor='slateblue', elinewidth=1, capsize=3, label='DLED (Our)')
plt.axhline(y=0, color='k', linestyle='--')

plt.xlabel("Error Rate (%)")
plt.ylabel("Macro F1 improvement")
plt.legend()
plt.show()
plt.legend()
fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\ALC\robustness_macro.png')

########### PLOT precision #############
fig=plt.figure()
ax=fig.add_subplot(111)


plt.errorbar(x, acc_y_confident, list(map(abs,acc_e_confident)), c='orange', fmt='o-', ecolor='orange', elinewidth=1, capsize=3, label='Confident')
plt.errorbar(x, acc_y_retag, list(map(abs,acc_e_retag)), c = 'darkblue', fmt='o-', ecolor='darkblue', elinewidth=1, capsize=3, label='Retag')

plt.errorbar(x, acc_y_datamap, list(map(abs,acc_e_datamap)), c='r', fmt='o-', ecolor='r', elinewidth=1, capsize=3, label='Datamap Confidence')
plt.errorbar(x, acc_y_curriculum, list(map(abs,acc_e_curriculum)), c = 'g', fmt='o-', ecolor='g', elinewidth=1, capsize=3, label='Curriculum Spotter')
plt.errorbar(x, acc_y_leitner, list(map(abs,acc_e_leitner)), c = 'palegreen', fmt='o-', ecolor='palegreen', elinewidth=1, capsize=3, label='Leitner Spotter')

plt.errorbar(x, acc_y_dist, list(map(abs,acc_e_dist)), c='lawngreen', fmt='o-', ecolor='lawngreen', elinewidth=1, capsize=3, label='Mean Distance')
plt.errorbar(x, acc_y_knn, list(map(abs,acc_e_knn)), c = 'cyan', fmt='o-', ecolor='cyan', elinewidth=1, capsize=3, label='KNN Entropy')
plt.errorbar(x, acc_y_gold, list(map(abs, acc_e_gold)), c = 'gold', fmt='o-', ecolor='gold', elinewidth=1, capsize=3, label='Gold standard')
plt.errorbar(x, acc_y_dled, list(map(abs,acc_e_dled)), c = 'slateblue', fmt='o-', ecolor='slateblue', elinewidth=1, capsize=3, label='DLED (Our)')
plt.axhline(y=0, color='k', linestyle='--')

plt.xlabel("Error Rate (%)")
plt.ylabel("Accuracy improvement")
plt.legend()
plt.show()
plt.legend()
fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\ALC\robustness_acc.png')

