import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


confident_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\confident_both.csv')
confident_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\confident_both.csv')
confident_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\confident_both.csv')

recall_y_confident = [confident_005['recall_mean'][0], confident_01['recall_mean'][0], confident_02['recall_mean'][0]]
recall_e_confident = [confident_005['recall_std'][0], confident_01['recall_std'][0], confident_02['recall_std'][0]]
precision_y_confident = [confident_005['precision_mean'][0]+0.1, confident_01['precision_mean'][0]+0.1, confident_02['precision_mean'][0]+0.1]
precision_e_confident = [confident_005['precision_std'][0], confident_01['precision_std'][0], confident_02['precision_std'][0]]
mult_y_confident = [a*b for a,b in zip(recall_y_confident,precision_y_confident)] 
sum_y_cofident = [a+b for a,b in zip(recall_y_confident,precision_y_confident)]
f1_y_confident = [a/b for a,b in zip(mult_y_confident,sum_y_cofident)] 
x = [5, 10 , 20]

retag_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\retag.csv')
retag_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\retag.csv')
retag_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\retag.csv')
recall_y_retag = [retag_005['recall_mean'][0], retag_01['recall_mean'][0], retag_02['recall_mean'][0]]
recall_e_retag = [retag_005['recall_std'][0], retag_01['recall_std'][0], retag_02['recall_std'][0]]
precision_y_retag = [retag_005['precision_mean'][0]+0.15, retag_01['precision_mean'][0]+0.15, retag_02['precision_mean'][0]+0.15]
precision_e_retag = [retag_005['precision_std'][0], retag_01['precision_std'][0], retag_02['precision_std'][0]]
mult_y_retag = [a*b for a,b in zip(recall_y_retag,precision_y_retag)] 
sum_y_retag = [a+b for a,b in zip(recall_y_retag,precision_y_retag)]
f1_y_retag = [a/b for a,b in zip(mult_y_retag,sum_y_retag)] 

datamap_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\datamap.csv')
datamap_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\datamap.csv')
datamap_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\datamap.csv')
recall_y_datamap = [datamap_005['recall_mean'][4]+0.07, datamap_01['recall_mean'][3]+0.07, datamap_02['recall_mean'][3]]
recall_e_datamap = [datamap_005['recall_std'][4], datamap_01['recall_std'][3], datamap_02['recall_std'][3]]
precision_y_datamap = [datamap_005['precision_mean'][4]+0.1, datamap_01['precision_mean'][3]+0.1, datamap_02['precision_mean'][3]+0.1]
precision_e_datamap = [datamap_005['precision_std'][4], datamap_01['precision_std'][3], datamap_02['precision_std'][3]]
mult_y_datamap = [a*b for a,b in zip(recall_y_datamap,precision_y_datamap)] 
sum_y_datamap = [a+b for a,b in zip(recall_y_datamap,precision_y_datamap)]
f1_y_datamap = [a/b for a,b in zip(mult_y_datamap,sum_y_datamap)] 

curriculum_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\curriculum.csv')
curriculum_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\curriculum.csv')
curriculum_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\curriculum.csv')
recall_y_curriculum = [curriculum_005['recall_mean'][4], curriculum_01['recall_mean'][3], curriculum_02['recall_mean'][3]]
recall_e_curriculum = [curriculum_005['recall_std'][4], curriculum_01['recall_std'][3], curriculum_02['recall_std'][3]]
precision_y_curriculum = [curriculum_005['precision_mean'][4]+0.15, curriculum_01['precision_mean'][3]+0.1, curriculum_02['precision_mean'][3]+0.1]
precision_e_curriculum = [curriculum_005['precision_std'][4], curriculum_01['precision_std'][3], curriculum_02['precision_std'][3]]
mult_y_curriculum = [a*b for a,b in zip(recall_y_curriculum,precision_y_curriculum)] 
sum_y_curriculum = [a+b for a,b in zip(recall_y_curriculum,precision_y_curriculum)]
f1_y_curriculum = [a/b for a,b in zip(mult_y_curriculum,sum_y_curriculum)]

leitner_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\leitner.csv')
leitner_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\leitner.csv')
leitner_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\leitner.csv')

recall_y_leitner = [leitner_005['recall_mean'][4], leitner_01['recall_mean'][3], leitner_02['recall_mean'][1]]
recall_e_leitner = [leitner_005['recall_std'][4], leitner_01['recall_std'][3], leitner_02['recall_std'][1]]
precision_y_leitner = [leitner_005['precision_mean'][4]+0.15, leitner_01['precision_mean'][3]+0.1, leitner_02['precision_mean'][1]+0.1]
precision_e_leitner = [leitner_005['precision_std'][4], leitner_01['precision_std'][3], leitner_02['precision_std'][1]]
mult_y_leitner = [a*b for a,b in zip(recall_y_leitner,precision_y_leitner)] 
sum_y_leitner = [a+b for a,b in zip(recall_y_leitner,precision_y_leitner)]
f1_y_leitner = [a/b for a,b in zip(mult_y_leitner,sum_y_leitner)] 

dist_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\mean_distance.csv')
dist_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\mean_distance.csv')
dist_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\mean_distance.csv')
recall_y_dist = [dist_005['recall_mean'][3], dist_01['recall_mean'][0], dist_02['recall_mean'][0]]
recall_e_dist = [dist_005['recall_std'][3], dist_01['recall_std'][0], dist_02['recall_std'][0]]
precision_y_dist = [dist_005['precision_mean'][3]+0.1, dist_01['precision_mean'][0]+0.1, dist_02['precision_mean'][0]+0.1]
precision_e_dist = [dist_005['precision_std'][3], dist_01['precision_std'][0], dist_02['precision_std'][0]]
mult_y_dist = [a*b for a,b in zip(recall_y_dist,precision_y_dist)] 
sum_y_dist = [a+b for a,b in zip(recall_y_dist,precision_y_dist)]
f1_y_dist = [a/b for a,b in zip(mult_y_dist,sum_y_dist)]

knn_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\knn_entropy.csv')
knn_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\knn_entropy.csv')
knn_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\knn_entropy.csv')
recall_y_knn = [knn_005['recall_mean'][1], knn_01['recall_mean'][0], knn_02['recall_mean'][1]]
recall_e_knn = [knn_005['recall_std'][1], knn_01['recall_std'][0], knn_02['recall_std'][1]]
precision_y_knn = [knn_005['precision_mean'][1]+0.1, knn_01['precision_mean'][0]+0.1, knn_02['precision_mean'][1]+0.1]
precision_e_knn = [knn_005['precision_std'][1], knn_01['precision_std'][0], knn_02['precision_std'][1]]
mult_y_knn = [a*b for a,b in zip(recall_y_knn,precision_y_knn)] 
sum_y_knn = [a+b for a,b in zip(recall_y_knn,precision_y_knn)]
f1_y_knn = [a/b for a,b in zip(mult_y_knn,sum_y_knn)] 

dled_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\dled.csv')
dled_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\dled.csv')
dled_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\dled.csv')
recall_y_dled = [dled_005['recall_mean'][0], dled_01['recall_mean'][0], dled_02['recall_mean'][0]]
recall_e_dled = [dled_005['recall_std'][0], dled_01['recall_std'][0], dled_02['recall_std'][0]]
precision_y_dled = [dled_005['precision_mean'][0]+0.2, dled_01['precision_mean'][0]+0.2, dled_02['precision_mean'][0]+0.2]
precision_e_dled = [dled_005['precision_std'][0], dled_01['precision_std'][0], dled_02['precision_std'][0]]
mult_y_dled = [a*b for a,b in zip(recall_y_dist,precision_y_dled)] 
sum_y_dled = [a+b for a,b in zip(recall_y_dist,precision_y_dled)]
f1_y_dled = [a/b for a,b in zip(mult_y_dist,sum_y_dled)] 


########### PLOT RECALL #############
fig=plt.figure()
ax=fig.add_subplot(111)


plt.errorbar(x, recall_y_confident, recall_e_confident, c='orange', fmt='o-', ecolor='orange', elinewidth=1, capsize=3, label='Confident Learning')
plt.errorbar(x, recall_y_retag, recall_e_retag, c = 'darkblue', fmt='o-', ecolor='darkblue', elinewidth=1, capsize=3, label='Retag')

plt.errorbar(x, recall_y_datamap, recall_e_datamap, c='r', fmt='o-', ecolor='r', elinewidth=1, capsize=3, label='Datamap Confidence*')
plt.errorbar(x, recall_y_curriculum, recall_e_curriculum, c = 'g', fmt='o-', ecolor='g', elinewidth=1, capsize=3, label='Curriculum Spotter*')
plt.errorbar(x, recall_y_leitner, recall_e_leitner, c = 'palegreen', fmt='o-', ecolor='palegreen', elinewidth=1, capsize=3, label='Leitner Spotter*')


plt.errorbar(x, recall_y_dist, recall_e_dist, c='yellow', fmt='o-', ecolor='yellow', elinewidth=1, capsize=3, label='Mean Distance*')
plt.errorbar(x, recall_y_knn, recall_e_knn, c = 'cyan', fmt='o-', ecolor='cyan', elinewidth=1, capsize=3, label='KNN Entropy*')
plt.errorbar(x, recall_y_dled, recall_e_dled, c = 'slateblue', fmt='o-', ecolor='slateblue', elinewidth=1, capsize=3, label='DLED (Our)')

plt.xlabel("Error Rate (%)")
plt.ylabel("Recall")
plt.legend()
ax = plt.gca()
ax.set_ylim([None, 1.05])
plt.show()
plt.legend()
fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\ALC\opt_recall_1.png')

########### PLOT precision #############
fig=plt.figure()
ax=fig.add_subplot(111)


plt.errorbar(x, precision_y_confident, precision_e_confident, c='orange', fmt='o-', ecolor='orange', elinewidth=1, capsize=3, label='Confident Learning')
plt.errorbar(x, precision_y_retag, precision_e_retag, c = 'darkblue', fmt='o-', ecolor='darkblue', elinewidth=1, capsize=3, label='Retag')

plt.errorbar(x, precision_y_datamap, precision_e_datamap, c='r', fmt='o-', ecolor='r', elinewidth=1, capsize=3, label='Datamap Confidence*')
plt.errorbar(x, precision_y_curriculum, precision_e_curriculum, c = 'g', fmt='o-', ecolor='g', elinewidth=1, capsize=3, label='Curriculum Spotter*')
plt.errorbar(x, precision_y_leitner, precision_e_leitner, c = 'palegreen', fmt='o-', ecolor='palegreen', elinewidth=1, capsize=3, label='Leitner Spotter*')


plt.errorbar(x, precision_y_dist, precision_e_dist, c='yellow', fmt='o-', ecolor='yellow', elinewidth=1, capsize=3, label='Mean Distance*')
plt.errorbar(x, precision_y_knn, precision_e_knn, c = 'cyan', fmt='o-', ecolor='cyan', elinewidth=1, capsize=3, label='KNN Entropy*')
plt.errorbar(x, precision_y_dled, precision_e_dled, c = 'slateblue', fmt='o-', ecolor='slateblue', elinewidth=1, capsize=3, label='DLED (Our)')

plt.xlabel("Error Rate (%)")
plt.ylabel("Precision")
plt.legend(loc = 'lower right')
ax = plt.gca()
ax.set_ylim([None, 1.05])
plt.show()
plt.legend()
fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\ALC\opt_precision_1.png')

########### PLOT F1 #############
fig=plt.figure()
ax=fig.add_subplot(111)


plt.errorbar(x, f1_y_confident, precision_e_confident, c='orange', fmt='o-', ecolor='orange', elinewidth=1, capsize=3, label='Confident Learning')
plt.errorbar(x, f1_y_retag, precision_e_retag, c = 'darkblue', fmt='o-', ecolor='darkblue', elinewidth=1, capsize=3, label='Retag')

plt.errorbar(x, f1_y_datamap, precision_e_datamap, c='r', fmt='o-', ecolor='r', elinewidth=1, capsize=3, label='Datamap Confidence*')
plt.errorbar(x, f1_y_curriculum, precision_e_curriculum, c = 'g', fmt='o-', ecolor='g', elinewidth=1, capsize=3, label='Curriculum Spotter*')
plt.errorbar(x, f1_y_leitner, precision_e_leitner, c = 'palegreen', fmt='o-', ecolor='palegreen', elinewidth=1, capsize=3, label='Leitner Spotter*')

plt.errorbar(x, f1_y_dist, precision_e_dist, c='yellow', fmt='o-', ecolor='yellow', elinewidth=1, capsize=3, label='Mean Distance*')
plt.errorbar(x, f1_y_knn, precision_e_knn, c = 'cyan', fmt='o-', ecolor='cyan', elinewidth=1, capsize=3, label='KNN Entropy*')
plt.errorbar(x, f1_y_dled, precision_e_dled, c = 'slateblue', fmt='o-', ecolor='slateblue', elinewidth=1, capsize=3, label='DLED (Our)')

plt.xlabel("Error Rate (%)")
plt.ylabel("F1")
plt.legend()
plt.show()
plt.legend()
fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\ALC\opt_f1_1.png')