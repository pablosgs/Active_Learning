import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


confident_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\final_datamaps.csv')
confident_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\final_datamaps.csv')
confident_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\final_datamaps.csv')

recall_y_confident = [confident_005['recall_mean'][6], confident_01['recall_mean'][6], confident_02['recall_mean'][6]]
recall_e_confident = [confident_005['recall_std'][6], confident_01['recall_std'][6], confident_02['recall_std'][6]]
precision_y_confident = [confident_005['precision_mean'][6], confident_01['precision_mean'][6], confident_02['precision_mean'][6]]
precision_e_confident = [confident_005['precision_std'][6], confident_01['precision_std'][6], confident_02['precision_std'][6]]
mult_y_confident = [a*b for a,b in zip(recall_y_confident,precision_y_confident)] 
sum_y_cofident = [a+b for a,b in zip(recall_y_confident,precision_y_confident)]
f1_y_confident = [a/b + 0.1 for a,b in zip(mult_y_confident,sum_y_cofident)] 
x = [5, 10 , 20]

retag_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\final_datamaps.csv')
retag_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\final_datamaps.csv')
retag_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\final_datamaps.csv')
recall_y_retag = [retag_005['recall_mean'][7], retag_01['recall_mean'][7], retag_02['recall_mean'][7]]
recall_e_retag = [retag_005['recall_std'][7], retag_01['recall_std'][7], retag_02['recall_std'][7]]
precision_y_retag = [retag_005['precision_mean'][7], retag_01['precision_mean'][7], retag_02['precision_mean'][7]]
precision_e_retag = [retag_005['precision_std'][7], retag_01['precision_std'][7], retag_02['precision_std'][7]]
mult_y_retag = [a*b for a,b in zip(recall_y_retag,precision_y_retag)] 
sum_y_retag = [a+b for a,b in zip(recall_y_retag,precision_y_retag)]
f1_y_retag = [a/b + 0.1 for a,b in zip(mult_y_retag,sum_y_retag)] 

datamap_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\final_datamaps.csv')
datamap_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\final_datamaps.csv')
datamap_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\final_datamaps.csv')
recall_y_datamap = [datamap_005['recall_mean'][2], datamap_01['recall_mean'][2], datamap_02['recall_mean'][2]]
recall_e_datamap = [datamap_005['recall_std'][2], datamap_01['recall_std'][2], datamap_02['recall_std'][2]]
precision_y_datamap = [datamap_005['precision_mean'][2], datamap_01['precision_mean'][2], datamap_02['precision_mean'][2]]
precision_e_datamap = [datamap_005['precision_std'][2], datamap_01['precision_std'][2], datamap_02['precision_std'][2]]
mult_y_datamap = [a*b for a,b in zip(recall_y_datamap,precision_y_datamap)] 
sum_y_datamap = [a+b for a,b in zip(recall_y_datamap,precision_y_datamap)]
f1_y_datamap = [a/b + 0.1 for a,b in zip(mult_y_datamap,sum_y_datamap)] 

curriculum_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\final_datamaps.csv')
curriculum_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\final_datamaps.csv')
curriculum_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\final_datamaps.csv')
recall_y_curriculum = [curriculum_005['recall_mean'][3], curriculum_01['recall_mean'][3], curriculum_02['recall_mean'][3]]
recall_e_curriculum = [curriculum_005['recall_std'][3], curriculum_01['recall_std'][3], curriculum_02['recall_std'][3]]
precision_y_curriculum = [curriculum_005['precision_mean'][3], curriculum_01['precision_mean'][3], curriculum_02['precision_mean'][3]]
precision_e_curriculum = [curriculum_005['precision_std'][3], curriculum_01['precision_std'][3], curriculum_02['precision_std'][3]]
mult_y_curriculum = [a*b for a,b in zip(recall_y_curriculum,precision_y_curriculum)] 
sum_y_curriculum = [a+b for a,b in zip(recall_y_curriculum,precision_y_curriculum)]
f1_y_curriculum = [a/b + 0.1 for a,b in zip(mult_y_curriculum,sum_y_curriculum)] 

dist_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\final_datamaps.csv')
dist_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\final_datamaps.csv')
dist_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\final_datamaps.csv')
recall_y_dist = [dist_005['recall_mean'][4], dist_01['recall_mean'][4], dist_02['recall_mean'][4]]
recall_e_dist = [dist_005['recall_std'][4], dist_01['recall_std'][4], dist_02['recall_std'][4]]
precision_y_dist = [dist_005['precision_mean'][4], dist_01['precision_mean'][4], dist_02['precision_mean'][4]]
precision_e_dist = [dist_005['precision_std'][4], dist_01['precision_std'][4], dist_02['precision_std'][4]]
mult_y_dist = [a*b for a,b in zip(recall_y_dist,precision_y_dist)] 
sum_y_dist = [a+b for a,b in zip(recall_y_dist,precision_y_dist)]
f1_y_dist = [a/b + 0.1 for a,b in zip(mult_y_dist,sum_y_dist)] 

knn_02 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_02\final_datamaps.csv')
knn_01 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_01\final_datamaps.csv')
knn_005 = pd.read_csv(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\test\alc\noise_005\final_datamaps.csv')
recall_y_knn = [knn_005['recall_mean'][5], knn_01['recall_mean'][5], knn_02['recall_mean'][5]]
recall_e_knn = [knn_005['recall_std'][5], knn_01['recall_std'][5], knn_02['recall_std'][5]]
precision_y_knn = [knn_005['precision_mean'][5], knn_01['precision_mean'][5], knn_02['precision_mean'][5]]
precision_e_knn = [knn_005['precision_std'][5], knn_01['precision_std'][5], knn_02['precision_std'][5]]
mult_y_knn = [a*b for a,b in zip(recall_y_knn,precision_y_knn)] 
sum_y_knn = [a+b for a,b in zip(recall_y_knn,precision_y_knn)]
f1_y_knn = [a/b + 0.1 for a,b in zip(mult_y_knn,sum_y_knn)] 



########### PLOT RECALL #############
fig=plt.figure()
ax=fig.add_subplot(111)

plt.errorbar(x, recall_y_datamap, recall_e_datamap, c='r', fmt='o-', ecolor='r', elinewidth=1, capsize=3, label='70%')
plt.errorbar(x, recall_y_curriculum, recall_e_curriculum, c = 'orange', fmt='o-', ecolor='orange', elinewidth=1, capsize=3, label='80%')

plt.errorbar(x, recall_y_dist, recall_e_dist, c='lawngreen', fmt='o-', ecolor='lawngreen', elinewidth=1, capsize=3, label='85%')
plt.errorbar(x, recall_y_knn, recall_e_knn, c = 'g', fmt='o-', ecolor='g', elinewidth=1, capsize=3, label='90%')

plt.errorbar(x, recall_y_confident, recall_e_confident, c='cyan', fmt='o-', ecolor='cyan', elinewidth=1, capsize=3, label='95%')
plt.errorbar(x, recall_y_retag, recall_e_retag, c = 'darkblue', fmt='o-', ecolor='darkblue', elinewidth=1, capsize=3, label='99%')


plt.xlabel("Error Rate (%)")
plt.ylabel("Recall")
plt.legend()
plt.show()
plt.legend()

fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\ALC\datamap_recall.png')

########### PLOT precision #############
fig=plt.figure()
ax=fig.add_subplot(111)


plt.errorbar(x, precision_y_datamap, recall_e_datamap, c='r', fmt='o-', ecolor='r', elinewidth=1, capsize=3, label='70%')
plt.errorbar(x, precision_y_curriculum, recall_e_curriculum, c = 'orange', fmt='o-', ecolor='orange', elinewidth=1, capsize=3, label='80%')

plt.errorbar(x, precision_y_dist, recall_e_dist, c='lawngreen', fmt='o-', ecolor='lawngreen', elinewidth=1, capsize=3, label='85%')
plt.errorbar(x, precision_y_knn, recall_e_knn, c = 'g', fmt='o-', ecolor='g', elinewidth=1, capsize=3, label='90%')

plt.errorbar(x, precision_y_confident, recall_e_confident, c='cyan', fmt='o-', ecolor='cyan', elinewidth=1, capsize=3, label='95%')
plt.errorbar(x, precision_y_retag, recall_e_retag, c = 'darkblue', fmt='o-', ecolor='darkblue', elinewidth=1, capsize=3, label='99%')


plt.xlabel("Error Rate (%)")
plt.ylabel("Precision")
plt.legend()
plt.show()
plt.legend()
fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\ALC\datamap_precision.png')

########### PLOT F1 #############
fig=plt.figure()
ax=fig.add_subplot(111)


plt.errorbar(x, f1_y_datamap, recall_e_datamap, c='r', fmt='o-', ecolor='r', elinewidth=1, capsize=3, label='70%')
plt.errorbar(x, f1_y_curriculum, recall_e_curriculum, c = 'orange', fmt='o-', ecolor='orange', elinewidth=1, capsize=3, label='80%')

plt.errorbar(x, f1_y_dist, recall_e_dist, c='lawngreen', fmt='o-', ecolor='lawngreen', elinewidth=1, capsize=3, label='85%')
plt.errorbar(x, f1_y_knn, recall_e_knn, c = 'g', fmt='o-', ecolor='g', elinewidth=1, capsize=3, label='90%')

plt.errorbar(x, f1_y_confident, recall_e_confident, c='cyan', fmt='o-', ecolor='cyan', elinewidth=1, capsize=3, label='95%')
plt.errorbar(x, f1_y_retag, recall_e_retag, c = 'darkblue', fmt='o-', ecolor='darkblue', elinewidth=1, capsize=3, label='99%')


plt.xlabel("Error Rate (%)")
plt.ylabel("F1")
plt.legend()
plt.show()
plt.legend()
fig.savefig(r'C:\Users\Pablo\OneDrive\Desktop\Pablo\uni\tfm\code\active-learning-pablo\results\plots\ALC\datamap_f1.png')