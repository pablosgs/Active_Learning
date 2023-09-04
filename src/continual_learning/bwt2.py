import json
import numpy as np

folder = '/home/pablo/active-learning-pablo/results/test/continual_learning/dark_wo_duplicates2/DA3/'
method = 'warm'

with open(folder + method + '_abuse.json') as f:
    results = json.load(f)

count = 1

final_bwt = []
final_fwt = []
final_tpi = []

for key, value in results.items():

    count += 1

    bwt_cyber = value['4']['Financial Crime']['f1-score'] - value['0']['Financial Crime']['f1-score']
    bwt_drugs = value['4']['Cybercrime']['f1-score'] - value['0']['Cybercrime']['f1-score']
    bwt_goods = value['4']['Violent Crime']['f1-score'] - value['0']['Violent Crime']['f1-score']
    bwt_abuse = value['4']['Drugs / Narcotics']['f1-score'] - value['0']['Drugs / Narcotics']['f1-score']
    bwt_goods = value['4']['Goods and Services']['f1-score'] - value['0']['Goods and Services']['f1-score']
    bwt = bwt_cyber + bwt_drugs + bwt_goods + bwt_abuse + bwt_goods
    
    

    tpi_financial = value['4']['Sexual Abuse']['f1-score'] - value['0']['Sexual Abuse']['f1-score']


    fwt_financial = value['4']['Sexual Abuse']['f1-score']


    final_bwt.append(bwt)
    final_fwt.append(fwt_financial)
    final_tpi.append(tpi_financial)

final_bwt_abuse = sum(final_bwt)/(count-1)
final_fwt_abuse = sum(final_fwt)/(count-1)
final_tpi_abuse = sum(final_tpi)/(count-1)


with open(folder + method + '_cyber.json') as f:
    results = json.load(f)

count = 1

final_bwt = []
final_fwt = []
final_tpi = []

for key, value in results.items():

    count += 1

    bwt_cyber = value['4']['Financial Crime']['f1-score'] - value['0']['Financial Crime']['f1-score']
    bwt_drugs = value['4']['Sexual Abuse']['f1-score'] - value['0']['Sexual Abuse']['f1-score']
    bwt_goods = value['4']['Violent Crime']['f1-score'] - value['0']['Violent Crime']['f1-score']
    bwt_abuse = value['4']['Drugs / Narcotics']['f1-score'] - value['0']['Drugs / Narcotics']['f1-score']
    bwt_goods = value['4']['Goods and Services']['f1-score'] - value['0']['Goods and Services']['f1-score']
    bwt = bwt_cyber + bwt_drugs + bwt_goods + bwt_abuse + bwt_goods
    
    

    tpi_financial = value['4']['Cybercrime']['f1-score'] - value['0']['Cybercrime']['f1-score']


    fwt_financial = value['4']['Cybercrime']['f1-score']


    final_bwt.append(bwt)
    final_fwt.append(fwt_financial)
    final_tpi.append(tpi_financial)

final_bwt_cyber = sum(final_bwt)/(count-1)
final_fwt_cyber = sum(final_fwt)/(count-1)
final_tpi_cyber = sum(final_tpi)/(count-1)

with open(folder + method + '_drugs.json') as f:
    results = json.load(f)

count = 1

final_bwt = []
final_fwt = []
final_tpi = []

for key, value in results.items():

    count += 1

    bwt_cyber = value['4']['Financial Crime']['f1-score'] - value['0']['Financial Crime']['f1-score']
    bwt_drugs = value['4']['Sexual Abuse']['f1-score'] - value['0']['Sexual Abuse']['f1-score']
    bwt_goods = value['4']['Violent Crime']['f1-score'] - value['0']['Violent Crime']['f1-score']
    bwt_abuse = value['4']['Cybercrime']['f1-score'] - value['0']['Cybercrime']['f1-score']
    bwt_goods = value['4']['Goods and Services']['f1-score'] - value['0']['Goods and Services']['f1-score']
    bwt = bwt_cyber + bwt_drugs + bwt_goods + bwt_abuse + bwt_goods
    
    

    tpi_financial = value['4']['Drugs / Narcotics']['f1-score'] - value['0']['Drugs / Narcotics']['f1-score']


    fwt_financial = value['4']['Drugs / Narcotics']['f1-score']


    final_bwt.append(bwt)
    final_fwt.append(fwt_financial)
    final_tpi.append(tpi_financial)

final_bwt_drugs = sum(final_bwt)/(count-1)
final_fwt_drugs = sum(final_fwt)/(count-1)
final_tpi_drugs = sum(final_tpi)/(count-1)

with open(folder + method + '_financial.json') as f:
    results = json.load(f)

count = 1

final_bwt = []
final_fwt = []
final_tpi = []

for key, value in results.items():

    count += 1

    bwt_cyber = value['4']['Drugs / Narcotics']['f1-score'] - value['0']['Drugs / Narcotics']['f1-score']
    bwt_drugs = value['4']['Sexual Abuse']['f1-score'] - value['0']['Sexual Abuse']['f1-score']
    bwt_goods = value['4']['Violent Crime']['f1-score'] - value['0']['Violent Crime']['f1-score']
    bwt_abuse = value['4']['Cybercrime']['f1-score'] - value['0']['Cybercrime']['f1-score']
    bwt_goods = value['4']['Goods and Services']['f1-score'] - value['0']['Goods and Services']['f1-score']
    bwt = bwt_cyber + bwt_drugs + bwt_goods + bwt_abuse + bwt_goods
    
    

    tpi_financial = value['4']['Financial Crime']['f1-score'] - value['0']['Financial Crime']['f1-score']


    fwt_financial = value['4']['Financial Crime']['f1-score']


    final_bwt.append(bwt)
    final_fwt.append(fwt_financial)
    final_tpi.append(tpi_financial)

final_bwt_financial = sum(final_bwt)/(count-1)
final_fwt_financial = sum(final_fwt)/(count-1)
final_tpi_financial = sum(final_tpi)/(count-1)



with open(folder + method + '_goods.json') as f:
    results = json.load(f)

count = 1

final_bwt = []
final_fwt = []
final_tpi = []

for key, value in results.items():

    count += 1

    bwt_cyber = value['4']['Drugs / Narcotics']['f1-score'] - value['0']['Drugs / Narcotics']['f1-score']
    bwt_drugs = value['4']['Sexual Abuse']['f1-score'] - value['0']['Sexual Abuse']['f1-score']
    bwt_goods = value['4']['Violent Crime']['f1-score'] - value['0']['Violent Crime']['f1-score']
    bwt_abuse = value['4']['Cybercrime']['f1-score'] - value['0']['Cybercrime']['f1-score']
    bwt_goods = value['4']['Financial Crime']['f1-score'] - value['0']['Financial Crime']['f1-score']
    bwt = bwt_cyber + bwt_drugs + bwt_goods + bwt_abuse + bwt_goods
    
    

    tpi_financial = value['4']['Goods and Services']['f1-score'] - value['0']['Goods and Services']['f1-score']


    fwt_financial = value['4']['Goods and Services']['f1-score']


    final_bwt.append(bwt)
    final_fwt.append(fwt_financial)
    final_tpi.append(tpi_financial)

final_bwt_goods = sum(final_bwt)/(count-1)
final_fwt_goods = sum(final_fwt)/(count-1)
final_tpi_goods = sum(final_tpi)/(count-1)

with open(folder + method + '_violent.json') as f:
    results = json.load(f)

count = 1

final_bwt = []
final_fwt = []
final_tpi = []

for key, value in results.items():

    count += 1

    bwt_cyber = value['4']['Financial Crime']['f1-score'] - value['0']['Financial Crime']['f1-score']
    bwt_drugs = value['4']['Cybercrime']['f1-score'] - value['0']['Cybercrime']['f1-score']
    bwt_goods = value['4']['Sexual Abuse']['f1-score'] - value['0']['Sexual Abuse']['f1-score']
    bwt_abuse = value['4']['Drugs / Narcotics']['f1-score'] - value['0']['Drugs / Narcotics']['f1-score']
    bwt_goods = value['4']['Goods and Services']['f1-score'] - value['0']['Goods and Services']['f1-score']
    bwt = bwt_cyber + bwt_drugs + bwt_goods + bwt_abuse + bwt_goods
    
    

    tpi_financial = value['4']['Violent Crime']['f1-score'] - value['0']['Violent Crime']['f1-score']


    fwt_financial = value['4']['Violent Crime']['f1-score']


    final_bwt.append(bwt)
    final_fwt.append(fwt_financial)
    final_tpi.append(tpi_financial)

final_bwt_violent = sum(final_bwt)/(count-1)
final_fwt_violent = sum(final_fwt)/(count-1)
final_tpi_violent = sum(final_tpi)/(count-1)

tpi = (final_tpi_abuse +final_tpi_cyber + final_tpi_drugs + final_tpi_financial + final_tpi_goods + final_tpi_violent)/6
bwt = (final_bwt_abuse +final_bwt_cyber + final_bwt_drugs + final_bwt_financial + final_bwt_goods + final_bwt_violent)/6
fwt = (final_fwt_abuse +final_fwt_cyber + final_fwt_drugs + final_fwt_financial + final_fwt_goods + final_fwt_violent)/6

print('BWT: ', bwt)
print('FWT: ', fwt)
print('TPI: ', tpi)



