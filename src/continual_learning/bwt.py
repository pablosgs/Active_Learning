import json
import numpy as np

with open('results/test/active_learning/dark_wo_duplicates/DA/clue.json') as f:
    results = json.load(f)

count = 1

final_bwt = []
final_fwt = []
final_tpi = []

for key, value in results.items():

    count += 1

    bwt_financial = 1/(5) * (value['9']['Financial Crime']['f1-score'] + value['14']['Financial Crime']['f1-score'] + value['19']['Financial Crime']['f1-score'] + value['24']['Financial Crime']['f1-score'] + value['29']['Financial Crime']['f1-score']  - 5*value['5']['Financial Crime']['f1-score'])
    bwt_cyber = 1/(4) * (value['14']['Drugs / Narcotics']['f1-score'] + value['19']['Drugs / Narcotics']['f1-score'] + value['24']['Drugs / Narcotics']['f1-score'] + value['29']['Drugs / Narcotics']['f1-score'] - 4*value['9']['Drugs / Narcotics']['f1-score'])
    bwt_drugs = 1/(3) * (value['19']['Cybercrime']['f1-score'] + value['24']['Cybercrime']['f1-score'] + value['29']['Cybercrime']['f1-score'] - 3*value['14']['Cybercrime']['f1-score'])
    bwt_goods = 1/(2) * (value['24']['Goods and Services']['f1-score'] + value['29']['Goods and Services']['f1-score'] - 2*value['19']['Goods and Services']['f1-score'])
    bwt_abuse = 1/(1) * (value['29']['Sexual Abuse']['f1-score'] - 1*value['24']['Sexual Abuse']['f1-score'])
    bwt = bwt_financial + bwt_cyber + bwt_drugs + bwt_goods + bwt_abuse
    

    tpi_financial = value['4']['Financial Crime']['f1-score'] - value['0']['Financial Crime']['f1-score']
    tpi_cyber = value['9']['Drugs / Narcotics']['f1-score'] - value['4']['Drugs / Narcotics']['f1-score']
    tpi_drugs = value['14']['Cybercrime']['f1-score'] - value['9']['Cybercrime']['f1-score']
    tpi_goods = value['19']['Goods and Services']['f1-score'] - value['14']['Goods and Services']['f1-score']
    tpi_abuse = value['24']['Sexual Abuse']['f1-score'] - value['19']['Sexual Abuse']['f1-score']
    tpi_violent = value['29']['Violent Crime']['f1-score'] - value['24']['Violent Crime']['f1-score']
    tpi = tpi_financial + tpi_cyber + tpi_drugs + tpi_goods + tpi_abuse + tpi_violent

    fwt_financial = 1/(1) * (value['0']['Financial Crime']['f1-score'])
    fwt_cyber = 1/(2) * (value['0']['Drugs / Narcotics']['f1-score'] + value['4']['Drugs / Narcotics']['f1-score'])
    fwt_drugs = 1/(3) * (value['0']['Cybercrime']['f1-score'] + value['4']['Cybercrime']['f1-score'] + value['9']['Cybercrime']['f1-score'])
    fwt_goods = 1/(5) * (value['0']['Goods and Services']['f1-score'] + value['4']['Goods and Services']['f1-score'] + value['9']['Goods and Services']['f1-score'] + value['14']['Goods and Services']['f1-score'])
    fwt_abuse = 1/(5) * (value['0']['Sexual Abuse']['f1-score'] + value['4']['Sexual Abuse']['f1-score'] + value['9']['Sexual Abuse']['f1-score'] + value['14']['Sexual Abuse']['f1-score'] + value['19']['Sexual Abuse']['f1-score'])
    fwt_violent = 1/(6) * (value['0']['Violent Crime']['f1-score'] + value['4']['Violent Crime']['f1-score'] + value['9']['Violent Crime']['f1-score'] + value['14']['Violent Crime']['f1-score'] + value['19']['Violent Crime']['f1-score'] + value['24']['Violent Crime']['f1-score'])
    
    fwt = fwt_financial + fwt_cyber + fwt_drugs + fwt_goods + fwt_abuse

    final_bwt.append(bwt)
    final_fwt.append(fwt)
    final_tpi.append(tpi)

final_bwt = sum(final_bwt)/count
final_fwt = sum(final_fwt)/count
final_tpi = sum(final_tpi)/count

print('BWT: ', final_bwt)
print('FWT: ', final_fwt)
print('TPI: ', final_tpi)

