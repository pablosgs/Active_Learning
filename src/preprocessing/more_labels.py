import pandas as pd

from os.path import exists
import bs4
import codecs
import numpy as np
import hashlib
import sys 
sys.setrecursionlimit( 10000 )


def bs42element_tree( parent ) :
	if isinstance( parent, bs4.BeautifulSoup ) :
		return ''.join( bs42element_tree( child ) for child in parent.children )
	if isinstance( parent, bs4.Doctype ) :
		return ''
	if isinstance( parent, bs4.element.Comment ) :
		return ''
	if isinstance( parent, bs4.element.NavigableString ) :
		return ''
	parent_name = parent.name
	children = ''.join( bs42element_tree( child ) for child in parent.children )
	if len( children ) == 0 :
		return f' {parent_name}'
	return ' {' + parent_name + children + '}'

from src.preprocessing.text_preprocessing import TextPreprocessor
df2 = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/script_data_v2/final.csv', on_bad_lines='warn')
df3 = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/cflw_04_05_2023/metadata.csv', on_bad_lines='warn')
df4 = pd.read_json('/home/pablo/fastStorage/data/pablo_30_05_2023/metadata.json')
df = pd.concat([pd.concat([df2, df3], ignore_index=True),df4], ignore_index=True)
df['RelatedTagsArray'] = df['RelatedTagsArray'].str.split(',')
tag_counts = df['RelatedTagsArray'].explode().value_counts().copy()
#print(tag_counts)

abuse_labels = tag_counts.index.values.tolist()
#df['RelatedTagsArray'][:10]
""""
abuse_labels = [
        'Cybercrime',
       'Drugs / Narcotics', 'Financial Crime', 'Goods and Services',
       'Sexual Abuse', 'Violent Crime']
"""
for i in abuse_labels:
    df2[i] = 0

#df_copy = df.filter(['page_created_at', 'host_id', 'page_id', 'page_version_id','RelatedTagsArray', 'Cybercrime',
#       'Drugs / Narcotics', 'Financial Crime', 'Goods and Services',
#       'Sexual Abuse', 'Violent Crime'], axis=1)
#df_copy = df.filter(['domain_id', 'page_id', 'page_version_id','RelatedTagsArray', 'Cybercrime',
#       'Drugs / Narcotics', 'Financial Crime', 'Goods and Services',
#       'Sexual Abuse', 'Violent Crime'], axis=1)
#df_copy = df.filter(['domain_id', 'page_id', 'snapshot_id', 'page_version_id','RelatedTagsArray', 'Cybercrime',
#       'Drugs / Narcotics', 'Financial Crime', 'Goods and Services',
#       'Sexual Abuse', 'Violent Crime'], axis=1)

df_copy = df2.filter(['host_id', 'page_id', 'page_version_id','RelatedTagsArray'] + abuse_labels, axis=1)

for i in range(len(df_copy['RelatedTagsArray'])):
    for label in abuse_labels:
        if label in df_copy['RelatedTagsArray'][i]:
            df_copy[label][i] = 1

documents = []
df_copy['html'] = '-'
df_copy['text'] = '-'
#df_copy['hash'] = '-'
folder_path = '/home/pablo/active-learning-pablo/data/datasets/script_data_v2/html/'
#folder_path = '/home/pablo/active-learning-pablo/data/datasets/cflw_04_05_2023/html/'
#folder_path = '/home/pablo/fastStorage/data/pablo_30_05_2023/html/'
df = df_copy.copy(deep = True)


html_list = []
count_found = 0
count_copied = 0

for i in range(len(df_copy['RelatedTagsArray'])):
    file = str(df_copy.loc[i,'host_id']) + '_' + str(df_copy.loc[i,'page_id']) + '_' + str(df_copy.loc[i,'page_version_id']) + '.html'
    #file = str(df_copy.loc[i,'domain_id']) + '_' + str(df_copy.loc[i,'page_id']) + '_' + str(df_copy.loc[i,'page_version_id']) + '.html'
    file_path = folder_path + file
    if exists(file_path):
        #with open(file_path, "r") as html_file:
        count_found += 1
        #html = html_file.read()
        #f=codecs.open(file_path, 'r', 'utf-8').read()
        df_copy.loc[i,'html'] = codecs.open(file_path, 'r', 'utf-8').read()
        df_copy.loc[i,'text'] = bs4.BeautifulSoup(codecs.open(file_path, 'r', 'utf-8').read()).get_text()

        #df_copy.loc[i,'html'] = bs4.BeautifulSoup(codecs.open(file_path, 'r', 'utf-8').read()).get_text()
        #df_copy.loc[i,'tree'] = bs42element_tree(bs4.BeautifulSoup(codecs.open(file_path, 'r', 'utf-8').read()))
        #df_copy.loc[i,'hash'] = hashlib.md5(codecs.open(file_path, 'r', encoding = 'utf-8').read().encode('utf-8')).hexdigest()
        count_copied += 1
        #html_file.close()

df_copy = df_copy.filter(['html', 'text','RelatedTagsArray', 'Cybercrime',
       'Drugs / Narcotics', 'Financial Crime', 'Goods and Services',
       'Sexual Abuse', 'Violent Crime'], axis=1)

print('Found files: ', count_found)
print('Copied files: ', count_copied)

#df_copy.to_csv('/home/pablo/active-learning-pablo/data/datasets/cflw_large_with_trees_v3.csv', index=False)





#df_copy = pd.read_csv('/home/pablo/active-learning-pablo/data/datasets/cflw_large_with_trees.csv', on_bad_lines='warn')
print('shape before dropping: ', df_copy.shape)
#df_copy['html'] = html_list
df_copy = df_copy.dropna()
df_copy = df_copy.drop(df_copy[df_copy["html"].map(len) == 0].index)
df_copy = df_copy.drop(df_copy[df_copy["html"] == '-'].index)
df_copy = df_copy.drop(df_copy[df_copy["text"].map(len) == 0].index)
df_copy = df_copy.drop(df_copy[df_copy["text"] == '-'].index)
#df_copy = df_copy.drop(df_copy[df_copy["hash"].map(len) == 0].index)
#df_copy = df_copy.drop(df_copy[df_copy["hash"] == '-'].index)
print('shape after removing empty: ', df_copy.shape)

df_copy = df_copy.drop_duplicates(subset=['html', 'RelatedTagsArray'], keep='first')
print("\n-Number of remainig samples after deduplication of html text", df_copy.shape)

df_copy = df_copy.drop_duplicates(subset=['text', 'RelatedTagsArray'], keep='first')
print("\n-Number of remainig samples after deduplication of html tree", df_copy.shape)

#df_copy['dup1'] = df_copy.duplicated(subset = ['tree'], keep='first')
#df_copy['dup2'] = df_copy.duplicated(subset = ['tree'], keep='last')
#df_copy.dup1.ne(False).idxmax()
#dups1 = df_copy.loc[df_copy.dup1==True]['html'].values.tolist()
#dups2 = df_copy.loc[df_copy.dup2==True]['html'].values.tolist()
#alineated = pd.DataFrame({'1':dups1,'2':dups2})
#df_dup = df_copy[df_copy.duplicated(subset = ['tree'], keep=False)]
#df_dup.sort_values('html',inplace=True, ascending=False)
#df = df_copy
#import difflib
# Threshold filter based on Percentage similarity
#thr = 0.9
#df['Flag'] = 0
#for text in df['html'].tolist():
    #df['temp'] = [difflib.SequenceMatcher(None, text1,text).ratio() for text1 in df['html'].tolist()]
    #df.loc[df['temp'].gt(thr),['Flag']] = df['Flag'].max()+1
#df.drop('temp',1)

#df.loc[~df['Flag'].duplicated(keep='first')]
#print("\n-Number of remainig samples after deduplication of html tree", df.shape)

#df_copy = df_copy.drop_duplicates(subset=['hash'], keep='first')
#print("\n-Number of remainig samples after deduplication of html hash", df_copy.shape)
df_copy.to_csv('/home/pablo/active-learning-pablo/data/datasets/more_labels_2.csv', index=False)
#df_copy.to_csv('/home/pablo/active-learning-pablo/data/datasets/cflw_large_without_duplicates_v3.csv', index=False)
