import pandas as pd

from os.path import exists
import bs4
import codecs
import numpy as np
import hashlib
from src.preprocessing.text_preprocessing import TextPreprocessor


def bs42element_tree( parent ) :
	""" Return the structure of an HTML page as a tree with only the element names """
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


df_copy = pd.DataFrame('-', index=np.arange(7), columns=['html', 'tree', 'hash'])
folder_path = '/home/pablo/active-learning-pablo/prueba_duplicados/prueba'

count_found = 0
count_copied = 0

for i in range(7):
    try:
        file = str(i + 1) + '.html'
        file_path = folder_path + file
        if exists(file_path):
            count_found += 1
            df_copy['tree'][i] = bs42element_tree(bs4.BeautifulSoup(codecs.open(file_path, 'r', 'utf-8').read()))
            df_copy['html'][i] = bs4.BeautifulSoup(codecs.open(file_path, 'r', 'utf-8').read()).get_text()
            df_copy['hash'][i] = hashlib.md5(codecs.open(file_path, 'r', encoding = 'utf-8').read().encode('utf-8')).hexdigest()
            count_copied += 1
    except:
        continue #html_list.append('-')
print('Found files: ', count_found)
print('Copied files: ', count_copied)

df_copy = df_copy.dropna()
df_copy = df_copy.drop(df_copy[df_copy["html"].map(len) == 0].index)
df_copy = df_copy.drop(df_copy[df_copy["html"] == '-'].index)
df_copy = df_copy.drop(df_copy[df_copy["tree"].map(len) == 0].index)
df_copy = df_copy.drop(df_copy[df_copy["tree"] == '-'].index)
df_copy = df_copy.drop(df_copy[df_copy["hash"].map(len) == 0].index)
df_copy = df_copy.drop(df_copy[df_copy["hash"] == '-'].index)
print('shape after removing empty: ', df_copy.shape)

df_copy = df_copy.drop_duplicates(subset=['html'], keep='first')
print("\n-Number of remainig samples after deduplication of html text", df_copy.shape)

df_copy = df_copy.drop_duplicates(subset=['tree'], keep='first')
print("\n-Number of remainig samples after deduplication of html tree", df_copy.shape)

df_copy = df_copy.drop_duplicates(subset=['hash'], keep='first')
print("\n-Number of remainig samples after deduplication of html hash", df_copy.shape)
