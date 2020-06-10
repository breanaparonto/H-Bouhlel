import nltk
nltk.download('punkt')

import sklearn
import csv
import re
import unicodedata
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize


def replace_str(df,col,entree,sortie):
    for i in entree : 
        df[col]=df[col].str.replace(i,sortie)
    return 
    
def replace(df,col,list_entree,sortie):
    for i in list_entree : 
        df[col]=df[col].str.replace(i,sortie)
    return 

def replace_list(df,col,list_entree,list_sortie):
    for i,j in zip(list_entree,list_sortie):
        df[col]=df[col].str.replace(i,j)
    return 

def DFtoList(df,col,result):
    result=df.reset_index()[col].values.tolist()
    return

def nan_remove(liste):
    liste = [i for i in liste if str(i) != 'nan']

def sup_doublons(liste):
    liste = list(dict.fromkeys(liste))
    return
def sup_espace(liste):
    liste=[i.lstrip() for i in liste]
    

def modification(df,col,fct):
    df[col]=df[col].apply(lambda x: [item for item in x if item not in fct])
    return


def preprocessing(df,col,liste):
    df[col]=[untokenize(entry) for entry in liste]
    df[col]=[accent_remove(entry) for entry in df['example']]
    df[col] =df['example'].str.replace(' +', ' ')
    return 
    
def stem(df,col):
    liste=[]
    stemmer = FrenchStemmer()
    for i in df[col]:
        liste.append ([stemmer.stem(word) for word in i])
    return liste 

def untokenize(words):
    text = ' '.join(words) 
    return text.strip()

def accent_remove(s):

    text = ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')) 
    return text    
