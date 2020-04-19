import nltk
nltk.download('punkt')

import sklearn
import csv
import re
import unicodedata
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix

from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import word_tokenize

from preprocessing import*
from file import*
from LSA import LSA


df = pd.read_excel(r"C:\Users\ahmed\Desktop\Use_case_ASSU\Final_results\input_file.xlsx")
#df=df.rename(columns = {'is linked to [Local Data Element] > Name':'Donnee_metier'})


col1=df.columns[0]
col2=df.columns[1]

#proxy sur les outliers ( les anomalies)
proxy1=["N° Personne Sogessur","Numéro de personne","N° de personne,SI - N° Personne","SI - N° Personne Sogessur","N° Personne"]
proxy2=["CO - Coefficient Commercial","CO - Coefficient Commercial (Réduction Salariés)"]

proxy3=["SI - Date Saisie Montant","SI - Montant Provisions Restantes","Numéro de Police SOGESSUR","N° Devis","ALD - N° Police SOG","SI - Amount Incurred",
        "CO - Acceptation FID O/N","SI - Montant Total Paiements","CO - CSP de l'assuré principal","SI - City of Accident"]

ch=["Date mouvement technique sinistre","Montant de la provision cédée restante","Numéro de contrat","Numéro de contrat","Numéro de contrat",
    "Provision brute à la garantie du sinistre","Coefficient de fidélisation auto","Règlement net de recours à la garantie du sinistre",
    "Catégorie socio-professionnelle de l'assuré","Ville du sinistre MRH"]

proxy4=["CO - Commercial Premium Amount","CO - Gross Premium","CO - Montat Prime HT"]

replace(df,col1,proxy1,"Numéro de client assuré")
replace(df,col1,proxy2,"Ancienne réduction salariés (RED)")
replace_list(df,col1,proxy3,ch)
replace(df,col2,proxy4,"Montant de la prime commerciale d'assurance hors taxes en devise de reporting")

#preprocessing

df=df.apply(lambda x: x.astype(str).str.lower())

l1=['n°client','n°',"d'",'[^\w\s]']
l2=['numero client','numero','',' ']
replace_list(df,col1,l1,l2)

li1=['n°',"d'",'[^\w\s]']
li2=['numero','',' ']
replace_list(df,col2,li1,li2)

#Dictionnaire
liste1=['number','region','contract','name','customer','guarantee','address',"claim","closed","opened","status","HT","amendment","somme","crmr","csdp","sdp","product"]
liste2=['numero','pays','contrat','nom','client','garantie','adresse',"sinistre","cloture","ouverture","etat","Hors Taxes","avenant","montant","crm recalculé","coefficient de surveillance du portefeuille","surveillance du portefeuille","produit"]
replace_list(df,col1,liste1,liste2)

example_des=[]
example_m=[]    
DFtoList(df,col1,example_des)
DFtoList(df,col2,example_m)

sup_doublons(example_m)
nan_remove(example_m)
nan_remove(example_des)

sup_espace(example_des)
#sup_espace(example_m)

#Chunk les données

path=r"C:\Users\ahmed\Desktop\Use_case_ASSU\Final_results"

create_file(path,"\output_file.csv")
add_row(path,"\output_file.csv","Data1","Data2","correlation")
    
chunk_size=3000
chunks = [example_des[x:x+chunk_size] for x in range(0, len(example_des),chunk_size)]

for i,j in zip(chunks,range(len(chunks))):
    example=example_m+i
    
    print(len(example))
    
    df = pd.DataFrame({'example':example})
    df['example']= [word_tokenize(entry) for entry in df['example']]
    stopWords = set(stopwords.words('French'))
    stopWords_ang = set(stopwords.words('English'))
    l=["-","d","co","si"]
    modification(df,"example",stopWords_ang)
    modification(df,"example",stopWords)
    modification(df,"example",l)
    #stemming_frensh

    stem_list= stem(df,"example")

    preprocessing(df,'example',stem_list)
    example=df.values.tolist()
    
    flat_list = lambda l: [item for sublist in l for item in sublist]
    example=[i.lstrip() for i in df['example']]
    
    #Model
    dtm_lsa= LSA(example)

    df = pd.read_excel(path+"\input_file.xlsx")
    #example_des_b=df.reset_index()['Description'].values.tolist()
    
    #Label des Données brutes
    df[col2] = df[col2][(~df[col2].duplicated()) | df[col2].isna()] 
    #df to list
    example_des_b=df.reset_index()[col1].values.tolist()
    example_des_b = [i for i in example_des_b if str(i) != 'nan']
    example_m_b=df.reset_index()[col2].values.tolist()
    example_m_b = list(dict.fromkeys(example_m_b))

    
    example_m_b = [i for i in example_m_b if str(i) != 'nan']
    chunks_b= [example_des_b[x:x+chunk_size] for x in range(0, len(example_des_b), chunk_size)]
    example_num_b=example_m_b+chunks_b[j]

    
    similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
    CrossValid=pd.DataFrame(similarity,index=example_num_b,columns=example_num_b)
    Matrice_M_Phy=CrossValid.iloc[len(example_m_b):, 0:len(example_m_b)]
    
    with open(path+"\output_file.csv", mode='a',newline='') as file:       
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for index, row in Matrice_M_Phy.iterrows():  
            writer.writerow([index,row.idxmax(),round(row.loc[row.idxmax()],4)])