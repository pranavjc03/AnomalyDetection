# -*- coding: utf-8 -*-
"""
Created on Tue May  2 17:58:41 2023

@author: user
"""
'''
import pandas as pd

import numpy as np

train_url = 'NSL_KDD_Train.csv'


col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


df = pd.read_csv(train_url)

print(df)

df.to_csv("labelled_NSL_KDD_Train.csv")


'''

import pandas as pd

import numpy as np

train_url = 'labelled_NSL_KDD_Train.csv'


df = pd.read_csv(train_url)

feat_col_names = ["src_bytes","dst_bytes","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"]

print(len(feat_col_names))

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['label'] = le.fit_transform(df['label'])

print(df)


from sklearn.ensemble import RandomForestClassifier


alg = RandomForestClassifier()


train_x = df[feat_col_names].iloc[:,:]

train_y = df['label'].iloc[:]

alg.fit(train_x, train_y)

print("RF alg got trained")

inpdata = [0,0,117,16,1.0,1.0,0.0,0.0,0.14,0.06,0.0,255,15,0.06,0.07,0.0,0.0,1.0,1.0,0.0,0.0]


ypred = alg.predict([inpdata])


print(ypred)