# -*- coding: utf-8 -*-
"""
Created on Tue May  2 17:54:54 2023

@author: user
"""

import tkinter as tk

from tkinter import *

from PIL import Image, ImageTk

window = tk.Tk()

window.title("INTRUSION DETECTION PROJECT")

window.configure(background = "green")

window.geometry('1260x680')

window.grid_rowconfigure(0, weight =1)

window.grid_columnconfigure(0, weight =1)

IMAGE_PATH = 'imagebg.jpg'

WIDTH, HEIGTH = 1260, 680

canvas = tk.Canvas(window, width=WIDTH, height=HEIGTH)

canvas.pack()

img = ImageTk.PhotoImage(Image.open(IMAGE_PATH).resize((WIDTH, HEIGTH), Image.ANTIALIAS))

canvas.background = img  # Keep a reference in case this code is put in a function.

bg = canvas.create_image(0, 0, anchor=tk.NW, image=img)

lb = tk.Label(window, text = "INTRUSION DETECTION PROJECT", bg = 'black', fg = 'white', font = ('times', 30, 'bold'))
lb.place(x=400, y=20)


l1 = tk.Label(window, text = "Enter src_bytes: ", bg = 'black', fg = 'white')
l1.place(x=100, y=100) 

tt1 = StringVar()

t1 = tk.Entry(window, textvariable = tt1)
t1.place(x = 350, y =100)



l2 = tk.Label(window, text = "Enter dst_bytes: ", bg = 'black', fg = 'white')
l2.place(x=100, y=150) 

tt2 = StringVar()

t2 = tk.Entry(window, textvariable = tt2)
t2.place(x = 350, y =150)



l3 = tk.Label(window, text = "Enter count: ", bg = 'black', fg = 'white')
l3.place(x=100, y=200) 

tt3 = StringVar()

t3 = tk.Entry(window, textvariable = tt3)
t3.place(x = 350, y =200)


l4 = tk.Label(window, text = "Enter srv_count: ", bg = 'black', fg = 'white')
l4.place(x=100, y=250) 

tt4 = StringVar()

t4 = tk.Entry(window, textvariable = tt4)
t4.place(x = 350, y =250)


l5 = tk.Label(window, text = "Enter serror_rate: ", bg = 'black', fg = 'white')
l5.place(x=100, y=300) 

tt5 = StringVar()

t5 = tk.Entry(window, textvariable = tt5)
t5.place(x = 350, y =300)


l6 = tk.Label(window, text = "Enter srv_serror_rate: ", bg = 'black', fg = 'white')
l6.place(x=100, y=350) 

tt6 = StringVar()

t6 = tk.Entry(window, textvariable = tt6)
t6.place(x = 350, y =350)


l7 = tk.Label(window, text = "Enter rerror_rate: ", bg = 'black', fg = 'white')
l7.place(x=100, y=400) 

tt7 = StringVar()

t7 = tk.Entry(window, textvariable = tt7)
t7.place(x = 350, y =400)

l8 = tk.Label(window, text = "Enter srv_rerror_rate: ", bg = 'black', fg = 'white')
l8.place(x=100, y=450) 

tt8 = StringVar()

t8 = tk.Entry(window, textvariable = tt8)
t8.place(x = 350, y =450)

l9 = tk.Label(window, text = "Enter same_srv_rate: ", bg = 'black', fg = 'white')
l9.place(x=100, y=500) 

tt9 = StringVar()

t9 = tk.Entry(window, textvariable = tt9)
t9.place(x = 350, y =500)


l10 = tk.Label(window, text = "Enter diff_srv_rate: ", bg = 'black', fg = 'white')
l10.place(x=100, y=550) 

tt10 = StringVar()

t10 = tk.Entry(window, textvariable = tt10)
t10.place(x = 350, y =550)

l11 = tk.Label(window, text = "Enter srv_diff_host_rate: ", bg = 'black', fg = 'white')
l11.place(x=100, y=600) 

tt11 = StringVar()

t11 = tk.Entry(window, textvariable = tt11)
t11.place(x = 350, y =600)

l12 = tk.Label(window, text = "Enter dst_host_count: ", bg = 'black', fg = 'white')
l12.place(x=100, y=650) 

tt12 = StringVar()

t12 = tk.Entry(window, textvariable = tt12)
t12.place(x = 350, y =650)






l13 = tk.Label(window, text = "Enter dst_host_srv_count: ", bg = 'black', fg = 'white')
l13.place(x=600, y=100) 

tt13 = StringVar()

t13 = tk.Entry(window, textvariable = tt13)
t13.place(x = 800, y =100)


l14 = tk.Label(window, text = "Enter dst_host_same_srv_rate: ", bg = 'black', fg = 'white')
l14.place(x=600, y=150) 

tt14 = StringVar()

t14 = tk.Entry(window, textvariable = tt14)
t14.place(x = 800, y =150)


l15 = tk.Label(window, text = "Enter dst_host_diff_srv_rate: ", bg = 'black', fg = 'white')
l15.place(x=600, y=200) 

tt15 = StringVar()

t15 = tk.Entry(window, textvariable = tt15)
t15.place(x = 800, y =200)

l16 = tk.Label(window, text = "Enter dst_host_same_src_port_rate: ", bg = 'black', fg = 'white')
l16.place(x=600, y=250) 

tt16 = StringVar()

t16 = tk.Entry(window, textvariable = tt16)
t16.place(x = 800, y =250)


l17 = tk.Label(window, text = "Enter dst_host_srv_diff_host_rate: ", bg = 'black', fg = 'white')
l17.place(x=600, y=300) 

tt17 = StringVar()

t17 = tk.Entry(window, textvariable = tt17)
t17.place(x = 800, y =300)


l18 = tk.Label(window, text = "Enter dst_host_serror_rate: ", bg = 'black', fg = 'white')
l18.place(x=600, y=350) 

tt18 = StringVar()

t18 = tk.Entry(window, textvariable = tt18)
t18.place(x = 800, y =350)

l19 = tk.Label(window, text = "Enter dst_host_srv_serror_rate: ", bg = 'black', fg = 'white')
l19.place(x=600, y=400) 

tt19 = StringVar()

t19 = tk.Entry(window, textvariable = tt19)
t19.place(x = 800, y =400)

l20 = tk.Label(window, text = "Enter dst_host_rerror_rate: ", bg = 'black', fg = 'white')
l20.place(x=600, y=450) 

tt20 = StringVar()

t20 = tk.Entry(window, textvariable = tt20)
t20.place(x = 800, y =450)

l21 = tk.Label(window, text = "Enter dst_host_srv_rerror_rate: ", bg = 'black', fg = 'white')
l21.place(x=600, y=500) 

tt21 = StringVar()

t21 = tk.Entry(window, textvariable = tt21)
t21.place(x = 800, y =500)





def predict():
    print('Start Prediction!!!!!')
    import pandas as pd

    import numpy as np
    
    import boto3
    import botocore
    
    BUCKET_NAME = 'nwintrusion' # replace with your bucket name
    KEY = 'clouddata.csv' # replace with your object key
    
    s3 = boto3.client('s3', aws_access_key_id='AKIA3FLDXMYSXHGECICH' , aws_secret_access_key='8O0zZaLnKb24m47H9Zxt+NV2HER9cg5uaRnXQGGP')
    print('connection established')
    
    s3.download_file('adcnml','clouddata.csv','traindata.csv')

    
    print('file downloaded as traindata.csv file')

    train_url = 'traindata.csv'


    df = pd.read_csv(train_url)

    feat_col_names = ["src_bytes","dst_bytes","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate"]

    print(len(feat_col_names))
    
    '''

    from sklearn import preprocessing

    le = preprocessing.LabelEncoder()

    df['label'] = le.fit_transform(df['label'])

    print(df)
    
    '''


    from sklearn.ensemble import RandomForestClassifier


    alg = RandomForestClassifier()


    train_x = df[feat_col_names].iloc[:,:]

    train_y = df['label'].iloc[:]

    alg.fit(train_x, train_y)

    print("RF alg got trained")

    #inpdata = [0,0,117,16,1.0,1.0,0.0,0.0,0.14,0.06,0.0,255,15,0.06,0.07,0.0,0.0,1.0,1.0,0.0,0.0]


    #ypred = alg.predict([inpdata])


    #print(ypred)
    
    
    x1 = tt1.get()
    
    x2 = tt2.get()
    
    x3 = tt3.get()
    
    x4 = tt4.get()
    
    x5 = tt5.get()
    
    x6 = tt6.get()
    
    x7 = tt7.get()
    
    x8 = tt8.get()
    
    x9 = tt9.get()
    
    x10 = tt10.get()
    
    x11 = tt11.get()
    
    x12 = tt12.get()
    
    x13 = tt13.get()
    
    x14 = tt14.get()
    
    x15 = tt15.get()
    
    x16 = tt16.get()
    
    x17 = tt17.get()
    
    x18 = tt18.get()
    
    x19 = tt19.get()
    
    x20 = tt20.get()
    
    x21 = tt21.get()
    
    
    
    inp = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21]
    
    
    print(inp)
    
    ypred = alg.predict([inp])
    
    print(ypred)
    
    
    if ypred[0] == 0:
        print('Detected as Normal')
        
        patout.set('Detected as Normal')
        
    elif ypred[0] == 1:
        print('Detected as DOS')
        
        patout.set('Detected as DOS')
    elif ypred[0] == 2:
        print('Detected as PRobe')
        
        patout.set('Detected as Probe')
    elif ypred[0] == 3:
        print('Detected as U2R')
        
        patout.set('Detected as U2R')
                  
    else:
        print('Detected as R2L')
        
        patout.set('Detected as R2L')
    
            
    

but = tk.Button(window, text = 'Predict', command = predict, bg = 'red', fg = 'white', width = 20, height =1)

but.place(x = 600, y = 550)


out = tk.Label(window, text = "Predicted Output is: ", bg = 'white', fg = 'white')
out.place(x=600, y=600) 

patout = StringVar()

outEntry = tk.Entry(window, textvariable = patout, width = 20)
outEntry.place(x = 800, y =600)

window.mainloop()