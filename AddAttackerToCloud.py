# -*- coding: utf-8 -*-
"""
Created on Tue May  2 19:28:08 2023

@author: user
"""


import tkinter as tk
import os
from tkinter import *
from PIL import Image, ImageTk
window = tk.Tk()

import boto3
from botocore.exceptions import NoCredentialsError

ACCESS_KEY = 'AKIA3FLDXMYSXHGECICH'
SECRET_KEY = '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'


IMAGE_PATH = 'imagebg1.jpg'

WIDTH, HEIGTH = 1260, 680

canvas = tk.Canvas(window, width=WIDTH, height=HEIGTH)

canvas.pack()

img = ImageTk.PhotoImage(Image.open(IMAGE_PATH).resize((WIDTH, HEIGTH), Image.ANTIALIAS))

canvas.background = img  # Keep a reference in case this code is put in a function.

bg = canvas.create_image(0, 0, anchor=tk.NW, image=img)

window.title("INTRUSION ATTACKER TO CLOUD")

window.configure(background = "white")

window.geometry('1260x680')

window.grid_rowconfigure(0, weight =1)

window.grid_columnconfigure(0, weight =1)


lb = tk.Label(window, text = "INTRUSION ATTACKER TO CLOUD", bg = 'black', fg = 'white', font = ('times', 30, 'bold'))
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



l22 = tk.Label(window, text = "(0:N::1:DOS::2:Probe::3:U2R::4:R2L)", bg = 'black', fg = 'white')
l22.place(x=600, y=550) 

tt22 = StringVar()

t22 = tk.Entry(window, textvariable = tt22)
t22.place(x = 800, y =550)

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def addtocloud():
    
    
    feat_col_names = ["src_bytes","dst_bytes","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

    print(len(feat_col_names))   
    
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
    
    x22 = tt22.get()
    
    inp = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22]
    
    f = open('clouddata.csv','a+')
    f.write(x1+','+x2+','+x3+','+x4+','+x5+','+x6+','+x7+','+x8+','+x9+','+x10+','+x11+','+x12+','+x13+','+x14+','+x15+','+x16+','+x17+','+x18+','+x19+','+x20+','+x21+','+x22)
    f.write(os.linesep)
    f.close()
    
    '''
    import csv    
      
    with open('clouddata.csv', 'w') as f:
          
        # using csv.writer method from CSV package
        write = csv.writer(f)
          
        write.writerow(feat_col_names)
        write.writerow(inp)
    
    '''
    upload_to_aws('clouddata.csv', 'adcnml', 'clouddata.csv')

            
    

but = tk.Button(window, text = 'Add to cloud', command = addtocloud, bg = 'red', fg = 'white', width = 20, height =1)

but.place(x = 600, y = 650)


window.mainloop()