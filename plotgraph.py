# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 17:45:26 2022

@author: user
"""
import pandas as pd

import matplotlib.pyplot as  plt

import numpy as np

fig  = plt.figure(2)

df = pd.read_csv('acc.csv')

al = df['algorithm']

ma = df['accuracy']

plt.bar(al,ma,align='center')

plt.xlabel('Algorithm Used')

plt.ylabel('Accuracy')


plt.show()

fig.savefig('accuracy.jpg')


plt.close()