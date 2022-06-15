# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 18:29:37 2020

@author: 91944
"""

import numpy as np
import pandas as pd
from scipy.stats import iqr
# import data
data = pd.read_csv('design.csv')
#Q1 = np.quantile(data.iloc[1:50,16],0.25)
#Q3 = np.quantile(data.iloc[1:50,16],0.75)

def get_average(feature):
	s=0
	c=0
	for ele in feature:
		ele = str(ele)
		if ele.lower() != 'nan':
			s += float(ele)
			c += 1

	return s/c

def find_and_replace(feat_num):
    global data
    cur_label=data['Label'][0]
    start_index=0
    Q1 = np.quantile(data.loc[1:50,feat_num],0.25)
    Q3 = np.quantile(data.loc[1:50,feat_num],0.75)
    IQR = Q3-Q1
    for i in range( len(data)):
        if (cur_label != data['Label'][i]):
            cur_label = data['Label'][i]
            start_index = i
            Q1 = np.quantile(data.loc[start_index:start_index+50,feat_num],0.25)
            Q3 = np.quantile(data.loc[start_index:start_index+50,feat_num],0.75)
            IQR=Q3-Q1
            #print('checking index : ',i)
            # resetting the current label
			# setting the starting index of current label
        if (data[feat_num][i])<= Q1-(1.5*IQR) or (data[feat_num][i])>= Q3+(1.5*IQR):
            print('old',data[feat_num][i],feat_num,i)
            data[feat_num][i] = get_average(data[feat_num][start_index:start_index+50])
            print('new', data[feat_num][i])
		# checking for NaN
		
			# if found then assign the average value of that feature corresponding
			# to that user       
			

for i in range(16,88):
    feat_num=data.columns[i]
    find_and_replace(feat_num)
data.to_csv('design_new.csv',index='false')