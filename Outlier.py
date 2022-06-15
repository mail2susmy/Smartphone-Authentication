import pandas as pd
# import data
data = pd.read_csv('D://PhD KTU/PhD/Keystroke Python/cleaned.csv')

from scipy.stats import iqr

import numpy as np
print(len(data))
Q1 = np.quantile(data.iloc[0:50,16],0.25)
Q3 = np.quantile(data.iloc[0:50:,16],0.75)
IQR = Q3 - Q1
#print(IQR)
print(Q1)
print(Q3)

def find_and_replace(feat_num):
    global data
    cur_label = data['Label'][0]
    
	#start_index = 0           
    Q1 = np.quantile(data.iloc[0:50,feat_num],0.25)
    Q3 = np.quantile(data.iloc[0:50,feat_num],0.75)
    
	for i in range( len(data)):
		print('checking index : ',i)
		if (cur_label != data['Label'][i]):
			# resetting the current label
			cur_label = data['Label'][i]
			# setting the starting index of current label
			start_index = i
            Q1 = np.quantile(data.iloc[start_index:start_index+50,feat_num],0.25)
            Q3 = np.quantile(data.iloc[start_index:start_index+50,feat_num],0.75)
            IQR=Q3-Q1
		# checking for NaN
		if (data[feature_number][i]).value()<= Q1-(1.5*IQR) or (data[feature_number][i]).value()>= Q3+(1.5*IQR):
			# if found then assign the average value of that feature corresponding
			# to that user       
			data[feature_number][i] = get_average(data[feature_number][start_index:start_index+50])
            
for i in range(16,88):
    feat_num=data.colums[i]
    find_and_replace(feat_num)
dat.to_csv('D://PhD KTU/PhD/Keystroke Python/cleaned.csv',index='false')
