import pandas as pd

# import data
data = pd.read_csv('data.csv')


# calculate average
def get_average(feature):
	s=0
	c=0
	for ele in feature:
		ele = str(ele)
		if ele.lower() != 'nan':
			s += float(ele)
			c += 1

	return s/c

def find_and_replace_nan(feature_number):
	print(feature_number)

	global data

	cur_label = data['Label'][0]
	start_index = 0

	for i in range(len(data)):
		print('checking index : ',i)
		if (cur_label != data['Label'][i]):
			# resetting the current label
			cur_label = data['Label'][i]
			# setting the starting index of current label
			start_index = i

		# checking for NaN
		if (str(data[feature_number][i]).lower() == 'nan'):
			# if found then assign the average value of that feature corresponding
			# to that user
			data[feature_number][i] = get_average(data[feature_number][start_index:start_index+50])


for i in range(1,88):
	feature_number = data.columns[i]
	find_and_replace_nan(feature_number)

data.to_csv('cleaned.csv',index=False)

