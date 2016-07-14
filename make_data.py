import numpy as np
import csv

data=[]
with open('origin_data/train.csv','rb') as f:
	series=csv.reader(f,skipinitialspace=True,delimiter='\t')
#set the input dim=1
	for row in series:
		data.append(row)
	data=np.array(data)
data=data[:,1:2]
#transpose column =>row
data=np.transpose(data)
#calculate data length
SERIES_LENGTH=np.size(data)
#set the window length
WINDOW_LENGTH=20
#set the prediction length
PREDICTION_LENGTH=5
SLICE_LENGTH=WINDOW_LENGTH+PREDICTION_LENGTH
#set the iteration
NEPOCH=2000
new_data=data[:,0:SLICE_LENGTH]
for epoch in range(NEPOCH):
	ran=np.random.randint(0,SERIES_LENGTH-SLICE_LENGTH)
	row=data[:,ran:ran+SLICE_LENGTH]
	new_data=np.concatenate((new_data,row))

np.save('data/train.npy',new_data)




