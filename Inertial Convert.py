import pandas as pd
from google.colab import drive
import numpy as np

drive.mount('/content/drive')

loc = 'Path to Multi-Sensor based Human Activity Recognition using Deep Learning folder'
loc1 = 'Data/Modified/Skeleton Points - Ori/'
loc2 = 'Data/Modified/Inertial/'

files = []
for i in range(1, 28):
  for j in range(1, 9):
    for k in range(1, 5):
      files.append('a'+str(i)+'_s'+str(j)+'_t'+str(k))

col1 = []
for i in range(18): #change to 14 if MPII model
  col1.append('rgb_x_'+str(i))
  col1.append('rgb_y_'+str(i))
  col1.append('rgb_z_'+str(i))

col2 = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
for i in files:
  try:
    data1 = pd.read_csv(loc+loc1+i+'.csv')
    data2 = pd.read_csv(loc+loc2+i+'.csv')
  except:
    print('File not found')
    continue
  d1 = {}
  d2 = {}
  for j in col2:
    print(i, j)
    l = list(data2[j])
    l_n = []
    for k in range(0, len(l)-2, 3):
      l_n.append(np.mean([l[k], l[k+1], l[k+2]]))
    print(len(data1))
    print(len(l_n))
    if len(data1) < len(l_n):
      d1[j] = l_n[:len(data1)]
    else:
      d1[j] = l_n
  for j in col1:
    l = list(data1[j])
    d1[j] = l[:len(d1['acc_x'])]
  d1 = pd.DataFrame(d1, columns=col1+col2)
  d1.to_csv(loc+loc1+i+'.csv', index=False)