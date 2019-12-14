import pandas as pd
from google.colab import drive
import numpy as np

def minMaxNorm(l):
  l_n = []
  maxi = max(l)
  mini = min(l)
  if maxi - mini == 0:
    l_n = [0 for i in l]
  else:
    l_n = [((i-mini)/(maxi-mini)) for i in l]
  return l_n

drive.mount('/content/drive')

loc = 'Path to Multi-Sensor based Human Activity Recognition using Deep Learning folder'
ori_loc = 'Data/Modified/Skeleton Points - Ori/'

col1 = []
col2 = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
for i in range(18):
  col1.append('rgb_x_'+str(i))
  col1.append('rgb_y_'+str(i))
  col1.append('rgb_z_'+str(i))
col = col1 + col2

data = pd.DataFrame()
for i in range(1, 28):
  for j in range(1, 9):
    for k in range(1, 5):
      print(i, j, k)
      try:
        d = pd.read_csv(loc+ori_loc+'a'+str(i)+'_s'+str(j)+'_t'+str(k)+'.csv')
      except:
        print('File not found')
        continue
      d['act'] = [i-1 for l in range(len(d))]
      d['sub'] = [j for l in range(len(d))]
      data = data.append(d)

d = {}
for i in col:
  print(i)
  d[i] = minMaxNorm(list(data[i]))
d['sub'] = list(data['sub'])
d['act'] = list(data['act'])
df = pd.DataFrame(d, columns=col+['sub', 'act'])

df.to_csv(loc+ori_loc+'combined.csv', index=False)