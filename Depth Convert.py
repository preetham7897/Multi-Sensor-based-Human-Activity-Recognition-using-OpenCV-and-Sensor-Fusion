import cv2
import pandas as pd
from google.colab import drive
import numpy as np
from scipy.io import loadmat
import pickle
from PIL import Image

drive.mount('/content/drive')

loc = 'Path to Multi-Sensor based Human Activity Recognition using Deep Learning folder'
ori_loc = 'Data/Original/Depth/'
new_loc1 = 'Data/Modified/Skeleton Points - MPI/'

files = []
for i in range(1, 28):
  for j in range(1, 9):
    for k in range(1, 5):
      files.append('a'+str(i)+'_s'+str(j)+'_t'+str(k))

col = []
for i in range(14): #change to 18 if COCO model
  col.append('rgb_x_'+str(i))
  col.append('rgb_y_'+str(i))

for l in files:
  try:
    m = loadmat(loc+ori_loc+l+'_depth.mat')
    d = pd.read_csv(loc+new_loc1+l+'.csv')
  except:
    print('File not found')
    continue
  print(len(d), l)
  data = m['d_depth']
  for k in range(len(data[0][0])):
    image = []
    for i in range(240):
      img = []
      for j in range(320):
        img.append(data[i][j][k])
      image.append(np.array(img))
    img_np = np.array(image)
    img_i = Image.fromarray(img_np)
    img_r = img_i.resize([640, 480])
    img_np = np.array(img_r)
    for m in range(14):
      n = []
      for o, p in zip(list(d['rgb_x_'+str(m)]), list(d['rgb_y_'+str(m)])):
        n.append(img_np[p][o])
      for o in range(len(n)):
        if n[o] == 0:
          n[o] = int(np.mean(n))
      d['rgb_z_'+str(m)] = n
  d.to_csv(loc+new_loc1+l+'.csv', index=False)