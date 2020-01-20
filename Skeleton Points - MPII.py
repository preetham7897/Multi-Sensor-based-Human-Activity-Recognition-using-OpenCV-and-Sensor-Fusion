import cv2
import pandas as pd
from google.colab import drive
import time

drive.mount('/content/drive')

loc = 'Path to Multi-Sensor based Human Activity Recognition using Deep Learning folder'
wf = 'Codes/Model Files/pose_iter_160000.caffemodel'
pf = 'Codes/Model Files/pose_deploy_linevec_faster_4_stages.prototxt'
ori_loc = 'Data/Original/RGB/'
new_loc = 'Data/Modified/Skeleton Points - MPI/'

net = cv2.dnn.readNetFromCaffe(loc+pf, loc+wf)
col_x, col_y = [], []
for i in range(14):
  col_x.append('rgb_x_'+str(i))
  col_y.append('rgb_y_'+str(i))
col = col_x + col_y
col.append('frame')
files = []
for i in range(1, 9):
  for j in range(1, 5):
    files.append('a27_s'+str(i)+'_t'+str(j))

for i in files:
  print(i)
  cap = cv2.VideoCapture(loc+ori_loc+i+'_color.avi')
  c = 0
  files = []
  d = {}
  for j in col:
    d[j] = []
  while True:
    r, f = cap.read()
    t1 = time.time()
    if not r:
      break
    inp = cv2.dnn.blobFromImage(f, 1/255, (640, 480), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    H = out.shape[2]
    W = out.shape[3]
    for j in range(14):
      probMap = out[0, j, :, :]
      minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
      x = (640 * point[0]) / W
      y = (480 * point[1]) / H
      d['rgb_x_'+str(j)].append(int(x))
      d['rgb_y_'+str(j)].append(int(y))
    c += 1
    t2 = time.time()
    print(i, c, t2-t1)
  d['frame'] = [j for j in range(c)]
  df = pd.DataFrame(d, columns=col)
  df.to_csv(loc+new_loc+i+'.csv', index=False)
  cap.release()

