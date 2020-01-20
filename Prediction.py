import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier #change to DecisionTreeClassifier
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.metrics import precision_score as ps
from sklearn.metrics import recall_score as rs
from sklearn.metrics import f1_score as f1
from google.colab import drive
from sklearn.model_selection import RepeatedKFold

drive.mount('/content/drive')

loc = 'Path to Multi-Sensor based Human Activity Recognition using Deep Learning folder'
loc1 = 'Data/Modified/Skeleton Points - MPI/'
loc2 = 'Results/Skeleton Points - MPI/'

col1, col2 = [], ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
for i in range(14):
  col1.append('rgb_x_'+str(i))
  col1.append('rgb_y_'+str(i))
  col1.append('rgb_x_'+str(i))
col = col1 + col2

res_col = ['acc_tr', 'acc_ts', 'bas_tr', 'bas_ts', 'ps_tr', 'ps_ts', 'rs_tr', 'rs_ts', 'f1_tr', 'f1_ts']
max_dep = [i for i in range(2, 60)]
cri = ['gini', 'entropy']
spl = ['best', 'random']
n_est = [i*10 for i in range(1, 12)]

data = pd.read_csv(loc+loc1+'combined.csv')
rkf = RepeatedKFold(n_splits=10, n_repeats=10)

d = {}
for i in res_col:
  d[i] = []
d['criterion'] = []
c = 0
for tr_i, ts_i in rkf.split(data):
  train, test = data.iloc[tr_i], data.iloc[ts_i]
  tr_x, ts_x = train[col], test[col]
  tr_y, ts_y = train['act'], test['act']
  for i in cri:
    print(c, i)
    model = RandomForestClassifier(n_estimators=110, criterion=i)
    model.fit(tr_x, tr_y)
    tr_p = model.predict(tr_x)
    ts_p = model.predict(ts_x)
    d['acc_tr'].append(acc(tr_y, tr_p))
    d['acc_ts'].append(acc(ts_y, ts_p))
    d['bas_tr'].append(bas(tr_y, tr_p))
    d['bas_ts'].append(bas(ts_y, ts_p))
    d['ps_tr'].append(ps(tr_y, tr_p, average='macro', labels=np.unique(tr_p)))
    d['ps_ts'].append(ps(ts_y, ts_p, average='macro', labels=np.unique(ts_p)))
    d['rs_tr'].append(rs(tr_y, tr_p, average='macro', labels=np.unique(tr_p)))
    d['rs_ts'].append(rs(ts_y, ts_p, average='macro', labels=np.unique(ts_p)))
    d['f1_tr'].append(f1(tr_y, tr_p, average='macro', labels=np.unique(tr_p)))
    d['f1_ts'].append(f1(ts_y, ts_p, average='macro', labels=np.unique(ts_p)))
    d['criterion'].append(i)
  c += 1

df = pd.DataFrame(d, columns=['criterion']+res_col)
dic = {}
for i in res_col:
  dic[i] = []
for i in cri:
  df1 = df[df['criterion'] == i]
  for j in res_col:
    dic[j].append(np.mean(df1[j]))
dic['criterion'] = cri
df1 = pd.DataFrame(dic, columns=['criterion']+res_col)
df1.to_csv(loc+loc2+'rfc_cri.csv', index=False)