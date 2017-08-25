# encoding: utf-8
import pandas as pd
import numpy as np

"""
    一个小思想，一个小trick
    思想：总共70中，A中有66种，B中有62种，如何在数值化后二者维度保持一致
    trick: [T,T,F,T,F,T,T,F]取反 -----> “-”，比用not还好使
"""
# col2 = pd.read_csv('./data/col2.csv', header=None)
# col2 = pd.Series(col2.iloc[:, 0])
# data_train = pd.read_csv('./data/train_25192.csv', header=None)
# data_test = pd.read_csv('./data/test_11850.csv', header=None)
# # print col2.unique()
# col2 = pd.Series(col2.unique())
# # print data_train.iloc[:, 2].unique()
# data_train = pd.Series(data_train.iloc[:, 2].unique())
# # print data_test.iloc[:, 2].unique()
# data_test = pd.Series(data_test.iloc[:, 2].unique())
# print col2[-col2.isin(data_train)]

data = pd.read_csv('./data/test_11850.csv', header=None)
data.drop([42], axis=1, inplace=True)
data['type'] = ''
for i in range(data.shape[0]):
    if data.iloc[i, -2] in {'normal'}:
        data.loc[i, 'type'] = 'Normal'
    if data.iloc[i, -2] in {'udpstorm', 'processtable', 'mailbomb', 'apache2', 'back', 'land', 'neptune', 'pod', 'smurf', 'teardrop'}:
        data.loc[i, 'type'] = 'DOS'
    if data.iloc[i, -2] in {'httptunnel', 'loadmodule', 'ps', 'sqlattack', 'xterm', 'buffer_overflow', 'perl', 'rootkit'}:
        data.loc[i, 'type'] = 'U2R'
    if data.iloc[i, -2] in {'xsnoop', 'xlock', 'worm', 'snmpguess', 'snmpgetattack', 'sendmail', 'named', 'ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster'}:
        data.loc[i, 'type'] = 'R2L'
    if data.iloc[i, -2] in {'saint', 'mscan', 'ipsweep', 'loadmodule', 'nmap', 'portsweep', 'satan'}:
        data.loc[i, 'type'] = 'PROBING'
print data.type.value_counts()
data = data.join(pd.get_dummies(data.iloc[:, 1]))
data = data.join(pd.get_dummies(data.iloc[:, 2]))
data = data.join(pd.get_dummies(data.iloc[:, 3]))
print data.shape
data.drop([1, 2, 3, 41], axis=1, inplace=True)
data['Type'] = data.type
data.drop(['type'], axis=1, inplace=True)
print data.head()
data.to_csv('./data/test_pred.csv', index=False)