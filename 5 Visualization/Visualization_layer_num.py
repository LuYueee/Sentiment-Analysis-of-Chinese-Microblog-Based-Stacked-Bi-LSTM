# -*- coding: utf-8 -*-
"""
Visualization layer num: 
    1. Load data from 'Data' folder
    2. Visualize loss and accuracy with different layer number

Created on Sat Apr 28 12:25:40 2018

@author: Yue, Lu
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns



#读取数据
path=os.path.abspath('../Data')

#(1) 3 layers
name='stackedBLSTM_3layer.txt'
file=open(path+'\\'+name,'r')
line=file.readlines()

layer3_train_acc_set=[]
layer3_test_acc_set=[]
layer3_loss_set=[]

for i in range(1200):
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    layer3_train_acc_set.append(train_test_loss[0])
    layer3_test_acc_set.append(train_test_loss[1])
    layer3_loss_set.append(train_test_loss[2])
    

#(2) 2 layers
name='stackedBLSTM.txt'
file=open(path+'\\'+name,'r')
line=file.readlines()

layer2_train_acc_set=[]
layer2_test_acc_set=[]
layer2_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    layer2_train_acc_set.append(train_test_loss[0])
    layer2_test_acc_set.append(train_test_loss[1])
    layer2_loss_set.append(train_test_loss[2])
    

    



#模型可视化
#(1) Loss
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(layer2_loss_set,sns.xkcd_rgb['green'],label='2 layers')
plt.plot(layer3_loss_set,sns.xkcd_rgb['blue'],label='3 layers')

plt.xlabel('Number of Epochs')
plt.ylabel('Average Loss')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='upper right', fontsize=12,frameon=True,shadow=True)
plt.show()

#(2) Training Accuracy
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(layer2_train_acc_set,sns.xkcd_rgb['green'],label='2 layers')
plt.plot(layer3_train_acc_set,sns.xkcd_rgb['blue'],label='3 layers')
plt.xlabel('Number of Epochs')
plt.ylabel('Training Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.show()


#(3) Validation Accuracy
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(layer2_test_acc_set,sns.xkcd_rgb['green'],label='2 layers')
plt.plot(layer3_test_acc_set,sns.xkcd_rgb['blue'],label='3 layers')
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.show()



