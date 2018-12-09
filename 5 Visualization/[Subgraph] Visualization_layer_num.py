# -*- coding: utf-8 -*-
"""
[Subgraph] Visualization layer num: 
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
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #print(train_test_loss[2])
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
    #print(train_test_loss[2])
    layer2_train_acc_set.append(train_test_loss[0])
    layer2_test_acc_set.append(train_test_loss[1])
    layer2_loss_set.append(train_test_loss[2])
    



#模型可视化

sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":1.5})

fig=plt.figure(figsize=(15,3))
plt1=fig.add_subplot(1,3,1)
plt2=fig.add_subplot(1,3,2)
plt3=fig.add_subplot(1,3,3)

#(1) Loss
#subplot1
plt1.plot(layer2_loss_set,sns.xkcd_rgb['green'],label='2 layers')
plt1.plot(layer3_loss_set,sns.xkcd_rgb['blue'],label='3 layers')


plt1.set_xlabel('Number of Epochs')
plt1.set_ylabel('Average Loss')



#(2) Training Accuracy
#subplot2
plt2.plot(layer2_train_acc_set,sns.xkcd_rgb['green'],label='2 layers')
plt2.plot(layer3_train_acc_set,sns.xkcd_rgb['blue'],label='3 layers')


plt2.set_xlabel('Number of Epochs')
plt2.set_ylabel('Training Accuracy')



#(3) Validation Accuracy
#subplot3
plt3.plot(layer2_test_acc_set,sns.xkcd_rgb['green'],label='2 layers')
plt3.plot(layer3_test_acc_set,sns.xkcd_rgb['blue'],label='3 layers')


plt3.set_xlabel('Number of Epochs')
plt3.set_ylabel('Validation Accuracy')




#图例设置
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.show()
