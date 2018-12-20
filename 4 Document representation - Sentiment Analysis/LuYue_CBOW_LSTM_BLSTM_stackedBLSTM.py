# -*- coding: utf-8 -*-
"""
Baseline CBOW + LSTM & Bi-LSTM & stacked Bi-LSTM: 
    Train and test baseline CBoW+LSTM & CBoW+Bi-LSTM & CBoW+stacked Bi-LSTM
    (1) Load Word2Vec Model
    (2) Load labeled comments and preprocess the comments
    (3) Define parameters of LSTM models
    (4) Construct LSTM models
    (5) Define loss function and optimizer
    (6) Calculate the accuracy
    (7) Define model saver
    (8) Train and save the model 
    (9) Load and test the model
    
Created on Mon Apr  16 13:17:12 2018

@author: Yue, Lu
"""

from tensorflow.contrib import rnn 
import tensorflow as tf
import os
import numpy as np
import re
import jieba
import jieba.analyse
from sklearn.model_selection import train_test_split

def loadModel(curr_path,model_filename,model_name):
    '''
    Function:
        Load word2vec model from specific path
        
    Parameters:
        curr_path: project path
        model_filename: folder name of models
        model_name: file name of word2vec model

    Return:
        loaded word2vec model
    '''
    model=np.load(os.path.join(curr_path,model_filename,model_name))
    return model

def cleanLabeledComment(labeled_comment_path,labeled_comment_file):
    '''
    Function:
        Load labeled comments and preprocess the comments
        (1) split labels and comments
        (2) segment Chinese word using jieba
        (3) remove punctuation and Chinese stop words
        (4) remove all unlabeled or neural comments
        
    Parameters:
        labeled_comment_file: name of .txt labeled comment file (for each line: [label comment])
        labeled_comment_path: absolute path of file

    Return:
        sentiment_list: a list of labels    [label1,label2,label3,...,labeln]
        sentence_list:  a list of comments  [[word1,word2,word3,...,wordn],[word1,word2,word3,...,wordn],...,[word1,word2,word3,...,wordn]]
        raw_word_list:  a list of all words [word1,word2,word3,...,wordm]
    '''
    
    # Step 1 read stop words
    stopWords=[]
    stopwords_path=os.path.abspath('..\Corpus')
    stopwords_name='stopwords.txt'
    with open(stopwords_path+'\\'+stopwords_name,encoding='utf-8') as f:
        line=f.readline()
        while line:
            #while line is not null
            stopWords.append(line[:-1])
                #[:-1] reserve \n of every line in stop words
            line=f.readline()           
            #remove repeat stop words
        stopWords=set(stopWords)
        #print(stopWords)
    
    print('Load {n} Chinese stop words'.format(n=len(stopWords)))
    
   
    raw_word_list   = []
    sentence_list   = []
    sentiment_list  = []
    
    label_comment=open(labeled_comment_path+'\\'+labeled_comment_file,encoding='utf-8')
    
    
    for line in label_comment:
    # Step 2 remove blank space
        while '\n' in line:
            line = line.replace('\n','')
        #print(line)
    
    # Step 3 split label and data and store into data_label [label,comment]          
        data_label=line.split('\t',1)
        #transfer utf-8 code
        if data_label[0]=='\ufeff1':
            data_label[0]='1'
            
    # Step 4 reserve all Chinese char, number and English char and remove all other char (去掉除中文、英文、数字以外的特殊符号)
        # \d number
        # \w English char and number
        # \u4e00-\u9fa5 Chinese char
        if re.findall(u'[^\u4e00-\u9fa5\da-zA-Z]+',data_label[1]):  
            tmp=re.findall(u'[^\u4e00-\u9fa5\da-zA-Z]+',data_label[1])
            for i in tmp:
                data_label[1]=data_label[1].replace(i,' ')  
        #分离label和data      
        label=data_label[0]
        data=data_label[1]
        # only load labeled comments
        if label not in ['-1','0','1']:
            continue
        
        if len(data)>=1: 
        # if current line is not null
    # Step 5 do the Chinese word segmentation using Jieba and remove Chinese stop words
            raw_words = list(jieba.cut(data,cut_all=False))
            dealed_words = []
            for word in raw_words:                 
                if word not in stopWords and word not in ['www','com','http']:
                    #print('Dealed word: ',word)
                    raw_word_list.append(word)
                    #raw_word_list: all words in file [word1,word2,....,wordn]
                    dealed_words.append(word)
                    #dealed_words: words in current line
                    
    # Step 6 remove null or neural comments after clean text
            if(len(dealed_words)>=1 and label !='0'):
                sentence_list.append(dealed_words)
                sentiment_list.append(int(label))
    return sentiment_list,sentence_list,raw_word_list
    
def sentToVec(sentence):
    '''
    Function:
        Transform a comment into a comment vector
        
    Parameters:
        sentence: a comment with normalized length  E.g ['同意', '自我', '集体', '高潮', 0, 0, 0]

    Return:
        sent_vec: a sentence vector after words are transformed into word vectors 
                    E.g[
                        word1[v1,v2,v3,...,vd]
                        word2[v1,v2,v3,...,vd]
                        ...
                        wordk[v1,v2,v3,...,vd]
                        ]
    '''

    sent_vec=np.zeros((1,numDimensions))
    #flag indicates if it's the 1st word in the sent    
    flag=0
    
    for word in sentence:
        word_vec=np.zeros((1,numDimensions))
        
        # current word can be found in word2vec model
        if word in model:
            if word!=0:               
                word_vec=np.array(model[word])
                # get word vector
                
                # current word is the 1st word in comment [ word1[v1,v2,v3,...,vd] ]
                if flag==0:
                     #拼接成句子矩阵
                    sent_vec+=word_vec
                    flag=1

                # current word is not the 1st word in comment [ word1[v1,v2,v3,...,vd]  word2[v1,v2,v3,...,vd] ...  wordn[v1,v2,v3,...,vd] ]
                else:
                    sent_vec=np.vstack((sent_vec,word_vec))                 


        # current word can not be found in word2vec model
        else:
            tmp=np.zeros((numDimensions,))
            word_vec=np.array(tmp)
            #   set zero vector
            
            # current word is the 1st word in comment [ word1[0,0,0,...,0] ]
            if flag==0:
                #拼接成句子矩阵
                sent_vec+=word_vec
                flag=1
            
            # current word is not the 1st word in comment [ word1[v1,v2,v3,...,vd]  word2[v1,v2,v3,...,vd] ...  wordn[0,0,0,...,0] ]
            else:
                sent_vec=np.vstack((sent_vec,word_vec))
         
    return sent_vec
    
def allInput(sentiment_list,sentence_list,raw_word_list):
    '''
    Function:
        Get all inputs of LSTM model
        (1) Calculate the average sentence length K
        (2) Normalize the length of sentence
        (3) Concat all comment vectors
        (4) Encode all labels using one-hot
        
    Parameters:
        sentiment_list: a list of labels    [label1,label2,label3,...,labeln]
        sentence_list:  a list of comments  [[word1,word2,word3,...,wordn],[word1,word2,word3,...,wordn],...,[word1,word2,word3,...,wordn]]
        raw_word_list:  a list of all words [word1,word2,word3,...,wordm]

    Return:
        all_sentiment_vec:
        [
            sentiment1[0,1]
            sentiment2[0,1]
            ...
        ]
        
        all_sentence_vec:
        [
            sentence1[
                        word1[v1,v2,v3,..,vd]
                        word2[v1,v2,v3,..,vd]
                        ...
                        wordk[v1,v2,v3,..,vd]
                    ]  
                    
            sentence2[
                        word1[v1,v2,v3,..,vd]
                        word2[v1,v2,v3,..,vd]
                        ...
                        wordk[v1,v2,v3,..,vd]
                    ] 
                ...
            
        ]

    '''
    # (1) Calculate the average sentence length K
    sentLength=len(raw_word_list)/len(sentence_list)+5


    # (2) Normalize the length of sentence:
    #       length of comment > average sentence length => extract key words using TF-IDF
    #       length of comment < average sentence length => 0-padding
    for i in range(len(sentence_list)):
        if len(sentence_list[i])>sentLength:
            # 1. 评论单词数>平均单词数 直接截取前K个单词，大于平均句长以后的单词不做处理
            #sentence_list[i]=jieba.analyse.extract_tags(str(sentence_list[i]),topK=int(sentLength+1))
            sentence_list[i]=sentence_list[i][:int(sentLength+1)]
            '''出现[12,100][11,100][12,100]问题时'''
            while(len(sentence_list[i])<sentLength+1):
            #while(len(sentence_list[i])<sentLength):
                sentence_list[i].append(0)


        else:
            # 2. 评论单词数<平均单词数 长度不足的部分使用'0'填充
            '''出现[12,100][11,100][12,100]问题时'''
            while(len(sentence_list[i])<sentLength+1):
            #while(len(sentence_list[i])<sentLength):
                sentence_list[i].append(0)

                
    
    # (3) Concat all comment vectors
    flag=0
    all_sentence_vec=np.zeros((round(sentLength+1),numDimensions))
    for sent in sentence_list:
        # if current comment is the 1st comment in the list
        if flag==0:
            #(1)首先取出评论中每个单词的D维向量空间值，拼接为K*D的句子矩阵
            sent_vec=sentToVec(sent)
            #(2)将所有句子向量拼接，拼接为commentNum*K*D的矩阵
            all_sentence_vec+=sent_vec
            flag=1
            
        # if current comment is not the 1st comment in the list
        else:
            sent_vec=sentToVec(sent)
            all_sentence_vec=np.vstack((all_sentence_vec,sent_vec))
    
    all_sentence_vec=all_sentence_vec.reshape(-1,round(sentLength+1),numDimensions)
    #label and input data
    
    
    # (4) Encode all labels using one-hot
    # 将所有one-hot编码后的评论拼接为commentNum*2的矩阵
    flag=0
    # 2 classes
    all_sentiment_vec=np.zeros((2,))
    for sent in sentiment_list:
        if sent==1:
        # Positive -> [0,1]
            sent_vec=np.array([0,1])
            # if current label is the 1st label in the list
            if flag==0:
                all_sentiment_vec+=sent_vec
                flag=1
            else:
                all_sentiment_vec=np.vstack((all_sentiment_vec,sent_vec))
                
                
        if sent==-1:
        # Negative -> [1,0]
            sent_vec=np.array([1,0])
            # if current label is the 1st label in the list
            if flag==0:
                all_sentiment_vec+=sent_vec
                flag=1
            else:
                all_sentiment_vec=np.vstack((all_sentiment_vec,sent_vec))
    
    #print(all_sentiment_vec.shape)
    return all_sentiment_vec,all_sentence_vec
   

def batchInput(train_data,train_label,batch_size):
    '''
    Function:
        Segment the training set according to the batch size
        
    Parameters:
        train_data: training comment vectors
        train_label: training comment labels
        batch_size: size of network input

    Return:
        batch_data: segmented training data [[batch1],[batch2],[batch3],...]
        batch_label: segmented training label [[batch1],[batch2],[batch3],...]
        batch_num: number of batch for one epoch
    '''

    train_num=train_data.shape[0]
    p=0
    batch_num=0
    batch_data=[]
    batch_label=[]
    
    #训练集数量可以被 batch size 整除
    if(train_num % batch_size!=0):
        batch_num=int(train_num/batch_size)+1
        for i in range(batch_num):
            if i==(batch_num-1):
                batch_size1=train_num-p
                batch_range=p+batch_size1
                tmp_data=train_data[p:batch_range,:,:]
                tmp_label=train_label[p:batch_range,:]
                #print(tmp_data.shape)
                #print(tmp_label.shape)
                batch_data.append(tmp_data)
                batch_label.append(tmp_label)
                p=p+batch_size1
            else:
                batch_range=p+batch_size
                tmp_data=train_data[p:batch_range,:,:]
                tmp_label=train_label[p:batch_range,:]
                #print(tmp_data.shape)
                #print(tmp_label.shape)
                batch_data.append(tmp_data)
                batch_label.append(tmp_label)
                p=p+batch_size
                          
            
    else:
        batch_num=int(train_num/batch_size)
        for i in range(batch_num):
            batch_range=p+batch_size
            tmp_data=train_data[p:batch_range,:,:]
            tmp_label=train_label[p:batch_range,:]
            #print(tmp_data.shape)
            #print(tmp_label.shape)
            batch_data.append(tmp_data)
            batch_label.append(tmp_label)
            p=p+batch_size
    
    return batch_data,batch_label,batch_num


def LSTM(lstmUnits,keepratio,data,numClasses):
    '''
    Function:
        Construct a LSTM model
        
    Parameters:
        lstmUnits:  hidden size for each layer
        keepratio:  keep ratio of drop out
        data:       Input data of model
        numClasses: Num of different classes

    Return:
        prediction: predicted value
        weight:     weight of last layer
    '''
    #LSTM
    # hidden cell
    lstmCell=tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    # dropout
    lstmCell=tf.contrib.rnn.DropoutWrapper(cell=lstmCell,output_keep_prob=keepratio)
    # last hidden vector
    value,_=tf.nn.dynamic_rnn(lstmCell,data,dtype=tf.float32)

    weight=tf.Variable(tf.truncated_normal([lstmUnits,numClasses]))
    bias=tf.Variable(tf.constant(0.1,shape=[numClasses]))
    value=tf.transpose(value,[1,0,2])
    #tf.gather(): get the value of the last cell
    last=tf.gather(value,int(value.get_shape()[0])-1)
    prediction=(tf.matmul(last,weight)+bias)
    return prediction,weight

def BiLSTM(lstmUnits,keepratio,data,numClasses):
    '''
    Function:
        Construct a Bi-LSTM model
        
    Parameters:
        lstmUnits:  hidden size for each layer
        keepratio:  keep ratio of drop out
        data:       Input data of model
        numClasses: Num of different classes

    Return:
        prediction: predicted value
        weight:     weight of last layer
    '''
    # Bi-LSTM
    # Forward Layer
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(lstmUnits,state_is_tuple=True)
    lstm_fw_cell=tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell,output_keep_prob=keepratio)
    # Backward Layer
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(lstmUnits,state_is_tuple=True)
    lstm_bw_cell=tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell,output_keep_prob=keepratio)
    
    # input of bi-rnn
    x = tf.transpose(data, [1, 0, 2])
    x = tf.reshape(x, [-1, numDimensions])
    x = tf.split(x, sentLength)
    value,_,_ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,inputs=x,dtype=tf.float32)


    weight=tf.Variable(tf.truncated_normal([2*lstmUnits,numClasses]))
    bias=tf.Variable(tf.constant(0.1,shape=[numClasses]))
    # get the value of the last cell
    last=value[-1]
    prediction=(tf.matmul(last,weight)+bias)
    return prediction,weight
    
def lstm_cell(hidden_size,keep_prob):  
    '''
    Function:
        Construct a hidden LSTM cell
        
    Parameters:
        hidden_size: hidden size for each layer
        keep_prob:   keep ratio of drop out

    Return:
        a hidden LSTM cell
    '''
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)  
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)  
        
def stacked_BiLSTM(lstmUnits,keepratio,data,numClasses,num_layers):
    '''
    Function:
        Construct a stacked Bi-LSTM model
        
    Parameters:
        lstmUnits:  hidden size for each layer
        keepratio:  keep ratio of drop out
        data:       Input data of model
        numClasses: Num of different classes
        num_layers: num of forward and backward layer

    Return:
        prediction: predicted value
        weight:     weight of last layer
    '''
    # 2-layer Bi-LSTM
    # Forward Layer (2 layers)
    mLstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstmUnits,keepratio) for _ in range(num_layers)], state_is_tuple=True)
    # Backward Layer (2 layers)  
    mLstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstmUnits,keepratio) for _ in range(num_layers)], state_is_tuple=True)
    
    # input of bi-rnn    
    x = tf.transpose(data, [1, 0, 2])
    x = tf.reshape(x, [-1, numDimensions])
    x = tf.split(x, sentLength)
    
    value,_,_ = tf.contrib.rnn.static_bidirectional_rnn(mLstm_fw_cell,mLstm_bw_cell,inputs=x,dtype=tf.float32)

     
    weight=tf.Variable(tf.truncated_normal([2*lstmUnits,numClasses]))
    bias=tf.Variable(tf.constant(0.1,shape=[numClasses]))
    # get the value of the last cell
    last=value[-1]
    prediction=(tf.matmul(last,weight)+bias)
    return prediction,weight
    
if __name__=='__main__':
    
    # (1) Load Word2Vec Model
    ###########################################################################
    # 词向量维度                                                               #
    numDimensions = 100                                                       # Wordvec dimension #
    ###########################################################################
    model_name='CBOW_{}dimension.model'.format(numDimensions)    
   
    curr_path=os.path.abspath('..')
    model_filename='Model'
      
    model = loadModel(curr_path,model_filename,model_name)
        # if you don't plan to train the model any further, calling
        # init_sims make the model much more memory-efficient
    model.init_sims(replace=True)
    

    
    # (2) Load labeled comments and preprocess the comments
    labeled_comment_path=os.path.abspath('..\Corpus')
    labeled_comment_file='Labeled Comments.txt'    
    sentiment_list,sentence_list,raw_word_list=cleanLabeledComment(labeled_comment_path,labeled_comment_file)
    


    tf.reset_default_graph()
    

    # (3) Define parameters of LSTM models
    ###########################################################################
    # Size of network inputs                                                  #
    sentLength      = round(len(raw_word_list)/len(sentence_list)+1)+5        # Sentence Length
    # Num of forward and backward layer of Bi-LSTM250                         #
    num_layers      = 2                                                       # Layer Num
    # Hidden size for each layers                                             #
    lstmUnits       = 64                                                      # LSTM Cell
    ###########################################################################
    # Num of different classes (sentiment_list不重复元素数)
    numClasses      = len(set(sentiment_list))
    # Num of epoch
    training_epochs = 1200   
    # Keep ratio of drop out
    keepratio = tf.placeholder(tf.float32)  
    # True label of inputs (batchSize,numClasses)
    labels=tf.placeholder(tf.float32,[None,numClasses])
    # Inputs of network (batchSize,sentLength,numDimensions)
    data=tf.placeholder(tf.float32,[None,sentLength,numDimensions])

    batch_size      = 300
    #每隔display_step 次迭代打印一次
    display_step    = 1
    #每隔save_step次迭代保存一次
    save_step       = 150

    # (4) Construct LSTM models
    ############################################################################
    #prediction,W=LSTM(lstmUnits,keepratio,data,numClasses)                    #
    #prediction,W=BiLSTM(lstmUnits,keepratio,data,numClasses)                  #
    prediction,W=stacked_BiLSTM(lstmUnits,keepratio,data,numClasses,num_layers)# 选择构建其中一个LSTM模型
    ############################################################################


    
    # (5) Define loss function and optimizer
    # Regularization term
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)    
    regularizer = tf.contrib.layers.l2_regularizer(scale=5.0/50000)    
    reg_term = tf.contrib.layers.apply_regularization(regularizer)
    
    #损失函数： softmax_cross_entropy_with_logits 交叉熵
    #预测值logits和实际label值比较，用reduce_mean算出来平均loss,最后返回给cost
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=labels)+reg_term)

    #定义优化：Adam自适应调整学习率，随迭代次数增加而减小
    optm = tf.train.AdamOptimizer().minimize(loss) 

    # (6) Calculate the accuracy
    #比较预测结果和实际结果，分类正确为TRUE
    corr = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)) 
    #布尔值转化成float，计算准确度   
    accr = tf.reduce_mean(tf.cast(corr, "float"))
    init = tf.global_variables_initializer() 
    

    # (7) Define model saver
    #max_to_keep=3表示最终只保留最后三组模型
    saver=tf.train.Saver(max_to_keep=3)
   
   
    sess = tf.Session()
    sess.run(init)
    all_labels,all_data=allInput(sentiment_list,sentence_list,raw_word_list)
    print(all_data.shape)
    print(all_labels.shape)


    
    #do_train=1 训练模型
    #do_train=0 测试模型
    do_train = 1

    # (8) Train and save the model
    if do_train==1: 
        # 可视化数据
        avg_cost_set=[] 
        epoch_set=[]
        
        train_accr_set=[]
        test_accr_set=[]       
        avg_accr_set=[]
        display_epoch_set=[]  
        
        # 分割训练、测试数据
        train_valid_data, test_data, train_valid_label, test_label = train_test_split(
                                    all_data,all_labels,test_size=0.25,random_state=3)                                    
##################################################################################################
#                                                       ↑ test_size 和 random_state 可以调节     #
#################################################################################################                                   
                                    
        for epoch in range(training_epochs):
            
            # 分割训练集、验证集
            train_data, valid_data, train_label, valid_label = train_test_split(
                                    train_valid_data,train_valid_label,test_size=0.3)
            total_cost = 0.
            # 根据 batch size分割训练数据
            batch_data,batch_label,batch_num=batchInput(train_data,train_label,batch_size)
            
            # Train the model batch by batch
            for i in range(batch_num):
                train_batch_data=batch_data[i]
                train_batch_label=batch_label[i]
                cost = 0.             
                sess.run(optm, feed_dict={labels:train_batch_label,data:train_batch_data,keepratio:0.7})
                # Compute loss
                cost = sess.run(loss, feed_dict={labels:train_batch_label,data:train_batch_data,keepratio:0.7})
                total_cost=total_cost+cost
            avg_cost=total_cost/batch_num
            avg_cost_set.append(avg_cost)
            epoch_set.append(epoch)
    
            # Display logs per epoch step
            if epoch % display_step == 0: 
                #batch_labels,batch_data=allInput(sentiment_list,sentence_list,raw_word_list)
                print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
                train_acc = sess.run(accr, feed_dict={labels:train_label,data:train_data,keepratio:1.0})
  
                print (" Training accuracy: %.3f" % (train_acc))
                test_acc = sess.run(accr, feed_dict={labels:valid_label,data:valid_data, keepratio:1.0})
                #所有训练集准确率
                train_accr_set.append(train_acc)
                #所有验证集准确率
                test_accr_set.append(test_acc)
                avg_accr_set.append(sum(test_accr_set)/len(test_accr_set))
                display_epoch_set.append(epoch)
                
                print (" Test accuracy: %.3f" % (test_acc))
                print(' Avg test accr:',sum(test_accr_set)/len(test_accr_set))
                print(' Max Accr:',max(test_accr_set), 'for',len(test_accr_set),'epoches')
                print(' ')

                
            #Save the model
            if epoch % save_step ==0 and epoch!=0:
                model_name='StackedBLSTM'
                saver.save(sess,'../Model/'+model_name+str(epoch)+'.ckpt')
        
        print ("OPTIMIZATION FINISHED")
        
        # 将用于可视化的损失值、准确率数据写入磁盘
        datapath=os.path.abspath('../Data')
        dataname=model_name+'.txt'
        file=open(datapath+'\\'+dataname,'w')
        for i in range(len(train_accr_set)):
            file.write(str(train_accr_set[i])+'\t'+str(test_accr_set[i])+'\t'+str(avg_cost_set[i])+'\n')
        file.close()
        

    # (9) Load and test the model
    if do_train==0:
        
        #根据模型名称加载模型
        #model_name='LSTM250.ckpt'
        #saver.restore(sess,'../Model/'+model_name)
        
        #或加载最近一次保存的模型
        saver.restore(sess,tf.train.latest_checkpoint('../Model'))   
        
        #模型测试   
        # 1. Get test data set
        train_valid_data, test_data, train_valid_label, test_label = train_test_split(
                                    all_data,all_labels,test_size=0.25,random_state=3)
        # 2. Test the loaded model
        test_acc = sess.run(accr, feed_dict={labels:test_label,data:test_data, keepratio:1.})
        print (" Test accuracy: %.3f" % (test_acc))

        

        

        




