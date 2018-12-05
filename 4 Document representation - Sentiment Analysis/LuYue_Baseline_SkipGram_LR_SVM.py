# -*- coding: utf-8 -*-

"""
Baseline Skip-Gram + LR & SVM: 
    Train and test baseline Skip-Gram+LR & Skip-Gram+SVM
    (1) Load Word2Vec Model
    (2) Segment and clean labeled comments
    (3) Represent a document(comment) by averaging all word vectors' value in a comment
    (4) Split train & test data
    (5) Train and test machine learning model (LR and SVM)
    
Created on Mon Apr  2 12:55:12 2018

@author: Yue, Lu
"""

import os
import numpy as np
import re
import jieba
from  sklearn.svm import SVC #support vector classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def cleanLabeledComment(labeled_comment_path,labeled_comment_file):
    '''
    Function:
        Segment and clean labeled comments
        
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
    path=os.path.abspath('..\Corpus')
    name='stopwords.txt'
    with open(path+'\\'+name,encoding='utf-8') as f:
        line=f.readline()
        while line:
            #while line is not null
            stopWords.append(line[:-1])
                #[:-1] reserve \n of every line in stop words
            line=f.readline()           
            #remove repeat stop words
        stopWords=set(stopWords)

    print('Load {n} Chinese stop words'.format(n=len(stopWords)))
    
    # Step 2 remove punctuation and Chinese stop words and do the segmentation using Jieba
    raw_word_list   = []
    sentence_list   = []
    sentiment_list  = []
    

    label_comment=open(labeled_comment_path+'\\'+labeled_comment_file,encoding='utf-8')
    
    for line in label_comment:
        while '\n' in line:
            line = line.replace('\n','')
    
                
        data_label=line.split('\t',1)
        # transfer utf-8 code
        if data_label[0]=='\ufeff1':
            data_label[0]='1'
        # remove all punctuation and number in comment
        # \d number
        # \w English char and number
        # \u4e00-\u9fa5 Chinese char
        if re.findall(u'[^\u4e00-\u9fa5\da-zA-Z]+',data_label[1]):  
            tmp=re.findall(u'[^\u4e00-\u9fa5\da-zA-Z]+',data_label[1])
            for i in tmp:
                data_label[1]=data_label[1].replace(i,' ')  
        #print(data_label)       
        label=data_label[0]
        data=data_label[1]
        # only load labeled comments
        if label not in ['-1','0','1']:
            #print(data_label)
            continue
        
        if len(data)>=1: 
        # if current line is not null
            # do the segmentation using Jieba
            raw_words = list(jieba.cut(data,cut_all=False))
            dealed_words = []
            for word in raw_words:                 
                if word not in stopWords and word not in ['www','com','http']:
                    raw_word_list.append(word)
                    #raw_word_list: all words in file [word1,word2,....,wordn]
                    dealed_words.append(word)
                    #dealed_words: words in current line
            # remove null list after clean text
            if(len(dealed_words)>=1 and label !='0'):
                sentence_list.append(dealed_words)
                sentiment_list.append(label)
    return sentiment_list,sentence_list,raw_word_list
    
    
def commentToVec(comment,model):
    '''
    Function:
        (1) Get vector values of each word in the current comment
        (2) Get current comment vector by averaging all words' the vector values
        
    Parameters:
        comment: single comment with a list of words [word1,word2,word3,...,wordn]
        model: word2vec model

    Return:
        current comment vector
    '''
    word_vec=np.zeros((1,num_dimension))
    for word in comment:
        if word in model:
            word_vec+=np.array([model[word]])


    # 评论单词每一维度的值求平均值返回，维度和词向量维度一致
    return word_vec.mean(axis=0)

    
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
    
if __name__=='__main__':
    
    # (1) Load Word2Vec Model
    num_dimension = 100
    model_name='Skip-Gram_{}dimension.model'.format(num_dimension)
    curr_path=os.path.abspath('..')
    model_filename='Model'       
    model = loadModel(curr_path,model_filename,model_name)
    # if you don't plan to train the model any further, calling
    # init_sims make the model much more memory-efficient
    model.init_sims(replace=True)

    
    # (2) Segment and clean labeled comments
    labeled_comment_path=os.path.abspath('..\Corpus')
    labeled_comment_file='Labeled Comments.txt'    
    sentiment_list,sentence_list,raw_word_list=cleanLabeledComment(labeled_comment_path,labeled_comment_file)


    # (3) Get comment vectors
    comment_vec=[]
    for sent in sentence_list:
        # get current comment vector
        comment_vec.append(commentToVec(sent,model))
    comment_vec=np.array(comment_vec)
    train_data_features=comment_vec
    
    # (4) 分割训练、测试数据 
    data_train, data_test, label_train, label_test = train_test_split(
                                                train_data_features,sentiment_list,test_size=0.25,random_state = 3)

    ####################################################################################################################
    #                                                                           ↑ test_size 和 random_state 可以调节   #
    ###################################################################################################################            
    
    # (5) 训练和测试模型     
    
    # 1. CBOW+LR           
    LR_model=LogisticRegression()
    #训练
    LR_model = LR_model.fit(data_train,label_train)
    #测试
    print('LR + Skip-Gram Accuracy:',LR_model.score(data_test,label_test))
    
        
         
    # 2. CBOW+SVM        
    #C代表决策便捷的容忍程度
    SVM_model=SVC()
    #训练
    SVM_model = SVM_model.fit(data_train,label_train)
    #测试
    print('SVM + Skip-Gram Accuracy:',SVM_model.score(data_test,label_test))
