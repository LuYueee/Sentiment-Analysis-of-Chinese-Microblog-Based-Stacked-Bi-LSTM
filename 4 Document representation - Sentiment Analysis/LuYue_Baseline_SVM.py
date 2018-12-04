# -*- coding: utf-8 -*-
"""
Baseline SVM: 
    Train and test baseline BoW+SVM
    (1) Segment and clean labeled comments
    (2) Represent documents(comments) using CountVectorizer (based on word frequency)
    (3) Split train & test data
    (4) Train and test machine learning model (SVM)
    
Created on Sun Mar 18 09:46:31 2018

@author: Yue, Lu
"""
import os
import re
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from  sklearn.svm import SVC #support vector classifier

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
        #transfer utf-8 code
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
        label=data_label[0]
        data=data_label[1]
        # only load labeled comments
        if label not in ['-1','0','1']:
            continue
        
        if len(data)>=1: 
        # if current line is not null
            # do the segmentation using Jieba
            raw_words = list(jieba.cut(data,cut_all=False))
            dealed_words = []
            for word in raw_words:                 
                if word not in stopWords and word not in ['www','com','http']:
                    raw_word_list.append(word)
                    # raw_word_list: all words in file [word1,word2,....,wordn]
                    dealed_words.append(word)
                    # dealed_words: words in current line
            # remove null list after clean text
            if(len(dealed_words)>=1 and label !='0'):
                sentence_list.append(dealed_words)
                sentiment_list.append(int(label))
                
    return sentiment_list,sentence_list,raw_word_list

def countSentiment(sentiment_list):
    '''
    Function:
        Count and print # of positive and negative comments
        
    Parameters:
        sentiment_list: a list of labels    [label1,label2,label3,...,labeln]

    '''
    negative=0
    positive=0
    for s in range(len(sentiment_list)):
        if sentiment_list[s]==-1:
            negative+=1
        if sentiment_list[s]==1:
            positive+=1
    print('For',len(sentiment_list),', it has',negative,'negative comments and', positive,' positive comments') 
    

# (1) Segment and clean labeled comments
labeled_comment_path=os.path.abspath('..\Corpus')
labeled_comment_file='Labeled Comments.txt' 
sentiment_list,sent_list,_=cleanLabeledComment(labeled_comment_path,labeled_comment_file)

# 准备sklearn模型的句子输入  
# sentence_list=[segmented comment1,segmented comment2,segmented comment3,...,segmented commentn]
sentence_list=[]
for sent in sent_list:
    temp_list=''
    count=0
    for word in sent:
        temp_list=temp_list+' '+word
        count+=1
    sentence_list.append(temp_list)


# (2) 使用CountVectorizer基于计数的文档表征
v=CountVectorizer()
# 文本->基于词频的特征数据
train_data_features=v.fit_transform(sentence_list)
print(train_data_features.shape)


'''
sklearn 分成训练集测试集 
test_size 测试集比例 
random_state 随机种子
'''
# (3) 分割训练、测试数据 
data_train, data_test, label_train, label_test = train_test_split(
                                        train_data_features,sentiment_list,test_size=0.25,random_state = 3)
####################################################################################################################
#                                                                           ↑ test_size 和 random_state 可以调节   #
################################################################################################################### 
                                        
#countSentiment(label_test)

# (4) 训练和测试模型     
                               
# 1.定义LR模型
LR_model=LogisticRegression()
# 训练
LR_model = LR_model.fit(data_train,label_train)
# 测试
print('LR Accuracy:',LR_model.score(data_test,label_test))


# 2.定义NB模型
NB_model=MultinomialNB()
# 训练
NB_model = NB_model.fit(data_train,label_train)
# 测试                                  
print('NB Accuracy:',NB_model.score(data_test,label_test))
 
 
# 3.定义SVM模型（默认RBF内核）
# C代表决策便捷的容忍程度
SVM_model=SVC()
# 训练
SVM_model = SVM_model.fit(data_train,label_train)
# 测试
print('SVM Accuracy:',SVM_model.score(data_test,label_test))

                                     


