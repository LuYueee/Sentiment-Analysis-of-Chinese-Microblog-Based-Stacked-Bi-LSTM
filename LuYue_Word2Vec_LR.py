from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
import os
import numpy as np
import re
import jieba
#保存路径时可能会用到
import os.path as path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
def cleanText():
    # step 1 read stop words
    stopWords=[]
    path=r'C:\Users\windows\Desktop\1801\数据分析\CBOW'
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
        #print(stopWords)

    #print('停用词读取完毕，共{n}个单词'.format(n=len(stopWords)))


    # step2 remove punctuation and Chinese stop words and do the segmentation using Jieba
    raw_word_list = []
    sentence_list = []
    comment_file='10KL.txt'
    #encoding='gb18030'
    with open(path+'\\'+comment_file,encoding='gb18030') as f:
        line = f.readline()
        while line:
            #remove \n in every line
            while '\n' in line:
                line = line.replace('\n','')

            # remove all punctuation and number in comment
            # \d number
            # \w English char and number
            # \u4e00-\u9fa5 Chinese char
            if re.findall(u'[^\u4e00-\u9fa5\da-zA-Z]+',line):  
                tmp=re.findall(u'[^\u4e00-\u9fa5\da-zA-Z]+',line)
                for i in tmp:
                    line=line.replace(i,' ')
            if len(line)>0: 
            # if current line is not null
                # do the segmentation using Jieba
                raw_words = list(jieba.cut(line,cut_all=False))
                dealed_words = []
                for word in raw_words:                 
                    if word not in stopWords and word not in ['www','com','http']:
                        print('Dealed word: ',word)
                        raw_word_list.append(word)
                        #raw_word_list: all words in file [word1,word2,....,wordn]
                        dealed_words.append(word)
                        #dealed_words: words in current line
                sentence_list.append(dealed_words)
            # Read the next line
            line = f.readline()
    return sentence_list
    #sentence_list = [[word1,word2,...],[word1,word2,..]]
    
def commentToVec(comment,model):
    word_vec=np.zeros((1,100))
    for word in comment:
        if word in model:
            #print(word)
            word_vec+=np.array([model[word]])
    #print(comment)
    return word_vec.mean(axis=0)

    
def fileToSentList(path,name):
    '''
    Function:
        Open a file with segmented words and get a list of sentence with a list of words
        
    Parameters:
        name:           name of file
        path:           absolute path of file
        
    Return:
         sentence_list:  a list of sentences [[word1,word2,...],[word1,word2,..]]
        raw_word_list:  a list of all words
    '''

    raw_word_list = []
    sentence_list = []

    with open(path+'\\'+name,'r',encoding='utf-8') as f:
        line = f.readline()
        while line:
            #remove \n in every line
            while '\n' in line:
                line = line.replace('\n','')
                #sentence_list.append(line)
                #'''
            if len(line)>0: 
            # if current line is not null
                raw_words = line.split()
                dealed_words = []
                for word in raw_words:                 
                    if word not in ['',' ']:
                        print(word)
                        raw_word_list.append(word)
                        #raw_word_list: all words in file [word1,word2,....,wordn]
                        dealed_words.append(word)
                        #dealed_words: words in current line
                sentence_list.append(dealed_words)
                #'''               
            # Read the next line
            line = f.readline()
            
    #sentence_list = [[word1,word2,...],[word1,word2,..]]
    return sentence_list,raw_word_list
if __name__=='__main__':

    path=r'C:\Users\windows\Desktop\1801\数据分析\CBOW'
    #name='Weibo_Comments_With_Segmentation.txt'
    name='Comment_with_segmentation.txt'
    
    #sentence_list = [[word1,word2,...],[word1,word2,..]]
    #sentence_list,_=fileToSentList(path,name)
    sentence_list=cleanText()
    #print(sentence_list)
    #print(_)

# Word2Vec参数
# 输入 sentence_list: [[word1,word2,...],[word1,word2,..]]
# sg:           0 CBOW ; 1 skip-gram
# size:         dimension of word vector 100-300
# window:       size of sliding window
# alpha:        learning rate
# seed:         initialize vetocr
# min_count:    minimum count of word
# max_vocab_size: slelct most common words and map to word vector (None when word num < 1,000,000)
# workers:      thread num
# hs:           1 hierarchica softmax 0 negative sampling (default)
# negative:      num of negative sample
# iter :         num of iter (default 5)
        
num_dimension    = 300   # Word vector dimensionality
min_word_count  = 10    # Minimum word count
num_workers     = 4     # Number of threads to run in parallel
window_len      = 10     # Context window size
model_name='CBOW.model'
'''
model = Word2Vec(sentence_list,
                 size       =   num_dimension,
                 window     =   window_len,
                 alpha      =   1,
                 min_count  =   min_word_count,
                 workers    =   num_workers,
                 hs         =   0,
                 negative   =   200
                 )
'''
model = Word2Vec(sentence_list)

# if you don't plan to train the model any further, calling
# init_sims make the model much more memory-efficient
model.init_sims(replace=True)

#save the model into local file
#../ 上级目录
#model.save(os.path.join(r'C:\Users\windows\Desktop\1801\数据分析\CBOW','models',model_name))
print('---------------------------')
#test_words=['中国','骄傲','冠军','北京','羽生','裁判','犯规','棒子','爱豆','辣鸡']
test_words=['中国']
for word in test_words:
    print('similar word to ',word,':')
    print(model.wv.most_similar(word))#相似度[0~1] 越接近1说明越相关
    
#print(model.wv.doesnt_match(['冠军','金牌','犯规']))
#返回最不相关的词
print('---------------------------')


comment_vec=[]
for sent in sentence_list:
    comment_vec.append(commentToVec(sent,model))
comment_vec=np.array(comment_vec)
train_data_features=comment_vec

sentiment_list=[]
for i in range(0,1000):
    sentiment_list.append(0)
for i in range(1000,len(sentence_list)):
    sentiment_list.append(1)
    
data_train, data_test, label_train, label_test = train_test_split(
                                    train_data_features,sentiment_list,test_size=0.2)

#训练分类器
LR_model=LogisticRegression()
LR_model = LR_model.fit(data_train,label_train)
label_pred=LR_model.predict(data_test)

cnf_matrix=confusion_matrix(label_test,label_pred)
#http://www.dataguru.cn/thread-461647-1-1.html
#C（Accuracy）=(a+d)/(a+b+c+d)
#TP（recall or true positive rate） = d/(c+d)
print('Recall:',cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
print('Accuracy:',(cnf_matrix[1,1]+cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[0,1]+cnf_matrix[1,0]+cnf_matrix[1,1]))
