"""
CBOW: 
    1. Read segmented comments in 'Segmented Comments.txt'
    2. Train a word embedding model using Gensim
    3. Store the model in 'Model\CBOW_()dimension.model'

Created on Mon Mar  5 12:58:28 2018

@author: Yue, Lu
"""

import os
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
import numpy as np

  
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

            if len(line)>0: 
            # if current line is not null
                raw_words = line.split()
                dealed_words = []
                for word in raw_words:                 
                    if word not in ['',' ']:
                        raw_word_list.append(word)
                        # raw_word_list: all words in file [word1,word2,....,wordn]
                        dealed_words.append(word)
                        # dealed_words: words in current line
                sentence_list.append(dealed_words)
            # Read the next line
            line = f.readline()
            
    #sentence_list = [[word1,word2,...],[word1,word2,..]]
    return sentence_list,raw_word_list
    
    
if __name__=='__main__':

    # (1) Get input of model
    path=os.path.abspath('..\Corpus')
    name='Segmented Comments.txt'
    #name='Segmented Comments.txt'
    
    #sentence_list = [[word1,word2,...],[word1,word2,..]]
    sentence_list,_=fileToSentList(path,name)
    #print(sentence_list)


    # (2) Train Word2Vec Model
    # Word2Vec参数
    # 输入 sentence_list: [[word1,word2,...],[word1,word2,..]]
    # sg:           0 CBOW ; 1 skip-gram
    # size:         dimension of word vector 词向量维度，Gensim默认为100
    # window:       size of sliding window
    # alpha:        learning rate
    # seed:         initialize vetocr
    # min_count:    minimum count of word
    # max_vocab_size: slelct most common words and map to word vector (None when word num < 1,000,000)
    # workers:      thread num
    # hs:           1 hierarchica softmax 0 negative sampling (default)
    # negative:     num of negative sample
    # iter :        num of iter (default 5)
            
    ##########################################################           
    num_dimension   = 100   # Word vector dimensionality    # 
    ##########################################################
    min_word_count  = 10    # Minimum word count
    num_workers     = 4     # Number of threads to run in parallel
    window_len      = 10    # Context window size

    
    '''
    e.g Set the parameter
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
    # Train the model with inputs and parameters
    model = Word2Vec(sentence_list,
                     size       =   num_dimension)
    
    # if you don't plan to train the model any further, calling
    # init_sims make the model much more memory-efficient
    model.init_sims(replace=True)
    
    
    # (3) Save Model
    #指定保存的模型名称
    model_name='CBOW_{}dimension.model'.format(num_dimension)
    model_path=path=os.path.abspath('..\Model')
    model.save(os.path.join(model_path,model_name))
    
    
    # (4) Test Model
    test_words=['中国','骄傲','冠军','北京','裁判','犯规','棒子','爱豆','辣鸡']
    #返回最相关的词  
    for word in test_words:
        print('similar word to ',word,':')
        print(model.wv.most_similar(word))
        #相似度[0~1] 越接近1说明越相关
      
    #返回最不相关的词  
    print(model.wv.doesnt_match(['冠军','金牌','犯规']))



