"""
Chinese Words Segmentation : 
    Do the pretreatment of Weibo comments:
    1. remove  punctuation
    2. do the segmentation using Jieba
    3. remove Chinese stop words
    
Created on Sun Mar  17 02:27:16 2018

@author: Yue,Lu
"""
import re
import jieba


if __name__=='__main__':
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
    comment_file='10K.txt'
    with open(path+'\\'+comment_file,encoding='gbk') as f:
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

    #sentence_list = [[word1,word2,...],[word1,word2,..]]

    #step3 write dealed words into a file comment by comment
    write_file='Comment_with_segmentation.txt'
    file=open(path+'\\'+write_file,'w',encoding='utf-8')
    for sent in sentence_list:
        for i in range(len(sent)):
            file.write(sent[i]+' ')
        file.write('\n')
        
    print('Segmentation words have been written into',path+'\\'+write_file)
    file.close()
