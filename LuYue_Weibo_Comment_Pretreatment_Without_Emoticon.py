# -*- coding: utf-8 -*-
"""
Weibo Comment Pretreatment Without Emoticon: 
    Do the pretreatment of crawled weibo statuses:
    1.remove [emoticon]
    2.remove '回复'
    3.remove '@user:'  and '@user'
    4.remove special symbol
    5.write comments in 'Corpus\Preprocessing Comments.txt'
    
Created on Sun Mar  4 02:27:16 2018

@author: Yue, Lu
"""
import os
import re 
from bs4 import BeautifulSoup


def weiboCommentPretreatment(path,name):
    '''
    Function:
        Read weibo comments from a file and do the pretreatment
        
    Parameters:
        name: name of file
        path: absolute path of file
        content: a list of comment e.g [comment1,comment2,...]
    
    Return:
        content: a list of comment e.g [comment1,comment2,...]
    '''
    content=[]
    f=open(path+'\\'+name,'r',encoding='utf-8')
    for markup in f.readlines():
        #print(markup)
        markup='<p>'+markup+'</p>'
        soup=BeautifulSoup(markup,'lxml')
        #remove [emoticon] in comment
        comment=soup.p.text
        

        #remove '转发微博' and 'Repost' in comment
        if '转发微博' in comment:
            comment=''
        if '轉發微博' in comment:
            comment=''
        if 'Repost' in comment:
            comment=''  
            
        # remove '图片评论' and 'Comment with pics' in comment
        if '图片评论' in comment:
            comment=''
        if '圖片評論' in comment:
            comment=''    
        if 'Comment with pics' in comment:
            comment=''
            
        # remove '回复' in comment
        if '回复' in comment:
            comment=comment.lstrip('回复')
                
        #remove '@user:' in comment
        if re.findall(r'@(.*?):',comment):  
            at=re.findall(r'@(.*?):',comment) 
            for i in at:
                comment=comment.replace('@'+i+':','') 
                
        #remove '@user' in comment      
        if re.findall(r'@(.*?)',comment):  
            at=re.findall(r'@(.*?)',comment) 
            for i in at:
                comment=comment.replace('@'+i,'')
         
        #remove special symbol in comment
        if re.findall(u'[^\u4e00-\u9fa5\d\w\[\]+――！，。？、~@#￥%……&*（）：.!/_,$%^*(+"\']+',comment):  
            tmp=re.findall(u'[^\u4e00-\u9fa5\d\w\[\]+――！，。？、~@#￥%……&*（）：.!/_,$%^*(+"\']+',comment)
            for i in tmp:
                comment=comment.replace(i,' ')
     
                
        #remove space   
        comment=comment.strip()
        #appen current comment to list t[]
        
        #remove null comment
        if comment != '':
            content.append(comment)
        
    print(content)
    #[comment1,comment2,comment3,...]
    return content
        
def writeTxt(name,path,content):
    '''
    Function:
        Write a list of comment into a '.txt' file
        
    Parameters:
        name: name of file
        path: absolute path of file
        content: a list of comment e.g [comment1,comment2,...]
    
    Return:
        length: number of comment write into file
    '''

    # open the file
    f=open(path+'\\'+name+'.txt','a',encoding='utf8')
    #f.seek(0)
    # get length of comment list
    length=len(content)
    for i in range(0,length):
        f.writelines([str(content[i]),'\n'])
        #f.writelines([str(i+1),',',str(content[i]),'\n'])
        f.flush()
      
    print('[Comments have written into '+str(path)+'\\'+str(name)+'.txt]')
    f.close()
    ##
    return length
    
if __name__=='__main__':
    #读取爬取评论路径
    path=os.path.abspath('..\Corpus\Raw Comments')
    os.chdir(path)
    
    #统计评论数
    length=0
    for i in range(0,120):      
        name='comment'+str(i)+'.txt'
        if os.path.exists(name):                    
            content=weiboCommentPretreatment(path,name)
            name='Preprocessing Comments'
            #写入评论路径
            wirtepath=os.path.abspath('..')
            #统计评论数
            length=length+writeTxt(name,wirtepath,content)
        else:
            continue
       
       
    #print(length)    
    
