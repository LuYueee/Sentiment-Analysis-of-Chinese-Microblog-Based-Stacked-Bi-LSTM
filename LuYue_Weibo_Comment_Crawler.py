# -*- coding: utf-8 -*-
"""
Weibo Comment Crawler: 
    Get 1000 comments of a status in weibo and write into a '.txt' file

Created on Fri Feb 23 14:29:27 2018

@author: Yue, Lu
"""
import requests
from bs4 import BeautifulSoup

def writeTxt(name,path,content):
    '''
    Function:
        Write a list of comment into a '.txt' file
        
    Parameters:
        name: name of file
        path: absolute path of file
        content: a list of comment e.g [comment1,comment2,...]
        
    '''
    # open the file
    f=open(path+'\\'+name+'.txt','w',encoding='utf8')
    f.seek(0)
    # get length of comment list
    length=len(content)
    for i in range(0,length):
        f.writelines([str(content[i]),'\n'])
        #f.writelines([str(i+1),',',str(content[i]),'\n'])
        f.flush()
      
    print('[Comments have written into '+str(path)+'\\'+str(name)+'.txt]')
    f.close()
    
def getComment(hot_comment,comment):
    '''
    Function:
        Analyse json file and get all comment (content of 'text' field in hot_comment{} and comment[])
        
    Parameters:
        hot_comment: a dict of json file from page 0 to page 1
        comment: a list of json file from page 2 to page 100
        
    Return: 
        file: a string list of all weibo comment
    '''
    file=[]
    j=0
    #check if dict hot_comment['data'] has key 'hot_data'
    if 'hot_data' in hot_comment['data'].keys():
        length=len(hot_comment['data']['hot_data'])
        #get the number of hot comment
        for i in range(0,length):
            if hot_comment['data']['hot_data'][i] is not None:
                file.append(hot_comment['data']['hot_data'][i]['text'])
                j=i+1
        j=j+1
    
    '''
    length=len(hot_comment['data']['data']) 
    #get the number of normal comment in 'hot_comment'
    for i in range(0,length):
        if hot_comment['data']['data'][i] is not None:
            file.append(hot_comment['data']['data'][i]['text'])
            j=j+1
    '''
    
    length=len(comment)   
    #get the number of normal comment in 'comment'
    for i in range(0,length):
        length1=len(comment[i]['data']['data'])
        for k in range(0,length1):
            if comment[i]['data']['data'][k] is not None:
                file.append(comment[i]['data']['data'][k]['text'])
                j=j+1
    return file
    
def getJson(weibo_id,name,path):
    '''
    Function:
        Get 100 json files according to weibo id e.g weibo_id=4210306257555878 
        And call getComment() to get comments in json files and call writeTxt() to write comments into a '.txt' file
        
    Parameters:
        weibo_id: id of weibo status
        name: name of file
        path: absolute path of file
        
    Retrun:
        if get comment of current weibo id, return 1
    '''
    # j: num of comment
    j=1
    ok=0
    url_comment = ['http://m.weibo.cn/api/comments/show?id={}&page={}'.format(str(weibo_id),str(i)) for i in range(0,1)]
    # get a list of hot comment url
    hot_comment={}
    comment=[]
    
    print('\nWeibo ID:',weibo_id)
    for url in url_comment:
        wb_data = requests.get(url=str(url),headers=headers).json()
        # get weibo hot comment and trasnfer into json dict

        #数据获取成功
        if wb_data['ok']==1:
            ok=1
             # store json file into a dict called 'hot_comment'
            hot_comment.update(wb_data)
            ###
            #check if dict wb_data['data'] has key 'hot_data'
            if 'hot_data' in wb_data['data'].keys():
                length=len(wb_data['data']['hot_data'])
                for i in range(0,length):
                    print(j,wb_data['data']['hot_data'][i]['text'])
                    j=j+1
            ###

        else: 
            break
    
    
    url_comment = ['http://m.weibo.cn/api/comments/show?id={}&page={}'.format(weibo_id,str(i)) for i in range(1,250)]
    # get a list of normal comment url
    for url in url_comment:
        wb_data1 = requests.get(url=str(url),headers=headers).json()
        # get weibo normal comment and trasnfer into json dict
      
        
        #数据获取成功
        if wb_data1['ok']==1:
            ok=1
            # store json file into a list called 'comment'
            comment.append(wb_data1)
            ###
            length=len(wb_data1['data']['data'])
            for i in range(0,length):             
                print(j,wb_data1['data']['data'][i]['text'])
                j=j+1
            ###  
        else:
            break
        
    if ok==1:
        # get a list of comments
        file=getComment(hot_comment,comment)    
        # write into a .txt file
        writeTxt(name,path,file)
        return 1


  
#main() 
# request header
headers = {
    "Cookies":'xxxxxxxxxxx',
    "User-Agent":'Mozilla/5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13B143 Safari/601.1'
}
   







weibo_id=[4211414132076265]
count=1  
webo_num=len(weibo_id)     
for i in range(0,webo_num):
    filename='comment{}'.format(count)
    path=r'E:\CBOW\new'
    yes=getJson(weibo_id[i],filename,path)
    if yes==1:
        count=count+1
'''    
count=3
for i in range(0,100000000000000):
    filename='weibo{}'.format(count)
    path=r'E:\微博评论'    
    yes=getJson(4206026838401415+i,filename,path)
    #getJson(4206000000000000+i,filename)
    if yes==1:
        count=count+1
'''

