# -*- coding: utf-8 -*-
"""
Weibo Comment Crawler: 
    Get 1000 comments of a Weibo status and write into a 'commnet().txt' file

Created on Fri Feb 23 14:29:27 2018

@author: Yue, Lu
"""
import os  
import requests

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
            
            #check if dict wb_data['data'] has key 'hot_data'
            if 'hot_data' in wb_data['data'].keys():
                length=len(wb_data['data']['hot_data'])
                for i in range(0,length):
                    print(j,wb_data['data']['hot_data'][i]['text'])
                    j=j+1
            

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
            length=len(wb_data1['data']['data'])
            for i in range(0,length):             
                print(j,wb_data1['data']['data'][i]['text'])
                j=j+1
        else:
            break
        
    if ok==1:
        # get a list of comments
        file=getComment(hot_comment,comment)    
        # write into a .txt file
        writeTxt(name,path,file)
        return 1



# request header
headers = {
    "Cookies":'xxxxxxxxxxx',
    "User-Agent":'Mozilla/5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13B143 Safari/601.1'
}
   



#特定微博号
weibo_id=[4211414132076265]
count=0
webo_num=len(weibo_id)     
for i in range(0,webo_num):
    #指定写入名称
    filename='comment{}'.format(count)
    path=os.path.abspath('..\Corpus\Raw Comments')
    yes=getJson(weibo_id[i],filename,path)
    if yes==1:
        count=count+1
'''    
for i in range(0,100000000000000):
    filename='weibo{}'.format(count)   
    yes=getJson(i,filename,path)
    #getJson(i,filename)
    if yes==1:
        count=count+1
'''


'''
140 Weibo ids:
4211410679942667,
4211414132076265,
4211425037519308,
4211236704456334,
4211409556211786,
4211416673834375,
4211421191186321,
4211409069828580,
4211455627211373,
4211417878316145,
4211423343488980,
4211408310132422,
4211433807964163,
4211413667048372,
4210715995362212, 
4210305762729160,
4210306257555878,
4210307100437543,
4210316567038821,
4210321067246926,
4210306554658359,
4210310170793939,
4210316474512596,
4210324129260188,
4210325298896941,
4210305959356493,
4210308945905704,
4210306483531451,
4210309478329580,
4210513330640196,
4210306610262758, 
4210359495385115, 
4210359495385115,
4208525867621289,
4207604131321430,
4211420571218173, 
4209593778789839,
4209587005951918,
4209627589474142,
4209653275649374,
4209588519617060,
4209593875589991,
4209598841740230,
4209777040805841,
4209643536400586,
4209796964005278,
4209588968158446,
4209595796811942,
4209617745798451,
4209605556316906,
4209816441763127,
4209618403753515,
4209623060366677,
4209885640097172,
4209621440695261,
4209591082459389,
4209626700331429,
4209614646215704,
4209870191925527,
4209587202598211,
4209886927086326,
4209603371719343,
4209941960817007,
4209591006620898,
4209594496325022,
4209587987058336,
4209589170366596,
4209589492658746,
4209594593166016,
4209589954607013,
4209887515470901,
4209589479953408,
4209588960235462,
4209616923785940,
4209585697447338,
4209595931337161,
4209886319062708,
4209592860746958,
4209591342053211,
4208705333981113,
4209596236766482,
4209588658440856,
4210318555170866,
4210320400760846,
4210319070456286,
4210319456648984,
4211659985930115,
4208868927871732,
4211213506063314,
4210707492352557,
4210331612014169,
4209563546490348,
4210310552218782,
4210312372230737,
4210311651620970,
4208408775095722,
4211267574898651,
4208034815778169,
4211229821696432,
4213083327566698,
4209905059240647,
4211279243514963,
4211415319534682,
4211418012104398,
4211463906658133,
4211420201246770,
4211407329583723,
4211427118065121,
4205634885153162,
4206643393312113,
4210309235558507,
4211437628979349,
4209824096575811,
4208401502160433,
4205623660679263,
4204521595185855,
4211418662449721,
4211418888437141,
4211220330551923,
4209148667864898,
211205684350067,
4211057021185113,
4211416091265951,
4211248834933626,
4207297166988970,
4208042985411077,
4208393549982701,
4208033993565565,
4211266300100431,
4210306470749527,
4210309189568123,
4210659736721168,
4210562650515123,
4210320400760846,
4210359167955588,
4210336616006422,
4210546611748737,
4210564228276996,
4210289148932093,
4210674957114777,
4210290097271756,
4210300314466671
'''
