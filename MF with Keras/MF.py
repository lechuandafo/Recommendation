# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:15:37 2019

@author: YLC
"""

import pandas as pd
import numpy as np


from keras import Model
import keras.backend as K
from keras.layers import Embedding,Reshape,Input,Dot,Dense,Dropout
from keras.models import load_model
from keras.utils import to_categorical
from keras import regularizers
from keras.constraints import non_neg
import math
K.clear_session()

def Recmand_model(num_user,num_item,k):
    input_uer = Input(shape=[None,],dtype="int32")
    model_uer = Embedding(num_user+1,k,input_length = 1,
                          embeddings_regularizer=regularizers.l2(0.001), #正则，下同
                          embeddings_constraint=non_neg() #非负，下同
                          )(input_uer)
    model_uer = Dense(k, activation="relu",use_bias=True)(model_uer) #激活函数
    model_uer = Dropout(0.1)(model_uer) #Dropout 随机删去一些节点，防止过拟合
    model_uer = Reshape((k,))(model_uer)
    
    input_item = Input(shape=[None,],dtype="int32")
    model_item  = Embedding(num_item+1,k,input_length = 1,
                            embeddings_regularizer=regularizers.l2(0.001),
                            embeddings_constraint=non_neg()
                            )(input_item)
    model_item = Dense(k, activation="relu",use_bias=True)(model_item)
    model_item = Dropout(0.1)(model_item)
    model_item = Reshape((k,))(model_item)
    
    out = Dot(1)([model_uer,model_item]) #点积运算
    model = Model(inputs=[input_uer,input_item], outputs=out)
    model.compile(loss= 'mse', optimizer='Adam')
    model.summary()
    return model
   
def split_data(rating,topk):
    rating.sort_values(by=['user','time'],axis=0,inplace=True)#先按用户、再按时间排序
    rating['isTest'] = 0 #增加一列，标记是否为测试数据
    rating = rating.reset_index(drop = True)#重新索引
    #print(rating)
    timestamp = rating['time']
    for i in range(1,num_user+1):
        rating_ui = rating[rating['user']==i] #用户i的记录
        idx = rating_ui[rating_ui['time']>=timestamp.quantile(.8)].index#按时间5分位数进行划分，若改为最后K个，使用[-topk:].index即可
        for j in range(0,len(idx)):#选定的数据标记为测试集
            rating.iloc[idx[j]]['isTest'] = 1    
    train = rating[rating['isTest']==0]
    test = rating[rating['isTest']==1]
    return train,test

def train(train_data):
    model = Recmand_model(num_user,num_item,100)
    train_user = train_data['user'].values
    train_item = train_data['item'].values
    train_x = [train_user,train_item]
    train_y = train_data['score'].values
    model.fit(train_x,train_y,batch_size = 100,epochs =10)
    model.save("model.h5")
    
def test(train_data,test_data,all_user,all_item,topk):
    model = load_model('model.h5')
    RMSE = 0
    MAE = 0
    PRE = 0
    REC = 0
    MAP = 0
    NDCG = 0
    MRR = 0 
    for i in range(0,len(all_user)):
        visited_item = list(train_data[train_data['user']==all_user[i]]['item'])
#        print(visited_item)
        testlist = list(test_data[test_data['user']==all_user[i]]['item'])
        rat_k = list(test_data[test_data['user']==all_user[i]]['score']) #项目的评分
        p_rating = [] #总的预测评分
        rankedlist = [] #项目推荐列表
        for j in range(0,len(all_item)): #让每个用户给所有项目打分
            p_rating.append(float(model.predict([[all_user[i]],[all_item[j]]])))
        MAE = MAE + sum([abs(rat_k[s]-p_rating[s]) for s in range(len(testlist))])
        RMSE = RMSE + sum([(rat_k[s]-p_rating[s])*(rat_k[s]-p_rating[s]) for s in range(len(testlist))]) 
        k = 0
        while k < topk:#取前topK个 
            idx = p_rating.index(max(p_rating))
            if all_item[idx] in visited_item: #排除掉访问过的
                p_rating[idx] = 0
                continue
            rankedlist.append(all_item[idx])
            p_rating[idx] = 0
            k = k + 1
        print("对用户",all_user[i])
        print("Topk推荐:",rankedlist)
        print("实际访问:",testlist)
        AP_u,PRE_u,REC_u,NDCG_u,MRR_u = cal_indicators(rankedlist, testlist,rat_k)
        PRE = PRE + PRE_u
        REC = REC + REC_u
        MAP = MAP + AP_u
        NDCG = NDCG + NDCG_u
        MRR = MRR + MRR_u
        print('--------')

    print('评价指标如下:')
    RMSE = math.sqrt(RMSE/float(len(test_data)))
    MAE = MAE/float(len(test_data))
    PRE = PRE/len(all_user) 
    REC = REC/len(all_user)
    MAP = MAP/len(all_user)
    NDCG = NDCG/len(all_user)
    MRR = MRR/len(all_user)
    print('RMSE:',RMSE)
    print('MAE:',MAE)
    print('PRE@',topk,':',PRE)
    print('REC@',topk,':',REC)
    print('MAP@',topk,':',MAP)
    print('NDCG@',topk,':',NDCG)
    print('MRR@',topk,':',MRR)
    
def cal_indicators(rankedlist, testlist,test_score):
    hits = 0
    sum_precs = 0
    AP_u = 0 
    PRE_u = 0 
    REC_u = 0
    NDCG_u = 0
    MRR_u = 0 
    ranked_score = []
    for n in range(len(rankedlist)):
        if rankedlist[n] in testlist:
            hits += 1
            sum_precs += hits / (n + 1.0)
            ranked_score.append(test_score[testlist.index(rankedlist[n])])
            if MRR_u == 0:
                MRR_u = float(1/(testlist.index(rankedlist[n])+1)) #测试集用的是时间序而非评分序
        else:
            ranked_score.append(0)
    if hits > 0:
        AP_u = sum_precs/len(testlist)
        PRE_u = float(hits/len(rankedlist))
        REC_u = float(hits/len(testlist))
        DCG_u = cal_DCG(ranked_score)
        IDCG_u = cal_DCG(sorted(test_score)[0:len(rankedlist)])
        NDCG_u = DCG_u/IDCG_u
    return AP_u,PRE_u,REC_u,NDCG_u,MRR_u

def cal_DCG(rec_list):
    s = 0
    for i in range(0,len(rec_list)):
        s = s + (math.pow(2,rec_list[i])-1)/math.log2((i+1)+1)
    return s

if __name__ == '__main__':
    rating = pd.read_csv('movie.txt',header = None,sep = '\t',names = ['user','item','score','time'])
    all_user = np.unique(rating['user'])
    num_user = len(all_user)
    all_item = np.unique(rating['item'])
    num_item = len(np.unique(rating['item']))
    num_record = len(rating)
    topk = 10
#    print("用户数为:",num_user,"项目数为:",num_item,"记录数为:",num_record)
#    filling_rate = num_record/(num_user*num_item)
#    print("填充率为:",filling_rate)
    train_data,test_data = split_data(rating,topk) #分割训练集、测试集
    train(train_data)
    test(train_data,test_data,all_user,all_item,topk)