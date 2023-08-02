# #!/usr/bin/env python
# # coding: utf-8

# # # Datasets Information

# # Tables and graphs presenting the datasets used in Chapter 3: Experimentation.

# # In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect,sys

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import datasets,partitions,utils,re


# In[2]:


dataset_classes=inspect.getmembers(sys.modules['datasets'], inspect.isclass)
dataset_classes=[(i,j) for i,j in dataset_classes if i!="Dataset"]


# # In[4]:


# datasets_summary=[]

# for name,dataset_class in dataset_classes:
#     dataset=dataset_class()
    
#     # Build a model
#     X,y=dataset.X_y
# #     m_type=RandomForestClassifier if dataset.task=='classification' else RandomForestRegressor
#     m_type=LogisticRegression if dataset.task=='classification' else LinearRegression
#     m=m_type(n_jobs=-1)
    
#     # Eval with whole dataset
#     cv=cross_val_score(m,X,y,)
    
#     # Eval with 1/20 of dataset
#     X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=1/20)
#     score_20=m.fit(X_train,y_train).score(X_test,y_test)
    
#     # No. of classes
#     n_classes=np.nan if dataset.task=='regression' else y.nunique()
    
#     # No. of attributes
#     n_attributes=dataset.load_raw().shape[1]
    
#     # Partitioned on Col
#     p_on=f'{dataset.partition_on_column} ({len(dataset.natural_partition)})' if hasattr(dataset,'partition_on_column') else '-'
    

#     datasets_summary.append([
#         dataset,dataset.task,y.size,n_attributes,n_classes,p_on,cv.mean(),score_20
#     ])
    


# # In[48]:


# datasets_df=pd.DataFrame(
#     datasets_summary,
#     columns=[
#         'dataset','task','N','# of attributes',
#         '# of classes','partitioned on','baseline score','1/20th score'
#     ]
# ).sort_values('task')

# #datasets_df[['N','# of attributes','# of classes']]=datasets_df[['N','# of attributes','# of classes']].astype('int')
# datasets_df=datasets_df.fillna('-')
# datasets_df['task']=datasets_df['task'].str.title()

# datasets_df['N/20']=(datasets_df['N']/20).astype('int')
# datasets_df['(N/20)0.25']=(0.25*datasets_df['N']/20).astype('int')
# datasets_df['N']=((datasets_df['N']/1000).astype('int')).astype('str')+'k'
# datasets_df


# # In[49]:


# datasets_df.to_csv('tmp/datasets_df.csv',index=False)


# # ## Power law partitions on N

# # In[3]:


# X,y=datasets.VehicleLoanDefaultDataset().X_y


# # In[4]:


# def power_partitions_(c,a):
#     partition={i:(X.iloc[ixs],y.iloc[ixs]) for i,ixs in partitions.power_partition_n(y,c,a).items()}
#     xs,ys=[],[]
#     p=sorted(partition,key=lambda i:len(partition[i][1]))
#     for island in p:
#         xs.append(str(island))
#         ys.append(len(partition[island][1]))
#     return xs,ys


# # In[6]:


# # Create two subplots and unpack the output array immediately
# f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True,sharex=True,figsize=(10,2))

# a=0.25
# xs,ys=power_partitions_(10,a)
# ax1.barh(xs, ys)
# ax1.set_yticks([])
# ax1.set_title('a = 0.25')

# a=0.50
# xs,ys=power_partitions_(10,a)
# ax2.barh(xs, ys)
# ax2.set_title('a = 0.50')

# a=0.75
# xs,ys=power_partitions_(10,a)
# ax3.barh(xs, ys)
# ax3.set_title('a = 0.75')

# a=1.0
# xs,ys=power_partitions_(10,a)
# ax4.barh(xs, ys)
# ax4.set_title('a = 1.0')


# f.suptitle(r'Vehicle Loan Default dataset partitioned with $PowN(a,10)$',y=1.20)
# f.supylabel('Islands',x=0.1)
# f.supxlabel('Number of datapoints',y=-0.20)


# # ## Dir partitions on Y

# # In[213]:


# X,y=datasets.VehicleLoanDefaultDataset().X_y


# # In[225]:


# def dir_partitions_(c,alpha,y,X):
#     partition={i:(X.iloc[ixs],y.iloc[ixs]) for i,ixs in partitions.dirichlet_partition(y,c,alpha).items()}

#     _,yAll=utils.join_partitions(partition,partition.keys())
#     labels=yAll.unique()
        
#     clients=list(partition.keys())

#     # Class -> array with no. of elements of class per client
#     # E.g. 0 -> [100,200,300] implies client 0 has 100 class 0 examples and so on

#     class_client_counts={label:[] for label in labels}

#     for label in labels:
#         counts_all=np.zeros(len(clients))
#         for c,(client,(_,y)) in enumerate(partition.items()):
#             counts_all[c]=(y==label).sum()
        
#         class_client_counts[label]=counts_all
    
#     return clients,class_client_counts
    


# # In[270]:


# # Create two subplots and unpack the output array immediately
# f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True,sharex=True,figsize=(10,2))


# alpha=0.5
# clients,class_client_counts=dir_partitions_(10,alpha,y,X)
# prev=None
# for k,v in class_client_counts.items():
#     ax1.barh(clients, v,label=k,left=prev)
#     ax1.set_yticks([])
#     prev=v
# ax1.set_title(r'$\alpha$'+f' = {alpha}')

# alpha=1
# clients,class_client_counts=dir_partitions_(10,alpha,y,X)
# prev=None
# for k,v in class_client_counts.items():
#     ax2.barh(clients, v,label=k,left=prev)
#     prev=v
# ax2.set_title(r'$\alpha$'+f' = {alpha}')


# alpha=10
# clients,class_client_counts=dir_partitions_(10,alpha,y,X)
# prev=None
# for k,v in class_client_counts.items():
#     ax3.barh(clients, v,label=k,left=prev)
#     prev=v
# ax3.set_title(r'$\alpha$'+f' = {alpha}')

# alpha=10000
# clients,class_client_counts=dir_partitions_(10,alpha,y,X)
# prev=None
# for k,v in class_client_counts.items():
#     ax4.barh(clients, v,label=k,left=prev)
#     prev=v
# ax4.set_title(r'$\alpha$'+f' = {alpha}')

# #f.figure.set_figwidth(10)
# #f.figure.set_figheight(2)
# f.suptitle(r'Vehicle Loan Default dataset partitioned with Dir($\alpha$,10)',y=1.20)
# f.supylabel('Islands',x=0.1)
# f.supxlabel('Class distribution',y=-0.20)


# # In[50]:


# X,y=datasets.BlackFridayDataset().X_y


# # In[51]:


# m=RandomForestRegressor(n_jobs=-1)


# # In[52]:


# import time


# # In[55]:


# start=time.time()
# cross_val_score(m,X,y)
# time.time()-start


# # In[56]:


# 367.7014729976654/60


# # In[61]:


# ((16*5*25*6)/60)/24


# # In[63]:


# from sklearn.linear_model import LogisticRegression


# # In[244]:





# # In[245]:


# [1,2,3,4,5][-2:]


# # In[246]:


# for prop in [0.1,0.5,0.9]:
#     m=m=Lasso(
#         alpha=1/C,
#         #penalty='l1',
#         #solver='liblinear',
#         max_iter=2000
#     )
#     n=len(y)
#     i=int(prop*n)
#     a=RandomForestRegressor(n_jobs=-1)
#     a.fit(X.iloc[:i],y.iloc[:i])
    
#     b=RandomForestRegressor(n_jobs=-1)
#     b.fit(X.iloc[-(n-i):],y.iloc[-(n-i):])
    
#     s=StackingRegressor(
#         estimators=[('a',a),('b',b)],
#         final_estimator=m,
#         cv='prefit',
#         n_jobs=-1
#     )
#     s.fit(X,y)
#     score=s.score(X,y)
#     #imp=s.final_estimator_.feature_importances_
#     imp=s.final_estimator_.coef_
    
#     print(prop,score,imp)


# # In[248]:


# from sklearn.ensemble import VotingClassifier


# # In[ ]:


# VotingClassifier()


# # In[276]:


# a=RandomForestRegressor(n_jobs=-1)
# a.fit(X,y)
# a.score(X,y)


# # In[280]:


# b=RandomForestRegressor(n_jobs=-1)
# b.fit(X.iloc[:5000],y.iloc[:5000])
# b.score(X,y)


# # In[281]:


# c=RandomForestRegressor(n_jobs=-1)
# c.fit(X.iloc[:2000],y.iloc[:2000])
# c.score(X,y)


# # In[282]:


# for C in [0.00001,0.001,0.1,1,100,1000]:
#     m=Lasso(
#         alpha=1/C,
#         #penalty='l1',
#         #solver='liblinear',
#         max_iter=2000
#     )
#     #m=RandomForestRegressor()
    
#     s=StackingRegressor(
#         estimators=[('a',a),('b',b),('c',c)],
#         final_estimator=m,
#         cv='prefit',
#         n_jobs=-1
#     )
#     s.fit(X,y)
#     score=s.score(X,y)
#     imps=s.final_estimator_.coef_#feature_importances_
#     print(C,score,imps)


# # In[230]:


# from sklearn.ensemble import StackingClassifier,StackingRegressor
# from sklearn.linear_model import Lasso


# # In[233]:


# s.score(X,y)


# # In[266]:


# s.final_estimator_.feature_importances_


# # In[267]:


# s.final_estimator_.coef_


# # In[152]:


# X,y=datasets.DiamondsDataset().X_y


# # In[3]:


from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


# In[4]:


model_types={
    'classification':[RandomForestClassifier, LogisticRegression,SGDClassifier, KNeighborsClassifier,GradientBoostingClassifier],
    'regression':[RandomForestRegressor, LinearRegression,SGDRegressor, KNeighborsRegressor, GradientBoostingRegressor]
}


# In[6]:


scores_by_model_type_and_dataset=[]

for name,dataset_class in dataset_classes:
    
    # Instantiate dataset
    dataset=dataset_class()
    X,y=dataset.X_y
    
    print(dataset)
    
    
    for model_type in model_types[dataset.task]:

        # Build a model
        print(model_type)
        if 'SG' not in model_type.__name__ and 'Grad' not in model_type.__name__:
            m=model_type(n_jobs=-1)
        else:
            m=model_type()

        # Eval with whole dataset
        cv=cross_val_score(m,X,y,scoring='balanced_accuracy' if dataset.task=='classification' else 'r2')

        scores_by_model_type_and_dataset.append([
            dataset,dataset.task,str(model_type),cv.mean()
        ])


utils.savePickeObj(scores_by_model_type_and_dataset,"tmp","1.pkl")


# In[ ]:




