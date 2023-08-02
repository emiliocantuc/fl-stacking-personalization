from itertools import chain, combinations
import pandas as pd
import numpy as np
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import os,pickle

# Viz
import matplotlib.pyplot as plt

def powerset(iterable):
    """
    Returns the iterable's powerset as a generator.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def get_powersets(iterable):
    """
    Computes the iterable's powerset and returns a dictionary
    mapping each member of iterable to the subsets it appears in.

    For example if iterable = (1,2), it's powerset is
    [(), (1,), (2,), (1, 2)] and this funtion will return
    {1: [(1,), (1, 2)], 2: [(2,), (1, 2)]}.
    """
    powersets={}
    for s in powerset(iterable):
        for i in s:
            powersets[i]=powersets.get(i,[])+[s]

    return powersets

def normalized(a, axis=-1, order=1):
    """
    Normalizes a numpy array along an axis using a norm of order.
    
    Inputs:
        a: The numpy array
        axis: The axis along which to normalize (default = -1)
        order: The order of the norm to use (default = 1)
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def safe_get_dummies(df,columns):
    """
    Checks what columns have more than 1 unique value before calling
    pd.get_dummies. Columns with only 1 unique value are droped.
    """
    # Find columns with more than 1 value
    u=df[columns].nunique()
    i=u[u>1].index.to_list()

    # Drop columns with only 1 value
    to_drop=[c for c in columns if c not in i]
    df=df.drop(to_drop,axis=1)

    # Return call to pd.get_dummies with qualifying columns
    return pd.get_dummies(df,columns=i)

def join_partitions(partition,iterable):
    X=pd.concat([partition[i][0] for i in iterable])
    y=pd.concat([partition[i][1] for i in iterable])
    
    return X,y

# SKLEARN
def score(estimator,X,y,scoring):
    """
    Sklearn's scoring functionality without using cross_validation. 
    Assumes estimator is fitted, computes metrics in scoring
    with X and y as inputs and returns the results as a dictionary.
    """
    score_maker=_check_multimetric_scoring(estimator, scoring)
    out={}
    for score,maker in score_maker.items():
        out['test_'+score]=maker(estimator,X,y)
    return out

def train_val_test_split(X,y,train_size,val_size,test_size,random_state=None):
    assert sum([train_size,val_size,test_size])==1,'Sizes must add up to 1'

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)
    X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=test_size,random_state=random_state)
    return X_train,X_val,X_test,y_train,y_val,y_test


def stratified_train_val_test_split(X,y,train_size,val_size,test_size,random_state=None):
    assert sum([train_size,val_size,test_size])==1,'Sizes must add up to 1'

    for train_ixs,test_ixs in StratifiedShuffleSplit(n_splits=1,train_size=train_size,random_state=random_state).split(X,y):
        pass

    test_size_=test_size/(test_size+val_size)
    X_test,y_test=X.iloc[test_ixs],y.iloc[test_ixs]

    for val_ixs,test_ixs in StratifiedShuffleSplit(n_splits=1,test_size=test_size_,random_state=random_state).split(X_test,y_test):
        pass

    return X.iloc[train_ixs],X.iloc[val_ixs],X.iloc[test_ixs],y.iloc[train_ixs],y.iloc[val_ixs],y.iloc[test_ixs]



def del_dict_keys(d,keys):
    for k in [i for i in keys if i in d]:
        del d[k]
    

def format_cross(island,cross,n):
    del_dict_keys(cross,['fit_time','score_time'])
    island=str(island)
    to_print=[f'{k.replace("test_","")}:{np.array(v).mean():.2f}' for k,v in cross.items()]
    to_print+=[f'n:{n}',island]
    print('\t'.join(to_print))

def results_df(results):
    out={}
    for island,cross in results.items():
        tmp={}
        for k,v in cross.items():
            tmp[k]=np.array(v).mean()
        out[island]=tmp
    return pd.DataFrame(out).T


# Visualize partition - classification
def visualize_partition(partition,labels=None,title='Partition',legend=False,verbose=False,fn=''):

    #if labels is None:
    _,yAll=join_partitions(partition,partition.keys())
    labels=yAll.unique()
        
    clients=list(partition.keys())
    if verbose:
        print(f'Labels ({labels.size}): {labels}')
        print(f'Islands ({len(clients)}: {clients}')

    # Class -> array with no. of elements of class per client
    # E.g. 0 -> [100,200,300] implies client 0 has 100 class 0 examples and so on

    class_client_counts={label:[] for label in labels}

    for label in labels:
        counts_all=np.zeros(len(clients))
        for c,(client,(_,y)) in enumerate(partition.items()):
            counts_all[c]=(y==label).sum()
        
        class_client_counts[label]=counts_all
    
    if verbose:
        print(f'Class client counts: {class_client_counts}')

    plt.figure()
    fig, ax = plt.subplots()
    prev=None
    for k,v in class_client_counts.items():
        ax.barh(clients, v,label=k,left=prev)
        prev=v

    ax.set_title(title)
    plt.xlabel('Number of data points')
    plt.ylabel('Islands')
    if legend: plt.legend()
    if fn:
        fig.savefig(fn)
    else:
        plt.show()

    plt.close()

def visualize_regression_partition(partition,title='Partition',fn=''):
    """
    Visualizes de distribution in the y column of every island
    as box plots. Does not contain any info. regarding the
    number of samples per island.
    """
    labels=[]
    data=[]
    for i,(_,y) in partition.items():
        labels.append(i)
        data.append(y)


    plt.figure()
    box=plt.boxplot(data,sym='',vert=False,patch_artist=True)
    
    # Change colors
    # for item in ['boxes', 'whiskers', 'fliers', 'caps']:
    #     plt.setp(box[item], color='black')
    # plt.setp(box['medians'],color='white')
    
    plt.yticks(list(range(1,len(labels)+1)), labels)
    #plt.yticks([0,len(data)-1],['0',str(len(data))])
    plt.ylabel('Clients/Islands')
    plt.xlabel('Local y column')
    plt.title(title)
    if fn:
        plt.savefig(fn)

    else:
        plt.show()

    plt.close()


def visualize_partition_n(partition,title='',show_y_ticks=True,fn='',sort=False,xlims=None):
    """
    Plots a horizontal bar plot showing the number of examples per
    client.
    """
    xs,ys=[],[]
    if sort:
        p=sorted(partition,key=lambda i:len(partition[i][1]))
    else:
        p=partition.keys()

    for island in p:
        xs.append(str(island))
        ys.append(len(partition[island][1]))

    plt.figure()
    plt.barh(xs,ys)
    x=plt.xlim()
    if xlims:
        plt.xlim(xlims[0],xlims[1])
    plt.xlabel('Number of examples')
    plt.ylabel('Clients')
    plt.title(title)
    #plt.yticks(ticks=sorted(list(partition.keys())) if show_y_ticks else [])
    
    if fn:
        plt.savefig(fn)
    else:
        plt.show()
    
    plt.close()
    return x

# PICKLE
def savePickeObj(obj,path,fname):
    """
    Saves obj as a pickle file.
    NOTE: Overwrites file if it exists.
    """
    os.makedirs(path,exist_ok=True)
    fname=os.path.join(path,fname)
    with open(fname,"wb+") as f:
        pickle.dump(obj,f)
    print(f'Saved obj in {fname}')


def loadPickeObj(path,fname=None):
    """
    Loads and return object from pickle file.
    """
    fname=os.path.join(path,fname) if fname else path
    with open(fname,"rb") as f:
        return pickle.load(f)
    
def verbose_print(s,v_level,ge_than):
    # Prints s if v_level is >= than ge_than
    if v_level>=ge_than:
        print(s)