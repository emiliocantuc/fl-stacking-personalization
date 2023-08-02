
# Local imports
import utils,islands

# Other imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import os,time,argparse

# First we define different train,validation,test splits methods:
TEST_POOL_PROP=0.5

def local_splits(partition,train_size,val_size,test_size,task,random_state):
    splits_f=utils.train_val_test_split if task=='regression' else utils.stratified_train_val_test_split
    return {
        island:splits_f(X,y,train_size,val_size,test_size,random_state)
        for island,(X,y) in partition.items()
    }

def global_splits(partition,train_size,val_size,test_size,task,random_state):
    splits={}
    test_pool_X,test_pool_y=None,None
    
    for island,(X_p,y_p) in partition.items():
        # Do a regular train,val,tests split
        splits_f=utils.train_val_test_split if task=='regression' else utils.stratified_train_val_test_split
        X_train,X_val,X_test,y_train,y_val,y_test=splits_f(
            X_p,y_p,train_size,val_size,test_size,random_state
        )
        # Add only the train and val sets
        splits[island]=[X_train,X_val,y_train,y_val]
        
        # And add the test set indeces to a "test" pool
        if test_pool_X is None:
            test_pool_X=X_test
            test_pool_y=y_test
        else:
            test_pool_X=pd.concat([test_pool_X,X_test])
            test_pool_y=pd.concat([test_pool_y,y_test])
        
    
    # We now sample TEST_POOL_PROP % of the test pool
    n=max(test_pool_y.size,int(len(test_pool_y)*TEST_POOL_PROP))
    ixs=np.random.choice(test_pool_y.index,size=n,replace=False)

    X_test=test_pool_X.loc[ixs]
    y_test=test_pool_y.loc[ixs]
    assert X_test.shape[0]==y_test.shape[0]
    
    # and set it as everyone's test set
    for island,(X_train,X_val,y_train,y_val) in splits.items():
        splits[island]=X_train,X_val,X_test,y_train,y_val,y_test
    
    return splits


def evaluate(partition,island_type,island_model_types,split_f,task,k_crosses=5,score_metrics=[],
        train_size=0.7,val_size=0.15,test_size=0.15,meta_fit_on_val=True,random_state=None,verbose_level=0,**kargs):
    
    # Make partition keys strings in case they are not
    partition={str(k):v for k,v in partition.items()}

    # Instantiate estimators
    estimators={}
    for k,v in island_model_types.items():
        mt=v.__name__
        if 'SG' not in mt and 'Grad' not in mt:
            estimators[k]=v(n_jobs=-1)
        else:
            estimators[k]=v()

    # Instantiate islands
    islands={island:island_type(island,task,est,**kargs) for island,est in estimators.items()}

    # Scores to output. island name -> { score_name -> [scores] }
    scores={island:{'test_'+score:[] for score in score_metrics} for island in islands.keys()}
    weights={island:[] for island in islands.keys()}
    avg_fit_time,n_fits,zeros,negatives=0,0,0,0

    for k in range(k_crosses):

        # Do train/test splits
        r_s=k*random_state if random_state else None
        splits=split_f(partition,train_size,val_size,test_size,task,random_state=r_s)

        # Fit all models on their own training data
        for island in islands.values():
            X_train,_,_,y_train,_,_=splits[island.name]
            island.fit_local(X_train,y_train)


        # Every island
        for island in islands.values():
            # Gets its val/test split
            X_train,X_val,X_test,y_train,y_val,y_test=splits[island.name]

            # Fits its voting model using the validation set
            except_current=[(i,obj) for i,obj in islands.items() if i!=island.name]

            start_fit=time.time()

            if meta_fit_on_val:
                w=island.fit(otherIslands=except_current,X=X_val,y=y_val)

            else:
                w=island.fit(otherIslands=except_current,X=X_train,y=y_train)

            fit_time=time.time()-start_fit
            avg_fit_time+=fit_time
            n_fits+=1

            # And scores it on its own test data
            s=utils.score(island.model,X_test,y_test,score_metrics)
            for score_name,value in s.items():
                scores[island.name][score_name].append(value)

            # And save the weights it used
            weights[island.name].append(w)
            zeros+=sum(1 for w_i in w.values() if w_i==0)
            negatives+=sum(1 for w_i in w.values() if w_i<0)

            m=np.array(scores[island.name]['test_'+score_metrics[0]]).mean()
            utils.verbose_print(f'Island: {island.name} \t\t\tAvg Score: {m}',verbose_level,2)

    avg_fit_time=avg_fit_time/n_fits
    
    avg_scores=0
    n_=0
    for island,by_scores in scores.items():
        for score_name, scores_ in by_scores.items():
            if 'balanced' in score_name or 'r2' in score_name:
                avg_scores+=np.array(scores_).mean()
                n_+=1
    

    utils.verbose_print(f'Avg Scores: {avg_scores/n_} \tAvg Fit Time: {avg_fit_time} \t 0s: {zeros} Negs: {negatives}',verbose_level,2)

                
    return {'scores':scores,'weights':weights,'avg_fit_time':avg_fit_time}


def run(partitions_path,results_path,islands_to_test,splits_to_use,island_model_types,k_crosses,meta_fit_on_val,random_state,verbose_level):

    # Get the partition file names
    fnames=[fname for fname in os.listdir(partitions_path) if '.pkl' in fname]

    # For every file in the partitions directory
    for ith_file,p_file in enumerate(fnames):

        if 'natural' in p_file:continue

        utils.verbose_print(f'\n({(ith_file+1)}/{len(fnames)}) Partition: {p_file}',verbose_level,0)

        # For every island type
        for ith_island,(save_name,(island_type,island_params)) in enumerate(islands_to_test.items()):

            utils.verbose_print(f'\n({(ith_island+1)}/{len(islands_to_test)}) Island type: {save_name}',verbose_level,0)
  
            # Retrieve the partition's info, instantiate the dataset and obtain X,y.
            p_info=utils.loadPickeObj(partitions_path,p_file)
            dataset=p_info['dataset']()

            # For every type of train/test split function to use
            for ith_split_f,splits_f in enumerate(splits_to_use):

                utils.verbose_print(f'({(ith_split_f+1)}/{len(splits_to_use)}) Split type: {splits_f.__name__}',verbose_level,0)

                # Evaluate
                try:
                    res=evaluate(
                        partition=p_info['partition'],
                        island_type=island_type,
                        island_model_types=island_model_types[dataset.task],
                        task=dataset.task,
                        split_f=splits_f,
                        k_crosses=k_crosses,
                        score_metrics=metrics[dataset.task],
                        meta_fit_on_val=meta_fit_on_val,
                        random_state=random_state,
                        verbose_level=verbose_level,
                        **island_params
                    )

                    utils.savePickeObj(res,os.path.join(results_path,save_name,splits_f.__name__),p_file)
                
                except Exception as e:
                    print(e)
                    raise e

    print('Done!')

if __name__=='__main__':

    # Parse passed arguments
    parser=argparse.ArgumentParser(
        prog='Personalization Methods Evaluation',
        description='Performs the evaluation described in Sec. XX on the different personalization methods' # TODO confirm
    )
    parser.add_argument('--partitions_dir',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--k_crosses',type=int,default=5)
    parser.add_argument('--meta_fit',choices=['val','train'],default='val')
    parser.add_argument('-s','--seed',default=7)
    parser.add_argument('-v','--verbose_level',choices=[0,1,2],type=int,default=1)

    # Add split arguments to parser
    parser.add_argument('--all_splits',action='store_true')
    split_fns=[local_splits,global_splits]
    for split_fn in split_fns:
        parser.add_argument(f'--{split_fn.__name__}',action='store_true')

    args=vars(parser.parse_args())

    # Parse split methods
    splits_to_use=[]
    if args['all_splits']:
        splits_to_use=split_fns
    else:
        l=locals()
        splits_to_use=[l[arg] for arg,v in args.items() if 'split' in arg and v and arg in l]
    assert len(splits_to_use)>0,'Select at least one split type'


    # Define what models to use depending on the task and island

    island_model_types={
        'classification':{
            '0':RandomForestClassifier,
            '1':RandomForestClassifier,
            '2':LogisticRegression,
            '3':LogisticRegression,
            '4':SGDClassifier,
            '5':SGDClassifier,
            '6':KNeighborsClassifier,
            '7':KNeighborsClassifier,
            '8':GradientBoostingClassifier,
            '9':GradientBoostingClassifier
        },
        'regression':{
            '0':RandomForestRegressor,
            '1':RandomForestRegressor,
            '2':LinearRegression,
            '3':LinearRegression,
            '4':SGDRegressor,
            '5':SGDRegressor,
            '6':KNeighborsRegressor,
            '7':KNeighborsRegressor,
            '8':GradientBoostingRegressor,
            '9':GradientBoostingRegressor

        }
    }

    # Define what metrics to report depengind on the task 
    metrics={
        'regression':['r2','neg_mean_squared_error'],
        'classification':['balanced_accuracy','accuracy'] 
    }
    
    # Island types to evaluate. Name to save results with  -> (island obj, param dict)
    islands_to_test={
        'Local':(islands.LocalIsland,{}), #1
        'Equally Weighted':(islands.Island,{}), #1
        'By N':(islands.ByNIsland,{}), #1
        'Random':(islands.RandomContIsland,{}), #2
        'Forest':(islands.StackingForestIsland,{'prop':1.0}), #7
    }


    utils.verbose_print(f'Evaling with {args}',args['verbose_level'],0)

    # Call main method
    run(
        partitions_path=args['partitions_dir'],
        results_path=args['output_dir'],
        islands_to_test=islands_to_test,
        splits_to_use=splits_to_use,
        island_model_types=island_model_types,
        k_crosses=args['k_crosses'],
        meta_fit_on_val=args['meta_fit']=='val',
        random_state=args['seed'],
        verbose_level=args['verbose_level']
    )

    # debug:
    # rm -r tmp/tmp ;nohup python3 -u model_experiments.py --partitions_dir partitions/model_ultra_quick --output_dir tmp/tmp --local_splits -v 2 --k_crosses 1 &
    

    # Used to perform rapid testing
    # python3 -u eval.py --partitions_dir partitions/model_ultra_quick --output_dir results/model_ultra_quick_train --local_splits --verbose_level 2 --k_crosses 1 --meta_fit train
    
    # Used to perform model_experiments
    # nohup python3 -u model_experiments.py --partitions_dir partitions/model_experiments --output_dir results/mult_model_types --global_splits --local_splits --verbose_level 2 --k_crosses 5 &




