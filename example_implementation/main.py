
import sys,os
sys.path.insert(0, '..')
import datasets,utils,partitions,pickle,itertools,partition_loader

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import StackingClassifier,StackingRegressor

from sklearn.dummy import DummyClassifier,DummyRegressor

import numpy as np

import random,shutil

import fl_client,fetching_strategies

SERVER_URL='localhost:50052'


class Island:
    
    def __init__(self,name,X_y,task,fetch_policy,local_model_type,meta_model_type,poisoning):

        self.name=name
        self.X,self.y=X_y
        self.task=task
        self.fetch_policy=fetch_policy
        self.local_model=local_model_type()
        self.meta_model_type=meta_model_type
        self.max_round_bandwith=None
        self.poisoning=poisoning

        self.fl=fl_client.Client(name,SERVER_URL)
        self.fl.request_admission()

    def submit_public_model(self):

        # We fit our local model and submit it
        self.local_model.fit(self.X,self.y)
        self.fl.submit_model(self.local_model,trained_on_n=len(self.y),model_type=self.local_model.__class__.__name__)

        # We will set max_round_bandwith to 10 times the local model size for simplicity
        self.max_round_bandwith=len(pickle.dumps(self.local_model))*20

        # print(f'{self.name}: Submitted model')

    def update_round(self):

        # Get meta data
        meta_data=self.fl.get_model_info()
        # print(f'{self.name}: Got meta data: {meta_data}')

        # Fetch contribution scores
        contribution_scores=self.fl.get_importances()

        # Fetch 
        fetched_models,bandwith=self.fetch_models(meta_data,contribution_scores)
        # print(fetched_models)


        # Perform splits
        split_f=utils.stratified_train_val_test_split if self.task=='classification' else utils.train_val_test_split
        X_train,X_val,X_test,y_train,y_val,y_test=split_f(self.X,self.y,train_size=0.7,val_size=0.15,test_size=0.15,random_state=None)

        # Train local base model
        self.local_model.fit(X_train,y_train)

        # Join local base model with fetched models
        base_models=[(self.name,self.local_model)]+fetched_models

        # Train meta model with hyper param search TODO
        # First with a simple fixed meta model type
        stacked_model_type=StackingClassifier if self.task=='classification' else StackingRegressor
        final_estimator_type=self.meta_model_type if self.task=='regression' or y_val.nunique()>1 else DummyClassifier

        self.meta_model=stacked_model_type(
            estimators=base_models,
            final_estimator=final_estimator_type(),
            cv='prefit',
            n_jobs=-1
        )

        self.meta_model.fit(X_val,y_val)

        # Evaluate
        metric='balanced_accuracy' if self.task=='classification' else 'r2'
        stacked_score=utils.score(self.meta_model,X_test,y_test,scoring=[metric])['test_'+metric]
        local_score=utils.score(self.local_model,X_test,y_test,scoring=[metric])['test_'+metric]

        # print('stacked score',stacked_score)
        # print('local_store',local_score)
        local_delta=stacked_score-local_score        

        # Use the robust way 
        # Calculate importances using feature permutation
        # Island -> normalized importance score

        if hasattr(self.meta_model.final_estimator_,'feature_importances_'):
            # Final est. is a forest
            w={island_name:w_i for (island_name,_),w_i in zip(base_models,self.meta_model.final_estimator_.feature_importances_)}
        
        elif hasattr(self.meta_model.final_estimator_,'coef_'):
            # Is linear

            w=self.meta_model.final_estimator_.coef_
            w=w[0] if w.ndim==2 else w
            w=np.abs(w) # Interpreting importace as absolute value of coefs
            w=w/w.sum() if w.sum()>0 else np.zeros(shape=w.shape)
            w={base_models[i][0]:w_i for i,w_i in enumerate(w)}
            assert len(w)==len(base_models)
        
        else:
            # Dummy classifier
            w={i:1/len(fetched_models) for i,_ in fetched_models}

        if 'Dummy' in self.local_model.__class__.__name__ and self.poisoning:
            w={i:1 for i,_ in base_models if 'Dummy' in meta_data[i]['model_type']}
            w={i:imp/len(w) for i,imp in w.items()}
        
        #assert sum(importances.values())-1<1e-3, f'Importances must sum to 1: {w}'

        #print('importances',w)
        
        # Submit importances
        if len(w)>0:
            self.fl.submit_importances(w)

        return {
            'local_score':local_score,
            'local_delta':local_delta,
            'used_bandwith':bandwith,
            'n_fetched_models':len(fetched_models),
            'importances':w,
            'local_model_type':self.local_model.__class__.__name__,
            'meta_model_type':self.meta_model_type.__name__,
            'fetching_policy':self.fetch_policy.__name__,
            'max_round_bandwith':self.max_round_bandwith
        }

    def fetch_models(self,meta_data,contribution_scores):

        which_to_fetch,bandwith=self.fetch_policy(self,self.max_round_bandwith,meta_data,contribution_scores)
        assert len(which_to_fetch)>0,'Empty which to fetch'
        # print(self.name,'which to fetch',which_to_fetch)

        #[(name,model)]
        out=[]
        for from_island,model in self.fl.get_models(which_to_fetch).items():
            out.append((from_island,model))
        
        return out,bandwith

def get_random_sample(X,y,prop,task):
    if task=='classification':
        min_class=0
        while min_class<50:
            X,_,_,y,_,_=utils.stratified_train_val_test_split(X,y,train_size=prop,val_size=(1-prop)/2,test_size=(1-prop)/2)
            classes,counts_y=np.unique(y,return_counts=True)
            min_class=min(counts_y)
    else:
        X,_,_,y,_,_=utils.train_val_test_split(X,y,train_size=prop,val_size=(1-prop)/2,test_size=(1-prop)/2)

    return X,y


def run(dataset,assignments,p_method,p_param,n_islands,n_rounds,poisoning,output_dir='ex_imp_output/'):

    # Check params
    assert p_method in ['dirY','powN']

    # Generate partition
    X,y=dataset.X_y

    # Make the partition
    skip=True
    n_tries=100
    while n_tries>0:
        try:
            if p_method=='dirY':
            
                p=partitions.dirichlet_bootstrap_partition(y,c_clients=n_islands,alpha=p_param)
        
            else:
                p=partitions.power_partition_n(y,c_clients=n_islands,a=p_param,bootstrap=True,min_n=500)

            p=partitions.slice_partitions_from_X_y(p,X,y)
            partition_loader.partition_validator(p,500,dataset.task)
            skip=False
            break
        except AssertionError as e:
            if n_tries>0:
                n_tries-=1
                
    if skip:
        raise Exception(f'Partition validation not met!')
    
    if dataset.task=='classification':
        utils.visualize_partition(p,fn='hello.png')
    else:
        utils.visualize_partition_n(p,fn='hello.png')


    # round, island, local_delta, used bandwith, local_model_type, fetching policy, importances
    island_stats=[]

    # round -> page rank and plain (sum) importances reported by server
    importances={}

    for round in range(n_rounds):
       
        print(f'Round {round+1}/{n_rounds}')

        # Distribute a random sample from their data
        islands={
            Island(
                str(name),
                X_y=get_random_sample(X,y,prop=0.5,task=dataset.task),
                task=dataset.task,
                fetch_policy=assignments[name]['fetching_policy'],
                local_model_type=assignments[name]['local_model_type'][dataset.task],
                meta_model_type=assignments[name]['meta_model_type'][dataset.task],
                poisoning=poisoning
            )
            for name,(X,y) in p.items()
        }

        # Have every island train its public model and submit it
        for island in islands:
            island.submit_public_model()


        # Have every island perform a round update
        for ith_island,island in enumerate(islands):
            print(f'Island {island.name} ({ith_island+1}/{len(islands)})')
            
            stats=island.update_round()
            island_stats.append([
                round,island.name,stats['local_delta'],stats['used_bandwith'],
                stats['n_fetched_models'],stats['local_model_type'],stats['meta_model_type'],
                stats['fetching_policy'],stats['importances']
            ])

            imp_resp=island.fl.get_importances()
            
            #print(f'Island ({ith_island+1}/{len(islands)}) {island.name}: {stats}')
        
        # Get stats from server
        importances[round]=imp_resp

    
    # Clear server
    island.fl.clear_server()
        

    utils.savePickeObj(island_stats,output_dir,'island_stats.pkl')
    utils.savePickeObj(importances,output_dir,'importances.pkl')


def assign_things(n_islands,local_model_types,meta_model_types,fetching_policies,random_state=None):

    if random_state:
        random.seed(random_state)

    # Island -> dict{local_model,meta_model_type,fetching strategy}
    assignments={}

    for i in range(n_islands):
        assignments[i]={
            'local_model_type':{
                'classification':random.choice(local_model_types['classification']),
                'regression':random.choice(local_model_types['regression'])
            },
            'meta_model_type':{
                'classification':random.choice(meta_model_types['classification']),
                'regression':random.choice(meta_model_types['regression'])
            },
            'fetching_policy':random.choice(fetching_policies),
        }

    random.seed()

    return assignments
    

if __name__=='__main__':

    local_model_types={
        'classification':[
            RandomForestClassifier,
            LogisticRegression,
            DecisionTreeClassifier,
            GradientBoostingClassifier,
            DummyClassifier,
        ],
        'regression':[
            RandomForestRegressor,
            LinearRegression,
            DecisionTreeRegressor,
            GradientBoostingRegressor,
            DummyRegressor
        ]
    }

    meta_model_types={
        'classification':[
            RandomForestClassifier,
            LogisticRegression
        ],
        'regression':[
            RandomForestRegressor,
            LinearRegression
        ]
    }

    

    fetching_policies=[
        fetching_strategies.by_contribution_score,
        fetching_strategies.by_contribution_score_plain,
        fetching_strategies.by_random,
        fetching_strategies.by_trained_on_n,
        fetching_strategies.by_same_model_type,
        fetching_strategies.by_different_model_type,
        fetching_strategies.by_max_no_of_models
    ]

    datasets_=[
        # datasets.CoverTypeDataset(fname='../datasets/covtype.csv'),
        # datasets.VehicleLoanDefaultDataset(fname='../datasets/car_loan_default.csv'), # 5
        datasets.VehicleDataset(fname='../datasets/vehicles.csv'),
        datasets.CensusIncome(fname='../datasets/adult.data'), # 2
        # datasets.CreditCardDefaultDataset(fname='../datasets/credit_card_default.csv'), # 3
        # datasets.HotelReservationsDataset(fname='../datasets/hotel_reservations.csv'),  # 4
        

        # datasets.BlackFridayDataset(fname='../datasets/black_friday.csv'), #1
        # datasets.FlightPriceDataset(fname='../datasets/flight_price.csv'), # 7
        # datasets.FriedDataset(fname='../datasets/fried_delve.csv'), #8
        # datasets.DiamondsDataset(fname='../datasets/diamonds.csv'), #8 
         # 3
    ]

    n_rounds=5
    reps_per_config=5
    
    ith_run=0

    # 50 clients : 1? alpha
    # 100: 10?
    for dataset,p_method,n_islands,poisoning in itertools.product(datasets_,['dirY','powN'],[10,50,100],[True,False]):

        if dataset.task=='regression' and p_method=='dirY':
            continue

        if dataset.task=='classification' and p_method=='powN':
            continue

        if p_method=='dirY':
            p_params=[2,20]
        
        else:
            p_params=[0.5,0.75,1]
        
        for p_param in p_params:

            for rep in range(reps_per_config):

                print(f'Run: {ith_run}/{len(datasets_*2*3*3*reps_per_config)} {dataset} {p_method} {p_param} {n_islands}')

                id_run_seed=f'{dataset}_{p_method}_{p_param}_{n_islands}'
                id_run=f'{id_run_seed}_{poisoning}_{rep}'

                assignments=assign_things(n_islands,local_model_types,meta_model_types,fetching_policies,random_state=id_run_seed)

                ok=False
                while not ok:
                    try:
                        run(
                            dataset=dataset,
                            assignments=assignments,
                            p_method=p_method,
                            p_param=p_param,
                            n_islands=n_islands,
                            n_rounds=n_rounds,
                            poisoning=False,
                            output_dir=f'outputs_/{id_run}'
                        )
                        ok=True

                    except Exception as e:
                        # Delete output files
                        shutil.rmtree(f'outputs/{id_run}',ignore_errors=True)
                        print(f'Error: {e}')
                        
                ith_run+=1
    
    print('Done!')
