
# Local imports
import partitions,utils,datasets

# Other
import os,inspect,itertools,argparse,sys,random
import numpy as np
import matplotlib
matplotlib.use('Agg') # Don't plot
import matplotlib.pyplot as plt

def partition_validator(partition,MIN_SAMPLES_PER_ISLAND,task):
    # Check that every island meets the minimum number of samples
    least=min(len(y) for _,(_,y) in partition.items())
    assert least>=MIN_SAMPLES_PER_ISLAND,f'Minimum # of samples/island not met ({least})'

    if task=='classification':
        for _,(_,y) in partition.items():
            assert y.nunique()>1,'Has a single class in y'
            assert y.value_counts().min()>200,'Least freq class must have at least 200 samples'


def power_n(dataset_classes,a_s,c_clients_iter,n_partitions,N_TRIES,IMG_PATH,PARTITION_PATH,MIN_SAMPLES_PER_ISLAND,bootstrap=False):

    for dataset_class in dataset_classes:
        dataset=dataset_class()
        print(dataset,dataset.task)
        X,y=dataset.X_y

        base_x_lim=None
        
        for a,c_clients in itertools.product(a_s,c_clients_iter):
            # Str that identifies the partition
            id_str_general=f'{dataset}_powN_a={a}_clients={c_clients}_'

            for ith_partition in range(n_partitions):
                id_str=id_str_general+str(ith_partition)
            
                # Make the partition
                skip=True
                n_tries=N_TRIES
                while n_tries>0:
                    try:
                        partition_ixs=partitions.power_partition_n(y,c_clients,a,bootstrap=bootstrap)
                        partition=partitions.slice_partitions_from_X_y(partition_ixs,X,y)
                        partition_validator(partition,MIN_SAMPLES_PER_ISLAND,dataset.task)
                        skip=False
                        break
                    except AssertionError as e:
                        if n_tries>0:
                            n_tries-=1
                            
                if skip:
                    print(f'Minumum # of samples not met. Skipping {id_str}')
                    raise Exception(f'Minumum # of samples not met for {id_str}')
                
                # Save partition visualization
                xlims=utils.visualize_partition_n(
                    partition,
                    f'{dataset} dataset partitioned by pow_n w/a = {a}',
                    False,
                    f'{IMG_PATH}/{id_str}.png',
                    sort=True,
                    xlims=base_x_lim
                ) 
                if a==a_s[0]:
                    base_x_lim=xlims   

                
                # Save partition with relevant info
                with_info={
                    'partition':partition,
                    'dataset':dataset.__class__,
                    'partitioned_with':'power_partition_n',
                    'params':{
                        'a':a,
                        'c_clients':c_clients
                    }
                }
                utils.savePickeObj(with_info,PARTITION_PATH,f'{id_str}.pkl')

def dirichlet_y(dataset_classes,alphas,c_clients_iter,n_partitions,N_TRIES,IMG_PATH,PARTITION_PATH,MIN_SAMPLES_PER_ISLAND,bootstrap=False):

    for dataset_class in dataset_classes:
        dataset=dataset_class()
        if dataset.task!='classification': continue
        print(dataset,dataset.task)
        X,y=dataset.X_y
        
        for alpha,c_clients in itertools.product(alphas,c_clients_iter):
            # Str that identifies the partition
            id_str_general=f'{dataset}_dirY_alpha={alpha}_clients={c_clients}_'

            for ith_partition in range(n_partitions):
                id_str=id_str_general+str(ith_partition)
            
                # Make the partition
                skip=True
                n_tries=N_TRIES
                while n_tries>0:
                    try:
                        if bootstrap:
                            partition_ixs=partitions.dirichlet_bootstrap_partition(y,c_clients,alpha)
                        else:
                            partition_ixs=partitions.dirichlet_partition(y,c_clients,alpha)

                        partition=partitions.slice_partitions_from_X_y(partition_ixs,X,y)
                        partition_validator(partition,MIN_SAMPLES_PER_ISLAND,dataset.task)
                        skip=False
                        break
                    except AssertionError as e:
                        if n_tries>0:
                            n_tries-=1
                            
                if skip:
                    print(f'Minumum # of samples not met. Skipping {id_str}')
                    raise Exception(f'Minumum # of samples not met for {id_str}')
                
                # Save partition visualization
                utils.visualize_partition(
                    partition,
                    labels=None,
                    title=f'{dataset} dataset partitioned by dir_y w/alpha = {alpha}',
                    legend=False,
                    verbose=False,
                    fn=f'{IMG_PATH}/{id_str}.png'
                )    
                
                # Save partition with relevant info
                with_info={
                    'partition':partition,
                    'dataset':dataset.__class__,
                    'partitioned_with':'dirichlet_partition',
                    'params':{
                        'alpha':alpha,
                        'c_clients':c_clients
                    }
                }
                utils.savePickeObj(with_info,PARTITION_PATH,f'{id_str}.pkl')


def natural(dataset_classes,IMG_PATH,PARTITION_PATH,MIN_SAMPLES_PER_ISLAND):

    for dataset_class in dataset_classes:
        
        dataset=dataset_class()
        if not hasattr(dataset,'partition_on_column'): continue
        print(dataset,dataset.task)
        
        # Str that identifies the partition
        id_str=f'{dataset}_natural_partition'

        # Make the partition
        partition=dataset.natural_partition
        partition_validator(partition,MIN_SAMPLES_PER_ISLAND,dataset.task)

        # Save partition visualization
        if dataset.task=='classification':
            utils.visualize_partition(
                partition,
                labels=None,
                title=f'{dataset} dataset natural partition',
                legend=False,
                verbose=False,
                fn=f'{IMG_PATH}/{id_str}.png'
            )
        elif dataset.task=='regression':    
            utils.visualize_regression_partition(
                partition,
                title=f'{dataset} dataset natural partition',
                fn=f'{IMG_PATH}/{id_str}.png'
            )
            utils.visualize_partition_n(
                partition,
                title=f'{dataset} dataset natural partition',
                fn=f'{IMG_PATH}/{id_str}_n.png'
            )
            

        # Save partition with relevant info
        with_info={
            'partition':partition,
            'dataset':dataset.__class__,
            'partitioned_with':'natural_partition',
            'params':{}
        }
        utils.savePickeObj(with_info,PARTITION_PATH,f'{id_str}.pkl')


if __name__=='__main__':
    # Parse arguments
    parser=argparse.ArgumentParser(
        prog='Partition creator and loader',
        description='Creates and saves the partitions used in the experiments'
    )
    parser.add_argument('-e','--experiment',choices=['data','model','mult_rounds','mult_models','bootstrap'],default='data')
    parser.add_argument('-o','--output_dir',default='test')
    parser.add_argument('-s','--seed',type=int,default=7)
    parser.add_argument('-n','--n_per_partition',default=5,type=int)
    args=vars(parser.parse_args())
    
    experiment=args['experiment']

    # CONSTANTS

    # The minimum number of examples an island can have in the generated partitions.
    MIN_SAMPLES_PER_ISLAND=500

    # Number of times a stocastic partition can be attemp to meet MIN_SAMPLES_PER_ISLANDS before being skipped.
    N_TRIES=500

    # The datasets with which to perform partitions
    dataset_classes=[
    
        datasets.CoverTypeDataset,
        datasets.VehicleLoanDefaultDataset, # 5
        # datasets.CensusIncome, # 2
        # datasets.CreditCardDefaultDataset, # 3
        # datasets.HotelReservationsDataset,  # 4
        

        datasets.BlackFridayDataset, #1
        datasets.FlightPriceDataset, # 7
        # datasets.FriedDataset, #8
        # datasets.DiamondsDataset, #8 
        # datasets.VehicleDataset, # 3

        # [j for i,j in inspect.getmembers(sys.modules['datasets'], inspect.isclass) if i!="Dataset"]
    ]

    # CONSTANTS THAT CHANGE FOR DATA/MODEL EXPERIMENTS

    # The no. of clients to among which divide the datasets.
    n_islands={
        'data':[5],
        'model': [5,10],
        'mult_rounds': [30],
        'mult_models':[10],
        'bootstrap':[50] #10,50,100
    }

    # Power law exponent parameter for PowN partitining.
    pow_as={
        'data':[0.1,0.25,0.5,0.75,1],
        'model': [0.5,0.75,1],
        'mult_rounds': [0.5,1],
        'mult_models':[0.5,1],
        'bootstrap':[0.5,0.75,1]
    }
    
    # The alpha concentration parameter for DirY partitioning.
    dir_alphas={
        'data':[0.5,0.75,1,10],
        'model': [0.5,0.75,1,10],
        'mult_rounds': [0.5,10],
        'mult_models':[0.5,10],
        'bootstrap':[0.5,10,100]

    }

    # DEFINE AND MAKE SURE DIRECTORIES EXIST
    PARTITION_PATH=args["output_dir"]
    IMG_PATH=f'{args["output_dir"]}/img'
    os.makedirs(IMG_PATH,exist_ok=True)

    # SET SEEDS FOR REPRODUCIBILITY
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    n_partitions=args['n_per_partition']

    power_n(
        dataset_classes,
        pow_as[experiment],
        n_islands[experiment],
        n_partitions,
        N_TRIES,
        IMG_PATH,
        PARTITION_PATH,
        MIN_SAMPLES_PER_ISLAND,
        bootstrap=experiment=='bootstrap'
    )

    dirichlet_y(
        dataset_classes,
        dir_alphas[experiment],
        n_islands[experiment],
        n_partitions,
        N_TRIES,
        IMG_PATH,
        PARTITION_PATH,
        MIN_SAMPLES_PER_ISLAND,
        bootstrap=experiment=='bootstrap'
    )

    if args['experiment'] not in ['mult_rounds','mult_models','bootstrap']:

        natural(
            dataset_classes,
            IMG_PATH,
            PARTITION_PATH,
            MIN_SAMPLES_PER_ISLAND
        )
    

    # Used to gen data_experiments:
    # python3 partition_loader.py -e data -o partitions/data_experiments -s 7 -n 5

    # Used to gen model_experiments:
    # python3 partition_loader.py -e model -o partitions/model_experiments -s 7 -n 10

    # Mult. rounds
    # python3 partition_loader.py -e mult_rounds -o partitions/mult_rounds -n 1




