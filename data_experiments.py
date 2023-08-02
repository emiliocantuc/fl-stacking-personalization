from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold,StratifiedKFold

import utils,os,argparse


def run(partitions_path,model_type,metrics,output_dir,verbose_level):
    """
    Runs the experiment described in Sec 3.X TODO
    """

    # Get the partition file names
    fnames=[fname for fname in os.listdir(partitions_path) if '.pkl' in fname]

    # For every partition file
    for i_partition,fname in enumerate(fnames):

        # We get the string that uniquely identifies the partition
        id_str=fname.replace('.pkl','')

        utils.verbose_print(f'({i_partition+1}/{len(fnames)}) Partition: {id_str}',verbose_level,0)

        # And load the partition information (that includes the partition and the dataset)
        p_info=utils.loadPickeObj(partitions_path,fname)

        # Unpack the partition and dataset objects
        partition,dataset=p_info['partition'],p_info['dataset']()

        # Calculate each island's powerset. I.e. island -> powerset of other islands. 
        powersets=utils.get_powersets(partition.keys())

        # A k cross validation fold generator with 5 splits
        fold=StratifiedKFold(n_splits=5) if dataset.task=='classification' else KFold(n_splits=5)

        # Results per partition
        island_results={}

        # For every island and its powerset
        for i_island,(current_island,island_sets) in enumerate(powersets.items()):

            utils.verbose_print(f'({i_island+1}/{len(powersets)}) Current Island: {current_island}',verbose_level,0)

            # Results per current_island's island set
            results={}

            for i_set,island_set in enumerate(island_sets):
                
                # We pool the datasets of the islands in island_set   
                X,y=utils.join_partitions(partition,island_set)
                X=X.reset_index(drop=True)
                y=y.reset_index(drop=True)

                # Initialize the model
                m=model_type[dataset.task](n_jobs=-1,random_state=1)
                
                # metric -> list of scores for each fold
                cross={'test_'+m:[] for m in metrics[dataset.task]}
                
                # Perfom k cross validation - notice we do so on the local and not the pooled dataset
                for (_,ixTest) in fold.split(X=partition[current_island][0],y=partition[current_island][1]):
                    
                    # Get the test set 
                    xTestLocal=X.loc[ixTest]
                    yTestLocal=y.loc[ixTest]
                    
                    # Training set is obtained by removing test set from the pooled dataset
                    xTrain=X.drop(ixTest, axis=0)
                    yTrain=y.drop(ixTest, axis=0)

                    # Fit the model
                    m.fit(xTrain,yTrain)

                    # Score model on every metric specified
                    cross_i=utils.score(m,xTestLocal,yTestLocal,metrics[dataset.task])
                    for metric,score in cross_i.items():
                        cross[metric].append(score)
                
                if verbose_level>0: utils.format_cross(f'({i_set}/{len(island_sets)}) {island_set}',cross,len(y))

                results[island_set]=cross

            island_results[current_island]=utils.results_df({str(k):v for k,v in results.items()})

        utils.savePickeObj(island_results,output_dir,id_str+'.pkl')
        

if __name__=='__main__':

    # Parse passed arguments
    parser=argparse.ArgumentParser(
        prog='Data Experiments',
        description='Performs the experiments described in Sec. 3.1' # TODO confirm
    )
    parser.add_argument('--partitions_dir',required=True)
    parser.add_argument('--output_dir',default='test')
    parser.add_argument('-v','--verbose_level',choices=[0,1,2],type=int,default=1)
    args=vars(parser.parse_args())

    # Define what models to use depending on the task
    model_type={
        'classification':RandomForestClassifier,
        'regression':RandomForestRegressor
    }

    # Define what metrics to report depengind on the task 
    metrics={
        'regression':['r2','neg_mean_squared_error'],
        'classification':['balanced_accuracy','accuracy'] 
    }

    # Run the experiment
    run(args['partitions_dir'],model_type,metrics,args['output_dir'],args['verbose_level'])
        
    