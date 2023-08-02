
# Includes partitioning functions

import numpy as np

# Partition on labels

def slice_partitions_from_df(df,partition):
    out={}
    for client,ix in partition.items():
        out[client]=df.iloc[ix]
        out[client]=out[client].reset_index(drop=True)
    return out

def slice_partitions_from_dataset(dataset,partition):
    return slice_partitions_from_X_y(partition,dataset.X,dataset.y)

def slice_partitions_from_X_y(partition,X,y):
    out={}
    for client,ix in partition.items():
        out[client]=X.iloc[ix],y.iloc[ix]
    return out


def samples_per_class_dirichlet(n_classes,c_clients,alpha,n=None,debug=False):
    """
    
    Returns the number of samples the nth client must sample from each class
    according to the Dirichlet distribution with concentration parameter alpha.

    Unless the proportion of samples the i-th client must draw is specified in n[i], 
    n is set such that the number of samples are distributed uniformly
    (equivalent to setting n[i] = y.size / c_clients).

    Parameters
    ----------
    y : numpy array
        The numpy array of labels to be partitioned, assumed to be of integers 0 to
        # of classes -1.

    c_clients : int
        The number of clients or number of segments to partition y among.

    alpha : float
        Dirichlet sampling's concentration parameter (0 < alpha <= 1)

    n : numpy array or None, optional
        n[i] specifies the *number* of elements of y that the i-th client must sample.

    debug : boolean, optional
        Whether to perform extra checks (which can be slow) 
    
    Returns
    -------
    A numpy array of shape(c,k) matrix where A[i,j] denotes
    the amount of instances of class j the client i must draw.

    """
    assert alpha>0
    
    # Sample from Dirichelts Dist.
    # proportions[i][j] indicates the proportion of class j that client i must draw
    proportions=np.random.dirichlet(alpha*np.ones(n_classes),c_clients)
    
    # Multiply by n and cast as int
    for client,client_i_n in enumerate(n):
        proportions[client,:]*=client_i_n

    out=proportions.astype('int')
    
    # Correct errors caused by truncation
    missing_by_client=n-out.sum(axis=1)
    assert all(missing_by_client>=0),'Possible overflow'
    for client,n_missed_by_client in enumerate(missing_by_client):
        where_to_add=np.random.choice(n_classes,size=n_missed_by_client)
        np.add.at(out[client,:],where_to_add,1)
    
    if debug:
        # Total of output must equal total of input
        assert out.sum()==sum(n)
    
    return out

def dirichlet_partition(y,c_clients,alpha,n=None,debug=False,bootstrap=False):
    """
    Randomly partitions an array of labels y into a # c_clients of clients
    according to Dirichelet sampling with concentration parameter alpha.

    Unless the proportion of samples the i-th client must draw is specified in n[i], 
    n is set such that the number of samples are distributed uniformly
    (equivalent to setting n[i] = y.size / c_clients).

    To guarantee that every 0 < alpha <= 1 can be met the total number of samples that can
    be sampled is set to the number of labels with the minimum frequency in y
    ('n_max_all_alphas' in the code). This may be too conservative but it's the
    easiest way to guarantee that samples_per_class_dirichlet doesn't over-assign a class
    (returning a matrix with a sum of column 0 that is greater than the # of instances of
    class 0, for example).

    alpha --> 0 implies very uneven sampling while alpha --> inf approaches uniform sampling.  

    Parameters
    ----------
    y : numpy array
        The numpy array of labels to be partitioned, assumed to be of integers 0 to
        # of classes -1.

    c_clients : int
        The number of clients or number of segments to partition y among.

    alpha : float
        Dirichlet sampling's concentration parameter (0 < alpha <= 1)

    n : numpy array or None, optional
        n[i] specifies the proportion of elements of y that the i-th client must sample.
        Therefore 

    debug : boolean, optional
        Whether to perform extra checks (which can be slow) 
        
    Returns
    -------
    The partition as a dictionary: client id (int) -> array of indices (np.array).

    """
    assert isinstance(c_clients,int) and c_clients>0
    assert alpha>0

    # The number of classes if y is assumed to be pandas' categorical codes.
    classes,counts_y=np.unique(y,return_counts=True)
    n_classes=len(counts_y)

    # Max n such that all alphas can be guaranteed
    # The worst case that can occur is if one client is assigned
    n_max_all_alphas=counts_y.min()

    # If n is None we distribute equally
    if n is None:
        n=[n_max_all_alphas//c_clients]*c_clients

    else:
        assert all([0<=i<=1 for i in n]) #sum(n)==1 and 
        n=[int(n_max_all_alphas*n_prop) for n_prop in n]    
    
    # Given how many examples each client must sample from each class
    how_many=samples_per_class_dirichlet(
        n_classes=n_classes,
        c_clients=c_clients,
        alpha=alpha,
        n=n,
        debug=debug
    )

    # Assert we have enough instances from each class
    assert all(counts_y-how_many.sum(axis=0)>=0),'Not enough instances from each class to compy with how_many'

    # Find indices for each class and shuffle them
    wheres={}
    for class_i in classes:
        w=np.where(y==class_i)[0]
        np.random.shuffle(w)
        wheres[class_i]=list(w)

    # Client -> list of indices
    partition={c:[] for c in range(c_clients)}

    if not bootstrap:

        # For every class
        for i,class_i in enumerate(classes):

            # We distribute the corresponding indices to the clients
            prev=0
            for client,ni in enumerate(how_many[:,i]):
                partition[client].extend(wheres[class_i][prev:prev+ni])
                added=len(wheres[class_i][prev:prev+ni])

                if debug:
                    assert added==ni,f'added: {added} ni:{ni}'

                prev+=ni 

    return partition



def k_clases_partition(y,k_classes,c_clients,k):
    """
    Each client is assigned k random classes from (0,k_classes-1).
    Then each class is divided up evenly among those assigned to it.

    Input:
        y: 1-D array of labels to partition.
        k_classes: int denoting the number of classes in a.
        c_clients: int denoting the number of clients to partition among.
        k: int denoting the limit of classes each client can sample from. 
    
    Notes:
        - k must be such that (1 <= k <= k_classes). Lower values of k imply non-iid partitioning. 
        - Partitioning does not guarentee that all indices of a are assigned to a client.

    Returns:
        partition: a dictionary {client:array of indices}.

    """
    assert 1<=k<=k_classes

    # Client -> classes they can sample from
    client_classes={c:np.random.choice(k_classes,size=k,replace=False) for c in range(c_clients)}
    
    # Class -> clients that can sample from it
    classes_clients={k:[] for k in range(0,k_classes)}
    for c,classes in client_classes.items():
        for k in classes:
            classes_clients[k].append(c)
    
    # Class -> shuffled indices of its locations in a
    wheres={}
    for l in range(0,k_classes):
        w=np.where(y==l)[0]
        np.random.shuffle(w)
        wheres[l]=list(w)

    # Client -> indices    
    partition={c:[] for c in range(c_clients)}
    
    # Splice shuffled class indices according to n
    for k in range(0,k_classes):
        if len(classes_clients[k])==0: continue
        n=len(wheres[k])//len(classes_clients[k])
        i=0
        for c in classes_clients[k]:
            partition[c].extend(wheres[k][i:i+n])
            i+=n
    
    return partition

def dirichlet_partition_n(y,c_clients,alpha):
    p=np.random.dirichlet(alpha*np.ones(c_clients),1)[0]
    n=(p*y.size).astype('int')
    return random_partition(y,c_clients,n)


def power_n(y,c_clients,a):
    # p=np.random.power(a,size=c_clients)
    # p=p/p.sum()
    bins = np.linspace(0, 1, c_clients+1)
    data = np.random.power(a=a,size=len(bins)*10000)
    counts,_=np.histogram(data,bins)
    p=counts/counts.sum()
    n=(p*y.size).astype('int')
    return n


def power_partition_n(y,c_clients,a,bootstrap=False,min_n=None):
    """
    Partition on size of n. Will follow power law with
    coeficient a in (0,1]. a = 1 is equivalent to the
    uniform distribution while a --> 0 concentrates the mass
    on a single client.
    """
    n=power_n(y,c_clients,a)
    if bootstrap:
        return random_bootstrap_partition(y,c_clients,n+min_n)
    else:
        return random_partition(y,c_clients,n)

def random_partition(y,c_clients,n=None):
    """
    Randomly partitions a into a # c_clients of clients.
    Shuffles np.random.arange(a.size) and slices it according to n.

    Input:
        y: 1-D array of labels to partition.
        c_clients: int denoting the number of clients to partition among.
        n: array or None, optional. Denotes the number of labels to be assined to each client. Must sum to a.size.

    Returns:
        partition: a dictionary {client:array of indices}.
    """
    # If no size distribution passed assume equal
    if n is None:
        #assert a.size%c_clients==0
        n=(np.ones(c_clients)*(y.size//c_clients)).astype('int')
    
    elif isinstance(n,list):
        n=np.array(n)
    
    
    # We're only interested in the indices
    ix=np.arange(y.size)
    
    # Shuffle the indices
    np.random.shuffle(ix)
    
    # The partition to output. Client -> [indices].
    partition={}
    
    # Splice shuffled a according to n
    i=0
    for c in range(c_clients):
        partition[c]=ix[i:i+n[c]]
        i+=n[c]
        
    # Sanity check
    
    return partition


def random_bootstrap_partition(y,c_clients,n=None):
    """
    Randomly partitions a into a # c_clients of clients.
    Shuffles np.random.arange(a.size) and slices it according to n.

    Input:
        y: 1-D array of labels to partition.
        c_clients: int denoting the number of clients to partition among.
        n: array or None, optional. Denotes the number of labels to be assined to each client. Must sum to a.size.

    Returns:
        partition: a dictionary {client:array of indices}.
    """
    # If no size distribution passed assume equal
    if n is None:
        #assert a.size%c_clients==0
        n=(np.ones(c_clients)*(y.size//c_clients)).astype('int')
    
    elif isinstance(n,list):
        n=np.array(n)
    
    
    # We're only interested in the indices
    ix=np.arange(y.size)
    

    # The partition to output. Client -> [indices].
    partition={}
    
    # Splice shuffled a according to n
    for c in range(c_clients):
        partition[c]=np.random.choice(ix,size=n[c],replace=False)
        
    return partition

def dirichlet_bootstrap_partition(y,c_clients,alpha,debug=False):

    # Get info on y
    classes,counts_y=np.unique(y,return_counts=True)
    n_classes=len(classes)

    # Get y distributions
    y_dists=np.random.dirichlet(alpha*np.ones(n_classes),c_clients)
    
    # Calculate n - the number of samples per client
    # We will use the smallest class frecuency to be able to guarante any
    # class imbalance
    n=counts_y.min()
    
    if debug:
        print(f'n: {n}')
        
    # Find indices for each class
    classes=y.unique()
    wheres={}
    for class_i in classes:
        w=np.where(y==class_i)[0]
        wheres[class_i]=list(w)
        
    # The partition to output. Client -> [indices].
    partition={}
    
    # For every client
    for c in range(c_clients):
        
        # For every class
        for i,class_i in enumerate(classes):
            
            # Randomly choose the proportion of n of the class
            n_i=int(n*y_dists[c][i])
            
            ixs_i=np.random.choice(wheres[class_i],size=n_i,replace=False)
        
            # Save in partition
            if c not in partition:
                partition[c]=[]
                
            partition[c].extend(ixs_i.tolist())
        
        # Save as numpy array and shuffle
        partition[c]=np.array(partition[c])
        np.random.shuffle(partition[c])
        
    return partition

    