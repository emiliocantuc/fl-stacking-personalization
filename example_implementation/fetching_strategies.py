import random
import numpy as np


def standard_sorted_election(island_name,max_bandwith,meta_data,list_of_sorted_islands):
    
    chosen=[]
    b=0
    for i in list_of_sorted_islands:

        # Skip own island's model
        if i==island_name: continue 

        # Check if constraint is violated
        i_bytes=meta_data[i]['bytes']
        if b+i_bytes>max_bandwith:
            continue

        # Add
        chosen.append(i)
        b+=i_bytes
    
    return chosen,b
    
def by_contribution_score(island,max_bandwith,meta_data,contribution_scores,score_type='page_rank'):

    islands=list(meta_data.keys())

    # Sort by contribution score
    s_islands=sorted(islands,key=lambda i:contribution_scores[score_type].get(i,0),reverse=True)

    # Choose
    chosen,bandwith=standard_sorted_election(island.name,max_bandwith,meta_data,s_islands)
    
    # Check
    m__= min(islands,key=lambda i:meta_data[i]['bytes'])
    assert len(chosen)>0, f'by contribution score: Empty which to fetch. max:{max_bandwith} min model size:{m__} {bandwith}'

    return chosen,bandwith

def by_contribution_score_plain(island,max_bandwith,meta_data,contribution_scores):
    return by_contribution_score(island,max_bandwith,meta_data,contribution_scores,score_type='plain')


def by_random(island,max_bandwith,meta_data,_):

    islands=list(meta_data.keys())

    # Shuffle randomly
    s_islands=islands
    random.shuffle(s_islands)

    # Choose
    chosen,bandwith=standard_sorted_election(island.name,max_bandwith,meta_data,s_islands)
    
    # Check
    m__= min(islands,key=lambda i:meta_data[i]['bytes'])
    assert len(chosen)>0, f'by random: Empty which to fetch. max:{max_bandwith} min model size:{m__} {bandwith}'

    return chosen,bandwith


def by_trained_on_n(island,max_bandwith,meta_data,_):

    islands=list(meta_data.keys())

    # Sort by dataset size
    s_islands=sorted(islands,key=lambda i:meta_data[i]['trained_on_n'],reverse=True)

    # Choose
    chosen,bandwith=standard_sorted_election(island.name,max_bandwith,meta_data,s_islands)
    
    # Check
    m__= min(islands,key=lambda i:meta_data[i]['bytes'])
    assert len(chosen)>0, f'by n: Empty which to fetch. max:{max_bandwith} min model size:{m__} {bandwith}'

    return chosen,bandwith

def by_max_no_of_models(island,max_bandwith,meta_data,_):

    islands=list(meta_data.keys())

    # Sort by dataset size
    s_islands=sorted(islands,key=lambda i:meta_data[i]['bytes'])

    # Choose
    chosen,bandwith=standard_sorted_election(island.name,max_bandwith,meta_data,s_islands)
    
    # Check
    m__= min(islands,key=lambda i:meta_data[i]['bytes'])
    assert len(chosen)>0, f'by max_no_of_models: Empty which to fetch. max:{max_bandwith} min model size:{m__} {bandwith}'

    return chosen,bandwith

def by_same_model_type(island,max_bandwith,meta_data,_):

    islands=list(meta_data.keys())

     # Place matching model types first
    own=island.local_model.__class__.__name__
    s_islands=sorted(islands,key=lambda i:meta_data[i]['model_type']==own,reverse=True)

    # Choose
    chosen,bandwith=standard_sorted_election(island.name,max_bandwith,meta_data,s_islands)
    
    # Check
    m__= min(islands,key=lambda i:meta_data[i]['bytes'])
    assert len(chosen)>0, f'by same model: Empty which to fetch. max:{max_bandwith} min model size:{m__} {bandwith}'

    return chosen,bandwith


def by_different_model_type(island,max_bandwith,meta_data,_):
    """
    Attempt to match own local model type
    """

    islands=list(meta_data.keys())

    # We first 
    added_model_types=set([])
    chosen=set([])
    bandwith=0
    for i in islands:

        if i==island.name: continue

        model_type=meta_data[i]['model_type']
        bytes=meta_data[i]['bytes']

        if model_type not in added_model_types and bandwith+bytes<=max_bandwith: 
            chosen.add(i)
            added_model_types.add(model_type)
            bandwith+=bytes

    # Add the rest randomly
    random.shuffle(islands)
    for i in islands:

        if i==island.name or i in chosen: continue

        model_type=meta_data[i]['model_type']
        bytes=meta_data[i]['bytes']

        if bandwith+bytes<=max_bandwith: 
            chosen.add(i)
            bandwith+=bytes
    assert len(chosen)>0, f'by_different_model_types: Empty which to fetch.'
    return list(chosen),bandwith