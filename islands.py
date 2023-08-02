
import numpy as np
from betterVotingEstimators import VotingClassifier,VotingRegressor
from sklearn.ensemble import StackingClassifier,StackingRegressor
import utils
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression,Lasso
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.feature_selection import SelectFromModel

class Island:
    def __init__(self,name,task,local_estimator):
        self.name=name
        self.task=task
        self.local_estimator=local_estimator
        self.model_type=VotingClassifier if self.task=='classification' else VotingRegressor
        

    def fit_local(self,X,y):
        assert y.nunique()>1,'Only one class in y'
        # Call fit on the local estimator
        self.local_estimator.fit(X,y)

        # Save data dependent attributes
        self.n=len(y)

    def fit(self,otherIslands,X,y):
        # We assume that fit_local has been called on self and on each of
        # otherIslands.

        # Adds itself to list of islands 
        islands=[(self.name,self)]+otherIslands

        # Get the weights
        w=self.weights(islands,X,y)

        self.model=self.model_type(
            fitted_estimators=[(name,island.local_estimator) for name,island in islands],
            weights=w,
            n_jobs=-1
        )

        # "Fits" the voting model - does nothing because all estimators are already fitted
        self.model.fit()

        # Returns the weights used
        return {islands[i][0]:w_i for i,w_i in enumerate(w)}


    def weights(self,islands,X,y):
        n=len(islands)
        return [1/n]*n

    def __repr__(self):
        return f'<Island {self.name}>'


class LocalIsland(Island):
    """
    Asigns all weight to local island.
    """
    def weights(self, islands,X,y):
        return np.array([1 if name==self.name else 0 for name,_ in islands])
    
class RandomContIsland(Island):
    """
    Assigns random weights to islands.
    """
    def weights(self,islands,X,y):
        w=np.random.random(size=len(islands))
        return w/w.sum()

class RandomBinaryIsland(Island):
    """
    Chooses prop % of islands randomly and weighs them equally.
    """
    def __init__(self, name, task, local_estimator,prop=1.0):
        self.prop=prop
        super().__init__(name, task, local_estimator)

    def weights(self, islands,X,y):
        k=max(1,round(self.prop*len(islands)))
        ixs=np.random.choice(len(islands),size=k)
        return np.array([1/k if i in ixs else 0 for i in range(len(islands))])
    
class RandomBinaryNIsland(RandomBinaryIsland):
    """
    Chooses prop % of islands randomly with probability proportional to N
    and weighs them according to N.
    """

    def weights(self, islands,X,y):

        # Get probabilities from which to sample
        ps=np.array([island.n for _,island in islands])
        ps=ps/ps.sum()

        # Select
        k=max(1,round(self.prop*len(islands)))
        ixs=np.random.choice(len(islands),size=k,p=ps)

        # Get ns and normalize
        w=np.array([islands[i][1].n if i in ixs else 0 for i in range(len(islands))])
        w=w/w.sum()

        return w

    
class ByNIsland(Island):
    """
    Weighs islands by the size of their local training data.
    max_models can be passed to only give positive weights to the
    top max_models in size.
    """
    def __init__(self, name, task, local_estimator,prop=1.0):
        super().__init__(name, task, local_estimator)
        self.prop=prop

    def weights(self, islands,X,y):
        k=max(1,round(self.prop*len(islands)))
        if self.prop<1:
            selected=sorted(islands,key=lambda x:x[1].n,reverse=True)[:k]
        else:
            selected=islands

        ns=np.array([i[1].n if i in selected else 0 for i in islands])
        return ns/ns.sum()
    

class StackingLinearIsland(Island):

    def __init__(self, name, task, local_estimator,C=0.5,positive=False,prop=1.0):
        """
        C: regularization param
        """
        super().__init__(name, task, local_estimator)
        self.model_type=StackingClassifier if task=='classification' else StackingRegressor
        self.C=C
        self.positive=positive
        self.prop=prop

    def fit(self,otherIslands,X,y):
        # We assume that fit_local has been called on self and on each of
        # otherIslands.
        
        used_dummy=False
        if self.task=='classification' and y.nunique()==1:
            final_estimator=DummyClassifier()
            used_dummy=True
        
        elif self.task=='classification':
            final_estimator=LogisticRegression(
                C=self.C,
                penalty='l1',
                solver='liblinear',
                max_iter=2000
            )
        else:
            final_estimator=Lasso(
                positive=self.positive,
                alpha=1/self.C,
                max_iter=2000
            )


        # Adds itself to list of islands 
        islands=[(self.name,self)]+otherIslands
        base_models=[(island_name,island.local_estimator) for island_name,island in islands]

        self.model=self.model_type(
            estimators=base_models,
            final_estimator=final_estimator,
            cv='prefit',
            n_jobs=-1
        )

        self.model.fit(X,y)

        # For debugging purposes
        converged=self.model.final_estimator_.n_iter_<2000 if not used_dummy else True
        if not converged:
            print(f'DID NOT CONVERGE!! n= {len(y)}')


        # We get the weights
        w=self.model.final_estimator_.coef_ if not used_dummy else np.array([0]*len(islands))
        w=w[0] if w.ndim==2 else w
        w=np.abs(w) # Interpreting importace as absolute value of coefs
        w=w/w.sum() if w.sum()>0 else np.zeros(shape=w.shape)

        # Returns the weights used
        w={islands[i][0]:w_i for i,w_i in enumerate(w)}
        assert len(w)==len(islands)


        if self.prop<1 and not used_dummy:
            # The no. of model to include
            k=max(1,round(len(islands)*self.prop))

            # We select the top k models according to their feature importances
            selector=SelectFromModel(
                estimator=self.model.final_estimator_,
                prefit=True,
                max_features=k,
                threshold=-np.inf
            )
            selected_islands=[i for i,s in zip(islands,selector.get_support()) if s]
           
            # Refit the meta model using only the selected models
            base_models=[(island_name,island.local_estimator) for island_name,island in selected_islands]

            self.model=self.model_type(
                estimators=base_models,
                final_estimator=final_estimator,
                cv='prefit',
                n_jobs=-1
            )

            self.model.fit(X,y)

            # Set all weights as 0
            w={island_name:0 for (island_name,_) in islands}

            # Insert importances of selected ones
            s_w=self.model.final_estimator_.coef_
            s_w=s_w[0] if s_w.ndim==2 else s_w
            s_w=np.abs(s_w) # Interpreting importace as absolute value of coefs
            s_w=s_w/s_w.sum() if s_w.sum()>0 else np.zeros(shape=s_w.shape)
            for (name,_),imp in zip(selected_islands,s_w):
                w[name]=imp
        
        return w
    

class StackingForestIsland(Island):

    def __init__(self, name, task, local_estimator,prop=1.0):
        super().__init__(name, task, local_estimator)
        self.model_type=StackingClassifier if task=='classification' else StackingRegressor
        print(task,self.model_type)
        self.prop=prop

    def fit(self,otherIslands,X,y):
        # We assume that fit_local has been called on self and on each of
        # otherIslands.

        used_dummy=False
        if self.task=='classification' and y.nunique()==1:
            final_estimator=DummyClassifier()
            used_dummy=True

        elif self.task=='classification':
            final_estimator=RandomForestClassifier()

        else:
            final_estimator=RandomForestRegressor()


        # Adds itself to list of islands 
        islands=[(self.name,self)]+otherIslands
        base_models=[(island_name,island.local_estimator) for island_name,island in islands]

        self.model=self.model_type(
            estimators=base_models,
            final_estimator=final_estimator,
            cv='prefit',
            n_jobs=-1
        )

        self.model.fit(X,y)

        if not used_dummy:
            w={island_name:w_i for (island_name,_),w_i in zip(islands,self.model.final_estimator_.feature_importances_)}
        
        else:
            w={island_name:0.0 for (island_name,_) in islands}

        if self.prop<1 and not used_dummy:
            # The no. of model to include
            k=max(1,round(len(islands)*self.prop))

            # We select the top k models according to their feature importances
            selector=SelectFromModel(
                estimator=self.model.final_estimator_,
                prefit=True,
                max_features=k,
                threshold=-np.inf
            )
            selected_islands=[i for i,s in zip(islands,selector.get_support()) if s]
           
            # Refit the meta model using only the selected models
            base_models=[(island_name,island.local_estimator) for island_name,island in selected_islands]

            self.model=self.model_type(
                estimators=base_models,
                final_estimator=final_estimator,
                cv='prefit',
                n_jobs=-1
            )

            self.model.fit(X,y)

            # Set all weights as 0
            w={island_name:0 for (island_name,_) in islands}


            # Insert importances of selected ones
            for (name,_),imp in zip(selected_islands,self.model.final_estimator_.feature_importances_):
                w[name]=imp

        return w
    

class byNPriorLinearIsland(StackingLinearIsland):

    def fit(self, otherIslands, X, y):

        islands=[(self.name,self)]+otherIslands

        # Get weights according to stacked model
        s_weights=super().fit(otherIslands, X, y)
        s_weights=np.array([s_weights[island_name] for island_name,_ in islands])
        s_score=self.model.score(X,y)
        
        # Get weights by N
        n_weights=ByNIsland('',self.task,self.local_estimator).weights(islands,X,y)


        w=((s_weights*s_score)+((1-s_score)*n_weights))

        self.model_type=VotingClassifier if self.task=='classification' else VotingRegressor

        self.model=self.model_type(
            fitted_estimators=[(name,island.local_estimator) for name,island in islands],
            weights=w,
            n_jobs=-1
        )

        # "Fits" the voting model - does nothing because all estimators are already fitted
        self.model.fit()

        w={island_name:w_i for (island_name,_),w_i in zip(islands,w)}

        return w
    
class byNPriorForestIsland(StackingForestIsland):

    def fit(self, otherIslands, X, y):

        islands=[(self.name,self)]+otherIslands

        # Get weights according to stacked model
        s_weights=super().fit(otherIslands, X, y)
        s_weights=np.array([s_weights[island_name] for island_name,_ in islands])
        s_score=self.model.score(X,y)
        
        # Get weights by N
        n_weights=ByNIsland('',self.task,self.local_estimator).weights(islands,X,y)


        w=((s_weights*s_score)+((1-s_score)*n_weights))

        self.model_type=VotingClassifier if self.task=='classification' else VotingRegressor

        self.model=self.model_type(
            fitted_estimators=[(name,island.local_estimator) for name,island in islands],
            weights=w,
            n_jobs=-1
        )

        # "Fits" the voting model - does nothing because all estimators are already fitted
        self.model.fit()

        w={island_name:w_i for (island_name,_),w_i in zip(islands,w)}

        return w
    

class MultipleRoundIsland():

    def __init__(self,island):
        self.island=island

    def filter_islands(self,otherIslands,round,max_models_per_round,n_rounds):
        return otherIslands
    
    # Returns scores pe
    def run(self,otherIslands,X_meta_train,y_meta_train,X_meta_test,
                        y_meta_test,n_rounds,max_models_per_round,score_metrics,verbose_level=2):
        
        print(max_models_per_round)
        
        # List of {island -> weight given by self.island} per round
        weights_per_round=[]

        # List of dicts metric -> score
        scores_per_round=[]

        filtered_islands_per_round=[]

        for round in range(n_rounds):

            utils.verbose_print(f'Island: {self.island.name} Round: {round+1}/{n_rounds}',verbose_level,2)

            # Filter otherIslands based on criteria
            filtered_islands=self.filter_islands(otherIslands,round,max_models_per_round,n_rounds)
            assert len(filtered_islands)==max_models_per_round or self.__class__.__name__ in ['MultipleRoundIsland','MultipleRoundLocalIsland']
            filtered_islands_per_round.append([name for name,_ in filtered_islands])

            # Fit the island
            w=self.island.fit(otherIslands=filtered_islands,X=X_meta_train,y=y_meta_train)
            for name,_ in otherIslands:
                if name not in w:
                    w[name]=0
            weights_per_round.append(w)

            # Save for use by filter algos
            self.weights_prev_round=w

            # Score the island
            # And scores it on its own test data
            scores={}
            s=utils.score(self.island.model,X_meta_test,y_meta_test,score_metrics)
            for score_name,value in s.items():
                if score_name not in scores:
                    scores[score_name]=[]
                scores[score_name].append(value)
            
            scores_per_round.append(scores)
 
        return weights_per_round,scores_per_round,filtered_islands_per_round
    
    def __repr__(self):
        return f'<Mult Round Island {self.name}>'
    
class MultipleRoundLocalIsland(MultipleRoundIsland):

    def filter_islands(self, otherIslands, round, max_models_per_round, n_rounds):
        return []
    

class MultipleRoundRandomIsland(MultipleRoundIsland):

    def filter_islands(self, otherIslands, round, max_models_per_round, n_rounds):
        a=list(range(len(otherIslands)))
        # print(max_models_per_round,a,otherIslands)
        chosen_ixs=np.random.choice(a,size=max_models_per_round,replace=False)
        return [otherIslands[i] for i in chosen_ixs]

class MultipleRoundIslandByNIsland(MultipleRoundIsland):

    def filter_islands(self, otherIslands, round, max_models_per_round, n_rounds):

        return sorted(otherIslands,key=lambda i:i[1].n,reverse=True)[:max_models_per_round]
    

class MultipleRoundEpsGreedyIsland(MultipleRoundIsland):
    """
    Starts with the max_models_per_round with the highest n.
    In each subsecuent round, it requests eps % of max_models_per_round
    randomly with (1-eps) % with the highest importance.
    """

    def __init__(self, island,eps):
        super().__init__(island)
        self.eps=eps

    def filter_islands(self, otherIslands, round, max_models_per_round, n_rounds):

        # Explored set
        if round==0 or round>0 and all(w_i==0 for w_i in self.weights_prev_round.values()):
            if round==0:
                self.explored=[]
                # Island -> weight assigned to it
                self.island_weights={}

            # Return by Ns logic
            return sorted(otherIslands,key=lambda i:i[1].n,reverse=True)[:max_models_per_round]

        else:
            # Update from weights from last round
            # print('prev weights',self.weights_prev_round)
            for island,w in self.weights_prev_round.items():
                if w>0 and island!=self.island.name:
                    self.explored.append(island)
                    self.island_weights[island]=w 

        # Choose eps * len(otherIslands) randomly
        n_random=min(max_models_per_round,int(max_models_per_round*self.eps))
        # print(f'n random {n_random} n other islands {len(otherIslands)}')
        a=[i for i in range(len(otherIslands)) if otherIslands[i][0] not in self.explored]
        n_random=min(n_random,len(a))
        chosen_ixs=np.random.choice(a,size=n_random,replace=False)
        chosen_random=[otherIslands[i] for i in chosen_ixs]

        # Best?
        
        n_best=max_models_per_round-n_random
        # print('best',n_best)
        # print('island weights',self.island_weights)
        assert n_best+n_random==max_models_per_round
        chosen_best_ixs=sorted(self.island_weights,key=self.island_weights.get,reverse=True)[:n_best]
        # print(chosen_best_ixs)
        chosen_best=[i for i in otherIslands if i[0] in chosen_best_ixs]
        # print(chosen_best,n_best)
        # print(otherIslands)

        chosen=chosen_random+chosen_best
        assert len(chosen)<=max_models_per_round,f'# chosen random: {len(chosen_random)} best: {len(chosen_best)}'
        # print(chosen)
        return chosen
    

    




