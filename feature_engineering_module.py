## This file defines the methods for feature engineering:
## > loading data
## > adding rolling window features by group (dt=number of minutes)
## > adding frame-wide aggregates by group (dt=None)
## > defining entity embeddings for categorical features
##
## TODO:
## 1 - recode the count feature using .groupby().transform(len)
## 2 - take care of var type in the Manager. parquet 1.0 returns int64 for uint32
## 3 - keep the raw index as a range somewhere in the Manager
## 4- Make sure each function generates only one variable and returns its name


## For each engine, create the subgraph of Set of all available features combination, and inclusion only.
## Remove arrows that are decreasing in score.
## try graph-tool if networkX isn't good enough

import pandas as pd
import networkx as nx
import random
import yaml
import inspect
from itertools import chain

class FeatureCrawler(object):
    # This class is a graph crawler for feature engineering. It maintains a graph of all sets of features,
    # their respective inclusions and scores. It incrementally learns the performance of feature sets to find the best model.
    # It can expand the feature graph at any time with new features and automatically removes feature combinations that will not improve the model
    # An option to prune all non optimal branches is provided.

    ''' 

    attributes:
        status_ : current proportion of scored nodes/ percentage crawled
        leaves_ : {current leaves : their score}
    
    '''
    null_set=set()
    unit_graph=nx.DiGraph()
    unit_graph.add_edge('s','t')
    nx.set_node_attributes(unit_graph,{'s':null_set,'t':null_set},'feats')
    nx.set_node_attributes(unit_graph,{'s':True,'t':False},'score')
    
    def __init__(self,file_path,crawler_file,incr_score=True):
        # initiate with a baseline and score order (incr, decr)
        self.status_=0
        self.leaves_={}
        self.file_=file_path+crawler_file+'.yaml'
        self.gt=incr_score # defines whether better scores are defined with the greater than or lesser than operator.
        
        if os.path.isfile(self.file):
            print('loading graph from file')
            self.G=nx.read_yaml(File)
            self.update_status()
        else:
            print('creating empty feature graph')
            self.G=nx.DiGraph()
            self.G.add_node(0,feats=self.null_set,score=0)

    def is_better(self,a,b):
        return a>b if self.gt else a<b

    def update_status(self):
        #getting the proportion of non empty feature sets evaluated
        self.status_=mean([y is not None for x,y in nx.get_node_attributes(self.G,'score').items() if x!=0])
        self.leaves_={n:self.G.node[n]['score'] for n,deg in self.G.out_degree if deg==0 and self.G.node[n]['score'] is not None}
        nx.write_yaml(self.G,File)
        return self

    def update_features(self,feature_list):
        ''' Takes a list of available features and updates the graph with the ones that weren't included yet
            Args- list of features
        '''
        print('adding features to graph')
        current_features=chain(*[features for n,features in nx.get_node_atributes(self.G,'feats') if len(features)==1])
        new_features=[x for x in feature_list if x not in current_features]
        for x in new_features:
            self.unit_graph.node['t']['feats']=set(x)
            self.G=nx.algorithms.operators.cartesian_product(self.G,self.unit_graph)
            self.unit_graph.node['t']['feats']=null_set
            
            new_labels={x:y[0]|y[1] for x,y in nx.get_node_attributes(self.G,'feats').items()}
            nx.set_node_attributes(self.G,new_labels,'feats')
            
            new_scores={x:y[0] if y[1] else None for x,y in nx.get_node_attributes(self.G,'score').items()}
            nx.set_node_attributes(self.G,new_scores,'score')
            
            self.G=nx.convert_node_labels_to_integers(self.G)

        self.update_status()
        return self

    def get_unscored_features(self):
        if self.status_==1:
            print('All features were estimated, returning a random leaf')
            choices=random.choice(self.leaves_)
        # the graph G may already be topologically ordered from 0 by the renaming of the nodes
        else:
            unscored_subgraph=nx.subgraph([n for n,score in nx.get_node_attrubutes(self.G,'score') if score is None])
            choices=[n for n,deg in unscored_subgraph.in_degree if deg==0]
        return make_node_dict(random.choice(choices))

    def make_node_dict(self,node):
        node_dict={'node':node,**self.G.node[node]}
        return node_dict

    def record_score(self,node_dict):
        # checks that the score that is about to be recorded corresponds to the given features
        # adds it to the graph and removes node and all descendants if  the score wasn't improved from all its predecessors.
        node=node_dict.pop('node')
        if self.G.node[node]['feats']!=node_dict.pop('feats'):
             print('The features do not correspond to the given node')
        else:
            self.G.node[node]['score']=node_dict.pop('score')
            if any([self.is_better(self.G.node[x]['score'],self.G.node[node]['score']) for x in nx.ancestors(self.G,node)]):
                self.G.remove_nodes_from({node}|nx.descendants(self.G,node))
        self.update_status()
        return self
    
    def prune(self):
        # Recursively removes all scored leaves that are not global max
        # to use only if needing the single best model, not good for bagging/blending.
        while len({score for leaf,score in self.leaves_.items()})>1:
            max_score=max(nx.get_node_attributes(self.G,'score').items(), key=operator.itemgetter(1))[1]
            suboptimal_features={feats for feats,score in self.leaves_.items() if self.is_better(max_score,score)}
            suboptimal_leaves=[n for n,feat in nx.get_node_attributes(self.G,'feats') if feats in suboptimal_features]
            self.G.remove_nodes_from(suboptimal_leaves)
            self.update_status()          
        return self

class FeatureManager(object):
    '''
    The methods to use here are:
    get_sample(feature_list,index=None)
    get_training_data(feature_list)
    get_test_data(feature_list)
    The attribute to look for are:
    feature_list_

    '''

    def __init__(self,feature_path='features/',config_path=''):
        # keeps a ditionary of available feature generators
        self.feature_path=feature_path
        self.feature_list_=[]
        self.raw_index=[]
        self.feature_generators={x:y for  x,y in inspect.getmembers(self, predicate=inspect.ismethod) if x.beginswith('feat_')}
        self.dict_file='{}FeatureManager_config.yaml'.format(config_path)

        if os.file.isfile(self.dict_file):
            print('loading feature dict')
            feature_dict=self.update_features()
        else:
            print('no config file in {}'.format(config_path))
            feature_dict=self.feat_initial()
        self.save_feature_dict(feature_dict)

    @staticmethod
    def force_list(*arg):
        ''' Takes a list of arguments and returns the same, 
        but where all items were forced to a list.

        example : list_1,list_2=force_list(item1,item2)
        '''
        Gen=(x if isinstance(x,list) else [x] for x in arg)
        return Gen if len(arg)>1 else next(Gen)

    def update_features(self):
        '''
            The feature dictionary is of the format
            {'method_name': {feature_name: kwargs}
            it is saved in yaml and parsed for new features. Any method name not that doesn't exist or start with 'feat_' will be dropped.
            Initial features are a dict key:None except for target and test/subscription features.
            To add new features, manually add to the file '{method: {new_XX: kwargs}}' where XX is an integer
        '''
        with open(self.dict_file,'r') as File:
            temp_dict=yaml.load(File)
        temp_dict={x:y for x,y in temp_dict.items() if x in self.feature_generators}
        for method in temp_dict:
            if method!='feat_initial':
                new_features={}
                for feature in temp_dict[method]:
                    if feature.beginswith('new_'):
                        #check for duplicates, we don't want to compute something that already exists 
                        kwargs=temp_dict[method].pop(feature)
                        new_feature=self.feature_generators[method](**kwargs)
                        new_features[new_feature.name]=kwargs

                        print('saving parquet file')
                        new_feature.to_parquet('{}{}.pqt'.format(self.feature_path,feature_name))
                        del(new_feature);gc.collect()
                temp_dict[method].update(new_features)
        return temp_dict

    def save_feature_dict(self,feat_dict):

        feat_dict['instructions']='Add new features by adding a row to the table of the function you \'re using, with the arguments you want and \'new_XX\' where XX is an integer as feature_name.'
        with open(self.dict_file,'w') as File:
            yaml.dump(feat_dict,File)
        #extracting raw index, target and sub file names before collecting the remaining features in an iterator
        extras={feat_dict['feat_initial'].pop(x):x for x,y in feat_dict['feat_initial'] if y is not None}
        self.raw_index=pd.RangeIndex(extras['index'])
        self.target_file=extras['target']
        self.submission_file=extras['sub']
        self.feature_list_=chain(*[y for x,y in feature_dict.items()])
        return self

    def get_sample(self,features,sample_index=None):
        if sample_index is None:
            data=self.get_sample_index(True)
        else:
            data=pd.DataFrame(sample_index).join(self.get_feature_series('target'))
        for feat in features:
            if feat not in self.feature_list_:
                print('feature not available')
            else:
                data=data.join(self.get_feature_series(feat))

        data=data.astype({'ip':np.uint32}).reset_index(drop=True)
        return data

    def get_sample_index(self,add_target=False):
        y=get_feature_series('target') # beware, there may be an issue here if train and test_supplement overlap in time.
        n=sum(~y)-sum(y) # number of negative target rows to remove to get to a 50/50 distribution in target values
        y=y.drop(data[~data].sample(n=n,random_state=seed).index) #sub-sampling the training dataset
        if add_target:
            return y
        return y.index

    def get_training_data(self,features):
        df=self.get_feature_series('target')
        for feat in features:
            if feat not in self.feature_dict_:
                print('feature not available')
            else:
                df.join(self.get_feature_series(feat))
        df=df.astype({'ip':np.uint32}).reset_index(drop=True)
        return df

    def get_test_data(self,features):
        df=self.get_feature_series('sub')
        for feat in features:
            if feat not in self.feature_dict_:
                print('feature not available')
            else:
                df.join(self.get_feature_series(feat))
        df=df.astype({'ip':np.uint32}).reset_index(drop=True)
        return df

    def get_feature_series(self,feature):
        if feature=='sub':
            feature_parquet='{}{}.pqt'.format(self.feature_path,self.submission_feature)
        elif feature=='target':
            feature_parquet='{}{}.pqt'.format(self.feature_path,self.target_feature)
        else:
            if feat not in self.feature_list:
                print('feature not available')
                return None
            feature_parquet='{}{}.pqt'.format(self.feature_path,feature)
        print('loading from {}'.format(feature_parquet))
        with open(feature_parquet,'rb') as File:
            feature_series=pd.read_parquet(File) # index cannot be a range index here until all features have been merged.
        return feature_series

    def s3_transfer_to(self):

        return self

    def s3_transfer_from(self):
        if True: #check if file exists
            #get the file to the local drive
            return True
        else:
            return False

    def wrapper_for_feat_methods(self):
        #Not in use yet, but could be a wrapper out of it.
        df=pd.DataFrame(index=self.raw_index)
        for feat in all_col:
            df=df.join(self.get_feature_series(feat))
        
        for feat in new_features.columns:
            #add file name to feature_dict_
            new_features[feat].to_parquet('{}.pqt'.format(feat))
        return self

    def feat_initial(self):

        print('making parquet files of the dataset for quicker loading')

        def load_csv(name):

            file_path='{}{}.csv.zip'.format(self.path,name)

            # Defining dtypes
            types = {
                    'ip':np.uint32,
                    'app': np.uint16,
                    'os': np.uint16,
                    'device': np.uint16,
                    'channel':np.uint16,
                    'click_time': object
                    }

            if name=='test':
                types['click_id']= np.uint32
            else:
                types['is_attributed']='bool'

            # Defining csv file reading parameters
            read_args={
                'parse_dates':['click_time'],
                'infer_datetime_format':True,
                'index_col':'click_time',
                'usecols':list(types.keys()),
                'dtype':types,
                'compression':'zip',
                'engine':'c',
                'sep':','
                }

            print('Loading {}'.format(file_csv))
            with open(file_csv,'rb') as File:
                df=(pd
                    .read_csv(File,**read_args)
                    .tz_localize('UTC')
                    .tz_convert('Asia/Shanghai')
                )
            return df

        print('merging testing and supplement data')
        test=load_csv('test')
        test1=test.loc[test.index.hour.isin([12,13,14]),:]
        test2=test.loc[test.index.hour.isin([17,18,19]),:]
        test3=test.loc[test.index.hour.isin([21,22,23]),:]
        del(test);gc.collect()

        supplement=load_csv('test_supplement').assign(click_id=-1)
        supplement1=supplement.loc[:test1.index.min()-one_sec,:]
        supplement2=supplement.loc[test1.index.max()+one_sec:test2.index.min()-one_sec,:]
        supplement3=supplement.loc[test2.index.max()+one_sec:test3.index.min()-one_sec,:]
        supplement4=supplement.loc[:test3.index.min()-one_sec,:]
        del(supplement);gc.collect()

        test=(pd
            .concat([supplement1,test1,supplement2,test2,supplement3,test3,supplement4])
            .assign(is_attributed=-1)
            .astype({'is_attributed':np.int8})
            )

        del(supplement1,test1,supplement2,test2,supplement3,test3,supplement4);gc.collect()
        
        print('merging with training data')
        train=(load_csv('train')
            .assign(click_id=-1)
            .astype({'is_attributed':np.int8})
            )

        X=(pd.concat([train,test])
            .reset_index(drop=True)
            .assign(day=lambda x: x.click_time.dt.day.values.astype(np.int8),
                hour=lambda x: x.click_time.dt.hour.values.astype(np.int8))
        )

        del([train,test]);gc.collect()

        print('saving featues, raw index, target and submission to parquet files')
        initial_features={len(X.index):'index'}
        for col in X:
            if col == 'is_attributed':
                initial_features[col]='target'
                X.pop(col)[lambda x: x>0].astype(bool).to_parquet('{}{}.pqt'.format(self.path,col))
            elif col == 'click_id':
                initial_features[col]='sub'
                X.pop(col)[lambda x: x>0].astype(np.uint32).to_parquet('{}{}.pqt'.format(self.path,col)) #note that parquet 1.0 doesn't keep int32 anyhow, so it'll be int64
            else:
                initial_features[col]=None
                X.pop(col).to_parquet('{}{}.pqt'.format(self.path,col))
        del(X);gc.collect()
        return {'feat_initial':initial_features}

    def feat_window(self,grouped,aggregated='dummy_var',aggregator='count',dt=None):
        ''' Returns a dataframe with the original index and rolling windows of dt minutes of the aggregated columns,
        per unique tuple of the grouped features, calculated with a given aggregator. The advantage of only returning the new values only is that one
        can then join them as one wishes and chain them in a pipeline.

        Warning ! At the moment, the dataframe passed must be starting at 0 and without gaps for the join to work.
        
            Args:
            grouped (list/string): column(s) to group by.
            aggregated (string): column to aggregate on the rolling window.
            aggregator (string): method to aggregate by.
            dt (int) : window size, in minutes
            time_col (datetime): column to use for the windows
            pickle_name (str): file name to save/load the new features.

        Returns:
            pd.DataFrame with similar index as input: rolling windows of the aggregated featured, relative to the grouped (i.e. fixed) ones.
        '''
        grouped,aggregated,aggregator=force_list(grouped,aggregated,aggregator)
        all_col=grouped+aggregated
        # deal with the case where all_col isn't a subset of feature_list_

       

        df=pd.DataFrame(index=self.raw_index)
        for feat in all_col:
            df=df.join(self.get_feature_series(feat))

        if dt is None:
            print('performing pivot')
            feature_name='{}_{}_by_{}'.format(aggregator,aggregated,'_'.join(grouped))

            if aggregator==['count']:
                df['dummy_var']=1
            new_feature=(df
                        .groupby(grouped)
                        .transform(aggregation)
                        .rename(columns={'dummy_var':feature_name})
                        .loc[:,feature_name]
                       )
            # not always necessary, so find a better way to deal with types.
            if aggregators==['count']:
                new_feature=new_feature.astype(np.uint16)

        else:
            time_col='click_time'
            df=df.join(self.get_feature_series(time_col))
            dt=str(dt)+'T'
            aggregation={aggregated:aggregator,'dummy_var':lambda x: x[-1]}

            print('computing rolling window')          
            # Alternative approach, note that it will always return the biggest dtype, but we can't predict if float or int before hand.
            new_feature=(
                pd.concat([(grp
                    .set_index(time_col)
                    .rolling(window=dt)
                    .agg(aggregation)
                    ) for name,grp in df.assign(dummy_var=lambda x:x.index.values).groupby(grouped)
                    ])
                .sort_values(by=('dummy_var','<lambda>'))
                .drop([('dummy_var','<lambda>')],axis=1)
                .reset_index(drop=True)
                )

            #to debug the transition from df to series
            print(new_feature.head())

            if aggregators ==['count']:
                new_feature=new_frame.astype(np.uint16)
            else:
                new_feature=new_frame.astype(np.float16)

            # example name 5T_os_count_by_ip
            new_feature.columns=['{}_'.format(dt)+'_'.join(list(col)+['by']+grouped) for col in new_feature.columns]
            feature_name=new_feature.columns[0]
            new_feature=new_feature[feature_name]

        del(df);gc.collect()
        return new_feature

    def feat_delta(self,grouped,offset=-1):
        grouped=force_list(grouped)
        all_col=grouped+['click_time']

        df=pd.DataFrame(index=self.raw_index)
        for feat in all_col:
            df=df.join(self.get_feature_series(feat))

        if offset<0:
            feature_name='next_click_by_{}'.format('_'.join(grouped))
        else:
            feature_name='previous_click_by_{}'.format('_'.join(grouped))

        # I notices in other works that assigning on an existing column can be very time/ressource expensive, avoided it, at the price of memory load
        new_feature=(df
           .assign(int_time=lambda x: x.click_time.astype(np.int64)//10**9)
           #.astype({'int_time':np.uint32}) #add this to optimize memory usage instead of speed
           .assign(click_delta=lambda x: x.groupby(grouped).int_time.shift(offset)-x.int_time)
           .fillna(-10)
           .assign(click_delta=lambda x: (x.next_click+10))
           #.astype({'click_delta':np.uint32}) #add this to optimize memory usage instead of speed
           .rename(columns={'click_delta':feature_name})
           .loc[:,feature_name]
           .reset_index(drop=True)
          )

        #add file name to feature_dict_
        del(df);gc.collect()
        return new_feature

    def feat_fourrier(self,grouped,dt=10):

        def fourrier_agg(data):
            if len(data)>0:
                A=pd.DataFrame(data).resample('20s').count()
                A=A-A.mean()
                if A.max().values>0:
                    A=A/A.max()
                Fk=np.fft.fftshift(np.fft.fft(A)/len(A))
                fourrier_std=np.std(np.absolute(Fk)**2)*10000
                del(A);gc.collect()
            else:
                fourrier_std=0
            return fourrier_std.astype(np.int32)
        grouped=force_list(grouped)
        all_col=grouped+['click_time']
        
        print('computing fourrier transform spread')
        time='{}Min'.format(dt)
        feature_name='{}_fourrier_by_{}'.format(time,'_'.join(grouped))
        new_feature=(data.set_index('click_time')
                .assign(dummy_var=1)
                #.astype({'dummy_var':np.uint16}) #add this to optimize memory usage instead of speed
                .groupby(grouped+[pd.Grouper(freq=time)])
                .transform(fourrier_agg)
                .rename(columns={'dummy_var':feature_name})
                .loc[:,feature_name]
                #.reset_index(drop=True)
        )
        print('if the index here is range, delete the reset_index line in the code',new_feature.index)
        return new_feature