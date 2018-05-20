## This file defines the methods for feature engineering:
## > loading data
## > adding rolling window features by group (dt=number of minutes)
## > adding frame-wide aggregates by group (dt=None)
## > defining entity embeddings for categorical features
##
## TODO:
## 1 - recode the count feature using .groupby().transform(len)
## 2 - deal with descending or ascending scores for the crawler


## For each engine, create the subgraph of Set of all available features combination, and inclusion only.
## Remove arrows that are decreasing in score.
## try graph-tool if networkX isn't good enough

import pandas as pd
import networkx as nx
import random
import json

local_path='input/'

def force_list(*arg):
    ''' Takes a list of arguments and returns the same, 
    but where all items were forced to a list.

    example : list_1,list_2=force_list(item1,item2)
    '''
    Gen=(x if isinstance(x,list) else [x] for x in arg)
    if len(arg)>1:
        return Gen
    else:
        return next(Gen)

class FeatureCrawler(object):
    # This class is a graph crawler for feature engineering. It maintains a graph of all sets of features,
    # their respective inclusions and scores. It incrementally learns the performance of feature sets to find the best model.
    # It can expand the feature graph at any time with new features and automatically removes feature combinations that will not improve the model
    # An option to prune all non optimal branches is provided.
    
    null_set=set()
    unit_graph=nx.DiGraph()
    unit_graph.add_edge('s','t')
    nx.set_node_attributes(unit_graph,{'s':null_set,'t':null_set},'feats')
    nx.set_node_attributes(unit_graph,{'s':True,'t':False},'score')
    
    def __init__(self,incr_score=True,crawler_file):
        # initiate with a baseline and score order (incr, decr)
        self.status_=0 # percentage crawled
        self.leaves_={} # dict of leaves and their scores
        self.file_=crawler_file+'.yaml'
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
        new_features=[x for x in feature_list if x not in nx.all_neighbors(self.G,0)]
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
            choices=list(self.leaves_)
        # the graph G may already be topologically ordered from 0 by the renaming of the nodes
        else:
            unscored_subgraph=nx.subgraph([n for n,score in nx.get_node_attrubutes(self.G,'score') if score is None])
            choices=[n for n,deg in unscored_subgraph.in_degree if deg=0]
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
            self.G.remove_nodes_from({n for n,score in self.leaves_.items() if self.is_better(max_score,score)})
            self.update_status()          
        return self

class FeatureManager():

    def __init__(self,file_path):
        self.path=file_path
        #initiates the dictionary of features and their pickles/parquet
        if os.file.isfile('{}.'.format(name))
            print('loading feature dict')

!!            # write the read and save properly
            with open('{}.pkl'.format(pickle_file),'rb') as File:
                self.feature_dict_=yaml.read(File)
            self.target_file=self.feature_dict_.pop('target')
            self.submission_template_=self.feature_dict_.pop('sub')
        else:
            self.make_initial_parquet()

    def make_initial_parquet(self):

        print('making parquet files of the dataset for quicker loading')

        def load_csv(name):

            file_path='{}{}.csv.zip'.format(local_path,name)

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
                data=(pd
                    .read_csv(File,**read_args)
                    .tz_localize('UTC')
                    .tz_convert('Asia/Shanghai')
                )

            return data

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

        X=pd.concat([train,test]).reset_index(drop=True)

        del([train,test]);gc.collect()

        print('saving featues, target and submission to parquet files')
        for col in X:
            if col == 'is_attributed':
                self.feature_dict_['target']='{}.pqt'.format(col)
                X.pop(col)[lambda x: x>0].astype(bool).to_parquet('{}{}'.format(local_path,self.feature_dict_['target']))
            elif col == 'click_id':
                self.feature_dict_['sub']='{}.pqt'.format(col)
                X.pop(col)[lambda x: x>0].astype(np.uint32).to_parquet('{}{}'.format(local_path,self.feature_dict_['target'])) #note that parquet 1.0 doesn't keep int32 anyhow, so it'll be int64
            self.feature_dict_[col]='{}.pqt'.format(col)
            X.pop(col).to_parquet('{}{}'.format(local_path,self.feature_dict_[col]))

        del(X);gc.collect()
        return self

    def save_feature_dict(self):
            dict_to_save=self.feature_dict_
            dict_to_save['target']=self.target_file
            dict_to_save['sub']=self.submission_template_

!!            # save in YAML
        return self

    def get_sample_index(self):
        '''
        Returns :
            pd.Index - index of a subsampled dataset for a balanced training set.
        '''
        y=self.get_feature_series('target') # beware, there may be an issue here if train and test_supplement overlap in time.
        n=sum(~y)-sum(y) # number of negative target rows to remove to get to a 50/50 distribution in target values
        y=y.drop(y[~y].sample(n=n,random_state=seed).index) #sub-sampling the training dataset
        return y.index

    def get_feature_series(self,feature):
        feature_parquet='{}{}'.format(self.local_path+self.feature_dict_[feature])
        with open(feature_parquet,'rb') as File:
            feature_series=pd.read_parquet(File) # index cannot be a range index here until all features have been merged.
        return feature_series

    def load_data():
        '''
        Returns:
            pd.DataFrame - The whole dataset
            pd.Series - the target feature and the submission 'template' (click_id), with int index corresponding to their place in
                    the dataset so they can be joined to it.
            It also saves the files in parquet file and loads the data from them if they exist.
        '''
        # Setting file path

        data_pqt='{}data.pqt'.format(local_path)
        target_pqt='{}target.pqt'.format(local_path)

        print('loading the data')

        with open(data_pqt,'rb') as File:
            X=(pd
                .read_parquet(File)
                .reset_index(drop=True) # index dtype: int -> range
                .astype({'ip':np.uint32}) # parquet 1.0 doesn't do 32 bit int
                 )

        return X
            
    def make_feature(feature):
        # define list of features to make
        # fiter out those whose pqt files exist
        # compute the rest
        # transfer to s3_bucket
        return False

    def s3_feature_list(self):
        # connect to s3 bucket
        # get feature list
        return self

    def s3_transfer_to(self):

        return self

    def s3_transfer_from(self):

        return self

    def feat_template(self):
        # load and join needed features from parquet
        # compute new feature
        # save parquet
        # add to feature_dict_

        return self

    def feat_window(self,data,grouped,aggregated=None,aggregators='count',dt=5):
        ''' Returns a dataframe with the original index and rolling windows of dt minutes of the aggregated columns,
        per unique tuple of the grouped features, calculated with a given aggregator. The advantage of only returning the new values only is that one
        can then join them as one wishes and chain them in a pipeline.

        Warning ! At the moment, the dataframe passed must be starting at 0 and without gaps for the join to work.
        
            Args:
            data (pd.DataFrame): dataframe to add features to.
            grouping (list/string): column(s) to group by.
            aggregated (list): columns to aggregate on the rolling window.
            aggregators (list/string): methods to aggregate by.
            dt (int) : window size, in minutes
            time_col (datetime): column to use for the windows
            pickle_name (str): file name to save/load the new features.

        Returns:
            pd.DataFrame with similar index as input: rolling windows of the aggregated featured, relative to the grouped (i.e. fixed) ones.
        '''
        time_col='click_time'
        grouped,aggregated,aggregators=force_list(grouped,aggregated,aggregators)
        
        if dt is None:
            print('performing pivot')
            all_col=grouped+aggregated
            aggregation={x:aggregators for x in aggregated}
            new_frame=(data[all_col]
                        .groupby(grouped)
                        .agg(aggregation)
                       )
            if aggregators==['count']:
                new_frame=new_frame.astype(np.uint16)
            new_frame.columns=['_'.join(list(col)+['by']+grouped) for col in new_frame.columns]
            new_frame=data.merge(new_frame.reset_index(),on=grouped,how='left')[new_frame.columns].reset_index(drop=True)


        else:
            
            dt=str(dt)+'T'
            pickle_file=local_path+dt+'_'.join(aggregated+['by']+grouped+[pickle_name])


            print('computing rolling window')
            all_col=grouped+aggregated+[time_col]
            aggregation={x:aggregators for x in aggregated}
            aggregation['click_id']=lambda x: x[-1]

            # Alternative approach, note that it will always return the biggest dtype, but we can't predict if float or int before hand.
            new_frame=(
                pd.concat([(grp
                    .set_index(time_col)
                    .rolling(window=dt)
                    .agg(aggregation)
                    ) for name,grp in data[all_col].assign(click_id=lambda x:x.index.values.astype(np.uint32)).groupby(grouped)
                    ])
                .sort_values(by=('click_id','<lambda>'))
                .drop([('click_id','<lambda>')],axis=1)
                .reset_index(drop=True)
                )

            if aggregators ==['count']:
                new_frame=new_frame.astype(np.uint16)
            else:
                new_frame=new_frame.astype(np.float16)

            # example name 5T_os_count_by_ip
            print('saving parquet file')
            new_frame.columns=['{}_'.format(dt)+'_'.join(list(col)+['by']+grouped) for col in new_frame.columns]
            new_frame.to_pickle('{}.pqt'.format())

            
        return new_frame

    def feat_delta(self,data,grouped,offset=-1):
        grouped=force_list(grouped)
        if offset<0:
            name='next_click_by_{}'.format('_'.join(grouped))
        else:
            name='previous_click_by_{}'.format('_'.join(grouped))

        if os.file.isfile('{}.pqt'.format(name))
            print('loading delta pickle')
            with open('{}.pkl'.format(pickle_file),'rb') as File:
                new_frame=pd.read_pickle(File)
            data=data.join(new_frame,how='inner')

        else:
            data=(data.assign(click_time=lambda x: x.click_time.astype(np.int64)//10**9)
               .astype({'click_time':np.uint32})
               .assign(next_click=lambda x: x.groupby(grouped).click_time.shift(offset)-x.click_time)
               .fillna(-10)
               .assign(next_click=lambda x: x.next_click+10)
               .astype(np.uint32)
               .rename(columns={'next_click':name})
              )
            data[name].to_pickle('{}.pkl'.format(name))
        gc.collect()
        return data

    def feat_fourrier(self,data,grouped):

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

        var=force_list(grouped)
        name='{}_fourrier_by_{}'.format(time,'_'.join(var))
        
        if os.path.isfile('{}.pkl'.format(name)):
            print('loading fourrier pickle')
            with open('{}.pkl'.format(name),'rb') as File:
                new_frame=pd.read_pickle(File)
        else:
            print('computing fourrier transform spread')
            dt=10
            time='{}Min'.format(dt)
            new_frame=(data.set_index('click_time')
                    .loc[:,grouped]
                    .assign(temp=1)
                    .astype({'temp':np.uint16})
                    .groupby(grouped+[pd.Grouper(freq=time)])
                    .transform(fourrier_agg)
                    .rename(columns={'temp':'{}_fourrier_by_{}'.format(time,var)})
                    .reset_index(drop=True)
              )
            new_frame.to_pickle('{}.pkl'.format(name))
        return data.join(new_frame)

# funtion below to be deleted
def add_features(self,pickle):
        X=(X.assign(day=lambda x: x.click_time.dt.day.values.astype(np.int8),
            hour=lambda x: x.click_time.dt.hour.values.astype(np.int8))
            #.pipe(get_window,['app','hour'],'day',dt=None,pickle_name=pickle)
            #.pipe(get_window,['channel','hour'],'day',dt=None,pickle_name=pickle)
            #.pipe(get_window,'app','day',dt=5,pickle_name=pickle)
            .pipe(get_window,['app','device'],'day',dt=5,pickle_name=pickle)
            .pipe(get_window,['channel','device'],'day',dt=5,pickle_name=pickle)
            .pipe(get_window,['app','channel','device'],'day',dt=5,pickle_name=pickle)
            .pipe(get_delta,['ip','app'])
            .pipe(get_delta,['ip','os'])
            .pipe(get_delta,['ip','app','device'])
            .pipe(get_delta,['ip','app'],1)
            .pipe(get_delta,['ip','os'],1)
            .pipe(get_delta,['ip','app','device'],1)
            .drop(['click_time','day','hour','ip'],axis=1)
        )
            #.pipe(get_window,['app','ip'],'hour',dt=120)
            #.pipe(get_window,['app','channel'],'hour',dt=120)
            #.pipe(get_window,['app','os'],'hour',dt=120)
            #.pipe(get_window,['app','device'],'hour',dt=120)
            #.pipe(get_window,['ip','channel'],'hour',dt=120)
            #.pipe(get_window,['ip','os'],'hour',dt=120)
            #.pipe(get_window,['ip','device'],'hour',dt=120)
            #.pipe(get_window,['channel','os'],'hour',dt=120)
            #.pipe(get_window,['channel','device'],'hour',dt=120)
            #.pipe(get_window,['os','device'],'hour',dt=120)
            #.pipe(get_window,['app','ip','channel'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['app','ip','os'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['app','ip','device'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['app','channel','os'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['app','channel','device'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['app','os','channel'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['ip','channel','os'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['ip','channel','device'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['ip','os','device'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['channel','os','device'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['app','ip','channel','os'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['app','ip','channel','device'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['app','ip','device','os'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['app','device','channel','os'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['device','ip','channel','os'],'hour',dt=120,pickle_name=pickle)
            #.pipe(get_window,['app','ip','channel','os','device'],'hour',dt=120,pickle_name=pickle)

        print('feature extraction done')
        return X


