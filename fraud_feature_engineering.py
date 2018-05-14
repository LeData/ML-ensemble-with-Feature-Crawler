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
local_path='input/'

feature_generator={
    'get_window':{
        'grouped': [],
        'aggregated': [],
        'aggregators':[]
        'dt':[]
    },
    'get_delta':[],
    'get_fourrier':[]
}

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
    # It can expand the feature graph at any time with new features and automatically prunes branches that will not improve the model,
    # based on recorded scores.
    
    # The handling of full trees is not optimized yet (should always return the unique leaf.
    
    null_set=set()
    unit_graph=nx.DiGraph()
    unit_graph.add_edge('s','t')
    nx.set_node_attributes(unit_graph,{'s':null_set,'t':null_set},'feats')
    nx.set_node_attributes(unit_graph,{'s':1,'t':0},'score')
    
    def __init__(self,crawler_file=None):
        # initiate with a list of features (str)
        print('creating feature graph')
        self.pointer={'node':0,**self.G.node[0]}
        if crawler_file is None:
            self.G=nx.DiGraph()
            self.G.add_node(0,feats=self.null_set,score=0)
        else:
            self.G=nx.read_yaml(crawler_file+'.yaml')

    def add_features(self,features):

        print('adding features to graph')
        for x in features:
            self.unit_graph.node['t']['feats']=set(x)
            self.G=nx.algorithms.operators.cartesian_product(self.G,self.unit_graph)
            self.unit_graph.node['t']['feats']=set()
            
            new_labels={x:y[0]|y[1] for x,y in nx.get_node_attributes(self.G,'feats').items()}
            nx.set_node_attributes(self.G,new_labels,'feats')
            
            new_scores={x:y[0]*y[1] for x,y in nx.get_node_attributes(self.G,'score').items()}
            nx.set_node_attributes(self.G,new_scores,'score')
            
            self.G=nx.convert_node_labels_to_integers(self.G)
        return self

    def get_features(self):
        # the graph G may already be topologically ordered from 0 by the ernaming of the nodes
        for child in list(nx.topological_sort(Feat_tree.G.subgraph(nx.descendants(Feat_tree.G,0)))):
            if self.G.node[child]['score']==0:
                self.pointer={'node':child,**self.G.node[child]}
                return self.pointer
        # if all nodes have been evaluated, returns the best set of features:
        # if all went well, the following are equivalent:
        #highest_leaf=min(self.G.out_degree.items(), key=operator.itemgetter(1))[0]
        highest_leaf=max(nx.get_node_attributes(self.G,'score').items(), key=operator.itemgetter(1))[0]
        self.pointer={'node':highest_leaf,**self.G.node[highest_leaf]}
        return self.pointer

    def record_score(self,node_dict):
        # params: updated pointer
        node=node_dict.pop('node')
        if self.G.node[node]['feats']!=node_dict.pop('feats'):
             print('The features do not correspond to the given node')
        else:
            self.G.node[node]['score']=node_dict.pop('score')
            if any([self.G.node[x]['score']> self.G.node[node]['score'] for x in nx.ancestors(self.G,node)]):
                self.G.remove_nodes_from({node}|nx.descendants(self.G,node))
        return self
    
    def prune(self):
        # Remove all leaves that have been estimated and are not global max
        # if not, deletes the nodes of the branch.

        while len({self.G.node[n]['score'] for n,deg in self.G.out_degree if deg==0 and self.G.node[n]['score']>0})>1:
            pos_leaves_scores={n:self.G.node[n]['score'] for n,deg in self.G.out_degree if deg==0 and self.G.node[n]['score']>0}
            max_score=max(nx.get_node_attributes(self.G,'score').items(), key=operator.itemgetter(1))[1]
            self.G.remove_nodes_from({n for n,score in pos_leaves_scores.items() if score < max_score})
        return self

class FeatureManager():

    def __init__(self):
        #initiates the dictionary of features and their pickles/parquet 
        self.feature_files={}

def load_data(train=True,target_only=False):
    '''
    Returns:
        pd.DataFrame - The whole dataset
        pd.Series - the target feature and the submission 'template' (click_id), with int index corresponding to their place in
                the dataset so they can be joined to it.
        It also saves the files in parquet file and loads the data from them if they exist.
    '''
    # Setting file path

    if training:
        data_pqt='{}data.pqt'.format(local_path)
        target_pqt='{}target.pqt'.format(local_path)

        # Reading file and setting the timezone
        if not (os.path.isfile(data_pqt) & os.path.isfile(target_pqt)):
            make_parquet()

        print('loading the data')
        with open(target_pqt,'rb') as File:
            y=pd.read_parquet(File).reset_index(drop=True) # beware, there may be an issue here if train and test_supplement overlap in time.

        if target_only:
            return y

        with open(data_pqt,'rb') as File:
            X=(pd
                .read_parquet(File)
                .reset_index(drop=True) # index dtype: int -> range
                .astype({'ip':np.uint32}) # parquet 1.0 doesn't do 32 bit int
                 )

        return X,y
    else:
        sub_pqt='{}sub.pqt'.format(local_path)
        with open(sub_pqt,'rb') as File:
            click_id=pd.read_parquet(File) # index cannot be a range index here until all features have been merged.
        return click_id

def make_parquet():
    print('making parquet files of the dataset for quicker loading')
    data_pqt='{}data.pqt'.format(local_path)
    target_pqt='{}target.pqt'.format(local_path)
    sub_pqt='{}sub.pqt'.format(local_path)


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

    print('saving dataset, target series and submission series to parquet files')
    X.pop('is_attributed')[lambda x: x>0].astype(bool).to_parquet(target_pqt)
    X.pop('click_id')[lambda x: x>0].astype(np.uint32).to_parquet(sub_pqt)
    X.to_parquet(data_pqt)
    del(X);gc.collect()
    return True

def get_training_sample():
    '''
    Returns :
        pd.Index - index of a subsampled dataset for a balanced training set.
    '''
    y=load_data(train=True,target_only=True)
    n=sum(~y)-sum(y) # number of negative target rows to remove to get to a 50/50 distribution in target values
    y=y.drop(y[~y].sample(n=n,random_state=seed).index) #sub-sampling the training dataset
    return y.index

# funtion below to be deleted
def add_features(self,pickle):
        print('Starting feature extraction')
        if pickle=='train':
            X=self.train
        elif pickle=='test':
            X=self.test
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


def get_feature_list():
    # connect to s3 bucket
    # get feature list
    feat_list=[]
    return False

def make_features(feature):
    # define list of features to make
    # fiter out those whose pqt files exist
    # compute the rest
    # save parquet files
    # transfer to s3_bucket
    return False

def get_window(data,grouped,aggregated=None,aggregators='count',dt=5,pickle_name=''):
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
        print('saving pickle file')
        new_frame.columns=['{}_'.format(dt)+'_'.join(list(col)+['by']+grouped) for col in new_frame.columns]
        new_frame.to_pickle('{}.pkl'.format(pickle_file))

        
    return new_frame

def get_delta(data,grouped,offset=-1):
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

def get_fourrier(data,grouped):

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

def transfer_to_s3():
