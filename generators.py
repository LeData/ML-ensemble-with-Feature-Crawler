import pandas as pd
import numpy as np
import gc

class FeatureGenerators(object):
	# this class should contain no __init__
	# but just the methods in the following format

    '''
    def feat_nameoffeature(self,*args):

        features= #list of needed columns, most likely from *args

        df=pd.DataFrame(index=self.raw_index)
        for feat in features:
            df=df.join(self.__get_series(feat))

        feature_name= #setting up the name of the feature

        new_feature= #Computation of the new feature. remember to rename it with feature_name

        del(df);gc.collect()
        return new_feature
'''
    def feat_initial(self):
        # target and test/submission files must only be of the corresponding 
        # index in the main dataframe

        print('making parquet files of the dataset for quicker loading')

        def load_csv(name):

            file_csv='{}{}.csv.zip'.format(self.path,name)

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
        initial_features={str(len(X.index)):'index'}
        for col in X:
            if col == 'is_attributed':
                initial_features[col]='target'
                X.pop(col)[lambda x: x>=0].astype(bool).to_frame().to_parquet('{}{}.pqt'.format(self.feature_path,col))
            elif col == 'click_id':
                initial_features[col]='sub'
                X.pop(col)[lambda x: x>=0].astype(np.uint32).to_frame().to_parquet('{}{}.pqt'.format(self.feature_path,col)) #note that parquet 1.0 doesn't keep int32 anyhow, so it'll be int64
            else:
                initial_features[col]=None
                X.pop(col).to_frame().to_parquet('{}{}.pqt'.format(self.feature_path,col))
        del(X);gc.collect()
        return {'__feat_initial':initial_features}

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
        grouped,aggregated,aggregator=self.__force_list(grouped,aggregated,aggregator)
        features=grouped+aggregated
        # deal with the case where all_col isn't a subset of feature_list_

       

        df=self.__get_dataframe(features)

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
            df=df.join(self.__get_series(time_col))
            dt=str(dt)+'T'
            aggregation={aggregated:aggregator,'dummy_var':lambda x: x[-1]}

            print('computing rolling window')          
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
        grouped=self.__force_list(grouped)
        features=grouped+['click_time']

        df=self.__get_dataframe(features)

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

        grouped=self.__force_list(grouped)
        features=grouped+['click_time']

        df=self.__get_dataframe(features)

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
        
        print('computing fourrier transform spread')
        time='{}Min'.format(dt)
        feature_name='{}_fourrier_by_{}'.format(time,'_'.join(grouped))
        new_feature=(df.set_index('click_time')
                .assign(dummy_var=1)
                #.astype({'dummy_var':np.uint16}) #add this to optimize memory usage instead of speed
                .groupby(grouped+[pd.Grouper(freq=time)])
                .transform(fourrier_agg)
                .rename(columns={'dummy_var':feature_name})
                .loc[:,feature_name]
                #.reset_index(drop=True)
        )
        print('if the index here is range, delete the reset_index line in the code',new_feature.index)
        del(df);gc.collect()
        return new_feature