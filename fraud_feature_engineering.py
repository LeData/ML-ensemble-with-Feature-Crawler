## This file defines the methods for feature engineering:
## > loading data
## > adding rolling window features by group (dt=number of minutes)
## > adding frame-wide aggregates by group (dt=None)
## > defining entity embeddings for categorical features
## > transforming categorical features to "entity embedded" ones.

import pandas as pd

def load_data(name,skip=None,rows=None):
    ''' Load the csv files into a TimeSeries dataframe with minimal data types to reduce the used RAM space. 
    It also saves the files in parquet file to reduce loading time by a factor of ~10.

    Arg:
    
        -name (str): ante_day, last_day, train, train_sample or test

    Returns:
        pd.DataFrame, with int index equal to 'click_id'
    '''

    # Setting file path
    file_path='{}{}'.format(local_path,name)

    # Reading file and setting the timezone
    if os.path.isfile('{}.pqt'.format(file_path)):
        print('loading {}.pqt'.format(file_path))
        with open('{}.pqt'.format(file_path),'rb') as File:
            data=(pd.read_parquet(File)
                  .reset_index(drop=True)
                  .astype({'ip':np.uint32})
                 )
    else:
        if skip!=None:
            skip=range(1,skip)

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
            'nrows':rows,
            'skiprows': skip,
            'parse_dates':['click_time'],
            'infer_datetime_format':True,
            'index_col':'click_time',
            'usecols':list(types.keys()),
            'dtype':types,
            'engine':'c',
            'sep':','
            }

        print('Loading {}.csv'.format(file_path))
        with open('{}.csv'.format(file_path),'rb') as File:
            data=(pd
                .read_csv(File,**read_args)
                .tz_localize('UTC')
                .tz_convert('Asia/Shanghai')
                .reset_index()
            )

        # Sorting frames
        if name=='test': # making sure index == click_id
            data=data.sort_values(by=['click_id']).reset_index(drop=True)
        elif name=='train_sample': # sorting time randomized by sampling
            data=data.sort_values(by=['click_time']).reset_index(drop=True)
            
        print('saving to parquet')
        data.to_parquet('{}.pqt'.format(file_path))

    return data

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

def get_window(data,grouped,aggregated,aggregators='count',dt=5,time_col='click_time',pickle_name=''):
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
        data=data.merge(new_frame.reset_index(),on=grouped,how='left').reset_index(drop=True)

    else:
        
        dt=str(dt)+'T'
        pickle_file=local_path+dt+'_'.join(aggregated+['by']+grouped+[pickle_name])

        if os.path.isfile('{}.pkl'.format(pickle_file)):
            print('loading window pickle')
            with open('{}.pkl'.format(pickle_file),'rb') as File:
                new_frame=pd.read_pickle(File)

        else:
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
            
        data=data.join(new_frame,how='inner')
        del(new_frame)
        gc.collect()
        
    return data

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