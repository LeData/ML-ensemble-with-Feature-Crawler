import pandas as pd
class Model(object):

    def __init__(self,train):
        self.train=load_data(train,current_skip)
        from sklearn.model_selection import train_test_split
        self.train.name='training set'
        self.train_pool=[]
        self.valid=[]
        self.valid_pool=[]
        gc.collect()
        self.test=[]
        self.test_pool=[]
        gc.collect()
        self.seed=714

    def reduce_train(self,from_i,to_i):
        print('starting reduction of the training set')
        self.train=self.train.iloc[from_i:to_i,:].reset_index(drop=True)
        gc.collect()
        print('reduction done')
  
    def cap(self):
        if not self.capped:
            self.train=(self.train.loc[(self.train['ip']<126413)&(self.train['app']<521)&(self.train['device']<3031)&(self.train['os']<604)&(self.train['channel']<498),:]
                        .reset_index(drop=True)
                       )
            self.capped=True
            gc.collect()
        else:
            print('capped validation set already created, use .val_cap')

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

    def pre_process(self,train=True,split_frac=.1):
        if train:
            self.train=self.add_features('train')
            print('making the training and evaluation pools')
            self.train,self.valid=train_test_split(self.train, test_size=split_frac, random_state=self.seed,stratify=self.train['is_attributed'])
            self.train=self.transform_to_native_file(self.train)
            self.valid=self.transform_to_native_file(self.valid)
            gc.collect()
        else:
            self.test=self.add_features('test')
            self.test=self.transform_to_native_file(self.test,False)
        gc.collect()


class CatClassifier(Model):

    def __init__(self,train,params):
        from catboost import CatBoostClassifier,Pool
        super().__init__(train)
        self.model=CatBoostClassifier(**params)

    def transform_to_native_file(self,X,training=True):
        print('Creating the pool')
        if training:
            return Pool(data=X.drop(['is_attributed'],axis=1),label=X['is_attributed'].astype(np.int8))
        else:
            return Pool(data=X)

    def fit_model(self):    
        print('training model')
        self.model.fit(
            self.train,
            eval_set=self.valid,
            verbose=False,
            plot=False,
            use_best_model=True
        )

    def get_submission(self):
        self.test = load_data('test')
        submission=self.test.pop('click_id')
        self.pre_process('test')
        submission['prediction']=self.model.predict_proba(self.test)[:,1]
        submission.to_csv('catboost_sub.csv',index=False)

    #there's an issue here with the column names not being taken before the pool transformation
    def feature_importance(self):
        feat_importance=pd.Series(index=self.test.columns)
        feat_importance['importance']=self.model.feature_importances_
        print(feat_importance)
        feat_importance.sort_values(ascending=False).to_csv('feature importance.csv')

class lgbmClassifier(Model):

    def __init__(self,train,params):
        import lightgbm as lgb
        pd.Series(params).to_csv('parameters.csv')
        super().__init__(train)
        self.params=params

    def transform_to_native_file(self,X,training=True):
        print('Creating the lgbDataset')
        if training:
            return lgb.Dataset(data=X.drop(['is_attributed'],axis=1),label=X['is_attributed'].astype(np.int8),free_raw_data=True)
        else:
            return X

    def fit_model(self):    
        print('training model')
        self.model=lgb.train(self.params, self.train, num_boost_round= 2000,early_stopping_rounds= 150,
        valid_sets=[self.train,self.valid], verbose_eval=50)
        self.model.save_model('lgbm_model.txt')

    def get_submission(self):
        self.test = load_data('test')
        submission=pd.DataFrame(data={'click_id':self.test.pop('click_id')})
        self.pre_process(train=False)
        submission['is_attributed']=self.model.predict(self.test,num_iteration=self.model.best_iteration)
        submission.to_csv('lgbm_sub.csv',index=False)
        return submission

    def feature_importance(self):
        print("Features importance...")
        gain = self.model.feature_importance('gain')
        ft = pd.DataFrame({'feature':self.model.feature_name(), 'split':self.model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('split', ascending=False)
        print(ft.head(25))
        plt.figure()
        ft[['feature','gain']].head(25).plot(kind='barh', x='feature', y='gain', legend=False, figsize=(10, 20))
        plt.gcf().savefig('features_importance.png')
        plt.figure()
        ft[['feature','split']].head(25).plot(kind='barh', x='feature', y='split', legend=False, figsize=(10, 20))
        plt.gcf().savefig('features_importance.png')

class XGBoostClassifier(Model):
    def __init(self,train,params):
        import xgboost as xgb
        super().__init__(train)
        self.model=xgb.XGBClassifier(**params)

    def transform_to_native_file(self,X,training=True):
        print('Creating the DMatrix')
        if training:
            return xgb.DMatrix(data=X.drop(['is_attributed'],axis=1),label=X['is_attributed'].astype(np.int8),free_raw_data=True)
        else:
            pass

    def fit_model(self):    
        print('training model')
        self.model.fit(X,y)

    def get_submission(self):
        self.test = load_data('test')
        submission=self.test.pop('click_id')
        self.pre_process('test')
        submission['prediction']=self.model.predict_proba(self.test)[:,1]
        submission.to_csv('xgb_sub.csv',index=False)
        return submission

    def feature_importance(self):
        feat_importance=self.model.feature_importances_
        xgb.plot_importance(self.model)
        plt.show()
        feat_importance.sort_values(ascending=False).to_csv('feature importance.csv')


#this is not working yet, it was just copy-pasted here
class EntityEmbedding(Model):

    def __init__(self, X_train, y_train, X_val, y_val):
        from keras.models import Sequential
        from keras.models import Model as KerasModel
        from keras.layers import Input, Dense, Activation, Reshape
        from keras.layers import Concatenate
        from keras.layers.embeddings import Embedding
        from keras.callbacks import ModelCheckpoint
        super().__init__()
        self.epochs = 10
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)

    def preprocessing(self, X):
        X_list = split_features(X)
        return X_list

    def __build_keras_model(self,feat_embed_shapes):
        ''' The feat_embed_shapes dictionary contains the categorical features to embed as keys
        and a tuple of dimensions defining the embedding as values
        '''
        inputs={}
        outputs={}
        for key in feat_embed_shapes:
            inputs[key] = Input(shape=(1,))
            outputs[key] = Embedding(feat_embed_shapes[key][0], feat_embed_shapes[key][1], name='{}_embedding'.format(key))(input[key])
            outputs[key] = Reshape(target_shape=(feat_embed_shapes[key][1],))(output[key])

        input_model = inputs.values()
        output_embeddings = outputs.values()

        output_model = Concatenate()(output_embeddings)
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(1)(output_model)
        output_model = Activation('sigmoid')(output_model)

        self.model = KerasModel(inputs=input_model, outputs=output_model)

        self.model.compile(loss='mean_absolute_error', optimizer='adam')


    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(self.preprocessing(X_train), y_train,
                       validation_data=(self.preprocessing(X_val), y_val),
                       epochs=self.epochs, batch_size=128,
                       # callbacks=[self.checkpointer],
                       )
        # self.model.load_weights('best_model_weights.hdf5')
        print("Result on validation data: AUC -", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = self.preprocessing(features)
        return self.model.predict(features).flatten()