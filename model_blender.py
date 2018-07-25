## TODO
## 1- Add a iteration based condition for learn_until()
## 2 - get a seed shared with the models in the blender
## 3 - Manage downsample


import yaml
import gc
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import feature_engineering_module as fem
import pickle
from abc import ABCMeta, abstractmethod

class Model(object):

    default_split=.1
    seed=714

    __metaclass__=ABCMeta

    def __init__(self,files_path,crawler_file,increasing_measure=True):
        self.X_train=[]
        self.y=[]
        self.X_test=[]
        self.crawler=fem.FeatureCrawler(files_path,crawler_file,increasing_measure)

    def set_training_data(self,X_train,y_train,split_frac=default_split):
        self.X_train=X_train
        self.y_train=y_train
        self.__pre_process(train=True,split_frac=split_frac)
        return self

    def set_test_data(self,X_test):
        self.X_test=X_test
        self.__pre_process(train=False)
        return self

    @abstractmethod
    def __transform_to_native_file(self):
        #transform training or test data to the native file of the model
        pass

    def __pre_process(self,train=True,split_frac=default_split):
        if train:
            print('making the training and evaluation pools')
            X_train,y_train,X_valid,y_valid=train_test_split(self.X_train, self.y_train, test_size=split_frac, random_state=self.seed, stratify=self.y_train)
            self.train=self.__transform_to_native_file(X_train,y_train)
            self.valid=self.__transform_to_native_file(X_valid,y_valid)
        else:
            self.test=self.__transform_to_native_file(self.test,False)
        gc.collect()
        return self

    def check_crawl_stop(self,condition):
        # condition must be {'number': int, 'threshold': float}. It asks for at least n leaves above a certain threshold.
        stop=self.crawler.check_condition(condition)        
        return stop

    def get_learning_features(self):
        feat_dict=self.crawler.get_unscored_node()
        return feat_dict

    def get_blending_features(self):
        feat_dict_list=self.crawler.leaves_
        return feat_dict_list

    def update_learned_features(self,feature_dict):
        self.crawler.record_score(feature_dict)
        return self

    def update_feature_space(self,feature_list):
        print('updating the feature space.')
        self.crawler.update_graph(feature_list)
        return self

    @abstractmethod
    def fit(self,X_train,y_train,split_frac=default_split):
        # fit the model on training data
        pass

    @abstractmethod
    def CV_score(self,X_train,y_train):
        # computes a CV and returns the obtained score
        pass

    @abstractmethod
    def predict(self,X_test):
        # predict output for X_test
        pass
 
    @abstractmethod
    def get_submission(self):
        # get a submission ready file for Kaggle contests
        pass

    @abstractmethod
    def plot_feature_importance(self):
        # get feature importance
        pass


class lgbmClassifier(Model):

    def __init__(self,params,files_path,crawler_file):
        self.lgb=__import__('lightgbm')
        super().__init__(files_path,crawler_file)
        self.params=params
        #pd.Series(params).to_csv('parameters.csv')

    def __transform_to_native_file(self,X,y=None):
        print('Creating the lgbDataset')
        if y is not None:
            return self.lgb.Dataset(data=X,label=y,free_raw_data=True)
        else:
            return X

    def fit(self,X_train,y_train,split_frac=None):
        if split_frac is None:
            split_frac=self.default_split
        self.set_training_data(X_train,y_train)
        print('training model')
        self.model=self.lgb.train(self.params, self.train, num_boost_round= 2000,early_stopping_rounds= 150,
        valid_sets=[self.train,self.valid], verbose_eval=50)
        #self.model.save_model('lgbm_model.txt')
        return self

    def CV_score(self,X_train,y_train):
        self.train=self.__transform_to_native_file(X_train,y_train)
        eval_hist=self.lgb.cv(self.params,self.train,num_boost_round=50,nfold=3)
        score=np.max(eval_hist[self.params['metric']+'-mean'])
        return score

    def get_submission(self,X_test,to_csv=True):
        self.set_test_data(X_test)
        submission=pd.DataFrame(data={'click_id':self.test.pop('click_id')})
        self.__pre_process(train=False)
        submission['is_attributed']=self.model.predict(self.test,num_iteration=self.model.best_iteration)
        if ~to_csv:
            return submission['is_attributed'].values()
        submission.to_csv('lgbm_sub.csv',index=False)
        return self

    def predict(self,data):
        self.__set_test_data(data)
        result=self.model.predict(self.test,num_iteration=self.model.best_iteration)
        return result

    def plot_feature_importance(self):
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
        return self


class CatClassifier(Model):

    def __init__(self,params,files_path,crawler_file):
        from catboost import CatBoostClassifier,Pool
        super().__init__(files_path,crawler_file)
        self.model=CatBoostClassifier(**params)

    def __transform_to_native_file(self,X,y):
        print('Creating the pool')
        if y is not None:
            return Pool(data=X,label=y)
        else:
            return Pool(data=X)

    def fit(self):    
        print('training model')
        self.model.fit(
            self.train,
            eval_set=self.valid,
            verbose=False,
            plot=False,
            use_best_model=True
        )
        return self

    def get_submission(self):
        submission=self.test.pop('click_id')
        self.__pre_process('test')
        submission['prediction']=self.model.predict_proba(self.test)[:,1]
        submission.to_csv('catboost_sub.csv',index=False)
        return self

    #there's an issue here with the column names not being taken before the pool transformation
    def feature_importance(self):
        feat_importance=pd.Series(index=self.test.columns)
        feat_importance['importance']=self.model.feature_importances_
        print(feat_importance)
        feat_importance.sort_values(ascending=False).to_csv('feature importance.csv')
        return self

class XGBoostClassifier(Model):

    def __init(self,params,files_path,crawler_file):
        import xgboost as xgb
        super().__init__(files_path,crawler_file)
        self.model=xgb.XGBClassifier(**params)

    def __transform_to_native_file(self,X,y):
        print('Creating the DMatrix')
        if y is not None:
            return xgb.DMatrix(data=X,label=y,free_raw_data=True)
        else:
            pass

    def fit(self):    
        print('training model')
        self.model.fit(X,y)
        return self

    def get_submission(self):
        submission=self.test.pop('click_id')
        self.__pre_process('test')
        submission['prediction']=self.model.predict_proba(self.test)[:,1]
        submission.to_csv('xgb_sub.csv',index=False)
        return self

    def feature_importance(self):
        feat_importance=self.model.feature_importances_
        xgb.plot_importance(self.model)
        plt.show()
        feat_importance.sort_values(ascending=False).to_csv('feature importance.csv')
        return self

#this is not working yet, it was just copy-pasted here from another project
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


class LayerOne(object):
    '''models is a triple of models:
        keys = 'lgbm','xgbm','catboost'
        values = int - 

    '''

    def __init__(self,model_dict,parquet_path,config_path):
        self.parquet_path=parquet_path
        self.config_path=config_path

        print('Setting up the Feature Manager')
        self.manager=fem.FeatureManager(self.parquet_path,self.config_path,)
        print('Setting up the models')
        self.models=[lgbmClassifier(model_dict['lgbm'],self.config_path,'lvl1_lgbm')]
        for model in self.models:
            print('Updating the Feature Crawler for {}'.format(model))
            model.update_feature_space(self.manager.feature_list_)

    def learn_until(self,condition):
        # condition must be {'number': int, 'threshold': float}. 
        # It asks for at least n leaves above a certain threshold.
        for model in self.models:
            print('Feature learning with {} model'.format(model))
            #id self.downsample_:
                #self.manager.set_downsample(0.2)
            while not model.check_crawl_stop(condition):
                feat_dict=model.get_learning_features()
                print('Evaluating feature set: {}'.format(feat_dict['feats']))
                X_train,y_train=self.manager.get_training_data(feat_dict['feats'])
                feat_dict['score']=model.CV_score(X_train,y_train.values)
                model.update_learned_features(feat_dict)
        return self

    def get_layer2_data(self):
        for model in self.models:
            for features in model.get_blending_features():
                X_train,y_train=self.manager.get_training_data(features)
                model.fit(X_train,y_train)
                del(X_train,y_train);gc.collect()

                data=self.manager.get_all_data(features)
                level1_feature=model.predict(data)
                del(data);gc.collect()
                # what is level1_feature.name?
                with open(level1_feature.name,'w') as File:
                    level1_feature.to_parquet(File)

        # Add sub and target features
        # Make predictions on new sample + sub file for level 2
        return train,test


class LayerTwo(object):
    def __init__(self,data):
        # load level 1 dataframe feature by feature
        # split training and testing data
        self.data=data

    def train(self):
        #train one model of each
        return self

    def get_layer3_data(self):
        # apply trained models on the submission dataset from level1
        return self

class LayerThree(object):
    def __init__(self,data):
        self.data=data

    def get_submission(self):
        # weighted average of results from level2
        return self

    def submit_to_kaggle(self):
        #kaggle competitions submit [-h] -c COMPETITION -f FILE -m MESSAGE
        return self

