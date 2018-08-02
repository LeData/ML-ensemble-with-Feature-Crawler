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

    def __init__(self,files_path,crawler_file,increasing_measure=True,verbose=False):
        self.X_train=[]
        self.y=[]
        self.X_test=[]
        self.crawler=fem.FeatureCrawler(files_path,crawler_file,increasing_measure)
        self.verboseprint= print if verbose else lambda *a,**k:None

    def set_training_data(self,X_train,y_train,split_frac=default_split):
        self.train=(X_train,y_train)
        self.__pre_process(train=True,split_frac=split_frac)
        #self.transform_to_native_file(X_train,y_train)
        return self

    def set_test_data(self,X_test):
        self.test=X_test
        self.__pre_process(train=False)
        return self

    @abstractmethod
    def transform_to_native_file(self,X,y):
        #transform training or test data to the native file of the model
        pass

    def __pre_process(self,train=True,split_frac=default_split):
        if train:
            self.verboseprint('making the training and evaluation pools')
            X_train,X_valid,y_train,y_valid=train_test_split(self.train[0], self.train[1], test_size=split_frac, random_state=self.seed, stratify=self.train[1])
            self.train=self.transform_to_native_file(X_train,y_train)
            self.valid=self.transform_to_native_file(X_valid,y_valid)
        else:
            self.test=self.transform_to_native_file(self.test)
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
        feat_dict_list=self.crawler.get_leaves_features()
        return feat_dict_list

    def update_learned_features(self,feature_dict):
        self.crawler.record_score(feature_dict)
        return self

    def update_feature_space(self,feature_list):
        self.verboseprint('updating the feature space.')
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

    def __init__(self,params,files_path,crawler_file,verbose=False):
        self.lgb=__import__('lightgbm')
        super().__init__(files_path,crawler_file,verbose)
        self.params=params
        #pd.Series(params).to_csv('parameters.csv')

    def transform_to_native_file(self,X,y=None):
        self.verboseprint('Creating the lgbDataset')
        if y is not None:
            return self.lgb.Dataset(data=X,label=y,free_raw_data=True)
        else:
            return X

    def fit(self,X_train,y_train,split_frac=None):
        if split_frac is None:
            split_frac=self.default_split
        self.set_training_data(X_train,y_train)
        self.verboseprint('training model')
        self.model=self.lgb.train(self.params, self.train, num_boost_round= 2000,early_stopping_rounds= 250,
        valid_sets=[self.train,self.valid], verbose_eval=50)
        #self.model.save_model('lgbm_model.txt')
        return self

    def CV_score(self,X_train,y_train):
        self.train=self.transform_to_native_file(X_train,y_train)
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
        self.set_test_data(data)
        result=self.model.predict(self.test,num_iteration=self.model.best_iteration)
        return result

    def plot_feature_importance(self):
        self.verboseprint("Features importance...")
        gain = self.model.feature_importance('gain')
        ft = pd.DataFrame({'feature':self.model.feature_name(), 'split':self.model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('split', ascending=False)
        self.verboseprint(ft.head(25))
        plt.figure()
        ft[['feature','gain']].head(25).plot(kind='barh', x='feature', y='gain', legend=False, figsize=(10, 20))
        plt.gcf().savefig('features_importance.png')
        plt.figure()
        ft[['feature','split']].head(25).plot(kind='barh', x='feature', y='split', legend=False, figsize=(10, 20))
        plt.gcf().savefig('features_importance.png')
        return self

class CatBoostClassifier(Model):

    def __init__(self,params,files_path,crawler_file,verbose=False):
        self.cat= __import__(catboost)
        super().__init__(files_path,crawler_file,verbose)
        self.model=self.cat.CatBoostClassifier(**params)

    def transform_to_native_file(self,X,y):
        self.verboseprint('Creating the pool')
        if y is not None:
            return self.cat.Pool(data=X,label=y)
        else:
            return self.cat.Pool(data=X)

    def fit(self):    
        self.verboseprint('training model')
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
        self.verboseprint(feat_importance)
        feat_importance.sort_values(ascending=False).to_csv('feature importance.csv')
        return self

class XGBoostClassifier(Model):

    def __init__(self,params,files_path,crawler_file,verbose=False):
        self.xgb=__import__('xgboost')
        super().__init__(files_path,crawler_file,verbose)
        self.params=params

    def transform_to_native_file(self,X,y):
        self.verboseprint('Creating the DMatrix')
        if y is not None:
            return self.xgb.DMatrix(data=X,label=y,free_raw_data=True)
        else:
            pass

    def fit(self,X_train,y_train,split_frac=None):
        if split_frac is None:
            split_frac=self.default_split
        self.set_training_data(X_train,y_train)
        self.verboseprint('training model')
        self.model=self.xgb.train(self.params,self.train,num_boost_rounds=2000,early_stopping_rounds=250,
            evals=[self.train,self.valid], verbose_eval=50)
        return self

    def CV_score(self,X_train,y_train):
        self.train=self.transform_to_native_file(X_train,y_train)
        eval_hist=self.xgb.cv(self.params,self.train,num_boost_round=50,nfold=3)
        score=np.max(eval_hist[self.params['metric']+'-mean'])
        return score

    def get_submission(self,X_test,to_csv=True):
        self.set_test_data(X_test)
        submission=pd.DataFrame(data={'click_id':self.test.pop('click_id')})
        self.__pre_process(train=False)
        submission['is_attributed']=self.model.predict_proba(self.test,num_iteration=self.model.best_iteration)
        if ~to_csv:
            return submission['is_attributed'].values()
        submission.to_csv('lgbm_sub.csv',index=False)
        return self

    def plot_feature_importance(self):
        feat_importance=self.model.feature_importances_
        xgb.plot_importance(self.model)
        plt.show()
        feat_importance.sort_values(ascending=False).to_csv('feature importance.csv')
        return self

class LayerOne(object):
    '''LayerOne takes a dictionary of models and their parameters current available models:
        'lgbm','xgb','catboost'
    '''

    def __init__(self,model_dict,parquet_path,config_path,verbose=False):
        self.parquet_path=parquet_path
        self.config_path=config_path

        self.verboseprint= print if verbose else lambda *a,**k:None

        self.verboseprint('Setting up the Feature Manager')
        self.manager=fem.FeatureManager(self.parquet_path,self.config_path,verbose)
        self.verboseprint('Setting up the models')

        self.models={}
        for name,params in model_dict.items():
            if name == 'lgbm':
                self.models[name]=lgbmClassifier(params,self.config_path,'lvl1_'+name,verbose)
            elif name == 'catboost':
                self.models[name]=CatBoostClassifier(params,self.config_path,'lvl1_'+name,verbose)
            elif name == 'xgb':
                self.models[name]=XGBoostClassifier(params,self.config_path,'lvl1_'+name,verbose)
            else:
                self.verboseprint('{} model does not exist'.format(name))
                continue
            self.verboseprint('Updating the Feature Crawler for {}'.format(name))
            self.models[name].update_feature_space(self.manager.feature_list_)

    def learn_until(self,condition):
        # condition must be {'number': int, 'threshold': float}. 
        # It asks for at least n leaves above a certain threshold.
        for name,model in self.models.items():
            self.verboseprint('Feature learning with {} model'.format(name))
            learning_curve=[model.crawler.status_]
            #id self.downsample_:
                #self.manager.set_downsample(0.2)
            while not model.check_crawl_stop(condition):
                feat_dict=model.get_learning_features()
                self.verboseprint('Evaluating feature set: {}'.format(feat_dict['feats']))
                X_train,y_train=self.manager.get_training_data(feat_dict['feats'])
                feat_dict['score']=model.CV_score(X_train,y_train.values)
                model.update_learned_features(feat_dict)
                learning_curve.append(model.crawler.status_)
            pd.DataFrame(learning_curve).plot(title='learning curve for {}'.format(name))
        return self

    def get_lvl2_data(self):
        no_feature=self.manager.get_all_data(['sub','target'])
        lvl2_data={'sub':no_feature.iloc[:,0],'target':no_feature.iloc[:,1]}
        for name,model in self.models.items():
            for features in model.get_blending_features():
                X_train,y_train=self.manager.get_training_data(features)
                model.fit(X_train,y_train)
                del(X_train,y_train);gc.collect()

                data=self.manager.get_all_data(features)
                feature_name='_'.join([name]+list(features))
                lvl2_data[feature_name]=model.predict(data)
                del(data);gc.collect()
        output=pd.DataFrame(lvl2_data)


        # Add sub and target features
        #with open('{}lvl1_output.pqt'.format(self.parquet_path),'w') as File:
        #            output.to_parquet(File)
        return output

class LayerTwo(object):
    def __init__(self,data,model_dict,verbose=False):
        # load level 1 dataframe feature by feature
        # split training and testing data
        self.data=data
        self.verboseprint= self.verboseprint if verbose else lambda *a,**k:None
        self.models={}
        for name,params in model_dict.items():
            if name == 'lgbm':
                self.models[name]=lgbmClassifier(params,self.config_path,'lvl2_'+name,verbose)
            elif name == 'catboost':
                self.models[name]=CatBoostClassifier(params,self.config_path,'lvl2_'+name,verbose)
            elif name == 'xgb':
                self.models[name]=XGBoostClassifier(params,self.config_path,'lvl2_'+name,verbose)
            else:
                self.verboseprint('{} model does not exist'.format(name))
                continue


    def train(self):
        #train one model of each
        for name,model in self.models:
            print(name)

        return self

    def get_layer3_data(self):
        # apply trained models on the submission dataset from level1
        return self

class LayerThree(object):
    def __init__(self,data,verbose=False):
        self.data=data
        self.verboseprint= self.verboseprint if verbose else lambda *a,**k:None

    def get_submission(self):
        # weighted average of results from level2
        return self

    def submit_to_kaggle(self):
        #kaggle competitions submit [-h] -c COMPETITION -f FILE -m MESSAGE
        return self

