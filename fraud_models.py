# make and save dataframe with features + engine + 50 round score as columns
# add a columns checking if there isn't a model with the same features but less that has a bigger score.
#


import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import feature_engineering_module as fem
import pickle

class Model(object):

    seed=714

    def __init__(self,index,file_path,crawler_file,increasing_measure=True):
        self.train=[]
        self.sample_index=index
        self.crawler=fem.FeatureCrawler(file_path,crawler_file,increasing_measure)

    def check_crawl_stop(self,condition):
        # condition must be {'number': int, 'threshold': float}. It asks for at least n leaves above a certain threshold.
        num_leaves=len(self.crawler.leaves_)
        print('The crawler explored {}% of the feature space.'.format(self.crawler.status*100))
        print('There are {} leaves available.'.format(num_leaves))
            if self.crawler.status==1:                
                return True
            else:
                return len({x:y for x,y in self.crawler.leaves_.items() if self.crawler.is_better(y,condition['threshold'])})>=condition['number']

    def get_features_to_learn(self):
        feat_dict=self.crawler.get_unscored_features()
        return feat_dict

    def update_learned_features(self,features):
        self.crawler.update_features(features)
        return self

    def CV_score(self,data):
        score=0
        return score

    def pre_process(self,train=True,split_frac=.1):
        if train:
            print('making the training and evaluation pools')
            self.train,self.valid=train_test_split(self.train.join(self.target), test_size=split_frac, random_state=self.seed, stratify=self.target)
            self.train=self.transform_to_native_file(self.train)
            self.valid=self.transform_to_native_file(self.valid)
            gc.collect()
        else:
            self.test=self.transform_to_native_file(self.test,False)
        gc.collect()
        return self

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
        return self

    def get_submission(self):
        self.test = load_data('test')
        submission=self.test.pop('click_id')
        self.pre_process('test')
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

class lgbmClassifier(Model):

    def __init__(self,index,params,crawler_file=None):
        import lightgbm as lgb
        super().__init__(index,crawler_file)
        self.params=params
        #pd.Series(params).to_csv('parameters.csv')

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
        #self.model.save_model('lgbm_model.txt')

        #update crawler with best score
        self.current_model['score']=self.model.score
        self.crawler()
        return self

    def get_submission(self,to_csv=True):
        self.test = load_data(train=False)
        submission=pd.DataFrame(data={'click_id':self.test.pop('click_id')})
        self.pre_process(train=False)
        submission['is_attributed']=self.model.predict(self.test,num_iteration=self.model.best_iteration)
        if ~to_csv:
            return submission['is_attributed'].values()
        submission.to_csv('lgbm_sub.csv',index=False)
        return self

    def fit_predict():
        # getting training and testing data
        data=load_data()
        self.train=data.loc[self.sample_index]
        self.test=data.loc[self.sub.index]
        del(data);gc.collect()

        # getting a list of features from the crawler:
        self.current_model=self.crawler.get_features()
        self.build_features()
        self.fit_model()
        self.get_submission()
        return self

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
        return self

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
        return self

    def get_submission(self):
        self.test = load_data('test')
        submission=self.test.pop('click_id')
        self.pre_process('test')
        submission['prediction']=self.model.predict_proba(self.test)[:,1]
        submission.to_csv('xgb_sub.csv',index=False)
        return self

    def feature_importance(self):
        feat_importance=self.model.feature_importances_
        xgb.plot_importance(self.model)
        plt.show()
        feat_importance.sort_values(ascending=False).to_csv('feature importance.csv')
        return self

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


class LevelOne(object):
    '''models is a triple of models:
        keys = 'lgbm','xgbm','catboost'
        values = int - number of 

    '''

    def __init__(self,model_dict,parquet_path,config_path):
        self.parquet_path=parquet_path
        self.config_path=config_path
        self.manager=fem.FeatureManager(self.parquet_path,self.config_path)

        kwargs={index,self.parquet_path,self.config_path}
        self.models=[lgbmClassifier(**kwargs)]
        for model in self.models:
            model.update_crawler_features(self.manager.feature_list_)

    def learn_until(self,condition):
        # condition must be {'number': int, 'threshold': float}. It asks for at least n leaves above a certain threshold.
        for model in self.models:
            print('Feature learning with {} model'.format(model))
            while ~model.check_crawl_stop(condition):
                feat_dict=model.get_features_to_learn()

                print('Evaluating feature set: {}'.format(feat_dict['feats']))
                train=self.manager.get_sample(feat.dict['feats'])
                feat_dict['score']=model.CV_score(train)
                model.update_learned_features(feat_dict)
        return self

    def get_level2_data(self):
        for model in self.models:
            for features in model.get_engineered_features()
                train=self.manager.get_training_data(leaf_features)
                test=self.manager.get_test_data(leaf_features)
                level1_feature=model.fit_predict(train,test)
                del(train,test);gc.collect()
                with open(level1_feature.name,'w') as File:
                    level1_feature.to_parquet(File)

        # Make predictions on new sample + sub file for level 2
        return train,test


class LevelTwo(object):
    def __init__(self,data):
        # load level 1 dataframe feature by feature 
        # split training and testing data

    def train(self):
        #train one model of each

    def get_level3_data(self):
        # apply trained models on the submission dataset from level1

class LevelThree(object):
    def __init__(self,data):

    def get_submission(self):
        # weighted average of results from level2
    return self

    def submit_to_kaggle(self):
        #kaggle competitions submit [-h] -c COMPETITION -f FILE -m MESSAGE

    return self

