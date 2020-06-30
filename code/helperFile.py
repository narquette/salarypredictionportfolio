import pandas as pd
import numpy as np
import logging
import datetime
import time 
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,  cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
import xgboost as xgb
import lightgbm as lgb

# store dictionaries for 

# set dictionaries for ordinal encoding
jobtype_ord_map = {
    'JANITOR': 1,
    'JUNIOR': 2,
    'SENIOR': 3,
    'MANAGER': 4,
    'VICE_PRESIDENT': 5,
    'CFO': 6,
    'CTO': 7,
    'CEO': 8
}

degree_ord_map = {
    'NONE': 1,
    'HIGH_SCHOOL': 2,
    'BACHELORS': 3,
    'MASTERS': 4,
    'DOCTORAL': 5
}

# set dictionaries for tuning models
random_tune = {   
    'max_depth' : [1 ,2 ,3 ,4 ,5 ,10 ,20 ,50],
    'n_estimators':  [1, 2, 3, 4, 5, 15, 20, 25, 40, 50, 70, 100],
    'max_features' : ["auto", None, "sqrt", "log2", 0.7, 0.2],
    'min_samples_leaf' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
}

xgb_tune = {   
    'learning_rate' : [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
    'max_depth':  [1 ,2 ,3 ,4 ,5 ,10 ,20 ,50],
    'n_estimators' : [1, 2, 3, 4, 5, 15, 20, 25, 40, 50, 70, 100]    
}

lgb_tune = {   
    'learning_rate' : [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
    'max_depth':  [1 ,3 ,5 ,7 ,9 ,11, 13, 15], # maximum value is 17 based on num leaves constraint
    'num_leaves' : [2, 7, 31, 126, 2020, 8000, 10000], # recommended to not be larger than 2^(max depth), max value is 131072
    'n_estimators' : [1, 2, 3, 4, 5, 15, 20, 25, 40, 50, 70, 100],
    'min_data_in_leaf': [100, 250, 500, 750, 1000]
}

# set column names
numeric_columns =  ['yearsExperience', 'milesFromMetropolis']
category_columns = ['industry', 'jobType', 'degree', 'major']

class MachineLearning():
    
    def __init__(self, data, label, log_file):
        
        self.data = data
        self.label = label
        self.train_data = pd.read_csv(self.data, low_memory=False)
        self.train_data[label] = np.log(self.train_data[label]) + 1
        
        # set X and y
        self.X = self.train_data.drop(self.label, axis=1)
        self.y = self.train_data[[self.label]]
        
        # Split Train and Test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                    self.y,
                                                    test_size=0.33,
                                                    random_state=42)
        # Set log file name and start time
        self.start = time.time()
        now = datetime.datetime.now()
        self.file_name = f"{log_file}_{now.year}_{now.month}_{now.day}.log"
        
        # Start Logging
        self.logging = logging.basicConfig(format='%(asctime)s %(message)s', filename=self.file_name, level=logging.DEBUG)
        
    def PreProcessing(self, scale_cols, one_hot_cols):#, ordinal_cols):
        
        # set transformer methods
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
        label_encode = LabelEncoder()
#         job_ordinal_encode = OrdinalEncoder(categories=[np.array(list(jobtype_ord_map.keys()), dtype=object)])
#         deg_ordinal_encode = OrdinalEncoder(categories=[np.array(list(degree_ord_map.keys()), dtype=object)])
        imputer = SimpleImputer(add_indicator=True, verbose=1)
        scaler = PowerTransformer()

        # Make Transformer
        self.preprocessing = make_column_transformer(
            (make_pipeline(imputer,scaler), scale_cols),
             #(job_ordinal_encode, ordinal_cols[0]),
             #(deg_ordinal_encode, ordinal_cols[1]),
            (ohe, one_hot_cols),
            remainder='drop')
        
       
    def CrossValidation(self):
        
        # start logging
        start = time.time()
        logging.info(f"{self.model_name} Start")
        
        # model
        model = TransformedTargetRegressor(regressor=self.pipe, transformer=PowerTransformer())
          
        # evaluate model
        cross_score = cross_val_score(model, self.X, self.y, scoring='neg_mean_absolute_error', cv=None, n_jobs=-1)
           
        # convert scores to positive
        abs_score = np.absolute(cross_score)
           
        # summarize the result
        score = np.mean(abs_score)
            
        # set log for finishing
        logging.info(f"Score for {self.model_name} is {score}")
        logging.info(f"Run Time for {self.model_name} is {(time.time() - self.start) // 60} minutes")
        
        return score
        

    def Scoring(self):
        
        # start logging
        start = time.time()
        logging.info(f"{self.model_name} Start")
        
        # Score
        score = self.pipe.score(self.X_test, self.y_test.values.ravel()) 

        # set log for finishing
        logging.info(f"Score for {self.model_name} is {score}")
        logging.info(f"Run Time for Linear Regression is {(time.time() - self.start) // 60} minutes")
        
        # close logging file
        logging.FileHandler(self.file_name).close()
        
        return score
        

    def LinearRegression(self, cross_validation=False):

        # set model name
        self.model_name = 'Linear Regression'
               
        # put model into an object
        linear_regression = LinearRegression()

        # Make pipeline
        self.pipe = make_pipeline(self.preprocessing, linear_regression)        

        # Fit model
        self.pipe.fit(self.X_train, self.y_train.values.ravel())

        if cross_validation:
            
            score = self.CrossValidation()
            
        else:
            
            score = self.Scoring()
            
        # return best score
        return score
        
    def RandomForest(self, parameter_dict, cross_validation=False):
        
        # set model name
        self.model_name = 'RandomForest'
        
        # put model into an object
        random_forest = RandomForestRegressor(criterion='mse',
                                               oob_score=True, 
                                               n_jobs=-1, 
                                               random_state=44,
                                               bootstrap=True,
                                               max_samples=0.2)
        
        # set random forest parameter
        random_forest.set_params(**parameter_dict)
            
        # Make pipeline
        self.pipe = make_pipeline(self.preprocessing, random_forest)        

        # Fit model
        self.pipe.fit(self.X_train, self.y_train.values.ravel())

        if cross_validation:
            
            score = self.CrossValidation()
            
        else:
            
            score = self.Scoring()
            
        # return best score
        return score
        

    def XGboost(self, parameter_dict, cross_validation=False):
        
        # set model name
        self.model_name = 'XGboost'
        
        # put model into an object
        xg_boost = xgb.XGBRegressor(base_score=0.5, 
                                booster='gbtree', 
                                colsample_bylevel=1,
                                colsample_bytree=1, 
                                max_delta_step=0,
                                missing=None, 
                                n_jobs=-1,
                                nthread=None, 
                                objective='reg:squarederror', 
                                random_state=44,
                                reg_alpha=0, 
                                reg_lambda=1, 
                                scale_pos_weight=1, 
                                seed=None,
                                silent=True, 
                                subsample=1)
        
        # set random forest parameter
        xg_boost.set_params(**parameter_dict)
            
        # Make pipeline
        self.pipe = make_pipeline(self.preprocessing, xg_boost)        

        # Fit model
        self.pipe.fit(self.X_train, self.y_train.values.ravel())

        if cross_validation:
            
            score = self.CrossValidation()
            
        else:
            
            score = self.Scoring()
            
        # return best score
        return score

    def LGboost(self, parameter_dict, cross_validation=False):
        
        # set model name
        self.model_name = 'LGboost'
        
        # put model into an object
        lg_boost = lgb.LGBMRegressor(boosting_type='gbdt', 
                                feature_fraction=.8, 
                                n_jobs=-1,
                                nthread=None, 
                                objective='regression', 
                                random_state=44,
                                scale_pos_weight=1, 
                                bagging_fraction=1)
        
        # set random forest parameter
        lg_boost.set_params(**parameter_dict)
            
        # Make pipeline
        self.pipe = make_pipeline(self.preprocessing, lg_boost)        

        # Fit model
        self.pipe.fit(self.X_train, self.y_train.values.ravel())

        if cross_validation:
            
            score = self.CrossValidation()
            
        else:
            
            score = self.Scoring()
            
        # return best score
        return score
        
