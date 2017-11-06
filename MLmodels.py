#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import datetime

import xgboost as xgb

class MLmodels(object):

    def __init__(self, P_or_M, X_train, X_test, y_train, y_test):
        self.P_or_M = P_or_M
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def XGmodel(self, test=False):

        parameters = {'max_depth': [2,5,7],
                      'n_estimators': [100,200,500,1000],
                      'subsample': [0.95],
                      'colsample_bytree': [1.0]
                  }

        test_parameters = {'max_depth': [2],
                           'n_estimators': [50],
                           'subsample': [0.95],
                           'colsample_bytree': [1.0]
                       }

        if test==True:
            parameters = test_parameters

        xgb_model = xgb.XGBClassifier()

        clf = GridSearchCV(xgb_model,
                parameters,
                cv=10,
                scoring="log_loss",
                n_jobs=10,
                verbose=2)
        return clf


    def RFmodel(self, test=False):
        # RFのGridSearch用パラメータ
        parameters = {
                'n_estimators'      : [700, 1000, 1250, 1500, 2000],
                'max_features'      : [1, 3, 10],
                'random_state'      : [0],
                'n_jobs'            : [-1],
                'min_samples_split' : [3, 5, 20],
                'max_depth'         : [3, 5, 20, 30],
                'class_weight'      : ['balanced']
        }

        # テスト用
        test_parameters = {
                'n_estimators'      : [1500],
                'max_features'      : ['auto'],
                'random_state'      : [0],
                'n_jobs'            : [-1],
                'min_samples_split' : [5],
                'max_depth'         : [20],
                'class_weight'      : ['balanced']
        }

        if test==True:
            parameters = test_parameters

        # GridSearchを実行
        clf = GridSearchCV(estimator= RandomForestClassifier(),
                           param_grid= parameters,
                          scoring= 'accuracy',
                          cv= 10,
                          n_jobs= -1)

        return clf

    def fit_and_prediction(self, clf, save=False):

        clf.fit(self.X_train,self.y_train)

        predict = clf.predict(self.X_test)

        print('Accuracy: {}'.format(accuracy_score(self.y_test, predict)))
        print('Best parameters:{}'.format(clf.best_params_))
        print('Confusion matrix:{}'.format(confusion_matrix(self.y_test, predict)))
        print(classification_report(self.y_test, predict))

        if save == True:
            # モデルの保存
            todaydetail = datetime.datetime.today()
            td = todaydetail.strftime('%y%m%d_%H%M')
            # モデルの名前は patient, medicine で変更する
            joblib.dump(clf, './pickled_model/RF_{}_{}.pkl'.format(self.P_or_M ,td))
            print('model saved.')

            # 保存したモデル名を返す
            return 'RF_{}_{}.pkl'.format(self.P_or_M ,td)


    def RFmodel_call(self, modelname):
        '''
        作成したモデルを読み込む
        '''
        clf = joblib.load('./pickled_model/{}'.format(modelname + '.pkl'))

        predict = clf.predict(self.X_test)
        print('Best parameters:{}'.format(clf.best_params_))
        print('Confusion matrix:{}'.format(confusion_matrix(self.y_test, predict)))
        print(classification_report(self.y_test, predict))

        # 重要度をCSV保存
        fti = clf.best_estimator_.feature_importances_
        features = pdf.iloc[:,1:].columns

        f = pd.DataFrame({'features': features,
                 'feature_importances': fti
                 })
        f = f.sort_values('features',ascending=False)

        f.to_csv('./importances/{}.csv'.format(modelname))
