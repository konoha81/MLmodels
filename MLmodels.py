#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import xgboost as xgb
import pandas as pd
import datetime, os


class MLmodels(object):
    """
    機械学習モデルのセット。
    作成済み：XGboost, RandomForest, ExtraTree,
    （追加作成予定：SVM, Logistic regression)
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def XGmodel(self, test=False):
        """
        XGboost。 test=Trueのときはテスト用パラメータで動く。
        """

        # GridSearch用パラメータ
        parameters = {'max_depth': [2,5,7],
                      'n_estimators': [100,200,500,1000],
                      'subsample': [0.95],
                      'colsample_bytree': [1.0]
                  }
        # テスト用
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
        """
        RandomForest test=Trueのときはテスト用パラメータで動く。
        """

        # GridSearch用パラメータ
        parameters = {
                'n_estimators'      : [50, 100, 500, 1000, 1500],
                'max_features'      : [1, 3, 10],
                'random_state'      : [0],
                'n_jobs'            : [-1],
                'min_samples_split' : [3, 5, 20],
                'max_depth'         : [3, 5, 20, 30],
                'class_weight'      : ['balanced']
        }

        # テスト用
        test_parameters = {
                'n_estimators'      : [100],
                'max_features'      : ['auto'],
                'random_state'      : [0],
                'n_jobs'            : [-1],
                'min_samples_split' : [3],
                'max_depth'         : [3],
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


    def EXTmodel(self, test=False):
        # Extremely Randomized Trees

        # GridSearch用パラメータ
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
        clf = GridSearchCV(estimator= ExtraTreesClassifier(),
                           param_grid= parameters,
                          scoring= 'accuracy',
                          cv= 10,
                          n_jobs= -1)

        return clf

    def LGmodel(self, test=False):
        # Logistic Regression

        ### 数値の正規化をする
        print('Unimplemented')
        sys.exit()

        parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

        test_parameters = {'C': [0.001, 0.01] }

        if test==True:
            parameters = test_parameters

        # GridSearchを実行
        clf = GridSearchCV(estimator= LogisticRegression(),
                           param_grid= parameters,
                          scoring= 'accuracy',
                          cv= 10,
                          n_jobs= -1)

        return clf


    def fit_and_prediction(self, clf, save=False):
        """
        モデルの fit と テストデータでの predict を行う。
        save=True でモデルをpickle化して保存する。
        """

        clf.fit(self.X_train,self.y_train)

        predict = clf.predict(self.X_test)

        print('Accuracy: {}'.format(accuracy_score(self.y_test, predict)))
        print('Best parameters:{}'.format(clf.best_params_))
        print('Confusion matrix:{}'.format(confusion_matrix(self.y_test, predict)))
        print(classification_report(self.y_test, predict))

        # モデルの保存
        if save == True:
            # 実行時の日付と時間を取得
            todaydetail = datetime.datetime.today()
            td = todaydetail.strftime('%y%m%d_%H%M')

            # モデルの保存先がない場合はディレクトリを作成する。
            if os.path.exists('./pickled_model/'):
                os.mkdir("pickled_model")

            # pkl化する
            joblib.dump(clf, './pickled_model/{}_{}.pkl'.format(td))

            # 保存したモデル名をprint
            print('Model Saved: {}_{}.pkl'.format(td))


    def RFmodel_call(self, modelname):
        '''
        作成したモデル（modelnameで指定）を読み込む
        '''
        clf = joblib.load('./pickled_model/{}'.format(modelname + '.pkl'))

        predict = clf.predict(self.X_test)

        print('Best parameters:{}'.format(clf.best_params_))
        print('Confusion matrix:{}'.format(confusion_matrix(self.y_test, predict)))
        print(classification_report(self.y_test, predict))

        # 重要度をCSV保存
        fti = clf.best_estimator_.feature_importances_
        features = pdf.iloc[:,1:].columns

        # 特徴量とその重要度を DataFrame に入力する。
        f = pd.DataFrame({'features': features,
                 'feature_importances': fti
                 })
        f = f.sort_values('features',ascending=False)

        # CSVの保存先がない場合はディレクトリを作成する。
        if os.path.exists('./importances/'):
            os.mkdir('importances')

        f.to_csv('./importances/{}.csv'.format(modelname))
