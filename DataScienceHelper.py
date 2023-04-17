import re
import math
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression

from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from numpy.lib.twodim_base import triu_indices_from
random.seed(1)

df = pd.read_csv("sample_data/california_housing_train.csv")
df_train = pd.read_csv("sample_data/california_housing_train.csv")
df_test = pd.read_csv("sample_data/california_housing_test.csv")
df.head()

response_variable_name = "median_house_value"
categorical_col_names = ""


def warn(*args, **kwargs):
    pass
warnings.warn = warn


class DataScienceHelper(object):
  def __init__(self, df, response_variable_name, categorical_col_names):
    self.df = df

    self.A = np.asarray(df)
    self.X = np.asarray(df.drop([response_variable_name], axis=1))
    self.y = np.asarray(df[response_variable_name]).reshape(len(self.X), 1)

    self.X_tr = None
    self.y_tr = None
    self.X_ts = None
    self.y_ts = None

    self.response_variable_name = response_variable_name
    self.categorical_col_names = {col_name: i for i, col_name in enumerate(categorical_col_names.split(","))}
    self.col_names = {name: i for i, name in enumerate(list(df.columns))}
    self.encoders = {"OneHot": OneHotEncoder, "Ordinal": OrdinalEncoder}
    self.regressionModels = {
        "linear models": [LinearRegression(), Lasso(), Ridge()],
        "logistic_models": [LogisticRegression()],
        "decision_tree_ensembles": [AdaBoostRegressor(), GradientBoostingRegressor()],
        "neural_networks": [MLPRegressor()]
    }
    self.classificationModels = {
        "control_models": [LinearRegression(), Ridge(), RidgeClassifier()],
        "logistic_models": [PassiveAggressiveClassifier(), LogisticRegression()],
        "decision_tree_ensembles": [AdaBoostClassifier(), GradientBoostingClassifier()],
        "neural_networks": [MLPClassifier()]
    }
    
    self.initialize()

  def initialize(self):
    self.A = np.asarray(self.df)
    self.y = np.asarray(self.df[self.response_variable_name])
    self.y = self.y.reshape(len(self.df), 1)
    self.df = self.df.drop([self.response_variable_name], axis=1)
    self.X = np.asarray(self.df)
    self.handleCategorical()
    return

  def unravel(self, lst):
    return [int(num) for num in np.ravel(lst)]

  def nearestInt(self, lst):
    return [int(num+0.5) for num in lst]

  def getAccuracy(self, pred, ans):
    return sum([1 if a == b else 0 for a,b in zip(pred, ans)])/len(pred)

  def showStatistic(self, selection, pred, ans):
    selection = selection.lower()
    if selection == "mse":
      mse = sum([(p-a)**2 for p,a in zip(pred, ans)]) / len(pred)
      print("----Mean Squared Error: " + str(mse))
    elif selection == "rmse":
      mse = math.sqrt(sum([(p-a)**2 for p,a in zip(pred, ans)]) / len(pred))
      print("----Root Mean Squared Error: " + str(mse))
    elif selection in ("error rate"):
      error_rate = sum([abs(p-a)/a for p,a in zip(pred, ans)]) / len(pred)
      print("----Error Rate: " + str(error_rate))
    elif selection in ("me", "mean error"):
      mean_error = sum([abs(p-a) for p,a in zip(pred, ans)]) / len(pred)
      print("----Mean Error: " + str(mean_error))
    elif selection == "accuracy":
      accuracy = self.getAccuracy(pred, ans)
      print("----Accuracy: " + str(accuracy))
    return

  def ordinallyEncode(self, j):
    if j == self.categorical_col_names[self.response_variable_name]:
      step = 0
      vis = {}
      for i in range(len(self.y)):
        if self.y[i][0] not in vis:
          vis[self.y[i][0]] = step
          step += 1
      
      new_y = []
      for i in range(len(self.y)):
        new_y += [vis[self.y[i][0]]]
      self.y = np.array(new_y).reshape(len(new_y), 1)
      return self.y
        
    ordinal_value = 0
    vis = {}
    for i in range(len(self.X)):
      if self.X[i,j] not in vis:
        vis[self.X[i,j]] = ordinal_value
        ordinal_value += 1
    for i in range(len(self.X)):
      self.X[i,j] = vis[self.X[i,j]]
    return
  
  def get_models(self):
    if self.response_variable_name in self.categorical_col_names:
      models = self.classificationModels
    else:
      models = self.regressionModels
    
    modelList = []
    for model_category in models:
      for model in models[model_category]:
        modelList.append(model)
    return modelList

  def handleCategorical(self):
    if not self.categorical_col_names:
      return
    for col_name in self.col_names:
      if col_name in self.categorical_col_names:
        self.ordinallyEncode(self.categorical_col_names[col_name])
    return

  def train_predict(self):
    if self.response_variable_name in self.categorical_col_names:
      models = self.classificationModels
    else:
      models = self.regressionModels
      
    for model_category in models.keys():
      for i in range(len(models[model_category])):
        print(models[model_category][i])
        models[model_category][i].fit(self.X_tr, self.y_tr)
        pred = models[model_category][i].predict(self.X_ts)
        pred = self.unravel(pred)
        ans = self.unravel(self.y_ts)
        if self.response_variable_name in self.categorical_col_names:
          self.showStatistic("accuracy", pred, ans)
        else:
          self.showStatistic("error rate", pred, ans)
          self.showStatistic("rmse", pred, ans)
    return self.get_models()


  def train_models(self, df_tests=None):
    if not df_tests:
      self.X_tr, self.X_ts, self.y_tr, self.y_ts = train_test_split(self.X, self.y,  test_size=0.25, random_state=1)
      self.models = self.train_predict()
    else:
      # Note: Categorical handling is not implemented for this yet
      for df_test in df_tests:
        self.X_tr = self.X
        self.y_tr = self.y
        df_cache = df_test.drop([self.response_variable_name], axis=1)
        self.X_ts = np.asarray(df_cache)
        self.y_ts = np.asarray(df_test[self.response_variable_name])
        self.y_ts = self.y_ts.reshape(len(self.y_ts), 1)
        self.models = self.train_predict()
    return self.get_models()


dataScienceHelper = DataScienceHelper(df_train, response_variable_name, categorical_col_names)
trained_models = dataScienceHelper.train_models([])
