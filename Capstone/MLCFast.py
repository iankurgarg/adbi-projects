import pandas as pd
import numpy as np
import pickle
import itertools as it
import operator


class MLCFast:
  def __init__(self):
    self.priorProb = []
    self.classes = None
    self.means = []
    self.covs = []
    self.predProbs = []
    self.bmaweight = 0.0
    return

  def get_params(self, deep=False):
    return {}

  def compute_apriori (self, df):
    counts = df.value_counts(sort=False)
    prior = counts/len(df)
    return np.array(prior)

  def compute_means(self, df):
    return np.mean(df, axis=0)

  def compute_covs (self, df):
    return np.matrix(np.cov(np.transpose(df)))

  def fit(self, X, y):
    self.classes = np.unique(y)

    for c in self.classes:
      classData = X[y == c]
      self.priorProb.append(len(classData))
      mu = self.compute_means(classData)
      self.means.append(mu)
      sigma = self.compute_covs(classData)
      self.covs.append(sigma)
    self.priorProb = list(np.array(self.priorProb)/float(len(X)))

  def calcProb(self, X, c):
    prior = self.priorProb[c]
    cv = self.covs[c]
    T1 = np.matrix((X - self.means[c]))
    T2 = np.linalg.inv(cv)
    T3 = np.transpose(T1)
    likelihood = (T1*T2)
    likelihood = likelihood*T3
    likelihood = np.diag(likelihood)

    #likelihood = np.exp(-likelihood)/np.sqrt(np.linalg.det(cv))
    D = 0.0
    if (np.linalg.det(cv) == 0.0):
      D = np.finfo(float).tiny
    else:
      D = np.sqrt(np.linalg.det(cv))
    likelihood = -likelihood - np.log(D)
    return np.log(prior) + likelihood

  def normalizeProbs(self, probs):
    probArray = np.array(probs)
    sumProb = np.sum(probArray, axis=1)
    sumProb = sumProb.reshape(len(sumProb),1)

    out = np.divide(probs, sumProb)
    return pd.DataFrame(out)


  def predict(self, testX, type='default'):
    L = len(testX)
    if len(np.matrix(testX)) == 1:
      L = 1

    x = np.empty([L,len(self.classes)])
    probList = pd.DataFrame(x)

    for j in range(len(self.classes)):
      prob = self.calcProb(testX, j)
      prob = np.array(prob).reshape(L,1)
      prob = pd.DataFrame(prob)
      probList[j] = prob
    
    self.predProbs = probList

    #probList = self.normalizeProbs(probList)

    if (type == 'default'):
      predIndex = np.argmax(np.matrix(probList), axis=1)
      predictions = list(pd.DataFrame(np.array(self.classes)[predIndex]).iloc[:,0])
      return predictions
    elif (type == 'raw'):
      return probList
    else:
      print ('Invalid type. Can be either default or raw')
    return None

  def score(self, testX, actual):
    predictions = self.predict(testX)
    accuracy = sum(predictions == actual)/float(len(actual))
    return accuracy

def class_accuracies (predictions, actual):
  predictions = np.array(predictions)
  actual = np.array(actual)
  tab = pd.crosstab(predictions, actual)
  l = len(tab)
  accuracies = []
  for i in range(l) :
    acc = tab.iloc[i, i]/float(sum(actual == (i+1)))
    accuracies.append(acc)

  return accuracies


def saveModel(model, filename):
  with open(filename,'wb') as f:
    pickle.dump(model,f)
