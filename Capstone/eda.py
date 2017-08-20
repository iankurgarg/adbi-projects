import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import multiclass

from MLCFast import *




def plotCorrMatrix(data,title='Continuous Feature Correlation Figure 1'):
	fig = plt.figure(figsize = (50,50) )
	ax1 = fig.add_subplot(111)
	cmap = cm.get_cmap('jet', 30)
	cax = ax1.imshow(data.corr(), interpolation="nearest", cmap=cmap)
	ax1.grid(True)
	plt.title(title)
	labels=list(data.columns)
	ax1.set_xticklabels(labels,fontsize=6)
	ax1.set_yticklabels(labels,fontsize=6)
	# Add colorbar, make sure to specify tick locations to match desired ticklabels
	fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
	plt.show()

data = pd.read_csv('../Data/train.csv')
shuffle_vector = np.random.randint(low=0, high=data.shape[0], size=data.shape[0])
data = data.iloc[shuffle_vector,]
data.reset_index(drop=True, inplace=True)

#test = pd.read_csv('../Data/test.csv')

msk = np.random.rand(len(data)) < 0.8

train = data[msk]
cvData = data[~msk]

X = train.iloc[:,1:]
Y = train['label']
Y.reset_index(drop=True, inplace=True)

#plotCorrMatrix(X)
# No Correlation found among variables. 

normZ = Normalizer()
scaledX = normZ.fit_transform(X)
scaledTestX = normZ.transform(test)

pca = PCA()
scaledPCAX = pca.fit_transform(scaledX)
scaledPCAX = pd.DataFrame(scaledPCAX)
pcaX = scaledPCAX.iloc[:,:300]


testX = cvData.iloc[:,1:]
testY = cvData.iloc[:,0]#cvData['label']
testY.reset_index(drop=True, inplace=True)
scaledTestX = normZ.transform(testX)
pcaTestX = pd.DataFrame(pca.transform(scaledTestX)).iloc[:,:100]


#model = LogisticRegression()
#model = LinearDiscriminantAnalysis()
model = RandomForestClassifier(n_estimators = 100, n_jobs=-1)
#model = multiclass.OneVsRestClassifier(svm.SVC(kernel='rbf'),n_jobs=-1)


scores = cross_val_score(model, pcaX, Y, cv=10)

model.fit(pcaX, Y)


preds = model.predict(pcaTestX)
accuracy = sum(preds == testY)/float(len(preds))
print accuracy



model = runModel(pcaX, Y)
classes = np.unique(Y)
preds = predictFromModel(model, pcaTestX, classes)

calcAccuracy(preds, testY)


