from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import KFold

def runModel(X, Y):
	X = X.values
	Y = Y.values
	# encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	# convert integers to dummy variables (i.e. one hot encoded)
	dummy_y = np_utils.to_categorical(encoded_Y)
	d = X.shape[1]
	c = len(np.unique(Y))
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=d, kernel_initializer='uniform', activation='relu'))
	model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
	model.add(Dense(c, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X, dummy_y, nb_epoch=100, batch_size=800,  verbose=2)
	print 'Training Complete'
	return model

def saveModel(model, path):
	model.save(path)
	print 'Model saved'
	#model.save('../trained_models/ff_nn_3.h5')

def predictFromModel(model, testX, classes):
	testX = testX.values.astype(float)
	predictions = model.predict(testX)
	#print predictions
	predictions = np.argmax(predictions,axis=1)
	p = classes[predictions]
	return p


def calcAccuracy(testY, p):
	# print the accuracy
	#testY = testY.values
	accuracy = sum(testY == p)/float(len(testY))
	return accuracy


if __name__ == '__main__':
	kf = KFold(n_splits=10)
	scores = []
	for train_index, test_index in kf.split(pcaX):
		X_train, X_test = pcaX.iloc[train_index,], pcaX.iloc[test_index,]
		y_train, y_test = Y[train_index], Y[test_index]
		model = runModel(X_train, y_train)
		classes = np.unique(y_train)
		pred = predictFromModel(model, X_test, classes)
		acc = calcAccuracy(y_test, pred)
		scores.append(acc)

