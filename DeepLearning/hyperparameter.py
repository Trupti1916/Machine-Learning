# -*- coding: utf-8 -*-

#!pip install keras-tuner
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization
from keras.activations import relu, sigmoid
from sklearn.metrics import confusion_matrix, accuracy_score


#from keras.wrappers.scikit_learn import kerasClassifier

data = pd.read_csv('Churn_Modelling.csv')

data.head(5)

X = data.iloc[:,:-1] #independent features
y = data.iloc[:,-1]

y.head()

def model_build(para):
  model = keras.Sequential()
  for i in range(para.int('num_layers', 2, 20)):
    model.add(layers.Dense(units = para.int('units_', +str(i),
                                           min_value =32,
                                            max_value=512,
                                            step=32),
                                           activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizers= keras.optimizers.Adam(
            para.choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='mean_absolute_error',
            metrics=['mean_absolute_error'])
    return model

geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

X = pd.concat([X, geography, gender], axis=1)
X = X.drop(['Geography', 'Gender'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_train = X_train.drop(['Surname'], axis=1)
X_train.head(5)

X_test = X_test.drop(['Surname'], axis=1)
X_test.head(4)

X_train.shape[1]

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_test)

def create_model(Layers, activation):
  model = Sequential()
  for i, nodes in enumerate(Layers):
    if i == 0:
      model.add(Dense(nodes, input_dim = X_train.shape[1]))
      model.add(Activation=activation)
      model.add(Dropout(0.3))
    else:
      model.add(Dense(nodes))
      model.add(Activation = activation)
      model.add(Dropout(0.3))
  model.add(Dense(units=1, kernel_initializer='glorot_uniform', activtaion='sigmoid'))

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

model = kerasClassifier(build_fn= create_model, verbose=0)

Layers = [[20], [40, 20], [45,30, 15]]
activation = ['sigmoid', 'relu']
param_grid = dict(layers=Layers, activations= activation, batch_size= [128, 256], epochs = [30])

grid = GridSearchCV(estimator=  model, param_grid= param_grid, cv =5)

grid_result = grid.fit(X_train, y_train)

grid_result.best_score_, grid_result.best_params_

pred = grid.predict(X_test)
y_pred = (pred > 0.5)

cm = confusion_matrix(y_pred, y_test)
score = accuracy_score(y_pred, y_test)



