import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping,TensorBoard
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer,scale
import pickle
import numpy as np
from keras import backend as K
import time
from os import path

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
#print(train_df.head())
train_X = train_df.drop(columns=['Severity'])
train_y = train_df[['Severity']]

test_X = test_df.drop(columns=['Severity'])
test_y = test_df[['Severity']]

if path.exists('le.txt'):
    le=pickle.load(open('le.pickle','rb'))
else:
    le = LabelBinarizer()
    le.fit(train_y)
    pickle.dump(le,open("le.pickle","wb"))

train_X=scale(train_X)
train_y = le.transform(train_y)
test_X=scale(test_X)
test_y = le.transform(test_y)



dense_layers = [3,5,7,8]
layer_sizes = [128]
#get number of columns in training data
n_cols = train_X.shape[1]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:   
         NAME = "{}-nodes-{}-dense-{}".format(layer_size, dense_layer, int(time.time()))
         
         
         model = Sequential()

         #add layers to model

         model.add(Dense(250, activation='relu', input_shape=(n_cols,)))

         for i in range(dense_layer):
            model.add(Dense(layer_size, activation='relu'))
            model.add(Dropout(0.3))
        
         model.add(Dense(4, activation='sigmoid'))

         #compile model using accuracy to measure model performance
         tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

         #set early stopping monitor so the model stops training when it won't improve anymore
         #early_stopping_monitor = EarlyStopping(patience=3)

         model.fit(train_X, train_y, validation_split=0.2, epochs=500, callbacks=[tensorboard])

         print(NAME)
         pickle.dump(model,open(NAME+'.pickle',"wb"))
         print('SCORE:',model.evaluate(test_x, test_y, verbose=0))


'''
df=pd.read_csv('test.csv')
pred=le.inverse_transform(model.predict(df))
print(pred)

data={'Accident_ID':list(df['Accident_ID']),'Severity':pred}

dfs=pd.DataFrame(data, columns= ['Accident_ID', 'Severity'])

export_csv = dfs.to_csv ('export_dataframe.csv', index = None, header=True)
'''
