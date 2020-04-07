import pandas as pd
import pickle
import warnings
import os
from sklearn.preprocessing import scale
warnings.filterwarnings("ignore")

for i in os.listdir('top/'):
   model=pickle.load(open('top/'+i,'rb'))

   df=pd.read_csv('test.csv')

   '''
   X=scale(df.drop(columns=['Severity']))
   y=le.transform(df[['Severity']])
   print('SCORE:',model.evaluate(X, y, verbose=0))
  '''

   X=scale(df)
   pred=le.inverse_transform(model.predict(X))
   print(pred)

   data={'Accident_ID':list(df['Accident_ID']),'Severity':pred}

   dfs=pd.DataFrame(data, columns= ['Accident_ID', 'Severity'])

   export_csv = dfs.to_csv ('top/'+i+'.csv', index = None, header=True)
