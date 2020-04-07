from sklearn.model_selection import train_test_split
import pandas as pd

train_df = pd.read_csv('train.csv')
td=train_df[:8001]
ted=train_df[8001:]
td.to_csv('train1.csv')
ted.to_csv('test1.csv')
print('done')
