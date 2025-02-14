
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd


df = pd.read_csv('laptop_data.csv') # считываем таблицу

df = df.drop(["Unnamed: 0"], axis= 1) # удаляем ненужный столбец


encoder = OneHotEncoder(sparse_output=False, drop='first')

encoded_feathur = encoder.fit_transform(df[['Company', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys']])
encoded_df = pd.DataFrame(encoded_feathur, columns= encoder.get_feature_names_out(['Company', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys']))

df = df.drop(['Company', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys'], axis= 1)
df = pd.concat([df, encoded_df], axis= 1)

label_encoder = LabelEncoder()
df['Weight'] = label_encoder.fit_transform(df['Weight'])
df['TypeName'] = label_encoder.fit_transform(df['TypeName'])
df[ 'ScreenResolution'] = label_encoder.fit_transform(df[ 'ScreenResolution'])



