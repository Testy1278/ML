
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import pandas as pd


df = pd.read_csv('laptop_data.csv') # считываем таблицу

df = df.drop(["Unnamed: 0"], axis= 1) # удаляем ненужный столбец


encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

encoded_features = encoder.fit_transform(df[['Company', 'Cpu', 'Ram', 'Memory', 'OpSys']])
encoded_df = pd.DataFrame(encoded_features, columns= encoder.get_feature_names_out(['Company', 'Cpu', 'Ram', 'Memory', 'OpSys']))

df = df.drop(['Company', 'Cpu', 'Ram', 'Memory', 'OpSys'], axis= 1)
df = pd.concat([df, encoded_df], axis= 1)



target_encoder = TargetEncoder()
df['ScreenResolution'] = target_encoder.fit_transform(df['ScreenResolution'], df['Price'])
df['Weight'] = target_encoder.fit_transform(df['Weight'], df['Price'])
df['TypeName'] = target_encoder.fit_transform(df['TypeName'], df['Price'])
df['Gpu'] = target_encoder.fit_transform(df['Gpu'], df['Price'])



