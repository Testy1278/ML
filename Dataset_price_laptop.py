
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import pandas as pd


df = pd.read_csv('laptop_data.csv') # считываем таблицу

df = df.drop(["Unnamed: 0"], axis= 1) # удаляем ненужный столбец


encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore') # создаем объект кодироки OneHot

encoded_features = encoder.fit_transform(df[['Company', 'Cpu', 'Ram', 'Memory', 'OpSys']]) # кодируем с помощью ее
encoded_df = pd.DataFrame(encoded_features,
                         columns= encoder.get_feature_names_out(['Company', 'Cpu', 'Ram', 'Memory', 'OpSys'])) # создаем новый датафрейм на основе закодированных столбцов

df = df.drop(['Company', 'Cpu', 'Ram', 'Memory', 'OpSys'], axis= 1) # удаляем столбцы старого формата
df = pd.concat([df, encoded_df], axis= 1) # соединяем старые данные с новыми



target_encoder = TargetEncoder() 
df['ScreenResolution'] = target_encoder.fit_transform(df['ScreenResolution'], df['Price'])
df['Weight'] = target_encoder.fit_transform(df['Weight'], df['Price'])
df['TypeName'] = target_encoder.fit_transform(df['TypeName'], df['Price'])
df['Gpu'] = target_encoder.fit_transform(df['Gpu'], df['Price'])



