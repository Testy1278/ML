
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv("Expanded_data_with_more_features.csv")

df = df.drop(["Unnamed: 0"], axis= 1)

df['EthnicGroup'] = df["EthnicGroup"].fillna("Group C")
df['ParentEduc'] = df["ParentEduc"].fillna("some college")
df['LunchType'] = df["LunchType"].fillna("standard")
df['TestPrep'] = df["TestPrep"].fillna("none")
df['ParentMaritalStatus'] = df["ParentMaritalStatus"].fillna("married")
df['PracticeSport'] = df["PracticeSport"].fillna("sometimes")
df['IsFirstChild'] = df["IsFirstChild"].fillna("yes")
df['NrSiblings'] = df["NrSiblings"].fillna(0.0)
df['TransportMeans'] = df["TransportMeans"].fillna("school_bus")
df['WklyStudyHours'] = df["WklyStudyHours"].fillna("< 5")

Encoder = OneHotEncoder(sparse_output= False, drop= 'first', handle_unknown= 'ignore')

encoded_feature = Encoder.fit_transform(df[["ParentEduc", "TransportMeans", "LunchType", "EthnicGroup", "WklyStudyHours"]])
encoded_df = pd.DataFrame(encoded_feature, columns= Encoder.get_feature_names_out(["ParentEduc", "TransportMeans",
                                                                                    "LunchType", "EthnicGroup", "WklyStudyHours"]))

df = df.drop(["ParentEduc", "TransportMeans", "LunchType", "EthnicGroup", "WklyStudyHours"], axis= 1)
df = pd.concat([df, encoded_df], axis= 1)

label_encoder = LabelEncoder()

df['NrSiblings'] = label_encoder.fit_transform(df["NrSiblings"])
df['IsFirstChild'] = label_encoder.fit_transform(df["IsFirstChild"])
df['ParentMaritalStatus'] = label_encoder.fit_transform(df["ParentMaritalStatus"])
df['PracticeSport'] = label_encoder.fit_transform(df["PracticeSport"])
df['TestPrep'] = label_encoder.fit_transform(df["TestPrep"])
df['Gender'] = label_encoder.fit_transform(df["Gender"])
