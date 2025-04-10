import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#configuration the file
file_path = 'adult.data'  
train_out = 'train.data'    
test_out  = 'test.data'     
test_size = 0.2                   
random_state = 42

# 1. read data

df = pd.read_csv(file_path, header=None, sep=',', engine='python')


df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)




categorical_cols = [1, 3, 5, 6, 7, 8, 9, 13]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


def map_income(x):
    if x == '<=50K':
        return 0
    elif x == '>50K':
        return 1
    else:
        return x

df[14] = df[14].apply(map_income)


labels = df[14]
features = df.drop(columns=[14])
df_out = pd.concat([labels, features], axis=1)


train_df, test_df = train_test_split(df_out, test_size=test_size, random_state=random_state, stratify=labels)



def write_dataset(df_subset, output_path):
    n_samples = df_subset.shape[0]
  
    n_features = df_subset.shape[1] - 1
    n_classes = df_subset.iloc[:, 0].nunique()
    
    with open(output_path, 'w') as f:
        f.write(f"{n_samples}\n{n_features}\n{n_classes}\n")
      
        for _, row in df_subset.iterrows():
           
            label = int(row.iloc[0])
            feature_vals = row.iloc[1:].tolist()
        
            features_str = ' '.join(map(str, feature_vals))
            f.write(f"{label} {features_str}\n")


write_dataset(train_df, train_out)
write_dataset(test_df, test_out)


