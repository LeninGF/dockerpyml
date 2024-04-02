import pandas as pd
import os
import shutil

def create_file_dict(directory):
    file_dict = {}
    
    for root, dirs, files in os.walk(directory):
        key = root.split(os.sep)[-1]
        file_dict[key] = [os.path.join(root, file) for file in files]
            # print(root.split(os.sep)[-1], dirs, file)
    return file_dict



root = 'digits\\archive\\dataset'

file_dict = create_file_dict(root)
file_dict.pop('dataset')

# for key in file_dict.keys():
#     print(key, len(file_dict[key]))
data = [(key, path) for key, paths in file_dict.items() for path in paths]
# print(data)

df = pd.DataFrame(data, columns=['Digit', 'Path'])
df.to_csv('dataset.csv', index=False)
df_sample = df.groupby('Digit').sample(n=10, replace=True)
df_sample.to_csv('sample_dataset.csv', index=False)

for index, row in df_sample.iterrows():
    key = row['Digit']
    this_path = row['Path']
    new_dir = os.path.join('sample_dataset', key)
    os.makedirs(new_dir, exist_ok=True)
    shutil.copy(this_path, new_dir)        
