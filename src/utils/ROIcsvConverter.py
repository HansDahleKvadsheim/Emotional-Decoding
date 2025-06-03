import scipy.io
import numpy as np
import pandas as pd


mat_data = scipy.io.loadmat('Alice/Results/ROIsignal/sub-18_ROISignal.mat')
print(mat_data.keys())
data = mat_data['ROISignal'] 

print(data.shape)


n_rows = data.shape[0]
index = np.arange(n_rows)

# first column is empty (we’ll use it as row index), then 'img_timestamp' then 'ROISingal_0' ... 'ROISingal_399'
columns = ['img_timestamp'] + [f'ROISingal_{i}' for i in range(data.shape[1]-1)]
df = pd.DataFrame(data, columns=columns)

df.insert(0, '', index)

df.to_csv('ROIsignal.csv', index=False)

print("CSV file created as ROIsignal.csv")
