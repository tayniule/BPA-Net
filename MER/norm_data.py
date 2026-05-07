import json
import pickle
import numpy as np



def normalize(X,mean_value,std_value):
    mean_value, std_value = np.array(mean_value), np.array(std_value)
    mu, sigma = np.expand_dims(mean_value, axis=1), np.expand_dims(std_value, axis=1)
    X = (X - mu) / (sigma + 1e-8)
    return X

num_channels=20
for mode in ['train','val','test']:
    total_mean = np.zeros(num_channels)
    total_std = np.zeros(num_channels)
    with open(f'/code/clisa/clisa/Downstream_dataset/AdaBrain-Bench-LaBraM-fusion/preprocessing/MER/{mode}.json', 'rb') as f:
        files = json.load(f)

    print(mode)
    print(len(files['subject_data']))
    total = 0
    for file in files['subject_data']:
        with open(file,'rb') as f:
            data = pickle.load(f)

        for j in range(num_channels):
            total_mean[j] += data['sample'][j].mean()
            total_std[j] += data['sample'][j].std()
        total += 1

    mean = (total_mean / total).tolist()
    std = (total_std / total).tolist()


    for file in files['subject_data']:
        with open(file,'rb') as f:
            data = pickle.load(f)
        data['sample']=normalize(data['sample'],mean,std)

        with open(f'{file}', 'wb') as f:  # 注意是 'wb' (write binary) 模式
            pickle.dump(data, f)





