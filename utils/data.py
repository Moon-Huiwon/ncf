#%%
import torch
import numpy as np
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

#%%
user = pd.read_csv(R"C:\Users\문희원\Desktop\neural-collaborative-filtering-master\src\data\ml-1m\users.csv")
item = pd.read_csv(R"C:\Users\문희원\Desktop\neural-collaborative-filtering-master\src\data\ml-1m\movies.csv")
ratings = pd.read_csv(R"C:\Users\문희원\Desktop\neural-collaborative-filtering-master\src\data\ml-1m\ratings.csv")

user = user.drop(['gender', 'zipcode'], axis=1)
item = item.drop(['title', 'genre'], axis=1)

user = user.to_numpy()
item = item.to_numpy()
target = ratings.to_numpy()

user_tensor = torch.Tensor(user)
item_tensor = torch.Tensor(item)
target_tensor = torch.Tensor(target)

#%%
class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]
    
    def __len__(self):
        return self.user_tensor.size(0)

# %%
# ratings 파일의 rating column 값이 0보다 크면 1.0으로 값을 변환해라
ratings = deepcopy(ratings)
ratings['rating'][ratings['rating'] > 0] = 1.0
ratings

#%%
ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
ratings.head(10)
# %%
ratings[ratings['userId'] == 0]
#%%
