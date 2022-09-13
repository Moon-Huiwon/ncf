#%%
import torch
from torch import nn
import numpy as np
#%%
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(10, 15)
        self.linear2 = nn.Linear(10, 20)

    def forward(self, input):
        out = self.linear1(input)
        out = self.linear2(out)
        return out
#%%
mlp = MLP() # instancication 아이폰 생성
sample = torch.randn(10, 10)
out = mlp(sample)
#%%
linear1 = nn.Linear(10, 15)
linear2 = nn.Linear(15, 20)

input = torch.rand(10, 10)
input.shape
out = linear1(input) 
out.shape
out = linear2(out) 
out.shape
#%%
"""
<linear>
y = Wx + b

convolution

embedding

"""
#%%
B = 16
input_dim = 8 # p
output_dim = 4 # d

x = torch.randn(B, input_dim)
x.shape
linear = nn.Linear(input_dim, output_dim)
y = linear(x)
y.shape

linear.weight.data.shape
linear.bias.data.shape
#%%
B = 32
input_dim = 8
x = torch.randn(B, input_dim)

linear1 = nn.Linear(input_dim, 6)
linear2 = nn.Linear(6, 3)
linear3 = nn.Linear(3, 4)
linear4 = nn.Linear(4, 6)

y = linear4(linear3(linear2(linear1(x))))
y.shape
#%%
relu = nn.ReLU()
relu(y).shape
#%%
x = torch.randn(32, 3, 16, 16)
conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=4, stride=4, padding=1)
y = conv(x)
y.shape
#%%
embed = nn.Embedding(num_embeddings=100, embedding_dim=8)
idx = torch.from_numpy(np.array([[4, 8, 6, 8, 9],
                                 [4, 8, 6, 7, 6]]))
idx.shape
idx
y = embed(idx)
y.shape
#%%