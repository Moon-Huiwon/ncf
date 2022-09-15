
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import pandas as pd
import scipy.sparse as sp


##################################config.py#################################
config = {
    "model": "GMF",
    "model_path": "./models/",
    "train_rating": R"C:\Users\문희원\Desktop\neural_collaborative_filtering-master\Data\ml-1m.train.rating",
    "train_negative": R"C:\Users\문희원\Desktop\neural_collaborative_filtering-master\Data\ml-1m.train.negative",
    "test_negative": R"C:\Users\문희원\Desktop\neural_collaborative_filtering-master\Data\ml-1m.test.negative",
}

args = {
    "batch_size": 256,
    "dropout": 0.0,
    "epochs": 20,
    "factor_num": 32,
    "gpu": "0",
    "lr": 0.001,
    "num_layers": 3,
    "num_ng": 4,
    "out": True,
    "test_num_ng": 99,
    "top_k": 10,
}
os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]
cudnn.benchmark = True

#################################data_utils.py#################################
#학습데이터, 테스트 데이터 설정
train_data = pd.read_csv(
    config["train_rating"],
    sep="\t",
    header=None,
    names=["user", "item"],
    usecols=[0, 1],
    dtype={0: np.int32, 1: np.int32},
)



user_num = train_data["user"].max() + 1
item_num = train_data["item"].max() + 1

train_data = train_data.values.tolist()

# dok matrix 형식으로 저장
train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
for x in train_data:
    train_mat[x[0], x[1]] = 1.0
        
test_data = []
with open(config["test_negative"], "r") as fd:
    line = fd.readline()
    while line != None and line != "":
        arr = line.split("\t")
        u = eval(arr[0])[0]
        test_data.append([u, eval(arr[0])[1]])
        for i in arr[1:]:
            test_data.append([u, int(i)])
        line = fd.readline()
        

#################################data_utils.py#################################
class NCFData(data.Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()

        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training        
        self.labels = [0] * len(features)
        
    def set_ng_sample(self):
        assert self.is_training, "no need to sampling when testing"
        
        # negative sample 더하기
        self.features_ng = []
        for x in self.features_ps:
            # user
            u = x[0]
            for _ in range(self.num_ng):
                j = np.random.randint(self.num_item)
                #train set에 잇는 경우 다시 뽑기
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])
                
        labels_ps = [1] * len(self.features_ps)
        labels_ng = [0] * len(self.features_ng)
        
        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng
        
    def __len__(self):
        return(self.num_ng + 1) * len(self.labels)
    
    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels
        
        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label


###############################PREPARE DATASET################################
train_dataset = NCFData(train_data, item_num, train_mat, args["num_ng"], True)
test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(
    train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=0
)
test_loader = data.DataLoader(
    test_dataset, batch_size=args["test_num_ng"] + 1, shuffle=False, num_workers=0
)

##################################model.py#####################################
class GMF(nn.Module):
    def __init__(
        self, user_num, item_num, factor_num, num_layers, dropout, model,
    ):
        super(GMF, self).__init__()
        self.dropout = dropout
        self.model = model
        
        #임베딩 저장공간 확보; num_embeddings, embedding_dim
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        predict_size = factor_num
        
        self.predict_layer = nn.Linear(predict_size, 1)
        self._init_weight_()
        
    def _init_weight_(self):
        #weigth 초기화
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")
        
        #bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        concat = output_GMF
        
        prediction = self.predict_layer(concat)
        return prediction.view(-1)

#################################CREATE MODEL#################################

model = GMF(
    user_num,
    item_num,
    args["factor_num"],
    args["num_layers"],
    args["dropout"],
    config["model"],
)
model.cpu()
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args["lr"])



#####################################evaluate.py##############################
def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0

def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0

def metrics(model, test_loader, top_k):
    HR, NDCG = [], []
    
    for user, item, _ in test_loader:
        user = user.cpu()
        item = item.cpu()
        
        predictions = model(user, item)
        # 가장 높은 top_k개 선택
        _, indices = torch.topk(predictions, top_k)
        # 해당 상품 index 선택
        recommends = torch.take(item, indices).cpu().numpy().tolist()
        # 정답값 선택
        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))
        
    return np.mean(HR), np.mean(NDCG)


##################################TRAINING####################################
if __name__ == "__main__":
    count, best_hr = 0, 0
    writer = SummaryWriter()
    for epoch in range(args["epochs"]):
        model.train()
        start_time = time.time()
        train_loader.dataset.set_ng_sample()
        
        for user, item, label in train_loader:
            user = user.cpu()
            item = item.cpu()
            label = label.float().cpu()
            
            #gradient 초기화
            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            writer.add_scalar("data/loss", loss.item(), count)
            count += 1
            
        model.eval()
        HR, NDCG = metrics(model, test_loader, args["top_k"])
        
        elapsed_time = time.time() - start_time
        print(
            "The time elapse of epoch {:03d}".format(epoch)
            + " is: "
            + time.strftime("%H: %M: %S", time.gmtime(elapsed_time))
        )
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
        
        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args["out"]:
                if not os.path.exists(config["model_path"]):
                    os.mkdir(config["model_path"])
                torch.save(
                    model, "{}{}.pth".format(config["model_path"], config["model"])
                )
                
    print(
        "End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
            best_epoch, best_hr, best_ndcg
        )
    )

