import torch
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import random

import sys
from torch.utils.tensorboard import SummaryWriter

sys.path.append(r'2.NetworkStructure')
from Func_MyModel import *

writer = SummaryWriter()

torch.manual_seed(0)  # 为CPU设置随机种子
torch.cuda.manual_seed(0)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(0)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(0)  # Numpy module.
random.seed(0)  # Python random module
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ==========训练参数========================================================================
# 设置主要参数
epochs = 200
batchs = 32
validation_split = 0.2


MyNet = MyModel().cuda()

loss_func = torch.nn.BCELoss()
train_losses = []
val_losses = []
train_accs = []
val_accs = []

read_temp = h5py.File('1.TrainingDataset/Dataset_Second_Approach.mat')
TrainInputs = np.array(read_temp["Train_input"])
TrainLabels = np.array(read_temp["Train_output"])
TrainInputs = TrainInputs.transpose((2, 1, 0))
TrainLabels = TrainLabels.transpose((1, 0))

# ==========通过继承Dataset预处理训练数据集，以便使用pytorch提供的DataLoader函数加载数据集======================================
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train_inputs, train_labels):
        self.data = torch.tensor(train_inputs, dtype=torch.float32)
        self.label = torch.tensor(train_labels, dtype=torch.float32)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

ProDataset = MyDataset(TrainInputs, TrainLabels)


train_size = int((1 - validation_split) * len(ProDataset))
val_size = len(ProDataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(ProDataset, [train_size, val_size])
MyTrain_dataloader = DataLoader(dataset=train_dataset, batch_size=batchs, shuffle=True)
MyVal_dataloader = DataLoader(dataset=val_dataset, batch_size=batchs, shuffle=True)

# ==========设置优化器====================================================================
optimizer = torch.optim.Adam(MyNet.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                                total_steps=len(MyTrain_dataloader) * epochs, anneal_strategy='cos',
                                                pct_start=0.3)

# ==========开始训练====================================================================
for epoch in range(epochs):
    print("\n")
    print("\n")
    print("\n", "epoch=", epoch)

    train_loss = 0
    train_total = 0
    train_acc = 0
    MyNet.train()
    for i, (TrainInputs, TrainLabels) in tqdm(enumerate(MyTrain_dataloader), total=len(MyTrain_dataloader)):
        input = torch.tensor(TrainInputs, dtype=torch.float32, device='cuda')
        label = torch.tensor(TrainLabels, dtype=torch.float32, device='cuda')

        out = MyNet.forward(input)
        loss = loss_func(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        train_acc += sum(row.all().int().item() for row in (out.ge(0.5) == label))
        train_total += label.size(0)
        scheduler.step()
    print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[-1]['lr']))
    train_losses.append(train_loss / len(MyTrain_dataloader))
    train_accs.append(train_acc / train_total)
    print("训练：", "Loss=", train_loss / len(MyTrain_dataloader), "Acc=", train_acc / train_total)

    # 验证集
    val_loss = 0
    val_total = 0
    val_acc = 0
    MyNet.eval()
    for i, (TrainInputs, TrainLabels) in tqdm(enumerate(MyVal_dataloader), total=len(MyVal_dataloader)):
        input = torch.tensor(TrainInputs, dtype=torch.float32, device='cuda')
        label = torch.tensor(TrainLabels, dtype=torch.float32, device='cuda')

        # 计算前向传播，并且得到损失函数的值
        out = MyNet.forward(input)
        loss = loss_func(out, label)
        # 记录误差
        val_loss += loss.item()
        val_loss += loss.item()
        val_acc += sum(row.all().int().item() for row in (out.ge(0.5) == label))
        val_total += label.size(0)
        # 记录：预测错误的输入、标签、输出
        if epoch == 499:
            preds = (out > 0.5).float()
            false_val_out = preds.ne(label).float()
            index = torch.arange(1, label.size(0) + 1)
            for j in range(false_val_out.shape[0]):
                if false_val_out[j].sum() > 0:
                    writer.add_text('false_val_out',
                                    f'epoch{epoch}, batch{i}, sample{j}, input:{input[j]}, label:{label[j]}, out:{out[j]}',
                                    epoch * len(MyVal_dataloader) + i)

    val_losses.append(val_loss / len(MyVal_dataloader))
    val_accs.append(val_acc / val_total)
    print("验证：", "Loss=", val_loss / len(MyVal_dataloader), "Acc=", val_acc / val_total)

torch.save(MyNet, '3.TrainedNet/MyNet_Second_Approach.pt')

# ==========收敛曲线========================================================
plt.figure(1)
plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
plt.plot(np.arange(len(val_losses)), val_losses, label="val loss")
plt.plot(np.arange(len(train_accs)), train_accs, label="train accuracy")
plt.plot(np.arange(len(val_accs)), val_accs, label="val accuracy")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.show()
