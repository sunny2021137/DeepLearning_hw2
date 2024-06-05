import cv2
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import time
import tqdm
import random
import torch.nn.functional as F
import wandb
wandb.login()


def random_reshape(images):
    new_images = []
    channel_dict = {0:'RGB', 1:'RG', 2:'GB', 3:'R', 4:'G', 5:'B'}
    numbers = list(channel_dict.keys())
    for image in images:
        channel_idx = random.choice(numbers)
        # 修改通道
        if channel_dict[channel_idx] == 'RGB':
            img = image[:, :, :]
        elif channel_dict[channel_idx] == 'RG':
            img = image[:, :, 1:]
        elif channel_dict[channel_idx] == 'GB':
            img = image[:, :, :2]
        elif channel_dict[channel_idx] == 'R':
            img = image[:, :, 2:3]
        elif channel_dict[channel_idx] == 'G':
            img = image[:, :, 1:2]
        elif channel_dict[channel_idx] == 'B':
            img = image[:, :, 0:1]
        new_images.append(img)
    return new_images

# channel attention
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            # 相加
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # 將attention(batch, C, 1, 1)擴展成x的維度(batch, C, W, H)
        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        # 成上attention後的x
        return x * scale
    
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


# spatial attention
class ChannelPool(nn.Module):
    def forward(self, x):
        # ?? 這裡是對x的channel做max和mean  為甚麼是1   0是batch  2是W  3是H    
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # conv2d    in_planes:輸入的channel數  out_planes:輸出的channel數  kernel_size:卷積核大小  stride:步長  padding:填充  dilation:擴張  groups:分組卷積  bias:是否使用偏置
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        # x_out dim: (N, 1, W, H)
        x_out = self.spatial(x_compress)
        # scale dim: (N, 1, W, H)
        scale = torch.sigmoid(x_out) # broadcasting
        # out dim: (N, C, W, H)
        return x * scale

# channel attention + spatial attention
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
        
class Classifier(nn.Module):
    def __init__(self, in_size, out_dim):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 64, 64]
        self.input_size = in_size
        self.output_dim = out_dim
        
        self.cnn = nn.Sequential(
           nn.Conv2d(3, 32, 3, 1, 1),  # [ 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      #  [ 32, 32, 32]

            nn.Conv2d(32, 64, 3, 1, 1), # [ 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [ 64, 16, 16]
        )
        self.cbam = CBAM(64)
        self.fc = nn.Sequential(
            nn.Linear(64*16*16, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_dim),
        )
    

    def forward(self, x):
        out = self.cnn(x)
        out = self.cbam(out)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
        
    def resize_input(self, images, img_size):
        x = np.zeros((len(images), img_size, img_size, 3), dtype=np.uint8)
        for i, img in enumerate(images):
            img = cv2.resize(img, (self.input_size, self.input_size))

            if len(img.shape) == 2:
                img_3channel = np.stack((img,) * 3, axis=-1)
            elif img.shape[2] == 3:
                img_3channel = img
            elif img.shape[2] == 2:
                # 生成第三個通道，可以根據具體需求來決定其值
                third_channel = np.mean(img, axis=2)  # 這裡用前兩個通道的平均值作為示例
                # 合併成三通道圖像
                img_3channel = np.zeros((self.input_size, self.input_size, 3), dtype=img.dtype)
                img_3channel[:, :, :2] = img  # 前兩個通道保持不變
                img_3channel[:, :, 2] = third_channel  # 第三個通道為生成的數據
            # 變c, w, h
            # img_3channel = np.transpose(img_3channel, (2, 0, 1))
            x[i, :, :, :] = img_3channel

        return x

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
            
def load_img(f):
    shapes = []
    f=open(f)
    lines=f.readlines()
    imgs, lab=[], []
    for i in range(len(lines)):
        fn, label = lines[i].split(' ')
        im1=cv2.imread(fn)
        
        if im1.shape[2] not in shapes:
            shapes.append(im1.shape[2])
        # im1=cv2.resize(im1, (img_size,img_size))
        # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

        # im1 = preprocessing(im1, op_list)
        # vec = np.reshape(im1, [-1])

        imgs.append(im1)
        lab.append(int(label))
    print(i)

    # imgs= np.asarray(imgs, np.uint8)
    lab= np.asarray(lab, np.uint8)
    print(shapes)
    return imgs, lab

def main():
    #############
    eval_time = 10
    num_epoch = 100
    output_dim = 50
    img_size = 64
    batch_size = 128
    lr = 0.001

    x, y = load_img('train.txt')
    vx, vy = load_img('val.txt')
    tx, ty = load_img('test.txt')

    run = wandb.init(
    # Set the project where this run will be logged
    project="task1_3",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": num_epoch,
        "img_size": img_size,
    },)
    wandb.define_metric("Train/epoch")
    wandb.define_metric("Train/*", step_metric="Train/epoch")
    wandb.define_metric("Val/epoch")
    wandb.define_metric("Val/*", step_metric="Val/epoch")
    #############
    
    
    print("--1--")
    x_new = random_reshape(x)
    vx_new = random_reshape(vx)
    tx_new = random_reshape(tx)
    print("--2--")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    model = Classifier(img_size, output_dim).to(device)
    x_resize = model.resize_input(x_new, img_size=img_size)
    vx_resize = model.resize_input(vx_new, img_size=img_size)
    tx_resize = model.resize_input(tx_new, img_size=img_size)
    print("--3--")
    # training 時做 data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
        transforms.RandomRotation(15), # 隨機旋轉圖片
        transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    ])
    # testing 時不需做 data augmentation
    test_transform = transforms.Compose([
        transforms.ToPILImage(),                                    
        transforms.ToTensor(),
    ])
    
    
    train_set = ImgDataset(x_resize, y, train_transform)
    val_set = ImgDataset(vx_resize, vy, test_transform)
    test_set = ImgDataset(tx_resize, ty, test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    print("--4--")
    
    loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # optimizer 使用 Adam
    
    
    for epoch in range(num_epoch):
        print(epoch)
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
    
        model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
        for i, data in enumerate(train_loader):
            optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
            train_pred = model(data[0].to(device)) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
            batch_loss = loss(train_pred, data[1].to(device)) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
            batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
            optimizer.step() # 以 optimizer 用 gradient 更新參數值
            
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
    
        if True:
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    val_pred = model(data[0].to(device))
                    batch_loss = loss(val_pred, data[1].to(device))
        
                    val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                    val_loss += batch_loss.item()
        
                #將結果 print 出來
                # print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                #     (epoch + 1, num_epoch, time.time()-epoch_start_time, \
                #      train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
                wandb.log({"Train/epoch": epoch,
                            "Train/acc": train_acc/train_set.__len__(), 
                           "Train/loss": train_loss/train_set.__len__(),
                           "Val/epoch": epoch,
                           "Val/acc": val_acc/val_set.__len__(), 
                           "Val/loss": val_loss/val_set.__len__(),
                          })

    print("--5--")
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model(data[0].to(device))
            test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        print("test acc:", test_acc/test_set.__len__())   
    
if __name__ == '__main__':
    main()