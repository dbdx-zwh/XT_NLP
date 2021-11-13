import torch
import torch.nn as nn
import time as t

class MyLSTM(nn.Module):
    def __init__(self, emb_size, n_hidden, n_layer):
        super(MyLSTM, self).__init__()
        self.layers = n_layer
        # basc layer
        self.n_hidden = n_hidden
        self.Wxi0 = nn.Linear(emb_size, n_hidden, bias=False)
        self.Wxf0 = nn.Linear(emb_size, n_hidden, bias=False)
        self.Wxo0 = nn.Linear(emb_size, n_hidden, bias=False)
        self.Wxc0 = nn.Linear(emb_size, n_hidden, bias=False)
        self.Whi0 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Whf0 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Who0 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Whc0 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bi0 = nn.Parameter(torch.zeros([n_hidden]))
        self.bf0 = nn.Parameter(torch.zeros([n_hidden]))
        self.bo0 = nn.Parameter(torch.zeros([n_hidden]))
        self.bc0 = nn.Parameter(torch.zeros([n_hidden]))
        # add layer(层与层之间共享参数么？)
        self.Wxi = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Wxf = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Wxo = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Wxc = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Whi = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Whf = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Who = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Whc = nn.Linear(n_hidden, n_hidden, bias=False)
        self.bi = nn.Parameter(torch.zeros([n_hidden]))
        self.bf = nn.Parameter(torch.zeros([n_hidden]))
        self.bo = nn.Parameter(torch.zeros([n_hidden]))
        self.bc = nn.Parameter(torch.zeros([n_hidden]))
    
    def forward(self, inputs):
        # input : (n_step, batch_size, embedding)
        n_step = inputs.shape[0]
        batch_size = inputs.shape[1]
        H0 = torch.zeros(batch_size, self.n_hidden)
        C0 = torch.zeros(batch_size, self.n_hidden)
        H = torch.zeros(batch_size, self.n_hidden)
        C = torch.zeros(batch_size, self.n_hidden)
        
        for i in range(n_step):
            H_list = []
            for layer in range(self.layers):
                if layer == 0:
                    # t.sleep(10)
                    Xt = inputs[i]
                    # 1. 计算遗忘门、更新们、输出门参数 (self.H : Ht-1)
                    It = torch.sigmoid(self.Wxi0(Xt) + self.Whi0(H0) + self.bi0)
                    Ft = torch.sigmoid(self.Wxf0(Xt) + self.Whf0(H0) + self.bf0)
                    Ot = torch.sigmoid(self.Wxo0(Xt) + self.Who0(H0) + self.bo0)
                    # 2. 计算Ct_hat
                    Ct_hat = torch.tanh(self.Wxc0(Xt) + self.Whc0(H0) + self.bc0)
                    # 3. 更新Ht & Ct
                    H0 = Ot * torch.tanh(C0)
                    C0 = Ft * C0 + It * Ct_hat
                    Xt = H0
                else:
                    # 1. 计算遗忘门、更新们、输出门参数 (self.H : Ht-1)
                    It = torch.sigmoid(self.Wxi(Xt) + self.Whi(H) + self.bi)
                    Ft = torch.sigmoid(self.Wxf(Xt) + self.Whf(H) + self.bf)
                    Ot = torch.sigmoid(self.Wxo(Xt) + self.Who(H) + self.bo)
                    # 2. 计算Ct_hat
                    Ct_hat = torch.tanh(self.Wxc(Xt) + self.Whc(H) + self.bc)
                    # 3. 更新Ht & Ct
                    H = Ot * torch.tanh(C)
                    C = Ft * C + It * Ct_hat
                    Xt = H
            if layer == 0:
                H_list.append(H0)
            else:
                H_list.append(H)

        return H_list
