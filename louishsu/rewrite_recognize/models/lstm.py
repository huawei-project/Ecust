import math
import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

class BaseConv(nn.Module):
    def __init__(self, in_channels, n_classes, input_size):
        super(BaseConv, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AvgPool2d(kernel_size=(input_size//8, input_size//8)),
            Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class ConvLSTM(nn.Module):
    """ LSTM 
    Notes:
        - expressions
            f_t = \sigma(Mxf(x_t) + Lhf(h_{t-1}))
            i_t = \sigma(Mxi(x_t) + Lhi(h_{t-1}))

            c_t = \tanh (Mxc(x_t) + Lhc(h_{t-1}))
            C_t = f_t·C_{t-1} + i_t·c_t
            
            o_t = \sigma(Mxo(x_t) + Lho(h_{t-1}))
            H_t = o_t·tanh(C_t)
    """

    def __init__(self, in_channels, n_classes, input_size, n_times, isforward=True):
        super(ConvLSTM, self).__init__()
        
        self.in_channels = in_channels
        self.n_classes   = n_classes
        self.n_times     = n_times
        self.isforward     = isforward

        # 遗忘门
        self.Mxf = BaseConv(in_channels, n_classes, input_size)
        self.Lhf = nn.Linear (  n_classes, n_classes)
        # 输入门
        self.Mxi = BaseConv(in_channels, n_classes, input_size)
        self.Lhi = nn.Linear (  n_classes, n_classes)
        # 状态
        self.Mxc = BaseConv(in_channels, n_classes, input_size)
        self.Lhc = nn.Linear (  n_classes, n_classes)
        # 输出门
        self.Mxo = BaseConv(in_channels, n_classes, input_size)
        self.Lho = nn.Linear (  n_classes, n_classes)

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

    def forward(self, x, h_0=None, C_0=None):
        """
        Args:
            x:   {tensor(N, T,  C_{in},     H_{in}, H_{in})}
            h_0: {tensor(N,     N_{cls})}
            C_0: {tensor(N,     N_{cls})}
        Returns:
            h_t: {tensor(N,     N_{cls})}
            C_t: {tensor(N,     N_{cls})}
        """
        N = x.shape[0]

        if h_0 is None:
            h_0 = torch.zeros([N, self.n_classes])
        if C_0 is None:
            C_0 = torch.zeros([N, self.n_classes])

        for t in range(self.n_times):
            idx = t if self.isforward else (self.n_times-t-1)
            x_t = x[:, idx].unsqueeze(1)
            f_t = self.sigmoid(self.Mxf(x_t) + self.Lhf(h_0))
            i_t = self.sigmoid(self.Mxi(x_t) + self.Lhi(h_0))
            c_t = self.tanh   (self.Mxc(x_t) + self.Lhc(h_0))
            C_t = f_t*C_0 + i_t*c_t
            o_t = self.sigmoid(self.Mxo(x_t) + self.Lho(h_0))
            h_t = o_t*self.tanh(C_t)
            h_0 = h_t; C_0 = C_t

        return h_t
    
    def cuda(self):
        for m in self.children():
            if isinstance(m, BaseConv):
                m.cuda()
        return self._apply(lambda t: t.cuda(device))

class BiConvLSTM(nn.Module):
    def __init__(self, in_channels, n_classes, input_size, n_times):
        super(BiConvLSTM, self).__init__()

        self.n_classes = n_classes
        self.n_times   = n_times

        self.f_cell = ConvLSTM(in_channels, n_classes, input_size, n_times, True )
        self.b_cell = ConvLSTM(in_channels, n_classes, input_size, n_times, False)
        self.linear = nn.Linear(n_classes*2, n_classes)

    def forward(self, x, h_0=None, C_0=None):
        """
        Params:
            x:   {tensor(N, T,  C_{in},     H_{in}, H_{in})}
            h_0: {tensor(N,     N_{cls})}
            C_0: {tensor(N,     N_{cls})}
        Returns:
            out: {tensor(N, N_{cls})}
        """
        N = x.shape[0]

        if h_0 is None:
            h_0 = torch.zeros([N, self.n_classes])
        if C_0 is None:
            C_0 = torch.zeros([N, self.n_classes])
        
        out_f, _ = self.f_cell(x, h_0, C_0)
        out_b, _ = self.b_cell(x, h_0, C_0)

        out = self.linear(torch.cat([out_f, out_b], 1))
        
        return out

