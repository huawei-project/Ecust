import math
import torch
import torch.nn as nn

def _conv_global(in_channels, n_classes, input_size):
    m = nn.Sequential(
        nn.Conv2d(in_channels, n_classes, kernel_size=input_size),
        Flatten(),
    )
    return m

def _conv_base(in_channels, n_classes, input_size):
    m = nn.Sequential(
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

            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=(input_size//8, input_size//8)),
            Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, n_classes),
        )
    return m

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

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
        self.Mxf = _conv_global(in_channels, n_classes, input_size)
        self.Lhf = nn.Linear   (n_classes, n_classes)
        # 输入门
        self.Mxi = _conv_global(in_channels, n_classes, input_size)
        self.Lhi = nn.Linear   (n_classes, n_classes)
        # 状态
        self.Mxc = _conv_base  (in_channels, n_classes, input_size)
        self.Lhc = nn.Linear   (n_classes, n_classes)
        # 输出门
        self.Mxo = _conv_global(in_channels, n_classes, input_size)
        self.Lho = nn.Linear   (n_classes, n_classes)

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()
    
    def base(self, in_channels, n_classes, input_size):
        return nn.Sequential(
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
            if torch.cuda.is_available(): h_0 = h_0.cuda()
        if C_0 is None:
            C_0 = torch.zeros([N, self.n_classes])
            if torch.cuda.is_available(): C_0 = C_0.cuda()

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

# class BiConvLSTM(nn.Module):
#     def __init__(self, in_channels, n_classes, input_size, n_times):
#         super(BiConvLSTM, self).__init__()

#         self.n_classes = n_classes
#         self.n_times   = n_times

#         self.f_cell = ConvLSTM(in_channels, n_classes, input_size, n_times, True )
#         self.b_cell = ConvLSTM(in_channels, n_classes, input_size, n_times, False)
#         self.linear = nn.Linear(n_classes*2, n_classes)

#     def forward(self, x, h_0=None, C_0=None):
#         """
#         Params:
#             x:   {tensor(N, T,  C_{in},     H_{in}, H_{in})}
#             h_0: {tensor(N,     N_{cls})}
#             C_0: {tensor(N,     N_{cls})}
#         Returns:
#             out: {tensor(N, N_{cls})}
#         """
#         N = x.shape[0]

#         if h_0 is None:
#             h_0 = torch.zeros([N, self.n_classes])
#         if C_0 is None:
#             C_0 = torch.zeros([N, self.n_classes])
        
#         out_f, _ = self.f_cell(x, h_0, C_0)
#         out_b, _ = self.b_cell(x, h_0, C_0)

#         out = self.linear(torch.cat([out_f, out_b], 1))
        
#         return out





def _conv3x3(in_channels, out_channels):
    layer = [
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    ]
    return nn.Sequential(*layer)

def _get_weight(c, h, w):
    weight = nn.Parameter(torch.zeros(c, h, w))
    weight.data.normal_(0, math.sqrt(2. / (c*h*w)))
    return weight

def _get_bias(c, h, w):
    bias = nn.Parameter(torch.Tensor(c, h, w))
    bias.data.zero_()
    return bias

def _hadamard(x, weight):
    """
    Params:
        x:      {tensor(batch_size, out_channels, height, width)}
        weight: {tensor(            out_channels, height, width)}
    Returns:
        out:    {tensor(batch_size, out_channels, height, width)}
    """
    N = x.shape[0]
    out = torch.zeros_like(x)
    for n in range(N):
        out[n] = x[n] * weight
    return out

class ConvLSTMCell(nn.Module):
    """ Convolutional LSTM Cell
    Notes:
        - expressions
            f_t = \sigma(W_{xf}*X_t + W_{hf}*H_{t-1} + W_f·C_{t-1} + b_f)
            i_t = \sigma(W_{xi}*X_t + W_{hi}*H_{t-1} + W_i·C_{t-1} + b_i)
            C_t = f_t·C_{t-1} + i_t·tanh(W_{xc}*X_t + W_{hc}*H_{t-1} + b_c)
            o_t = \sigma(W_{xo}*X_t + W_{ho}*H_{t-1} + W_o·C_t     + b_o)
            H_t = o_t·tanh(C_t)
        - reference
            Convolutional LSTM Network: A Machine Learing Approach for Precipitation Nowcasting.
    """

    def __init__(self, in_channels, out_channels, input_size, n_times):
        super(ConvLSTMCell, self).__init__()
        self.init = False

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_times    = n_times

        # 遗忘门
        self.conv_xf = _conv3x3(in_channels, out_channels)
        self.conv_hf = _conv3x3(out_channels, out_channels)
        # 输入门
        self.conv_xi = _conv3x3(in_channels, out_channels)
        self.conv_hi = _conv3x3(out_channels, out_channels)
        # 状态
        self.conv_xc = _conv3x3(in_channels, out_channels)
        self.conv_hc = _conv3x3(out_channels, out_channels)
        # 输出门
        self.conv_xo = _conv3x3(in_channels, out_channels)
        self.conv_ho = _conv3x3(out_channels, out_channels)

        self.W_f = _get_weight(out_channels, input_size, input_size)
        self.b_f = _get_bias  (out_channels, input_size, input_size)
        self.W_i = _get_weight(out_channels, input_size, input_size)
        self.b_i = _get_bias  (out_channels, input_size, input_size)
        self.b_c = _get_bias  (out_channels, input_size, input_size)
        self.W_o = _get_weight(out_channels, input_size, input_size)
        self.b_o = _get_bias  (out_channels, input_size, input_size)

    def forward(self, x, h_0=None, C_0=None):
        """
        Args:
            x:   {tensor(N,     C_{in},     W, H)}
            h_0: {tensor(N,     C_{hidden}, W, H)}
            C_0: {tensor(N,     C_{hidden}, W, H)}
        Returns:
            h_t: {tensor(N,     C_{hidden}, W, H)}
        """
        x = torch.unsqueeze(x, 2)
        (N, T, C, W, H) = x.shape
        
        if h_0 is None:
            h_0 = torch.zeros([N, self.out_channels, H, W])
            if torch.cuda.is_available(): h_0 = h_0.cuda()
        if C_0 is None:
            C_0 = torch.zeros([N, self.out_channels, H, W])
            if torch.cuda.is_available(): C_0 = C_0.cuda()


        for t in range(self.n_times):
            x_t = x[:, t]
            f_t = torch.sigmoid(
                self.conv_xf(x_t) + self.conv_hf(h_0) + _hadamard(C_0, self.W_f) + self.b_f)    # 遗忘系数
            i_t = torch.sigmoid(
                self.conv_xi(x_t) + self.conv_hi(h_0) + _hadamard(C_0, self.W_i) + self.b_i)    # 输入系数
            c_t = torch.tanh(self.conv_xc(x_t) + self.conv_hc(h_0) + self.b_c)                  # 输入
            C_t = f_t*C_0 + i_t*c_t
            o_t = torch.sigmoid(
                self.conv_xo(x_t) + self.conv_ho(h_0) + _hadamard(C_t, self.W_o) + self.b_o)    # 输出系数
            h_t = o_t * torch.tanh(C_t)

            h_0 = h_t; C_0 = C_t

        return h_t

class ConvLSTMNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, input_size, n_times):
        super(ConvLSTMNet, self).__init__()

        self.n_classes = n_classes

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)

        self.lstm1 = ConvLSTMCell(1,  8, input_size,    in_channels)
        self.lstm2 = ConvLSTMCell(1, 16, input_size//2,     8)
        self.lstm3 = ConvLSTMCell(1, 32, input_size//4,     16)
        self.lstm4 = ConvLSTMCell(1, 32, input_size//8,     32)
        self.lstm5 = ConvLSTMCell(1, 64, input_size//16,    32)
    
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, n_classes),
            nn.ReLU(True),
        )

    def forward(self, x):
        """
        Params:
            x:  {tensor(N, C, H, W)}
        Returns:
            out:  {tensor(N, C, n_classes)}
        """
        (N, C, H, W) = x.shape
        
        x = self.lstm1(x)
        x = self.maxpool(x)
        x = self.lstm2(x)
        x = self.maxpool(x)
        x = self.lstm3(x)
        x = self.maxpool(x)
        x = self.lstm4(x)
        x = self.maxpool(x)
        x = self.lstm5(x)
        x = self.maxpool(x)

        x = x.view([N, -1])
        x = self.classifier(x)
            
        return x

if __name__ == '__main__':
    x = torch.rand(5, 23, 64, 64)

    model = ConvLSTMNet(1, 63, 23, 64)
    x = model(x)
    pass