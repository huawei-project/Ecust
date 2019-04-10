import math
import torch
import torch.nn as nn

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

    def __init__(self, in_channels, n_classes, input_size, n_times, base_model, isforward=True):
        super(ConvLSTM, self).__init__()
        
        self.in_channels = in_channels
        self.n_classes   = n_classes
        self.n_times     = n_times
        self.isforward     = isforward

        # 遗忘门
        self.Mxf = base_model(in_channels, n_classes)
        self.Lhf = nn.Linear (  n_classes, n_classes)
        # 输入门
        self.Mxi = base_model(in_channels, n_classes)
        self.Lhi = nn.Linear (  n_classes, n_classes)
        # 状态
        self.Mxc = base_model(in_channels, n_classes)
        self.Lhc = nn.Linear (  n_classes, n_classes)
        # 输出门
        self.Mxo = base_model(in_channels, n_classes)
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
            x_t = x[:, idx]
            f_t = self.sigmoid(self.Mxf(x_t) + self.Lhf(h_0))
            i_t = self.sigmoid(self.Mxi(x_t) + self.Lhi(h_0))
            c_t = self.tanh   (self.Mxc(x_t) + self.Lhc(h_0))
            C_t = f_t*C_0 + i_t*c_t
            o_t = self.sigmoid(self.Mxo(x_t) + self.Lho(h_0))
            h_t = o_t*self.tanh(C_t)
            h_0 = h_t; C_0 = C_t

        return h_t

class BiConvLSTM(nn.Module):
    def __init__(self, in_channels, n_classes, input_size, n_times, base_model):
        super(BiConvLSTM, self).__init__()

        self.n_classes = n_classes
        self.n_times   = n_times

        self.f_cell = ConvLSTM(in_channels, n_classes, input_size, n_times, base_model, True )
        self.b_cell = ConvLSTM(in_channels, n_classes, input_size, n_times, base_model, False)
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

