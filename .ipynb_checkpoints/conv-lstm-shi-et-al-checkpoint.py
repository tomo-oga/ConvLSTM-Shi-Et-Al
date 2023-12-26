import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, in_dims, hidden_dims, img_size, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        # initializing weights as described by Shi et. al.
        # note: cell_state does not need weights as handled by conv layer
        self.hidden_dims = hidden_dims
        h, w = img_size
        padding = kernel_size // 2
        
        dist = np.sqrt(1/(in_dims * kernel_size * kernel_size))
        
        # input gate weights
        self.W_xi = nn.Conv2d(in_dims, hidden_dims, kernel_size, padding=padding, bias=False)
        self.W_hi = nn.Conv2d(in_dims, hidden_dims, kernel_size, padding=padding, bias=False)
        self.W_ci = nn.Parameter(torch.Tensor(hidden_dims, h, w).uniform_(-dist, dist))
        
        self.b_i = nn.Parameter(torch.zeros(hidden_dims)) #bias term
        
        # forget gate weights
        self.W_xf = nn.Conv2d(in_dims, hidden_dims, kernel_size, padding=padding, bias=False)
        self.W_hf = nn.Conv2d(in_dims, hidden_dims, kernel_size, padding=padding, bias=False)
        self.W_cf = nn.Parameter(torch.Tensor(hidden_dims, h, w).uniform_(-dist, dist))
        
        self.b_f = nn.Parameter(torch.zeros(hidden_dims)) #bias term
        
        # cell state memory gate weights
        self.W_xc = nn.Conv2d(in_dims, hidden_dims, kernel_size, padding=padding, bias=False)
        self.W_hc = nn.Conv2d(in_dims, hidden_dims, kernel_size, padding=padding, bias=False)
        
        self.b_c = nn.Parameter(torch.zeros(hidden_dims)) #bias term
        
        # output gate weights
        self.W_xo = nn.Conv2d(in_dims, hidden_dims, kernel_size, padding=padding, bias=False)
        self.W_ho = nn.Conv2d(in_dims, hidden_dims, kernel_size, padding=padding, bias=False)
        self.W_co = nn.Parameter(torch.empty(hidden_dims, h, w))
        
        self.b_o = nn.Parameter(torch.Tensor(hidden_dims, h, w).uniform_(-dist, dist))
        
                    
    def forward(self, x_t, prev):
        b, c, h, w = x_t.shape
        h_t, c_t = prev
        
        # not shown well but h_t and c_t are of previous time steps, x_t is of current
              
        i_t = F.sigmoid(self.W_xi(x_t) + self.W_hi(h_t) + self.W_ci * c_t + self.b_i.view(1,self.hidden_dims,1,1))
        f_t = F.sigmoid(self.W_xf(x_t) + self.W_hf(h_t) + self.W_cf * c_t + self.b_f.view(1,self.hidden_dims,1,1))
        c_t = f_t * c_t + i_t * F.tanh(self.W_xc(x_t) + self.W_hc(h_t) + self.b_c.view(1,self.hidden_dims,1,1))
        o_t = F.sigmoid(self.W_xo(x_t) + self.W_ho(h_t) + self.W_co * c_t + self.b_o.view(1,self.hidden_dims,1,1))
        h_t = o_t * F.tanh(c_t)
        
        return (h_t, c_t)