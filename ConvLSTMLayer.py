import torch
import torch.nn

class ConvLSTMLayer(nn.Module):
    def __init__(self, in_dims, hidden_dims, img_size):
        super(ConvLSTMLayer, self).__init__()
        self.hidden_dims = hidden_dims
        self.img_size = img_size
        self.conv_lstm = ConvLSTMCell(in_dims, hidden_dims, img_size)
    
    def forward(self, x):
        b, t, c, h, w = x.shape
        
        #initializing tensor for initial hidden and cell states
        zeros = torch.zeros(b, self.hidden_dims, h, w)
        h_t, c_t = zeros.to(x.device), zeros.to(x.device)
        
        # record hidden_states
        hidden_states = torch.zeros(t, b, self.hidden_dims, h, w).to(x.device)
        
        #unroll ConvLSTMCell over time
        for i in range(t):
            h_t, c_t = self.conv_lstm(x[:, i, :, :, :] , (h_t, c_t))
            hidden_states[i] = h_t
        
        return hidden_states.permute(1, 0, 2, 3, 4)