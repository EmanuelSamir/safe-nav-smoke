import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size, device):
        h, w = image_size
        return (torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
                torch.zeros(batch_size, self.hidden_dim, h, w, device=device))


class ConvLSTM(nn.Module):
    """
    ConvLSTM Layer (Sequence to Sequence).
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i] if isinstance(self.hidden_dim, list) else self.hidden_dim,
                kernel_size=self.kernel_size,
                bias=self.bias
            ))
            
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input_tensor, hidden_state=None):
        """
        Args:
            input_tensor: (B, T, C, H, W)
            hidden_state: List of (h, c) tuples for each layer
        Returns:
            layer_output_list: List of outputs for each layer (if return_all_layers)
            last_state_list: List of last states
        """
        b, t, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(b, (h, w), input_tensor.device)
            
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h_c = hidden_state[layer_idx]
            output_inner = []
            
            for t_idx in range(t):
                inp = cur_layer_input[:, t_idx, :, :, :]
                
                h_c = self.cell_list[layer_idx](inp, h_c)
                output_inner.append(h_c[0])
                
            layer_output = torch.stack(output_inner, dim=1) # (B, T, C, H, W)
            cur_layer_input = layer_output # Next layer input is this layer output
            
            layer_output_list.append(layer_output)
            last_state_list.append(h_c)
            
        if not self.return_all_layers:
            return layer_output_list[-1], last_state_list
        else:
            return layer_output_list, last_state_list
            
    def _init_hidden(self, batch_size, image_size, device):
        init_states = []
        for i in range(self.num_layers):
            hd = self.hidden_dim[i] if isinstance(self.hidden_dim, list) else self.hidden_dim
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size, device))
        return init_states
