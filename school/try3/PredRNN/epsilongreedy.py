import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import EpsilonGreedyLSTMCell
import core.layers.SpatioTemporalLSTMCell

# Assuming SpatioTemporalLSTMCell and EpsilonGreedyLSTM are defined as provided

class CombinedModel(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm, input_size, hidden_size, epsilon):
        super(CombinedModel, self).__init__()
        self.spatio_temporal_lstm = SpatioTemporalLSTMCell(in_channel, num_hidden, width, filter_size, stride, layer_norm)
        self.epsilon_greedy_lstm = EpsilonGreedyLSTM(input_size, hidden_size, epsilon)

    def forward(self, x_t, h_t, c_t, m_t, a_t, input, hidden):
        # Process with SpatioTemporalLSTMCell
        h_new, c_new, m_new = self.spatio_temporal_lstm(x_t, h_t, c_t, m_t, a_t)

        # Process with EpsilonGreedyLSTM
        hx, cx = self.epsilon_greedy_lstm(input, hidden)

        return h_new, c_new, m_new, hx, cx

# Example usage
in_channel = 3
num_hidden = 64
width = 32
filter_size = 3
stride = 1
layer_norm = True

input_size = 10
hidden_size = 20
epsilon = 0.1

model = CombinedModel(in_channel, num_hidden, width, filter_size, stride, layer_norm, input_size, hidden_size, epsilon)

# Example inputs
x_t = torch.randn(1, in_channel, width, width)
h_t = torch.randn(1, num_hidden, width, width)
c_t = torch.randn(1, num_hidden, width, width)
m_t = torch.randn(1, num_hidden, width, width)
a_t = torch.randn(1, num_hidden, width, width)

input = torch.randn(1, input_size)
hidden = (torch.zeros(1, hidden_size), torch.zeros(1, hidden_size))

h_new, c_new, m_new, hx, cx = model(x_t, h_t, c_t, m_t, a_t, input, hidden)

print(h_new.shape, c_new.shape, m_new.shape, hx.shape, cx.shape)