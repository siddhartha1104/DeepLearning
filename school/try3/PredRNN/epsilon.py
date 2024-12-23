import torch
import torch.nn as nn
import random

class EpsilonGreedyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, epsilon=0.1):
        super(EpsilonGreedyLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        
        # Define LSTM gates
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

    def epsilon_greedy_gate(self, gate_value):
        """Applies epsilon-greedy policy to a gate."""
        if random.random() < self.epsilon:
            # Exploration: generate random gate values between 0 and 1
            return torch.rand_like(gate_value)
        else:
            # Exploitation: use calculated gate values
            return gate_value

    def forward(self, x_t, hidden):
        h_t, c_t = hidden
        
        # Concatenate input and hidden state
        x_t = x_t.view(-1, 5)  # assuming h_t has shape (batch_size, 5)
        combined = torch.cat((h_t, x_t), dim=1)
        h_t = h_t.view(-1, 2)  # assuming x_t has shape (batch_size, 2)
        combined = torch.cat((h_t, x_t), dim=1)

        # Calculate gate values
        f_t = torch.sigmoid(self.W_f(combined))  # Forget gate
        i_t = torch.sigmoid(self.W_i(combined))  # Input gate
        o_t = torch.sigmoid(self.W_o(combined))  # Output gate
        c_hat_t = torch.tanh(self.W_c(combined)) # Candidate cell state

        # Apply epsilon-greedy policy to the gates
        f_t = self.epsilon_greedy_gate(f_t)
        i_t = self.epsilon_greedy_gate(i_t)
        o_t = self.epsilon_greedy_gate(o_t)

        # Update the cell state
        c_t = f_t * c_t + i_t * c_hat_t

        # Update the hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class EpsilonGreedyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, epsilon=0.1):
        super(EpsilonGreedyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = EpsilonGreedyLSTMCell(input_size, hidden_size, epsilon)

    def forward(self, input_seq):
        h_t = torch.zeros(input_seq.size(0), self.hidden_size)
        c_t = torch.zeros(input_seq.size(0), self.hidden_size)

        outputs = []
        for x_t in input_seq:
            h_t, c_t = self.lstm_cell(x_t, (h_t, c_t))
            outputs.append(h_t.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (h_t, c_t)

# Example Usage:
input_size = 10  # Size of the input features
hidden_size = 20  # Size of the hidden state
sequence_length = 5  # Number of time steps in the sequence
batch_size = 1  # Number of sequences in the batch

# Create the EpsilonGreedyLSTM model
epsilon = 0.2  # Exploration rate
model = EpsilonGreedyLSTM(input_size, hidden_size, epsilon)

# Dummy input sequence (sequence_length, batch_size, input_size)
input_seq = torch.randn(sequence_length, batch_size, input_size)

# Forward pass through the model
output, (h_n, c_n) = model(input_seq)

print("Output:", output)
print("Final hidden state:", h_n)
print("Final cell state:", c_n)
