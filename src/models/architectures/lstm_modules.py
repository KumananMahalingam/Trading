"""
LSTM building blocks: ResidualLSTM, Attention, TCN
"""
import torch
import torch.nn as nn


class ResidualLSTM(nn.Module):
    """LSTM with residual connections"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        self.bidirectional = bidirectional
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.residual_fc = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        residual = self.residual_fc(x)
        # Ensure shapes match
        if residual.dim() == 2:
            residual = residual.unsqueeze(1)
        if residual.shape[1] != lstm_out.shape[1]:
            # Repeat or interpolate residual to match sequence length
            residual = residual.repeat(1, lstm_out.shape[1], 1)
        return lstm_out + residual


class MultiHeadAttention(nn.Module):
    """Multi-head attention for time series"""
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = torch.softmax(attention, dim=-1)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.fc_out(out)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block"""
    def __init__(self, input_size, output_size, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        # Use 'same' padding to maintain sequence length
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size,
                              padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Skip connection
        self.downsample = nn.Conv1d(input_size, output_size, 1) if input_size != output_size else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)

        # Apply first convolution
        out = self.relu(self.conv1(x))
        out = self.dropout(out)

        # Apply second convolution
        out = self.conv2(out)

        # Ensure shapes match before adding residual
        if out.shape != residual.shape:
            # Crop or pad to match shapes
            if out.shape[2] > residual.shape[2]:
                out = out[:, :, :residual.shape[2]]
            elif out.shape[2] < residual.shape[2]:
                residual = residual[:, :, :out.shape[2]]

        out = self.relu(out + residual)
        return out