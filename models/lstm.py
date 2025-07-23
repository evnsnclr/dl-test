import torch
from torch import nn
import math


class LSTMFloodDetection(nn.Module):
    def __init__(self, input_size, output_size, device, hidden_dim=128, 
                 num_layers=2, dropout=0.1, bidirectional=False):
        """
        LSTM model for flood detection and water level prediction.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            device: Device to run the model on
            hidden_dim: Number of hidden units in LSTM
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMFloodDetection, self).__init__()
        
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,  # Add dropout
            bidirectional=bidirectional
        ).to(device)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Adjust FC layer for bidirectional
        lstm_output_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
        
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, src, h_0=None):
        """
        Forward pass of the LSTM model
        
        Args:
            src: Input sequence (batch_size, seq_len, input_size)
            h_0: Initial hidden state tuple (h_0, c_0)
        
        Returns:
            output: Predictions (batch_size, output_size)
                   For multi-step prediction, output_size represents the prediction horizon
        """
        batch_size = src.size(0)
        
        # Initialize hidden state if not provided
        if h_0 is None:
            h_0 = self.init_hidden(batch_size)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(src, h_0)
        
        # Get the last time step output
        # For bidirectional LSTM, we concatenate the last outputs from both directions
        if self.bidirectional:
            # Forward direction: last time step
            forward_out = lstm_out[:, -1, :self.hidden_dim]
            # Backward direction: first time step
            backward_out = lstm_out[:, 0, self.hidden_dim:]
            last_out = torch.cat([forward_out, backward_out], dim=1)
        else:
            # Use the output from the last time step
            last_out = lstm_out[:, -1, :]
        
        # Apply dropout
        last_out = self.dropout(last_out)
        
        # Project to output size
        # For multi-step prediction, output_size is the prediction horizon
        output = self.fc(last_out)
        
        # Ensure output shape matches expected format
        # If output_size > 1 (multi-step), reshape to (batch_size, output_size)
        # This already happens naturally with the FC layer
        
        return output
    
    def init_hidden(self, batch_size):
        """Initialize hidden and cell states"""
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim
        ).to(self.device)
        
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim
        ).to(self.device)
        
        return (h_0, c_0)
    
    def predict_sequence(self, src, n_steps=1):
        """
        Predict multiple steps into the future
        
        Args:
            src: Input sequence (batch_size, seq_len, input_size)
            n_steps: Number of steps to predict
        
        Returns:
            predictions: Future predictions (batch_size, n_steps, output_size)
        """
        self.eval()
        batch_size = src.size(0)
        predictions = []
        
        with torch.no_grad():
            # Get initial prediction
            current_input = src
            h = self.init_hidden(batch_size)
            
            for _ in range(n_steps):
                # Make prediction
                output = self.forward(current_input, h)
                predictions.append(output.unsqueeze(1))
                
                # Use prediction as next input (autoregressive)
                # Shift the input sequence and append the prediction
                current_input = torch.cat([
                    current_input[:, 1:, :],
                    output.unsqueeze(1)
                ], dim=1)
        
        return torch.cat(predictions, dim=1)