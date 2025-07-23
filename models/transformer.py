import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerWaterLevelPrediction(nn.Module):
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=8,
                 dim_feedforward=2048, num_layers_enc=6, 
                 num_layers_dec=6, dropout=0.1, max_length=512, ignore_index=1,
                 encoder_only=True):
        super(TransformerWaterLevelPrediction, self).__init__()
        
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.encoder_only = encoder_only
        
        # Input embedding layers
        self.input_embedding = nn.Linear(input_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_length)
        
        if encoder_only:
            # Use only encoder for efficiency
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=num_layers_enc
            ).to(device)
            self.output_embedding = None
            self.transformer_model = None
        else:
            # Full transformer with encoder-decoder
            self.output_embedding = nn.Linear(output_size, hidden_dim)
            self.transformer_model = nn.Transformer(
                d_model=hidden_dim,
                nhead=num_heads,
                num_encoder_layers=num_layers_enc,
                num_decoder_layers=num_layers_dec,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ).to(device)
            self.encoder = None
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        if self.output_embedding is not None:
            self.output_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)

    def embed(self, x, is_target=False):
        """
        Embed input sequences with positional encoding
        Args:
            x: input tensor of shape (batch_size, seq_len, features)
            is_target: whether this is target sequence (for decoder)
        """
        if is_target:
            embedded = self.output_embedding(x) * math.sqrt(self.hidden_dim)
        else:
            embedded = self.input_embedding(x) * math.sqrt(self.hidden_dim)
        
        # Apply positional encoding
        embedded = embedded.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        embedded = self.pos_encoder(embedded)
        embedded = embedded.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        return self.dropout(embedded)

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, memory_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the transformer
        Args:
            src: source sequence (batch_size, src_seq_len, input_size)
            tgt: target sequence (batch_size, tgt_seq_len, output_size) - optional for encoder-only mode
        """
        # Embed source sequence
        src_embedded = self.embed(src, is_target=False)
        
        if self.encoder_only:
            # Encoder-only mode for sequence regression
            memory = self.encoder(src_embedded, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            # Use mean pooling for multi-step prediction
            output = memory.mean(dim=1)  # Global average pooling
        else:
            if tgt is not None:
                # Encoder-Decoder mode
                tgt_embedded = self.embed(tgt, is_target=True)
                
                # Generate causal mask for target
                if tgt_mask is None:
                    tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
                
                # Apply transformer
                output = self.transformer_model(
                    src_embedded, tgt_embedded,
                    src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
            else:
                # Encoder-only mode using full transformer
                memory = self.transformer_model.encoder(src_embedded, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
                output = memory.mean(dim=1)  # Global average pooling
        
        # Project to output size
        output = self.output_projection(output)
        
        return output

    def predict(self, src, max_length=None, temperature=1.0):
        """
        Generate predictions autoregressively
        Args:
            src: source sequence (batch_size, src_seq_len, input_size)
            max_length: maximum length of generated sequence
            temperature: sampling temperature for generation
        """
        self.eval()
        if max_length is None:
            max_length = self.max_length
            
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        src_embedded = self.embed(src, is_target=False)
        memory = self.transformer_model.encoder(src_embedded)
        
        # Initialize target with start token (zeros)
        tgt = torch.zeros(batch_size, 1, self.output_size).to(device)
        
        outputs = []
        
        with torch.no_grad():
            for i in range(max_length):
                tgt_embedded = self.embed(tgt, is_target=True)
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
                
                output = self.transformer_model.decoder(
                    tgt_embedded, memory, tgt_mask=tgt_mask
                )
                
                # Project and get next token
                next_token_logits = self.output_projection(output[:, -1:, :])
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                outputs.append(next_token_logits)
                
                # Use predicted token as next input
                tgt = torch.cat([tgt, next_token_logits], dim=1)
        
        return torch.cat(outputs, dim=1)

