import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def sliding_window_attention(q, k, v, window_size, padding_mask=None):
    '''
    Computes the simple sliding window attention from 'Longformer: The Long-Document Transformer'.
    This implementation is meant for multihead attention on batched tensors. It should work for both single and multi-head attention.
    :param q - the query vectors. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param k - the key vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param v - the value vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param window_size - size of sliding window. Must be an even number.
    :param padding_mask - a mask that indicates padding with 0.  #[Batch, SeqLen]
    :return values - the output values. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :return attention - the attention weights. #[Batch, SeqLen, SeqLen] or [Batch, num_heads, SeqLen, SeqLen]
    '''
    assert window_size%2 == 0, "window size must be an even number"
    seq_len = q.shape[-2]
    embed_dim = q.shape[-1]
    batch_size = q.shape[0] 

    values, attention = None, None

    # TODO:
    #  Compute the sliding window attention.
    # NOTE: We will not test your implementation for efficiency, but you are required to follow these two rules:
    # 1) Implement the function without using for loops.
    # 2) DON'T compute all dot products and then remove the uneccessary comptutations
    #    (You can compute the dot products for any entry, even if it corresponds to padding, as long as it is within the window).
    # Aside from these two rules, you are free to implement the function as you wish. 
    ## HINT: There are several ways to implement this function, and while you are free to implement it however you may wish,
    ## some are more intuitive than others. We suggest you to consider the following:
    ## Think how you can obtain the indices corresponding to the entries in the sliding windows using tensor operations (without loops),
    ## and then use these indices to compute the dot products directly.
    # ====== YOUR CODE: ======
    
    device = q.device
    k = k.to(device)
    v = v.to(device)
    
    if padding_mask is not None:
        padding_mask = padding_mask.to(device)
    half_window_size = window_size // 2
    num_heads = q.size()[1]
    is_multihead = q.dim() == 4

    k_padded = F.pad(k, (0, 0, half_window_size, half_window_size), mode='constant', value=0).to(device)
    k_indices = torch.arange(0, k_padded.size()[-2], device=device)
    k_indices_unfolded = k_indices.unfold(0, window_size+1, 1)

    if is_multihead:
        expanded_indices = k_indices_unfolded.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_heads, 1, 1)
        gather_indices = expanded_indices.unsqueeze(-1).expand(-1, -1, -1, -1, embed_dim)
        k_padded_expanded = k_padded.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
    else:
        expanded_indices = k_indices_unfolded.unsqueeze(0).repeat(batch_size, 1, 1)
        gather_indices = expanded_indices.unsqueeze(-1).expand(-1, -1, -1, embed_dim)
        k_padded_expanded = k_padded.unsqueeze(1).expand(-1, seq_len, -1, -1)

    k_windows = torch.gather(k_padded_expanded, -2, gather_indices).to(device)
    q_reshaped = q.unsqueeze(-2)
    
    attention_scores = torch.matmul(q_reshaped, k_windows.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(embed_dim)
    attention_scores = attention_scores.squeeze(-2)  # [Batch, num_heads, SeqLen, window_size]

    idx = torch.arange(seq_len, device=device).unsqueeze(1) + torch.arange(-window_size // 2 , window_size // 2 + 1, device=device).unsqueeze(0)
    idx = idx.clamp(0, seq_len - 1)
    
    if is_multihead:
        idx = idx.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        full_attention_scores = torch.zeros((batch_size, num_heads, seq_len, seq_len), device=device)
        full_attention_scores = full_attention_scores.scatter_add_(3, idx, attention_scores)
    else:
        idx = idx.unsqueeze(0).expand(batch_size, -1, -1)
        full_attention_scores = torch.zeros((batch_size, seq_len, seq_len), device=device)
        full_attention_scores = full_attention_scores.scatter_add_(2, idx, attention_scores)

    full_attention_scores = torch.where(full_attention_scores == 0.0000,torch.tensor(float('-inf'),
                                                     dtype=full_attention_scores.dtype, device=device), full_attention_scores)

    full_attention_scores = full_attention_scores.detach()
    
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(-2).unsqueeze(-2) if is_multihead else padding_mask.unsqueeze(-2)
        full_attention_scores = full_attention_scores.masked_fill_(padding_mask == 0, float('-inf'))
        full_attention_scores = full_attention_scores.masked_fill_( padding_mask.transpose(-1, -2) == 0, float('-inf'))


    attention = F.softmax(full_attention_scores, dim=-1)
    attention[torch.isnan(attention)] = 0.0 
    values = torch.matmul(attention, v)
    # ========================

    return values, attention


class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        
        # Stack all weight matrices 1...h together for efficiency
        # "bias=False" is optional, but for the projection we learned, there is no teoretical justification to use bias
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation of the paper if you would like....
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, padding_mask, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, 3*Dims]
        
        q, k, v = qkv.chunk(3, dim=-1) #[Batch, Head, SeqLen, Dims]
        
        # Determine value outputs
        # TODO:
        # call the sliding window attention function you implemented
        # ====== YOUR CODE: ======
        values, attention = sliding_window_attention(q, k, v, self.window_size, padding_mask)
        # ========================

        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim) #concatination of all heads
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o
        
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000): 
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, window_size, dropout=0.1):
        '''
        :param embed_dim: the dimensionality of the input and output
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param num_heads: the number of heads in the multi-head attention
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, embed_dim, num_heads, window_size)
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask):
        '''
        :param x: the input to the layer of shape [Batch, SeqLen, Dims]
        :param padding_mask: the padding mask of shape [Batch, SeqLen]
        :return: the output of the layer of shape [Batch, SeqLen, Dims]
        '''
        # TODO:
        #   To implement the encoder layer, do the following:
        #   1) Apply attention to the input x, and then apply dropout.
        #   2) Add a residual connection from the original input and normalize.
        #   3) Apply a feed-forward layer to the output of step 2, and then apply dropout again.
        #   4) Add a second residual connection and normalize again.
        # ====== YOUR CODE: ======
        
        x1 = self.self_attn(x, padding_mask)
        x1 = self.dropout(x1)
        x2 = self.norm1(x + x1)
        
        feed_forward_output = self.feed_forward(x2)
        feed_forward_output = self.dropout(feed_forward_output)
        x = self.norm2(feed_forward_output + x2)

        # ========================
        
        return x
    
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout=0.1):
        '''
        :param vocab_size: the size of the vocabulary
        :param embed_dim: the dimensionality of the embeddings and the model
        :param num_heads: the number of heads in the multi-head attention
        :param num_layers: the number of layers in the encoder
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param max_seq_length: the maximum length of a sequence
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability

        '''
        super(Encoder, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, hidden_dim, num_heads, window_size, dropout) for _ in range(num_layers)])

        self.classification_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the logits  [Batch]
        '''
        output = None

        # TODO:
        #  Implement the forward pass of the encoder.
        #  1) Apply the embedding layer to the input.
        #  2) Apply positional encoding to the output of step 1.
        #  3) Apply a dropout layer to the output of the positional encoding.
        #  4) Apply the specified number of encoder layers.
        #  5) Apply the classification MLP to the output vector corresponding to the special token [CLS] 
        #     (always the first token) to receive the logits.
        # ====== YOUR CODE: ======
        # 1
        embedding_sentence = self.encoder_embedding(sentence)
        # 2
        embedding_sentence = self.positional_encoding(embedding_sentence)
        # 3
        x = self.dropout(embedding_sentence)
        # 4
        for layer in self.encoder_layers:
            x = layer(x, padding_mask)
        # 5
        cls_output = x[:, 0]
        output = self.classification_mlp(cls_output)
        # ========================
        return output  
    
    def predict(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the binary predictions  [Batch]
        '''
        logits = self.forward(sentence, padding_mask)
        preds = torch.round(torch.sigmoid(logits))
        return preds

    