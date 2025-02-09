import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator
import numpy as np


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # Get unique chars and sort them lexicographically
    unique_chars = sorted(list(set(text)))
    
    # Create the mappings
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
    
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # Initialize counter for removed characters
    n_removed = 0
    
    # Create a copy of the text to modify
    text_clean = text
    
    # Remove each character and count occurrences
    for char in chars_to_remove:
        # Count occurrences before removal
        n_removed += text_clean.count(char)
        # Remove the character
        text_clean = text_clean.replace(char, '')
    
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # Create a tensor of zeros with shape (sequence_length, num_unique_chars)
    N = len(text)
    D = len(char_to_idx)
    result = torch.zeros((N, D), dtype=torch.int8)
    
    # For each character in the text, set the corresponding index to 1
    for i, char in enumerate(text):
        idx = char_to_idx[char]
        result[i, idx] = 1
    
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # Get indices where value is 1 in each row
    indices = torch.argmax(embedded_text, dim=1)
    
    # Convert indices to characters and join them
    result = ''.join([idx_to_char[idx.item()] for idx in indices])
    
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # Embed the text
    embedded_text = chars_to_onehot(text, char_to_idx)
    
    # Calculate number of sequences (excluding last char as it has no label)
    N = (len(text) - 1) // seq_len
    V = len(char_to_idx)
    
    # Reshape embedded text into sequences
    samples = embedded_text[:N*seq_len].view(N, seq_len, V)
    
    # Create labels using the next character for each position
    # Convert characters to indices for labels
    label_indices = torch.tensor([char_to_idx[c] for c in text[1:N*seq_len + 1]], device=device)
    labels = label_indices.view(N, seq_len)
    
    # Move tensors to specified device
    samples = samples.to(device)
    
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # Scale the input by 1/temperature
    scaled = y / temperature
    
    # Apply softmax
    result = torch.softmax(scaled, dim=dim)
    
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence
    
    with torch.no_grad():
        # Initialize hidden state
        hidden_state = None
        
        # Process the start sequence
        # Convert to one-hot and add batch dimension
        x = chars_to_onehot(start_sequence, char_to_idx).to(device=device, dtype=torch.float)
        x = x.unsqueeze(0)  # Add batch dimension: (S, V) -> (1, S, V)
        
        # Get initial predictions
        y, hidden_state = model(x, hidden_state)
        
        # Generate remaining characters one at a time
        chars_to_generate = n_chars - len(start_sequence)
        for _ in range(chars_to_generate):
            # Get the prediction for the next character
            last_char_output = y[0, -1, :]  # Shape: (V,)
            
            # Convert to probability distribution with temperature
            char_probs = hot_softmax(last_char_output, dim=0, temperature=T)
            
            # Sample from the distribution
            next_char_idx = torch.multinomial(char_probs, num_samples=1).item()
            next_char = idx_to_char[next_char_idx]
            
            # Add to output text
            out_text += next_char
            
            # Prepare next input: convert char to one-hot and add batch and sequence dimensions
            next_input = chars_to_onehot(next_char, char_to_idx).to(device=device, dtype=torch.float)
            next_input = next_input.view(1, 1, -1)  # Shape: (1, 1, V)
            
            # Get next prediction
            y, hidden_state = model(next_input, hidden_state)
    
    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # Calculate number of complete batches
        n_batches = len(self.dataset) // self.batch_size
        
        # Create indices array that will give us sequential samples within each batch
        # For each position i in the batch, we want consecutive samples across batches
        # So for position i, we start at i and increment by batch_size for each batch
        indices = []
        for j in range(n_batches):
            for i in range(self.batch_size):
                indices.append(j + i * n_batches)
        
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # Create the parameters for each layer
        # For each layer we need 6 parameter tensors and 3 bias vectors
        self.layer_params = []
        
        for layer in range(n_layers):
            layer_input_dim = in_dim if layer == 0 else h_dim
            
            # Create parameters for this layer
            layer = nn.ParameterDict({
                # Update gate parameters
                'Wxz': nn.Parameter(torch.randn(layer_input_dim, h_dim) / np.sqrt(layer_input_dim)),
                'Whz': nn.Parameter(torch.randn(h_dim, h_dim) / np.sqrt(h_dim)),
                'bz': nn.Parameter(torch.zeros(h_dim)),
                
                # Reset gate parameters
                'Wxr': nn.Parameter(torch.randn(layer_input_dim, h_dim) / np.sqrt(layer_input_dim)),
                'Whr': nn.Parameter(torch.randn(h_dim, h_dim) / np.sqrt(h_dim)),
                'br': nn.Parameter(torch.zeros(h_dim)),
                
                # Candidate state parameters
                'Wxg': nn.Parameter(torch.randn(layer_input_dim, h_dim) / np.sqrt(layer_input_dim)),
                'Whg': nn.Parameter(torch.randn(h_dim, h_dim) / np.sqrt(h_dim)),
                'bg': nn.Parameter(torch.zeros(h_dim))
            })
            
            self.layer_params.append(layer)
            
        # Register the layer parameters with PyTorch
        for i, layer in enumerate(self.layer_params):
            self.add_module(f'layer_{i}', layer)
            
        # Output layer parameters
        self.Wy = nn.Parameter(torch.randn(h_dim, out_dim) / np.sqrt(h_dim))
        self.by = nn.Parameter(torch.zeros(out_dim))

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.n_layers, self.h_dim, device=input.device)

        # Lists to store layer states for each timestep
        layer_states = []
        for i in range(self.n_layers):
            layer_states.append(hidden_state[:, i, :])

        # Process input sequence
        layer_outputs = []
        for t in range(seq_len):
            layer_input = input[:, t, :]
            
            # Process each layer
            for k in range(self.n_layers):
                # Get parameters for current layer
                params = self.layer_params[k]
                h_prev = layer_states[k]
                
                # Update gate
                z_t = torch.sigmoid(
                    layer_input @ params['Wxz'] +
                    h_prev @ params['Whz'] +
                    params['bz']
                )
                
                # Reset gate
                r_t = torch.sigmoid(
                    layer_input @ params['Wxr'] +
                    h_prev @ params['Whr'] +
                    params['br']
                )
                
                # Candidate state
                g_t = torch.tanh(
                    layer_input @ params['Wxg'] +
                    (r_t * h_prev) @ params['Whg'] +
                    params['bg']
                )
                
                # New hidden state
                h_new = z_t * h_prev + (1 - z_t) * g_t
                
                # Store new state
                layer_states[k] = h_new
                
                # Apply dropout if not last layer
                if k < self.n_layers - 1 and self.dropout > 0:
                    layer_input = torch.dropout(h_new, p=self.dropout, train=self.training)
                else:
                    layer_input = h_new
            
            # Store output for this timestep
            layer_outputs.append(layer_input)
        
        # Stack outputs and apply output transformation
        layer_output = torch.stack(layer_outputs, dim=1)
        layer_output = layer_output @ self.Wy + self.by
        
        # Stack final hidden states
        hidden_state = torch.stack(layer_states, dim=1)
        
        return layer_output, hidden_state
