r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=128,  # Larger batch size since we have a large text corpus
        seq_len=128,     # Longer sequences to capture Shakespearean language patterns
        h_dim=512,       # Larger hidden dim to capture rich vocabulary and writing style
        n_layers=2,      # Reduced layers to prevent overfitting on specific language patterns
        dropout=0.3,     # Higher dropout since Shakespeare has repetitive patterns
        learn_rate=0.002,  # Slightly higher learning rate for faster initial learning
        lr_sched_factor=0.3,  # More aggressive LR reduction
        lr_sched_patience=2,  # Shorter patience due to Shakespeare's structured nature
    )
    return hypers


def part1_generation_params():
    start_seq = "ACT I\nSCENE I. "  # Start with typical play structure
    temperature = 0.7  # Balance between creativity and coherence
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    # A temperature of 0.7 provides a good balance:
    # - High enough to allow creativity and variation
    # - Low enough to maintain coherent language and structure
    # 
    # The start sequence "ACT I\nSCENE I. " is chosen because:
    # - It's a common opening in Shakespeare's plays
    # - It sets the proper context for play-like generation
    # - It includes proper formatting with newlines and punctuation
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split the corpus into sequences instead of training on the whole text for several important reasons:

1. **Memory Efficiency**: Training on the entire text at once would require enormous amounts of memory to store both the input data and the gradients during backpropagation. By splitting into sequences, we can process manageable chunks that fit in memory.

2. **Computational Feasibility**: RNNs process data sequentially, and computing gradients through very long sequences becomes computationally expensive. The computational complexity grows with sequence length, making it impractical to process the entire text at once.

3. **Vanishing/Exploding Gradients**: When training RNNs on very long sequences, gradients tend to either vanish or explode as they're propagated back through time. Splitting into shorter sequences helps mitigate this problem by limiting the number of time steps through which gradients must flow.

4. **Parallel Processing**: Working with smaller sequences allows us to process multiple sequences in parallel (batch processing), which significantly speeds up training by taking advantage of modern GPU architectures.

5. **Better Learning Dynamics**: Shorter sequences provide more frequent parameter updates, allowing the model to learn more efficiently. This leads to better convergence and more stable training dynamics.

6. **Local Pattern Learning**: Most linguistic patterns and dependencies occur within relatively short ranges. Training on sequence chunks allows the model to focus on learning these local patterns effectively while still capturing longer-range dependencies through the hidden state that persists between sequences.
"""

part1_q2 = r"""
**Your answer:**

The generated text can show memory longer than the sequence length due to several key mechanisms:

1. **Hidden State Persistence**: During generation, the RNN's hidden state carries information from previous sequences forward. Even though we train on fixed-length sequences, during generation, the hidden state is maintained and updated continuously, allowing the model to maintain context beyond the training sequence length.

2. **Learned Pattern Recognition**: The model learns general patterns and rules about the structure of Shakespearean text during training. These learned patterns (like play structure, character dialogue patterns, and language style) are encoded in the model's weights, not just in the hidden state. This allows the model to maintain consistency even across longer spans of text.

3. **Statistical Dependencies**: Through training, the model learns both local and global statistical dependencies in the text. While it processes text sequence by sequence, the weights of the network encode these longer-range dependencies, allowing it to generate coherent text that maintains consistency in:
   - Character names and roles
   - Plot elements
   - Act and scene structure
   - Language style and period-appropriate vocabulary

4. **Hierarchical Learning**: Even though we train on fixed-length sequences, the model learns hierarchical patterns at different scales. Lower layers might capture character-level patterns, while higher layers capture broader structural patterns, allowing the model to maintain coherence across longer spans of generated text.
"""

part1_q3 = r"""
**Your answer:**

"""

part1_q4 = r"""
**Your answer:**


"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**

"""

# Part 3 answers
def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======

    # ========================
    return hypers

part3_q1 = r"""
**Your answer:**


"""

part3_q2 = r"""
**Your answer:**


"""

part3_q3 = r"""
**Your answer:**



"""



PART3_CUSTOM_DATA_URL = None


def part4_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


part4_q3= r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""


# ==============
