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

Batches must remain in order because each batch depends on information from the previous ones. The model's hidden state carries forward important context that the next batch needs. This sequential processing is necessary for the model to learn and understand patterns that span across multiple batches.
"""

part1_q4 = r"""
**Your answer:**

Lower temperatures make the model play it safe -> it sticks to what it knows will work, leading to text that makes sense but might be a bit boring.

Higher temperatures make the model more random in its choices -> its willing to try any character, which usually creates text that doesn't make much sense.

Very low temperatures make the model very cautious -> it always goes for the safest choice, leading to text that is clear but might keep repeating the same patterns.
"""
# ==============


# ==============
# Part 2 answers

#PART2_CUSTOM_DATA_URL = None
PART2_CUSTOM_DATA_URL = 'https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip'


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size = 64,
        h_dim = 256,
        z_dim = 64,
        x_sigma2 = 0.0015,
        learn_rate = 0.0003,
        betas = (0.9, 0.9)
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The sigma value controls how much the model can experiment and deviate from its standard output
1. Low sigma values make the model generate images that are very close to what it learned during training -> basically staying close to its comfort zone.
2. High sigma values let the model take more risks and create images that are different from what it typically produces during training.


"""

part2_q2 = r"""
**Your answer:**

1. **Reconstruction loss** shows how well the decoder recreated the original input. **KL divergence** measures how close the VAE's learned patterns are to the expected distribution.
2. KL divergence loss adjusts the models internal values like mu, sigma. to make its output distribution match what we want.
3. Using KL divergence helps the model create images that look more like the original ones because it keeps the model's outputs close to what it was trained on.


"""

part2_q3 = r"""
**Your answer:**
By maximizing how well our model fits the data we see, we can better understand what kind of distribution produced that data in the first place. This helps us understand where the data originally came from and what patterns it follows.


"""

part2_q4 = r"""
**Your answer:**
1. Log numbers are smaller and easier to work with in calculations, which helps avoid computational errors.
2. When we use log, we can add numbers instead of multiplying them. Adding is simpler and more stable than multiplying lots of probabilities together.
"""

# Part 3 answers
def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=16,
        z_dim=8,
        data_label=1,
        label_noise=0.1,
        discriminator_optimizer=dict(
            type="Adam",
            lr=0.0002,
        ),
        generator_optimizer=dict(
            type="Adam",
            lr=0.0002,
        ),
    )
    # ========================
    return hypers

part3_q1 = r"""
**Your answer:**
GANs training process requires careful management of gradients depending on which part we are training:
1. **Discriminator training phase:**

    a. uses both real and generated data.
    
    b. we drop the generators gradients here since we are only training the discriminator to get better at spotting fakes.
    
    c. keeping generator gradients would mess with its learning process.

2. **Generator training phase:**

    a. generator creates fake data for the discriminator to evaluate.
    
    b. we keep the gradients this time because we need them to update the generators weights.
    
    c. these gradients tell the generator how to adjust to create more convincing fakes.

So gradients are enabled when updating the generator to help it improve, and disabled when training the discriminator to prevent unwanted changes to the generator.



"""

part3_q2 = r"""
**Your answer:**
1. looking at just the generator's loss isn't enough to decide when to stop training. Even if the generator seems to be doing well at fooling the discriminator, that doesn't mean it's creating good images. The quality and variety of the generated images matter more than just the loss numbers.

2. when we see the discriminators loss doesnt chnage while the generators loss drops, it suggests the generator is getting better at fooling a weak discriminator. This usually means there is a problem with the training balance, so we want both parts to improve together, not just one getting better while the other stays stuck.

"""

part3_q3 = r"""
**Your answer:**

VAE images are more blurry because they are trained to minimize the difference between input and output images. This makes the model play it safe and produce smooth images that look similar to each other.

GAN images are sharper because the generator has to fool a discriminator instead of matching pixels. Instead of trying to copy images directly, it learns what makes images look realistic by competing with the discriminator. This competition pushes it to create more detailed and clear outputs.

-> VAEs  aim for accurate reconstruction which leads to blurry but consistent images, while GANs aim for realism which leads to sharper, more varied results.

"""


#PART3_CUSTOM_DATA_URL = None
PART3_CUSTOM_DATA_URL = 'https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip'


def part3_transformer_encoder_hyperparams():
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
    
    hypers = dict(
        embed_dim = 256,     
        num_heads = 4,        
        num_layers = 4,
        hidden_dim = 300,
        window_size = 32,
        droupout = 0.1,
        lr = 0.005
   
    )
    
    hypers['embed_dim'] = 128
    hypers['num_heads'] = 4
    hypers['num_layers'] = 4
    hypers['hidden_dim'] = 300 #320
    hypers['window_size'] = 130 #180
    hypers['droupout'] = 0.2
    hypers['lr'] = 1e-4
    # ========================
    return hypers




part4_q1 = r"""
**Your answer:**
1. Each layer in the encoder can only look at a small window of nearby tokens at a time. But when we stack multiple layers, they work together to see a bigger picture. Each new layer builds on what the previous layer learned, to spread information further.

2. Each layer processes the output of the previous layer, allowing information to spread further than the initial window size.

    a. layer 1 processes a token with its direct neighbors. 
    
    b. layer 2 then processes those outputs, meaning each token now includes information from its neighbors' neighbors.
    
    c. as this continues through more layers, the context grows wider. 

3. In the final layer, each token has collected information from tokens far beyond its initial window size. This happens because each previous layer helped spread information further along the sequence, letting the last layer work with much more context than a single window.
"""

part4_q2 = r"""
**Your answer:**

we need to find a way to get more global context without affecting the computational cost which os O(nw). we can use a dialated window (keeping the same number of tokens but skipping some, and paying attention to ones that are further apart).

by spacing out which tokens we look at, we can capture patterns across longer distances without adding computation time.

"""


part5_q1 = r"""
**Your answer:**
The model in part 5 (BERT) performed better than our previous model for two main reasons:

1. Pre-training Advantage:

    a. BERT was trained on a bigger, more diverse dataset
    
    b. this allows it to learn rich, general language representations
    
    c. Fine-tuning allows the model to adapt these learned features to specific tasks


2. Fine-tuning Strategies:

   a.  Freezing last layers: Preserves learned features, adapts only final classification
    b. Retraining all layers: Allows full model adjustment to the specific task

However BERT isn ot perfect, If the new task is very different from its original training data, a model trained from scratch might work better. The key is matching the models background knowledge to the specific task requirements.

"""

part5_q2 = r"""
**Your answer:**
1. Purpose of Different Layers:

    a. Last layers (linear layers) are task-specific and directly responsible for classification
    
    b. Internal layers are more general, designed for language understanding


2. Why Freezing Last Layers is Problematic:

    a. Classification layers are specifically tailored to the task
    
    b. Preventing their adaptation limits the models ability to specialize


4. Consequence:

    a. The model would likely perform worse
    
    b. Less ability to adapt to the specific classification task
    
    c. Reduced effectiveness in learning task-specific nuances

-> last layers are crucial for adapting a pretrained model to a specific task, and limiting their training would affect (for the worse) the models performance

"""


part5_q3= r"""
**Your answer:**

BERT functions as an encoder, which means it is suitable fort understanding text input but cannot generate new sequences. This makes it effective for classification tasks but not suitable for machine translation.

To perform translation, a model needs encoding and decoding capabilities. The encoder processes the source language, while the decoder generates text in the target language. BERT only handles the **encoding**.

While BERT could be modified for translation by adding a decoder component, this would require retraining and changes to the architecture. Most translation systems instead use models specifically designed with both encoding and decoding capabilities from the start.

"""

part5_q4 = r"""
**Your answer:**

RNNs and transformers serve different purposes: 

1. RNNs are simpler models that need less data and computing power, They work well for basic tasks that involve time or sequence, and can process data one piece at a time.

2. Transformers are more powerful but need more data and computing power, They are better at understanding connections between words or elements that are far apart in a sequence.

-> we prefer to use RNNs when you have a small dataset or limited computing power, or when we need to process data piece by piece. 
-> we prefer transformers when we have plenty of data and computing power, especially if we need to understand complex relationships in the data.
"""

part5_q5 = r"""
**Your answer:**
NSP (Next Sentence Prediction) trains BERT to understand connections between sentences. The model gets two sentences and determines if they belong together or not.

BERT does this by adding special tokens and processing both sentences at once. It outputs a simple yes or no prediction about whether the sentences are related. This helps BERT grasp context and sentence relationships.

While NSP helps with understanding text relationships, many modern versions of BERT perform well without using this task in training -> It turned out to be less crucial than initially thought.

"""


# ==============
