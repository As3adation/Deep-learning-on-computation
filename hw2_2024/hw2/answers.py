r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**

1.A:    shape of X is (N, in_features) = (64, 1024)
        shape of y is (N, out_features) = (64, 512)
        and dimenstion of the jacbian are (N, out_features, N, in_features) = (64, 512,64, 1024)

1.B:    Yes, the jacobian is sparse because the matrix is mostly zeros, and the non-zero values are only in the diagonal of the matrix.
1.C:    No, we can calculate $\delta X = \frac{\partial L}{\partial X}$ by using the chain rule and the linear layer,
            based on the linear layer we can calculate $\delta X = \delta Y * W$.   
            the shape of $\delta y = (N, out_features) = (64, 512)$ and the shape of W is (512,1024) 
            so the shape of $\delta X = (N, in_features) = (64, 1024)$ and using the chain rule we can calculate $\delta X = \delta Y * W$.
            this way of calculation is more efficient and we dont need to calculate the jacobian.


2.A since $ y = XW^T + b $ the value of $ \frac{\partial y}{\partial W} $ will have dimensions of (N, out_features, in_features) = (64, 512, 1024)
    becuase for each i in the batch of size N we have a matrix of size (512, 1024) becuase each output dimenstion depends on all input features.
    there are N samples in the batch so the full jacobian will have the shape of (N, out_features, in_features) = (64, 512, 1024).

2.B No, the jacobian is not sparse because the matrix is not mostly zeros, the output Y depends directly on the input X and the weights W.
    and for a specific cell n in the batch, the deravative $\frac{\partial y_[n,j]}{\partial W[k,l]}$ is the input $X[n,l]$ becuase of the linear operation,            
    that's why the jacobian is not sparse, and is dense.

2.C No, we can calculate $\delta W = \frac{\partial L}{\partial W}$ by using the chain rule:
        $ \frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} * \frac{\partial Y}{\partial W} $
        and we know that $ \frac{\partial L}{\partial Y} = \delta Y $ and $ \frac{\partial Y}{\partial W} = X^T $, so no need to calculate the jacobian.
        so we can just calcualte $ \frac{\partial L}{\partial W} =  \frac{\partial L}{\partial Y} * X^T $.
**    

"""

part1_q2 = r"""
**
    As we learned in the tutorial the backpropagation algorithm is used to calculate the gradients of the loss function, which is crucial for the optimization process.
    but the backpropagation algorithm is not the only way to calculate the gradients, and thats why backpropagation is not required in order to train
    nuearal networks, but it is the most common way to calculate the gradients.
**
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.01
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.01
    lr_momentum = 0.01
    lr_rmsprop = 0.001
    reg = 0.001

    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. Graphs of no dropout have a higher training accuracy and a lower testing accuracy, and the graphs of dropout have a lower training accuracy and a higher testing accuracy.
    this is matches our expectation because the dropout technique is used to prevent overfitting, and it does so by randomly setting some of the neurons to zero during the training process, and thats why we see a lower training accuracy.
2. high-dropout rate can cause underfitting because the model is not learning enough from the data, and as indicated in the graph the 0.8 dropout rate has a slow convergence rate, high test loss, and lower overall accuracy.
    A low-dropout rate can cause overfitting because the model is learning too much from the data,
    the dropout rate of 0.4 make a good balance and achieve steady improvments in test performance while maintaining a reasonable training accuracy.


"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.001
    weight_decay = 0.01
    momentum = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""