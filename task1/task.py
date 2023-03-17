import torch
import numpy as np
import random
import time

def polynomial_fun(w, x):
    """
    Implement a polynomial function, that takes two input arguments, a weight vector w
    of size M + 1 and an input scalar variable x, and returns the function value y.

    Args:
        w (torch.Tensor): a weight vector of size M + 1
        x (torch.Tensor): an input scalar variable

    Returns:
        torch.Tensor: the function value
    """
    
    # Determine the degree of the polynomial and initialize a zeros tensor to store the terms of y
    M = len(w) - 1
    y = torch.zeros_like(x)
    
    # Compute the terms of the polynomial using a loop
    for m in range(M + 1): # m in [0,...,M]
        y += w[m] * x ** m # Compute the sum of the terms to get the function value
    
    # Return the function value as a torch tensor
    return y

def generate_data(sample_size, w, std=0.2):
    """
    Generate a training set and a test set of both x and t.

    Args:
        sample_size (int): sample size to generate data
        w (torch.Tensor): a weight vector of size M + 1
        std (float): standard deviation to generate Gaussian noise

    Returns:
        torch.Tensor: pairs of x and t
    """
    
    x = torch.linspace(-20, 20, sample_size * 2) # Generate a set of x values
    y = polynomial_fun(w, x) # Compute the corresponding y values using the polynomial function defined earlier
    noise = torch.normal(mean=0, std=std, size=y.shape) # Generate a set of noise values
    t = y+noise # Add the noise to the y values to obtain the target values t
    
    # Generate a list of indices corresponding to the x and t values and randomly shuffle the indices
    idx = list(range(sample_size * 2))
    random.shuffle(idx)
    
    # Split the shuffled indices into two sets for training and testing
    x_train = x[idx[:sample_size]]
    x_test = x[idx[sample_size:]]
    t_train = t[idx[:sample_size]]
    t_test = t[idx[sample_size:]]
    
    # Return the training and testing sets as torch tensors
    return x_train, x_test, t_train, t_test

def fit_polynomial_ls(x, t, M):
    """
    Using the linear algebra modules in PyTorch, implement a least square solver for fitting the polynomial functions.
    
    Args:
        x (torch.Tensor): a tensor containing input values
        t (torch.Tensor): obtained by adding Gaussian noise to y
        M (int): the degree of the polynomial

    Returns:
        torch.Tensor: an optimum weight vector
    """
    
    # Construct the Vandermonde matrix
    xm = torch.arange(M+1)
    X = torch.pow(x.unsqueeze(1), xm)

    # Compute the optimal weight vector using the least squares solution
    w = torch.linalg.lstsq(X, t).solution

    return w

def fit_polynomial_sgd(x, t, M, learning_rate, minibatch_size):
    """
    Using relevant functions/modules in PyTorch, implement a stochastic minibatch gradient
    descent algorithm for fitting the polynomial functions.

    Args:
        x (torch.Tensor): an input tensor
        t (torch.Tensor): a target tensor
        M (int): the degree of the polynomial
        learning_rate (float): the learning rate of the optimizer
        minibatch_size (int): the minibatch size used in each iteration

    Returns:
        torch.Tensor: the optimum weight vector
    """
    
    # Initialize the weight vector randomly
    w = torch.randn(M+1, requires_grad=True)

    # Define the loss function and the optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([w], lr=learning_rate)

    # Train the model using minibatch stochastic gradient descent
    num_samples = x.shape[0]
    num_minibatches = int(np.ceil(num_samples / minibatch_size))

    for epoch in range(40000):
        # Shuffle the data at the beginning of each epoch
        permutation = torch.randperm(num_samples)
        x = x[permutation]
        t = t[permutation]
        running_loss = 0

        for i in range(num_minibatches):
            # Get the minibatch of data
            start_idx = i * minibatch_size
            end_idx = min((i+1) * minibatch_size, num_samples)
            x_mb = x[start_idx:end_idx]
            t_mb = t[start_idx:end_idx]

            # Compute the model output and the loss
            y_mb = polynomial_fun(w, x_mb)
            loss = criterion(y_mb, t_mb)

            # Compute the gradients and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Print the loss at the end of each epoch
        if epoch % 5000 == 4999:
            print('Epoch {}: Loss = {:.3f}'.format(epoch+1, running_loss/minibatch_size))

    # Return the final weight vector
    return w


def print_report():
    """
    Report the loss, the mean and standard deviation, the RMSEs and time spent in fitting/training.
    """
    
    # Initialize the variables
    sample_sizes = [50, 100]
    w = torch.tensor(np.array([1,2,3,4,5,0]), dtype=torch.float32)
    
    for sample_size in sample_sizes:
        # Obtain the training and test data from previous function
        x_train, x_test, t_train, t_test = generate_data(sample_size, w)
        y_train = polynomial_fun(w, x_train)
        y_test = polynomial_fun(w, x_test)
        
        # Training LS model
        print('LS training starts...')
        start_ls = time.time()
        w_ls = fit_polynomial_ls(x_train, t_train, 5)
        end_ls = time.time()
        time_ls = end_ls - start_ls
        y_train_ls = polynomial_fun(w_ls, x_train)
        y_test_ls = polynomial_fun(w_ls, x_test)
        print('LS training finished.')
        
        # Training SGD model and print the loss and epoch
        print('SGD training starts...')
        start_sgd = time.time()
        w_sgd = fit_polynomial_sgd(x_train, t_train, 5, 0.001, 64)
        end_sgd = time.time()
        time_sgd = end_sgd - start_sgd
        y_train_sgd = polynomial_fun(w_sgd, x_train)
        y_test_sgd = polynomial_fun(w_sgd, x_test)
        print('SGD training finished.')
        print('\n')
        
        # Calculate the mean values
        mean_ori_train = torch.mean(torch.abs(t_train - y_train))
        mean_ls_train = torch.mean(torch.abs(y_train_ls - y_train))
        mean_ls_test = torch.mean(torch.abs(y_test_ls - y_test))
        mean_sgd_train = torch.mean(torch.abs(y_train_sgd - y_train))
        mean_sgd_test = torch.mean(torch.abs(y_test_sgd - y_test))
        
        # Calculate the standard deviation values
        std_ori_train = torch.std(torch.abs(t_train - y_train))
        std_ls_train = torch.std(torch.abs(y_train_ls - y_train))
        std_ls_test = torch.std(torch.abs(y_test_ls - y_test))
        std_sgd_train = torch.std(torch.abs(y_train_sgd - y_train))
        std_sgd_test = torch.std(torch.abs(y_test_sgd - y_test))
        
        # Calculate the RMSE values
        rmse_ls_test_y = torch.sqrt((torch.mean(y_test_ls - y_test)**2))
        rmse_sgd_test_y = torch.sqrt((torch.mean(y_test_sgd - y_test)**2))
        rmse_ls_test_w = torch.sqrt((torch.mean(w_ls - w)**2))
        rmse_sgd_test_w = torch.sqrt((torch.mean(w_sgd - w)**2))
        
        # Print the messages we need
        print('The mean and standard deviation in difference between the observed training data and the underlying "true" polynomial curve.')
        print('-----------------------------------------------------------')
        print('\t   Training Data with Sample Size = {}'.format(sample_size))
        print('-----------------------------------------------------------')
        print('\t| Original Data')
        print('-----------------------------------------------------------')
        print('   Mean |\t{:.3f}'.format(mean_ori_train))
        print('   std  |\t{:.3f} '.format(std_ori_train))
        print('\n')
        
        print('The mean and standard deviation in difference between the "LS-predicted" values and the underlying "true" polynomial curve and between the "SGD-predicted" values and the underlying "true" polynomial curve.')
        print('-----------------------------------------------------------')
        print('\t   Training Data with Sample Size = {}'.format(sample_size))
        print('-----------------------------------------------------------')
        print("\t| Least Squares | Minibatch SGD")
        print('-----------------------------------------------------------')
        print('   Mean |\t{:.3f}   |  {:.3f}'.format(mean_ls_train,mean_sgd_train))
        print('   std  |\t{:.3f}   |  {:.3f}'.format(std_ls_train,std_sgd_train))
        print('-----------------------------------------------------------')
        
        print('\t   Test Data with Sample Size = {}'.format(sample_size))
        print('-----------------------------------------------------------')
        print("\t| Least Squares | Minibatch SGD")
        print('-----------------------------------------------------------')
        print('   Mean |\t{:.3f}   |  {:.3f}'.format(mean_ls_test,mean_sgd_test))
        print('   std  |\t{:.3f}   |  {:.3f}'.format(std_ls_test,std_sgd_test))
        print('\n')
        
        print('The root-mean-square-errors (RMSEs) in both w and y.')
        print('-----------------------------------------------------------')
        print('\t   Test Data with Sample Size = {}'.format(sample_size))
        print('-----------------------------------------------------------')
        print("\t| Least Squares | Minibatch SGD")
        print('-----------------------------------------------------------')
        print(' RMSE y |\t{:.3f}   |  {:.3f}'.format(rmse_ls_test_y,rmse_sgd_test_y))
        print(' RMSE w |\t{:.3f}   |  {:.3f}'.format(rmse_ls_test_w,rmse_sgd_test_w))
        print('\n')

        print('The time spent in fitting/training in seconds.')
        print('-----------------------------------------------------------')
        print('\t   Data with Sample Size = {}'.format(sample_size))
        print('-----------------------------------------------------------')
        print("\t| Least Squares | Minibatch SGD")
        print('-----------------------------------------------------------')
        print('   Time |\t{:.3f} s |  {:.3f} s'.format(time_ls,time_sgd))
        print('\n')
        

if __name__ == '__main__':
    print_report()