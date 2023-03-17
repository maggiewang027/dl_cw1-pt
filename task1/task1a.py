import torch
import numpy as np
from task import generate_data, polynomial_fun

def fit_polynomial_sgd_M(x, t, M_max, learning_rate, minibatch_size):
    """
    Using relevant functions/modules in PyTorch, implement a stochastic minibatch gradient
    descent algorithm for fitting the polynomial functions.

    Args:
        x (torch.Tensor): an input tensor
        t (torch.Tensor): a target tensor
        M_max (int): Highest degree the polynomial function can have
        learning_rate (float): the learning rate of the optimizer
        minibatch_size (int): the minibatch size used in each iteration

    Returns:
        best_w (torch.Tensor): the optimum weight vector
        best_M (int): best degree of the polynomial
    """
    # Initialize the weight vector randomly
    M = torch.randint(1, M_max+1, size=(1,), dtype=torch.float).requires_grad_(True)
    w = torch.randn(M_max+1, requires_grad=True)

    # Define the loss function and the optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([w, M], lr=learning_rate)

    # Train the model using minibatch stochastic gradient descent
    num_samples = x.shape[0]
    num_minibatches = int(np.ceil(num_samples / minibatch_size))
    
    # Initialize the best loss and m
    best_loss = torch.inf
    m = torch.arange(0, M_max+1, dtype=torch.float32)

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
            
            # Update the weight
            w_m = torch.relu(M - m).unsqueeze(-1)
            w_m = torch.clamp(w_m, 0, 1)
            w_new =  w_m.t() * w
            w_new = w_new.reshape(-1,)

            # Compute the model output and the loss
            y_mb = polynomial_fun(w_new, x_mb)
            loss = criterion(y_mb, t_mb)

            # Compute the gradients and update the weights
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(w, max_norm=1)
            torch.nn.utils.clip_grad_norm_(M, max_norm=1)
            optimizer.step()
            
            # Update the loss, the weight vector and M
            if loss < best_loss:
                best_loss = loss
                best_w = w_new
                if torch.abs(M.int() - M) < 0.5:
                    best_M = M.int() - 1
                else: 
                    best_M = M.int()
            
            running_loss += loss.item()

        # Print the loss at the end of each epoch
        if epoch % 5000 == 4999:
            M_print = M.int()[0] - 1 if torch.abs(M.int() - M) < 0.5 else M.int()[0]
            print('Epoch {}: Loss = {:.3f}, M = {}'.format(epoch+1, running_loss/minibatch_size, M_print))

    # Return the final weight vector and best degree of polynomial
    return best_w[:best_M+1], best_M

def report():
    """
    Report the loss, the optimised M value, the mean and standard deviation.
    """
    
    # Initialize the variables
    sample_sizes = [50, 100]
    w = torch.tensor(np.array([1,2,3,4,5]), dtype=torch.float32)
    
    for sample_size in sample_sizes:
        # Obtain the training and test data from previous function
        x_train, x_test, t_train, t_test = generate_data(sample_size, w)
        
        # Training SGD model and print the loss and epoch
        print('SGD training starts...')
        w_sgd, M_sgd = fit_polynomial_sgd_M(x_train, t_train, 10, 0.001, 64)
        # Keep the size of true and predicted w same
        if w.shape[0]-w_sgd.shape[0] > 0:
            ele = torch.zeros(w.shape[0]-w_sgd.shape[0])
            w_sgd = torch.cat((w_sgd,ele))
        # Calculate predicted targets
        y_train_sgd = polynomial_fun(w_sgd, x_train)
        y_test_sgd = polynomial_fun(w_sgd, x_test)
        print('SGD training finished.')
        print('\n')
        
        # Keep the size of true and predicted w same
        if w_sgd.shape[0]-w.shape[0] > 0:
            ele = torch.zeros(w_sgd.shape[0]-w.shape[0])
            w = torch.cat((w,ele))
        # Calculate true targets
        y_train = polynomial_fun(w, x_train)
        y_test = polynomial_fun(w, x_test)
        
        # Calculate the mean values
        mean_sgd_train = torch.mean(torch.abs(y_train_sgd - y_train))
        mean_sgd_test = torch.mean(torch.abs(y_test_sgd - y_test))
        
        # Calculate the standard deviation values
        std_sgd_train = torch.std(torch.abs(y_train_sgd - y_train))
        std_sgd_test = torch.std(torch.abs(y_test_sgd - y_test))
        
        # Print the messages we need
        print("The optimised M value.")
        print('-----------------------------------------------------------')
        print('\t   Training Data with Sample Size = {}'.format(sample_size))
        print('-----------------------------------------------------------')
        print('   M value |\t{}'.format(M_sgd[0]))
        print('\n')
        
        print('The mean and standard deviation in difference between the model-predicted values and the underlying "true" polynomial curve.')
        print('-----------------------------------------------------------')
        print('\t   Sample Size = {}'.format(sample_size))
        print('-----------------------------------------------------------')
        print("\t| Training Data | Test Data")
        print('-----------------------------------------------------------')
        print('   Mean |\t{:.3f}   |  {:.3f}'.format(mean_sgd_train,mean_sgd_test))
        print('   std  |\t{:.3f}   |  {:.3f}'.format(std_sgd_train,std_sgd_test))
        print('\n')


if __name__ == '__main__':
    report()