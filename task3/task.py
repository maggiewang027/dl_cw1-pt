import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from task2.task import mixup
import time
import csv


def train(epoch, sampling_method, batch_size, train_size, val_size):
    """
    Train the model.

    Args:
        epoch (int): number of epoch
        sampling_method (int): select from 1 or 2
        batch_size (int): size of batch for training
        total_size (int): total size of the training data
    """
    
    # Initialize the values
    net.train()
    train_loss = 0
    train_correct = 0
    
    for idx, data in enumerate(trainloader, 0):
        # Obtain the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        # Apply the mixup algorithm to the inputs
        inputs, targets_a, targets_b, lam = mixup.mixup_data(inputs, targets, sampling_method, alpha=1.0)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Apply forward + backward + optimize
        outputs = net(inputs)
        # Calculate the loss by using the mixup algorithm defined before
        loss = mixup.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        # Calculate the number of correct predictions
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        loss.backward()
        optimizer.step()

        # Obtain total loss
        train_loss += loss.item()
        
    # Initialize the values
    net.eval()
    val_loss = 0
    val_correct = 0
    
    for idx, data in enumerate(valloader):
        # Obtain the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        # Obtain the test outputs and the loss from the trained model
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item()
        # Calculate the number of correct predictions
        _, predicted = torch.max(outputs.data, 1)
        val_correct += predicted.eq(targets.data).cpu().sum()
    
    # Print the train results
    print('Epoch {}: Train Accuracy = {:.3f} %, Train Loss = {:.3f}, Validation Accuracy = {:.3f} %, Validation Loss = {:.3f}'
          .format(epoch+1, 100.*train_correct/train_size, train_loss/batch_size, 100.*val_correct/val_size, val_loss/batch_size))
    
    return epoch+1, 100.*train_correct/train_size, train_loss/batch_size, 100.*val_correct/val_size, val_loss/batch_size
        


def test(epoch, batch_size, total_size):
    """
    Test the model.

    Args:
        epoch (int): number of epoch
        batch_size (int): size of batch for testing
        total_size (int): total size of the test data
    """
    
    # Initialize the values
    net.eval()
    test_loss = 0
    correct = 0
    
    for idx, data in enumerate(testloader):
        # Obtain the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        # Obtain the test outputs and the loss from the trained model
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        # Calculate the number of correct predictions
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum()

    # Print the test results
    #print('Epoch {}: Test Accuracy = {:.3f} %, Test Loss = {:.3f}'.format(epoch+1, 100.*correct/total_size, test_loss/batch_size))
    
    return epoch+1, 100.*correct/total_size, test_loss/batch_size



if __name__ == '__main__':
    # Check the device used to train
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    
    # Indicate the choice which is the first one
    print('An ablation study is used by comparing the performance in difference between training with the two λ sampling methods in Task 2.')
    
    # Load cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 20
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Concat the dataset
    dataset = torch.utils.data.ConcatDataset([trainset, testset])
    
    # Split the data into development set (80%) and holdout test set (20%)
    train_size1 = int(0.8 * len(dataset))
    test_size = int(0.2 * len(dataset))
    train_dataset, test_set = torch.utils.data.random_split(dataset, [train_size1, test_size])
    
    # Random-split the development set in the train (90%) and validation sets (10%)
    train_size2 = int(0.9 * len(train_dataset))
    val_size = int(0.1 * len(train_dataset))
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size2, val_size])
    print('Train Size: {}, Val Size: {}, Test Size: {}'.format(len(train_set),len(val_set),len(test_set)))

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load the initial resnet50 model
    net = torchvision.models.resnet50().to(device)

    # Initialize loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train for sampling method = 1
    print('Start training with sampling method = 1.')
    train_time = 0
    header_train = False
    header_test = False
    for epoch in range(10):  # Loop over the dataset multiple times
        start_time = time.time()
        train_epoch, train_acc, train_loss, val_acc, val_loss = train(epoch=epoch, sampling_method=1, batch_size=batch_size, train_size=len(train_set), val_size=len(val_set))
        end_time = time.time()
        train_time += end_time - start_time
        # Save the monitoring data to csv file
        with open('summary_train_val_sample_1.csv', 'a') as file:
            f = csv.writer(file, delimiter=',')
            if not header_train:
                f.writerow(['Epoch','Train Accuracy(%)','Train Loss','Validation Accuracy(%)','Validation Loss','Total Time Spent(s)'])
                header_train = True
            f.writerow([train_epoch, train_acc.item(), train_loss, val_acc.item(), val_loss, train_time])
        test_epoch, test_acc, test_loss = test(epoch=epoch, batch_size=batch_size, total_size=len(test_set))
        with open('summary_test_sample_1.csv', 'a') as file:
            f = csv.writer(file, delimiter=',')
            if not header_test:
                f.writerow(['Epoch','Test Accuracy(%)','Test Loss'])
                header_test = True
            f.writerow([test_epoch, test_acc.item(), test_loss])
    print('Training done.')
    print('Summary:')
    # Report a summary of loss values, speed, metric on training and validation
    print('Total training time: {:.3f} s'.format(train_time))
    print('Epoch {}: Train Accuracy = {:.3f} %, Train Loss = {:.3f}, Validation Accuracy = {:.3f} %, Validation Loss = {:.3f}'
          .format(train_epoch, train_acc, train_loss, val_acc, val_loss))
    # Report a summary of loss values and the metrics on the holdout test set
    print('Epoch {}: Test Accuracy = {:.3f} %, Test Loss = {:.3f}'.format(test_epoch, test_acc, test_loss))

    # Save trained model
    torch.save(net.state_dict(), 'model_sample_1.pt')
    print('Model saved.')
    
    # Reload the initial resnet50 model
    net = torchvision.models.resnet50().to(device)

    # Initialize loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train for sampling method = 2
    print('Start training with sampling method = 2.')
    train_time = 0
    header_train = False
    header_test = False
    for epoch in range(10):  # Loop over the dataset multiple times
        start_time = time.time()
        train_epoch, train_acc, train_loss, val_acc, val_loss = train(epoch=epoch, sampling_method=2, batch_size=batch_size, train_size=len(train_set), val_size=len(val_set))
        end_time = time.time()
        train_time += end_time - start_time
        with open('summary_train_val_sample_2.csv', 'a') as file:
            f = csv.writer(file, delimiter=',')
            if not header_train:
                f.writerow(['Epoch','Train Accuracy','Train Loss','Validation Accuracy','Validation Loss','Time Spent (s)'])
                header_train = True
            f.writerow([train_epoch, train_acc.item(), train_loss, val_acc.item(), val_loss, train_time])
        test_epoch, test_acc, test_loss = test(epoch=epoch, batch_size=batch_size, total_size=len(test_set))
        with open('summary_test_sample_2.csv', 'a') as file:
            f = csv.writer(file, delimiter=',')
            if not header_test:
                f.writerow(['Epoch','Test Accuracy','Test Loss'])
                header_test = True
            f.writerow([test_epoch, test_acc.item(), test_loss])
    print('Training done.')
    print('Summary:')
    # Report a summary of loss values, speed, metric on training and validation
    print('Total training time: {:.3f} s'.format(train_time))
    print('Epoch {}: Train Accuracy = {:.3f} %, Train Loss = {:.3f}, Validation Accuracy = {:.3f} %, Validation Loss = {:.3f}'
          .format(train_epoch, train_acc, train_loss, val_acc, val_loss))
    # Report a summary of loss values and the metrics on the holdout test set
    print('Epoch {}: Test Accuracy = {:.3f} %, Test Loss = {:.3f}'.format(test_epoch, test_acc, test_loss))

    # Save trained model
    torch.save(net.state_dict(), 'model_sample_2.pt')
    print('Model saved.')
    
    # Compare the results
    print('The summary results from last training and testing with sampling method = 1 is:')
    print('Train Accuracy: 42.777 %, Train Loss: 188.101, Validation Accuracy: 54.354 %, Validation Loss: 24.023, Time Spent: 879.609 s')
    print('Test Accuracy: 54.300 %, Test Loss: 61.076')
    print('The summary results from last training and testing with sampling method = 2 is:')
    print('Train Accuracy: 41.927 %, Train Loss: 189.581, Validation Accuracy: 50.333 %, Validation Loss: 49.068, Time Spent: 870.730 s')
    print('Test Accuracy: 51.217 %, Test Loss: 113.890')
    print('Both models performed roughly the same, but performed better overall when the sampling method is 1 which means lambda is sampled from a beta distribution.')