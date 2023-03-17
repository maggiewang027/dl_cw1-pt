import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable


class mixup:    
    def mixup_data(x, y, sampling_method, alpha=1.0):
        """
        Implement the mixup algorithm for training data.

        Args:
            x (torch.Tensor): input feature vectors
            y (torch.Tensor): input target vectors
            sampling_method (int): decide lambda is sampled from beta or uniform distribution
            alpha (float): defaults to 1.0

        Returns:
            torch.Tensor, float: mixed inputs, pairs of targets, and lambda
        """
        
        # Set the lambda value
        if alpha > 0:
            if sampling_method == 1:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = np.random.uniform(0, 1)
        else:
            lam = 1

        # Get the feature size to generate index
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        # Calculate mixed features and obtain pairs of targets
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """
        Calculate the loss of the mixup algorithm.

        Args:
            criterion (module): a criterion that measures the loss
            pred (torch.Tensor): predictions from the model
            y_a (torch.Tensor): target vectors drawn at random from the training data
            y_b (torch.Tensor): target vectors
            lam (float): lambda value

        Returns:
            torch.Tensor: the loss of the mixup algorithm
        """
        
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def mixup_visualise(images):
        """
        Visualise the implementation, by saving to a PNG file “mixup.png”, a montage of 16 images
        with randomly augmented images that are about to be fed into network training.

        Args:
            images (torch.Tensor): inputs to create montage
        """
        
        # Create a montage
        im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).cpu().numpy().astype('uint8'))
        im.save("mixup.png")


def train(epoch, sampling_method, batch_size, total_size):
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
    correct = 0
    
    for idx, data in enumerate(trainloader, 0):
        # Obtain the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        # Apply the mixup algorithm to the inputs
        inputs, targets_a, targets_b, lam = mixup.mixup_data(inputs, targets, sampling_method, alpha=1.0)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        
        # Save a montage of 16 images
        if epoch == 0 and idx == 0 and sampling_method == 1:
            mixup.mixup_visualise(inputs[:16])
            print('mixup.png saved.')
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Apply forward + backward + optimize
        outputs = net(inputs)
        # Calculate the loss by using the mixup algorithm defined before
        loss = mixup.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        # Calculate the number of correct predictions
        _, predicted = torch.max(outputs.data, 1)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        loss.backward()
        optimizer.step()

        # Obtain total loss
        train_loss += loss.item()
    
    # Print the train results
    #print('Epoch {}: Train Accuracy = {:.3f} %, Train Loss = {:.3f}'.format(epoch+1, 100.*correct/total_size, train_loss/batch_size))


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
    print('Epoch {}: Test Accuracy = {:.3f} %, Test Loss = {:.3f}'.format(epoch+1, 100.*correct/total_size, test_loss/batch_size))


if __name__ == '__main__':
    # Check the device used to train
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    
    # Load cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 20
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load the initial resnet50 model
    net = torchvision.models.resnet50().to(device)

    # Initialize loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train for sampling method = 1
    print('Start training with sampling method = 1.')
    for epoch in range(10):  # Loop over the dataset multiple times
        train(epoch=epoch, sampling_method=1, batch_size=batch_size, total_size=50000)
        test(epoch=epoch, batch_size=batch_size, total_size=10000)
    print('Training done.')

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
    for epoch in range(10):  # Loop over the dataset multiple times
        train(epoch=epoch, sampling_method=2, batch_size=batch_size, total_size=50000)
        test(epoch=epoch, batch_size=batch_size, total_size=10000)
    print('Training done.')

    # Save trained model
    torch.save(net.state_dict(), 'model_sample_2.pt')
    print('Model saved.')
    
    
    # Reload cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Create a montage of 36 test images
    batch_size = 36
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    dataiter = iter(testloader)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Load the trained model
    model = torchvision.models.resnet50()
    model.load_state_dict(torch.load('model_sample_1.pt'))

    # Inference the ground-truth and the predicted classes
    images, labels = next(dataiter)
    print('Ground-truth: ', ' '.join(classes[labels[j]] for j in range(36)))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(classes[predicted[j]] for j in range(36)))

    # Save result to images
    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("result.png")
    print('result.png saved.')