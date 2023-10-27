import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.init as init
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple feedforward neural network for the multicategory perceptron
class MulticategoryPerceptron(nn.Module):
    def __init__(self):
        super(MulticategoryPerceptron, self).__init__()
        self.fc = nn.Linear(784, 10)  # 784 input features, 10 output classes

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std dev. of MNIST dataset
])

full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# Load the test dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# define the activation function
def step_function(x):
    return torch.sign(F.relu(x))


def Train(First_N_Samples, ETA, EPS, Enable = 0):

    # Training loop (3.1)
    # epoch = 0
    errors = []

    # Define parameters
    n = First_N_Samples  # Number of training samples
    eta = ETA  # Learning rate
    epsilon = EPS  # Stop criterion threshold


    # Initialize the model and move it to the GPU
    model = MulticategoryPerceptron().to(device)

    for param in model.parameters():
        if param.requires_grad:
            init.uniform_(param, -1, 1)

    # Extract the first n samples
    train_dataset = torch.utils.data.Subset(full_train_dataset, range(n))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)


    while True:
        misclassified = 0

        # 3.1.1 Loop
        for images, labels in train_loader:

            # Move data to GPU
            images, labels = images.to(device), labels.to(device)

            #print(labels)
            # 3.1.1.1 Calculate the induced local fields with the current training sample and weights
            outputs = model(images.view(-1, 784))

            #print(outputs)
            # 3.1.1.2 Calculate the predicted labels
            _, predicted = torch.max(outputs.data, 1)

            #print(predicted)
            # 3.1.1.3 Check if j is not the same as the input label
            if predicted != labels:
                misclassified += 1

            if len(errors) == 0:
                pass
            else:
                # Weight update (3.1.3.1)
                dxi = torch.zeros(10, 1, device=device)
                dxi[labels] = 1
                Wxi = torch.mm(model.fc.weight, images.view(-1, 784).t())
                model.fc.weight.data += eta * torch.mm((dxi - step_function(Wxi)), images.view(-1, 784))

        errors.append(misclassified)
        #epoch += 1

        print(f'The normalized error of {n} samples are {errors[-1] / n} for epoch {len(errors)-1}.')

        # Checking 2
        CH = (len(errors) >= 130) * Enable
        # 3.1.3 Check the stopping criterion
        if errors[-1] / n <= epsilon or CH:
            break

    plt.figure()
    plt.plot(range(len(errors)), errors, color = 'b', marker = 'o', ms = 6)
    plt.xticks(range(0, len(errors), (len(errors) - 1)//19 + 1))
    plt.xlabel('Number of epochs')
    plt.ylabel('Misclassifications')
    plt.title(f'(Misclassifications) vs (Epoch number)')
    plt.show()

    return model, errors

def Test(model):

    # Testing loop (3.2)
    test_errors = 0

    for images, labels in test_loader:
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)

        # 3.2.1 Calculate the induced local fields with the current test sample and weights
        outputs = model(images.view(-1, 784))

        # 3.2.2 Calculate the predicted labels
        _, predicted = torch.max(outputs.data, 1)

        # Calculate the test misclassification errors
        if predicted != labels:
            test_errors += 1

    percentage_test_errors = (test_errors / len(test_loader)) * 100

    print(f"Percentage of Misclassified Test Samples: {percentage_test_errors}%")


model_updated_50, errors_50 = Train(50, 1, 0)
Test(model_updated_50)
print('\n\n\n\n\n\n----- End of one-time experiment with 50 samples (Step # f) -----\n\n\n\n\n\n')
model_updated_1000, errors_1000 = Train(1000, 1, 0)
Test(model_updated_1000)
print('\n\n\n\n\n\n----- End of one-time experiment with 1000 samples (Step # g) -----\n\n\n\n\n\n')
model_updated_6000, errors_6000 = Train(6000, 1, 0, Enable = 1)
Test(model_updated_6000)
print('\n\n\n\n\n\n----- End of one-time experiment with 6000 samples (Step # h) -----\n\n\n\n\n\n')
for i in range(3):
    model_updated_desired, errors_desired = Train(6000, 2, 0.03)
    Test(model_updated_desired)
    print(f'\n\n\n\n\n\n----- End of {i+1}-time/times experiment with 6000 samples (Step # i), with eta = 2, and epsilon = 0.3 -----\n\n\n\n\n\n')
