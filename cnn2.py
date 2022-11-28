import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np
from PIL import Image

IMAGE_SIZE = 16
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

#train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=composed)
#validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=composed)

class CNN(nn.Module):
    
    # Contructor
    def __init__(self, out_1=16, out_2=32):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 4 * 4, 10)
    
    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)#Flatten to 1D vector
        x = self.fc1(x)
        return x     

def predict(operationBytes):
    try:
            
        img = Image.open(operationBytes)
        img.save('tel.png')
        print(" ----------  here   ------------")
            
    except Exception as e:
        print(e)

    model = CNN(out_1=16, out_2=32)
    model.load_state_dict(torch.load('./mnist_cnn.pt'))
    img = Image.open("./tel.png")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    convert_tensor = transforms.ToTensor()

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    a = transform(img)
    model.eval()
    with torch.no_grad():
        output = model(a[3].view(1, 1, IMAGE_SIZE, IMAGE_SIZE))
        prediction = output.max(1, keepdim=True)[1]
    return prediction[0,0]

if __name__ == "__main__":
    n_epochs=100
    cost_list=[]
    accuracy_list=[]
    N_test=len(validation_dataset)
    COST=0

    model = CNN(out_1=16, out_2=32)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

    for epoch in range(n_epochs):
        COST=0
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST+=loss.data
        
        cost_list.append(COST)
        correct=0
        #perform a prediction on the validation  data  
        for x_test, y_test in validation_loader:
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)

    torch.save(model.state_dict(), "mnist_cnn.pt")

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(cost_list, color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('Cost', color=color)
    ax1.tick_params(axis='y', color=color)
        
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color) 
    ax2.set_xlabel('epoch', color=color)
    ax2.plot( accuracy_list, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()