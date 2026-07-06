import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform=transforms.ToTensor()
train_dataset=datasets.MNIST(root='./data',train=True,transform=transform,download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=transform,download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=1000)

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier,self).__init__()
        self.fc1=nn.Linear(28*28,128)
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=x.view(-1,28*28)
        x=F.relu(self.fc1(x))
        return self.fc2(x)

model=DigitClassifier()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(5):
    for image,label in train_loader:
        output=model(image)
        loss=criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss:{loss.item():.2f} for time:{epoch+1}")

correct=0
total=0
with torch.no_grad():
    for image,label in test_loader:
        output=model(image)
        loss=criterion(output,label)
        _,predicted=torch.max(output,1)
        total+=label.size(0)
        correct+=(predicted==label).sum().item()
    print(f"test accuracy is {(correct/total)*100}%")

