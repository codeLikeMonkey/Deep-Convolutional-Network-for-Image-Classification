import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable



#define 
#CNN
class CNN0(nn.Module):
    def __init__(self):
        super(CNN0,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(9*9*64,10)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,265,kernel_size=3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(265,265,kernel_size=3,padding = 1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(265,265,kernel_size=3,padding = 1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(265,265,kernel_size=3,padding = 1),
            nn.ReLU(),
        )
        self.fc6 = nn.Linear(16960,10)

#         self.sf = nn.Softmax

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0),-1)
#         print(out.data.cpu().numpy().shape)
        out = self.fc6(out)
        return out
    
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,64,kernel_size=3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,padding = 1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc5 = nn.Linear(256 * 8 *8,256)
        self.fc6 = nn.Linear(256,10)

#         self.sf = nn.Softmax

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
#         print(out.data.cpu().numpy().shape)
        out = out.view(out.size(0),-1)
        out = self.fc5(out)
        out = self.fc6(out)
        return out    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# train the model 
def trainning(net = None, train_loader = None,test_loader = None, criterion = None, optimizer = None, num_epochs = 0):
    net.train()
    train_loss_container = []
    train_acc_container = []
    test_loss_container = []
    test_acc_container =[]
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            #begin
            optimizer.zero_grad()
            outputs = net(images)
            train_loss = criterion(outputs,labels)
            train_loss.backward()
            optimizer.step()
#             print(1)

            if (i + 1) % 100 == 0:
                test_acc = testing(net = net,data_to_test = test_loader)
                train_acc = testing(net = net, data_to_test = train_loader)
                train_loss_container.append(train_loss.data[0])
                test_acc_container.append(test_acc)
                train_acc_container.append(train_acc)
#                 print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
#                        %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, train_loss.data[0]))
                print ('Epoch [%d/%d], Train: %.4f Test: %.4f'
                       %(epoch+1, num_epochs, train_acc,test_acc))

#                 print(test_acc)
#                 print("train:%s\ttest:%s"%(train_acc,test_acc))
#                
    return train_acc_container,test_acc_container
                
                
def testing(net = None, data_to_test = None):
    # Test the Model
    net.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in data_to_test:
        images = Variable(images).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    return 100 * correct / total
