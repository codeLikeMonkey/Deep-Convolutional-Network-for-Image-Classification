from network import *
import pickle

# Hyper Parameters
num_epochs = 12
batch_size = 100
learning_rate = 0.001

#CIFAR-10

train_dataset = dsets.CIFAR10(root = "./datasets/",train = True,transform = transforms.ToTensor(),download= True)
test_dataset = dsets.CIFAR10(root = "./datasets/",train = False,transform = transforms.ToTensor(),download= True)


#Data Loader

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size,shuffle = False)



cnn = CNN1()
#xavier initalize
nn.init.xavier_normal(cnn.conv1[0].weight)
# nn.init.xavier_normal(cnn.conv1[0].bias)
cnn.cuda()



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(),lr = learning_rate)

train_acc_container,test_acc_container = trainning(
    net = cnn, 
    train_loader = train_loader,
    
    test_loader = test_loader, 
    criterion = criterion,
    optimizer = optimizer, 
    num_epochs = num_epochs,
)

with open("CNN1_result.dat","wb") as f:
    pickle.dump([train_acc_container,test_acc_container],f)
    print("CNN1 data save")




import numpy as np
import matplotlib.pyplot as plt

plt.plot(np.arange(len(train_acc_container))*100,np.array(train_acc_container))
plt.plot(np.arange(len(test_acc_container))*100,np.array(test_acc_container))
plt.legend(["train","test"])
plt.grid()
plt.title("Accuracy over training")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.savefig("CNN1.jpg")
plt.show()