import torch, torchvision, datetime
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFile

################################################# FUNCTIONS #################################################À
def validate(model, val_loader) -> None:
    for name, loader in [("Knife", val_loader), ("Scissor", val_loader), ("camera", val_loader), ("cellphone", val_loader), ("electronic", val_loader), ("laptop", val_loader), ("lighter", val_loader), ("powerbank", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():  # <1>
            for imgs, labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1) # <2>
                total += labels.shape[0]  # <3>
                correct += int((predicted == labels).sum())  # <4>

        print("Accuracy {}: {:.2f}".format(name , correct / total))
    return


# training loop with l2 regularization
def training_loop_l2reg(n_epochs, optimizer, model, loss_fn,
                        train_loader) -> None:
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            # l2 regularization
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                          for p in model.parameters())  # <1>
            loss = loss + l2_lambda * l2_norm

            # zero out the gradient, propagate the loss, perform the optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))
    return

# a useful function to show an image that takes a tensor as an input
def show_img(t_img:torch.tensor) -> None:
    assert t_img.shape[0] <= 3 , "You're trying to display a batch of size > 1. Be wary of opening more than one image..."
    to_img = transforms.ToPILImage()
    img = to_img(t_img)
    img.show()
    return 

#################################### MODELS ####################################
class NetBatchNorm(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, 
                               padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1 // 2)
        self.fc1 = nn.Linear(2 * 8 * 8 * n_chans1, 32)
        self.fc2 = nn.Linear(32, 8)
        
    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = out.view(-1, 2 * 8 * 8 * self.n_chans1)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
    
class NetDepth(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2,
                               kernel_size=3, padding=1)
        self.fc1 = nn.Linear(2 * 4 * 4 * n_chans1, 32)
        self.fc2 = nn.Linear(32, 5)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = F.max_pool2d(torch.relu(self.conv3(out)), 2)
        out = out.view(-1, 2 * 4 * 4 * self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

####################################### GLOBAL VARIABLES ##########################################
path = "/home/rick/Ri/SecondYear/2ndSemester/AI-Lab/project_private/temp_train"
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) # train on gpu if available


################################## SCRIPT #######################################
# resize and transform images into tensors
# normalization is done later, so we don't include it in the first transforms
transformations = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])
datafolder = torchvision.datasets.ImageFolder(root=path, transform=transformations)
datafolder.classes = [(x, c) for c, x in enumerate(datafolder.classes)]
print(datafolder.classes)
# create array of indices for training and validation set
n_samples = len(datafolder)
n_val = int(0.3 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

print(train_indices.shape, val_indices.shape, sep="\n")
train_data = [datafolder[x] for x in train_indices]
assert len(train_data) == train_indices.shape[0]
val_data = [datafolder[x] for x in val_indices]
assert len(val_data) == val_indices.shape[0]
# create the training and validation tensors

train_imgs = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]} # can this be made a dictionary of tensors straight-away?
for c, (img_t, label) in enumerate(train_data):
    if c % 1000 == 0:
        print(f"processing img {c}")
    train_imgs[label].append(img_t)

for key in train_imgs:
    train_imgs[key] = torch.stack(train_imgs[key])

for key in train_imgs:
    print(train_imgs[key].shape, key)
    
val_imgs = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]} # can this be made a dictionary of tensors straight-away?
for c, (img_t, label) in enumerate(val_data):
    if c % 1000 == 0:
        print(f"processing img {c}")
    val_imgs[label].append(img_t)

for key in val_imgs:
    val_imgs[key] = torch.stack(val_imgs[key])

for key in val_imgs:
    print(val_imgs[key].shape, key)
# normalize the images belonging to each class! 
# NOTE: normalize for each class,
# not over all the images

ntrain_imgs = []
nval_imgs = []

# normalize the training set
class_means = {}
class_std = {}

for cl in train_imgs:
    class_means[cl] = torch.mean(train_imgs[cl], (0, 2, 3))
    class_std[cl] = torch.std(train_imgs[cl], (0, 2, 3))
    
for key in train_imgs:
    ntrain_imgs.append(((train_imgs[key] - class_means[key][None, :, None, None]) / class_std[key][None, :, None, None], key))

# normalize the validation set
class_means = {}
class_std = {}

for cl in val_imgs:
    class_means[cl] = torch.mean(val_imgs[cl], (0, 2, 3))
    class_std[cl] = torch.std(train_imgs[cl], (0, 2, 3))
    
for key in val_imgs:
    nval_imgs.append(((val_imgs[key] - class_means[key][None, :, None, None]) / class_std[key][None, :, None, None], key))
# further preprocess necessary to perfom forward pass
n_train_imgs = []

for label in range(len(ntrain_imgs)):
    for k in range(ntrain_imgs[label][0].shape[0]):
        n_train_imgs.append((ntrain_imgs[label][0][k], label))
print(len(n_train_imgs))

n_val_imgs = []
for label in range(len(nval_imgs)):
    for k in range(nval_imgs[label][0].shape[0]):
        n_val_imgs.append((nval_imgs[label][0][k], label))
print(len(n_val_imgs))
# create loaders for training set and validation set. Also perform another shuffle
train_loader = torch.utils.data.DataLoader(n_train_imgs, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(n_val_imgs, batch_size=64, shuffle=True)

for img_t, label in train_loader:
    print(img_t.shape, label, type(label), len(label), sep="\n")
    break
for imt_t, label in val_loader:
    print(img_t.shape, label, type(label), len(label), sep="\n")
    break
n_out = len(datafolder.classes)
print(f"[*] Dataset contains {n_out} classes)")



numel_list = [p.numel() for p in model.parameters()]
print(sum(numel_list), numel_list)



model = NetBatchNorm()  
optimizer = optim.SGD(model.parameters(), lr=.6e-2)  
loss_fn = nn.CrossEntropyLoss()  
n_epochs = 20

# train the model
training_loop_l2reg(  
    n_epochs = n_epochs,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
)

# validation
validate(model, val_loader)


model_path = "/home/rick/Ri/SecondYear/2ndSemester/AI-Lab/project_private/"
model_name = "model1.pt"
torch.save(model.state_dict(), model_path + model_name) 