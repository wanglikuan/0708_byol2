import torch
from byol_pytorch import BYOL
from torchvision import models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision

### transformer ###
class TransformsSimCLR:
    """
    一种随机数据扩充模块，它对任意给定的数据实例进行随机转换，
    得到同一实例的两个相关视图，
    记为x̃i和x̃j，我们认为这是一个正对。
    """

    def __init__(self, size, train=True):
        """
        :param size:图片尺寸
        """
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.train = train

    def __call__(self, x):
        """
        :param x: 图片
        :return: x̃i和x̃j
        """

        if self.train:
            return self.train_transform(x), self.train_transform(x)
        else:
            return self.test_transform(x)
### transformer end ###

device = "cuda"
round_num = 0

train_dataset = CIFAR10(
    root='dataset',
    train=True,
    transform=TransformsSimCLR(size=56),
    download=True
)  # 训练数据集

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    drop_last=True,
)  # 训练数据加载器

iter_trainloader = iter(train_loader)

#def get_next_train_batch():
#    try:
#        # Samples a new batch for persionalizing
#        (x, y) = next(iter_trainloader)
#    except StopIteration:
#        # restart the generator if the previous generator is exhausted.
#        iter_trainloader = iter(trainloader)
#        (x, y) = next(iter_trainloader)
#
#    if type(x) == type([]):
#        x[0] = x[0].to(device)
#    else:
#        x = x.to(device)
#    y = y.to(device)
#
#    return x, y

def get_next_train_batch():
    global iter_trainloader
    global train_loader
    try:
        # Samples a new batch for persionalizing
        ((x_i, x_j), y) = next(iter_trainloader)
    except StopIteration:
        # restart the generator if the previous generator is exhausted.
        iter_trainloader = iter(train_loader)
        ((x_i, x_j), y) = next(iter_trainloader)

    if type(x_i) == type([]):
        x_i[0] = x_i[0].to(device)
        x_j[0] = x_j[0].to(device)        
    else:
        x_i = x_i.to(device)
        x_j = x_j.to(device)        
    y = y.to(device)

    return x_i, x_j, y


resnet = models.resnet18(pretrained=True)

renet_device = resnet.to(device)

learner = BYOL(
    renet_device,
    image_size = 32,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 32, 32)

for _ in range(70000):
    #images = sample_unlabelled_images()
    print('epoch: ',round_num,' begin')
    round_num += 1
    images_i, images_j, y = get_next_train_batch()
    #print('loaded training data')    
    loss = learner(images_i,images_j)
    #print('training loss got')        
    opt.zero_grad()
    loss.backward()
    print('backward finished')    
    opt.step()
    learner.update_moving_average() # update moving average of target encoder
    print('---------------------------')    

# save your improved network
torch.save(resnet.state_dict(), './improved-net-70000-64.pt')
