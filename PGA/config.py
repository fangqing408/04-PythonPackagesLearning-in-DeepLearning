import torch
import torchvision.transforms as T

class Config:
    backbone = 'resnet153' # [timm]
    metric = 'arcface' # ['arcface', 'cosface]
    embedding_size = 512
    input_shape = [1, 128, 128]
    train_transform = T.Compose([ 
        T.Grayscale(num_output_channels=1),
        T.RandomHorizontalFlip(p=0.5),
        T.Resize((144, 144)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]) 
        # 需要注意的是，这个 mean 指的是均值减去 0.5 并不是把当前的像素点的均值变成 0.5，
        # 也就是默认了现在的均值为 0.5，标准差为 0.5，转化为了均值为 0，标准差为 1 的一般形式
    ])
    test_transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    train_root = "./mnist_train_torch"
    test_root = "./mnist_test_torch"
    test_list = "./lfw_test_pair.txt"
    checkpoints = "checkpoints"
    test_model = "checkpoints/0.pth"
    train_batch_size = 256
    test_batch_size = 1
    epochs = 10
    optimizer = "sgd" # ['sgd', 'adam']
    lr = 1e-1
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = "focal_loss" # ['focal_loss', 'cross_entropy']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory = True
    num_workers = 4
config = Config()


    