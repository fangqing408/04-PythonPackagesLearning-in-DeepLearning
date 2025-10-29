import torch
from torchvision import transforms as T

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    overlap_ratio = 0.4
    warmup_epochs = 5
    total_epochs = 40
    learning_rate = 1e-5
    embedding_size = 512
    input_shape = [1, 128, 128]

    train_transform = T.Compose([ 
        T.Grayscale(num_output_channels=1),
        # T.RandomHorizontalFlip(p=0.5),
        T.Resize((128, 128)),
        # T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.1307], std=[0.3081]) 
        # 需要注意的是，这个 mean 指的是均值减去 0.5 并不是把当前的像素点的均值变成 0.5，
        # 也就是默认了现在的均值为 0.5，标准差为 0.5，转化为了均值为 0，标准差为 1 的一般形式
    ])
    test_transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.1307], std=[0.3081])
    ])
    train_root = "./mnist_train_torch"
    test_root = "./mnist_test_torch"

config = Config()