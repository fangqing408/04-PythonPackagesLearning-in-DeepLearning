import torch
from torchvision import transforms as T

class Config:
    batch_size = 512

    pkmix_P = 8
    pkmix_K = 8
    pkmix_shuffle = True
    
    dataloader_shuffle = True
    num_workers = 6
    pin_memory = True
    drop_last = True
    tensorboard_name = "./train_casia_log_30_end/arcface_pga_448_8_8_Borigin_Nema"
    embedding_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cosine_scale = 16
    num_layers = 6
    topk = 5
    t_diff = 1
    learning_rate = 1e-3
    learning_rate_pga = 5e-4
    weight_decay = 5e-4
    weight_decay_pga = 0.0
    sgd_momentum = 0.9
    warmup_epochs = 8
    total_epochs = 28
    lambda_K = 1
    lambda_modify = False
    input_shape = [3, 112, 112]

    # train_root = "./mnist_train_torch"
    casia_train_root = "./CASIA-WebFace"
    train_transform = T.Compose([
        # T.Grayscale(num_output_channels=1),
        T.RandomHorizontalFlip(p=0.5),
        T.Resize((136, 136)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        # 需要注意的是，这个 mean 指的是均值减去 0.5 并不是把当前的像素点的均值变成 0.5，
        # 也就是默认了现在的均值为 0.5，标准差为 0.5，转化为了均值为 0，标准差为 1 的一般形式
    ])

    # test_root = "./mnist_test_torch"
    lfw_pair = "./data/lfw_test_pair.txt"
    lfw_test_root = "./data/lfw-align-128"
    test_transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.1307], std=[0.3081])
    ])
    
config = Config()