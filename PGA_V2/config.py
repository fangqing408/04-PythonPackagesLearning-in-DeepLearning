import torch
from torchvision import transforms as T

class Config:
    batch_size = 256
    pkmix_P = 4
    pkmix_K = 8
    batch_main_size = 224
    num_workers = 6
    pin_memory = True
    drop_last = True
    tensorboard_name = "./train_casia_log_30_1121/arcface_pga_448_8_8"
    embedding_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_layers = 6
    topk = 5
    t_diff = 1
    learning_rate = 0.01
    learning_rate_pga = 5e-4
    weight_decay = 5e-4
    weight_decay_pga = 0.0
    sgd_momentum = 0.9
    milestones = [20, 28]
    gamma = 0.1
    warmup_epochs = 8
    total_epochs = 32
    lambda_K = 1
    input_shape = [3, 112, 112]

    casia_train_root = "./webface_112x112"
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])
    lfw_pair = "./data/lfw_test_pair.txt"
    lfw_test_root = "./data/lfw-align-128"
    
config = Config()