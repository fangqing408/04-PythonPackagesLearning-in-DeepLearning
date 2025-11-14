import torch
from torchvision import transforms as T

class Config:
    # ==================================================
    # ==================default=========================
    # ==================================================
    batch_size = 512

    overlap_ratio = 0.4
    overlap_shuffle = True

    pk_P = 64
    pk_K = 8
    pk_shuffle = True
    pk_seed = 42
    
    # 当自己实现采样器的话，DataLoader 的 shuffle 参数失效，否则为 True 为随机采样，False 为顺序采样
    dataloader_shuffle = True
    num_workers = 6
    pin_memory = True
    drop_last = True
    tensorboard_name = "./train_casia_log_30/default"
    embedding_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cosine_scale = 16
    num_layers = 10
    topk = 5
    t_diff = 1
    use_ema = True
    ema_m = 0.8
    learning_rate = 1e-3
    learning_rate_pga = 5e-4
    weight_decay = 5e-4
    weight_decay_pga = 0.0
    sgd_momentum = 0.9
    warmup_epochs = 24
    total_epochs = 175
    lambda_K = 16
    lambda_Z = 8
    lambda_idea = 1
    lambda_modify = False
    input_shape = [3, 112, 112]
    # ==================================================
    # ===================train==========================
    # ==================================================
    train_root = "./mnist_train_torch"
    casia_train_root = "./CASIA-WebFace"
    train_transform = T.Compose([ 
        # T.Grayscale(num_output_channels=1),
        T.RandomHorizontalFlip(p=0.5), # MNIST 禁止使用，数字具有方向性，翻转会带来语义混乱
        T.Resize((136, 136)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        # 需要注意的是，这个 mean 指的是均值减去 0.5 并不是把当前的像素点的均值变成 0.5，
        # 也就是默认了现在的均值为 0.5，标准差为 0.5，转化为了均值为 0，标准差为 1 的一般形式
    ])
    # ==================================================
    # ===================test===========================
    # ==================================================
    test_root = "./mnist_test_torch"
    lfw_pair = "./data/lfw_test_pair.txt"
    lfw_test_root = "./data/lfw-align-128"
    test_transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.1307], std=[0.3081])
    ])
    
config = Config()

# 预处理影响太大，设置相同的预处理种子，将 pga 的学习率变成了之前的三倍
# sofmax_pga_log 是 train_mnist_log 的实验版本
# ======================================================
# =====pga_v1 静态 ema，不下降 lambda_K、lambda_Z========
# pga_v2 动态 ema，下降 lambda_K, lambda_Z，学习率增 0.5 倍
# pga_v3 静态 ema，不下降 lambda_K、lambda_Z，学习率 0.1 倍，构造 idea 的时候，负样本 sigma_out 给了很小值 0.01
# =====softmax_v1，第一次 softmax ======================
# =====softmax_v2，第二次 softmax ======================
# 预训练的差值会一直持续到训练结束，甚至都无法补正！！！！！