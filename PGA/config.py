import torch
from torchvision import transforms as T

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    overlap_ratio = 0.4
    warmup_epochs = 10
    total_epochs = 100
    learning_rate = 1e-5
    embedding_size = 512
    input_shape = [1, 128, 128]

    train_transform = T.Compose([ 
        T.Grayscale(num_output_channels=1),
        # T.RandomHorizontalFlip(p=0.5), # MNIST 禁止使用，数字具有方向性，翻转会带来语义混乱
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

# 预处理影响太大，设置相同的预处理种子，将 pga 的学习率变成了之前的三倍
# sofmax_pga_log 是 train_mnist_log 的实验版本
# ======================================================
# =====pga_v1 静态 ema，不下降 lambda_K、lambda_Z========
# pga_v2 动态 ema，下降 lambda_K, lambda_Z，学习率增 0.5 倍
# pga_v3 静态 ema，不下降 lambda_K、lambda_Z，学习率 0.1 倍，构造 idea 的时候，负样本 sigma_out 给了很小值 0.01
# =====softmax_v1，第一次 softmax ======================
# =====softmax_v2，第二次 softmax ======================
# 预训练的插值会一直持续到训练结束，甚至都无法补正！！！！！

