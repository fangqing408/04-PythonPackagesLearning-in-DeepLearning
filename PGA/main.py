from torch.utils.data import DataLoader
from torch import optim
from pga_v2 import PGAHead
from heads_arcface import ArcFaceHead
from models_resnet50 import ResNet50
# from dataset_overlap import OverlapSampler, OverlapDataset
# from dataset_pk import PKSampler, PKDataset
from dataset_pkrandom import PKMixDataset, PKMixSampler
# from train_and_evaluate_MNIST import train_MNIST
# from train_and_evaluate_CASIA import train_CASIA
from train_and_evaluate_CASIA_ResNet50 import train_CASIA_ResNet50
from config import config
import torch
    
def reset_all(num_classes):
    # 重新初始化所有对象
    torch.cuda.empty_cache()
    # backbone = MobileNetWithPGA(embedding_size=config.embedding_size).to(config.device)
    model = ResNet50().to(config.device)
    arcface_head = ArcFaceHead(in_features=config.embedding_size, out_features=num_classes).to(config.device)
    pga = PGAHead(num_layers=config.num_layers, device=config.device, topk=config.topk, t_diff=config.t_diff).to(config.device)

    backbone, fc, bn_bias = [], [], []
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith('fc'): fc.append(p)
        elif p.ndim==1 or n.endswith('.bias') or 'bn' in n.lower(): bn_bias.append(p)
        else: backbone.append(p)
        
    # optimizer = optim.AdamW([
    #     {"params": backbone.parameters(), "lr": config.learning_rate},
    #     {"params": softmax_head.parameters(), "lr": config.learning_rate},
    # ], weight_decay=config.weight_decay)
    
    optimizer = optim.SGD([
        {'params': backbone, 'lr': config.learning_rate, 'weight_decay': config.weight_decay},
        {'params': bn_bias, 'lr': config.learning_rate, 'weight_decay': 0.0},
        {"params": arcface_head.parameters(), "lr": 3 * config.learning_rate}
    ], momentum=config.sgd_momentum)

    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer=optimizer, 
    #     step_size=2,
    #     gamma=0.95
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,  
        T_max=config.total_epochs,
        eta_min=config.learning_rate * 0.01
    )

    optimizer_pga = optim.AdamW([
        {"params": pga.parameters(), "lr": config.learning_rate_pga}
    ], weight_decay=config.weight_decay_pga) 
    
    # optimizer_pga = optim.SGD([
    #     {"params": pga.parameters(), "lr": config.learning_rate_pga}
    # ], weight_decay=config.weight_decay_pga) 

    scheduler_pga = torch.optim.lr_scheduler.CosineAnnealingLR( 
        optimizer=optimizer_pga,
        T_max=config.total_epochs - config.warmup_epochs,
        eta_min=config.learning_rate_pga * 0.02
    )

    # scheduler_pga = torch.optim.lr_scheduler.StepLR( 
    #     optimizer=optimizer_pga,  
    #     step_size=2,
    #     gamma=0.95
    # )

    return model, arcface_head, pga, optimizer, scheduler, optimizer_pga, scheduler_pga

if __name__ == "__main__":
    dataset = PKMixDataset(root=config.casia_train_root, transform=config.train_transform)

    sampler = PKMixSampler(data=dataset, P=8, K=8, batch_main_size=448, shuffle=True, avoid_overlap=True)
    loader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    num_classes = dataset.num_classes
    model, arcface_head, pga, optimizer, scheduler, optimizer_pga, scheduler_pga = reset_all(num_classes=num_classes)
    train_CASIA_ResNet50(
        name=config.tensorboard_name,
        model=model,
        head=arcface_head,
        pga=pga,
        loader=loader,
        lfw_test_root=config.lfw_test_root,
        pairs_file=config.lfw_pair,
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_pga=optimizer_pga,
        scheduler_pga=scheduler_pga,
        device=config.device,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.total_epochs, 
        lambda_K=config.lambda_K,
        lambda_modify=config.lambda_modify
    )