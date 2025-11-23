from torch.utils.data import DataLoader
from torch import optim
from pga import PGAHead
from head_arcface import ArcFaceHead
from iresnet50_with_feats import ResNet50
from dataset_pkmix import PKMixDataset, PKMixSampler
from train import train_CASIA_ResNet50
from config import config
import torch
    
def reset_all(num_classes):
    torch.cuda.empty_cache()
    model = ResNet50().to(config.device)
    arcface_head = ArcFaceHead(in_features=config.embedding_size, out_features=num_classes).to(config.device)
    pga = PGAHead(num_layers=config.num_layers, device=config.device, topk=config.topk, t_diff=config.t_diff).to(config.device)

    params = list(model.parameters()) + list(arcface_head.parameters())
    optimizer = optim.SGD(
        params=params,
        lr=config.learning_rate,
        momentum=config.sgd_momentum,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.milestones,
        gamma=config.gamma
    )

    optimizer_pga = optim.AdamW([
        {"params": pga.parameters(), "lr": config.learning_rate_pga}
    ], weight_decay=config.weight_decay_pga) 

    scheduler_pga = torch.optim.lr_scheduler.CosineAnnealingLR( 
        optimizer=optimizer_pga,
        T_max=config.total_epochs - config.warmup_epochs,
        eta_min=config.learning_rate_pga * 0.02
    )

    return model, arcface_head, pga, optimizer, scheduler, optimizer_pga, scheduler_pga

if __name__ == "__main__":
    dataset = PKMixDataset(root=config.casia_train_root, transform=config.train_transform)

    sampler = PKMixSampler(data=dataset, P=config.pkmix_P, K=config.pkmix_K, batch_main_size=config.batch_main_size)
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
    )