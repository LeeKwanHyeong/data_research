import torch.optim as optim

def build_optimizer_and_scheduler(model, cfg):
    opt = optim.AdamW(model.parameters(), lr = cfg.lr, weight_decay = cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.t_max)
    return opt, sched