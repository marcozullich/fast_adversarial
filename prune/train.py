import torch
from torch.cuda import init
from torch.optim import lr_scheduler
import utils

def init_adv_hyperparams(eps, alpha, dataset_std, dataset_mean):
    epsilon_adv = (eps / 255) / dataset_std
    alpha_adv = (alpha / 255) / dataset_std
    pgd_alpha = (2 / 255) / dataset_std
    upper = (1 - dataset_mean) / dataset_std
    lower = (0 - dataset_mean) / dataset_std
    return {"eps": epsilon_adv, "alpha": alpha_adv, "pgd_alpha": pgd_alpha, "upper": upper, "lower": lower}

def init_cyclic_scheduler(optimizer, lr_min, lr_max, num_epochs, num_batches):
    num_steps = num_epochs * num_batches
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=lr_min,
        max_lr=lr_max,
        step_size_up=num_steps/2,
        step_size_down=num_steps/2
    )
    return scheduler


def train(
    model: torch.nn.Module,
    num_epochs: int,
    trainloader: torch.utils.data.DataLoader,
    dataset_mean:torch.Tensor,
    dataset_std:torch.Tensor,
    loss_fn,
    lr_min:float=0.0,
    lr_max:float=0.2,
    wd:float=5e-4,
    momentum:float=.9,
    eps_adv:int=8,
    alpha_adv:float=10,
    delta_init:str="random",
    device=None,
    mask=None,
    verbose=True
):
    adv_hyp = init_adv_hyperparams(eps_adv, alpha_adv, dataset_std)

    if device is None:
        device = utils.get_device()
    
    model = model.to(device).train()
    if verbose:
        print(f"--> Model is on {device} and in train mode")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=wd)
    
    lr_scheduler = init_cyclic_scheduler(optimizer, lr_min, lr_max, num_epochs, len(trainloader))

    for epoch in len(num_epochs):
        meters = {"loss": utils.AverageMeter(), "acc": utils.AverageMeter()}
        train_loop()

def train_loop(
    model,
    trainloader,
    optimizer,
    lr_scheduler,
    adversarial_hyp,
    device,
    mask,
    meters
):
    for i, (data, labels) in enumerate(trainloader):
        
        batch_loop(model, data, labels, optimizer, lr_scheduler, )

def perturb_clamp(data, upper, lower):
    return torch.max(torch.min(data, upper), lower)

def init_perturbation(adv_hyp, data, device):
    perturb = torch.zeros_like(data).to(device)
    for j in range(len(adv_hyp["eps"])):
        perturb[:, j, :, :].uniform_(-adv_hyp["eps"][j][0][0].item(), adv_hyp["eps"][j][0][0].item())
    perturb.data = perturb_clamp(perturb.data, adv_hyp["upper"]-data, adv_hyp["lower"]-data)
    perturb.requires_grad = True
    return perturb

def apply_perturbation(perturbation, data, adv_hyp):
    gradient = perturbation.grad.detach()
    perturbation.data = perturb_clamp(perturbation + adv_hyp["alpha"] * torch.sign(gradient), -adv_hyp["eps"], adv_hyp["eps"])
    perturbation.data = perturb_clamp(perturbation[:data.shape[0]], adv_hyp["upper"]-data, adv_hyp["lower"]-data)
    return perturbation.detach()


def batch_loop(
    model,
    data,
    labels,
    loss_fn,
    optimizer,
    optimizer_step:bool,
    lr_scheduler,
    adversarial_hyp,
    device,
    meters
    
):
    data = data.to(device)
    labels = labels.to(device)
    perturb = init_perturbation(adversarial_hyp, device)

    prediction = model(data + perturb[:data.shape[0]])
    loss = loss_fn(prediction, labels)
    loss.backward()
    apply_perturbation(perturb, data, adversarial_hyp)
    

    prediction = model(data + perturb[:data.shape[0]])
    loss = loss_fn(prediction, labels)
    loss.backward()
    if optimizer_step:
        optimizer.step()
    
    meters["train"].update(loss.item(), labels.shape[0])
    meters["acc"].update(utils.accuracy(prediction, labels).item(), labels.shape[0])

    lr_scheduler.step()














