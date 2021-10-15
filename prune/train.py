from pickle import load
import torch
from torch.cuda import init
from torch.nn.modules import loss
from torch.optim import lr_scheduler
import utils
import adversarial
from mask import Mask



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
    device=None,
    mask=None,
    ite_optimizer_step=1,
    checkpoint_save_location=None,
    load_from_checkpoint=False,
    verbose=True
):
    adv_hyp = adversarial.init_adv_hyperparams(eps_adv, alpha_adv, dataset_std, dataset_mean)

    if device is None:
        device = utils.get_device()
    
    model = model.to(device).train()
    if verbose:
        print(f"--> Model is on {device} and in train mode")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=wd)
    
    lr_scheduler = init_cyclic_scheduler(optimizer, lr_min, lr_max, num_epochs, len(trainloader))

    epoch_start, ite_start = 0, 0
    if load_from_checkpoint:
        epoch_checkpoint, ite_checkpoint = load_checkpoint(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            location=checkpoint_save_location
        )
        epoch_start, ite_start = calculate_start_epoch_and_iteration(epoch_checkpoint, ite_checkpoint, num_epochs, len(trainloader))
        print(f"Training will resume from epoch {epoch_start} and ite {ite_start}")

    for epoch in range(epoch_start, num_epochs):
        if epoch > epoch_start:
            ite_start = 0

        meters = {"loss": utils.AverageMeter(), "acc": utils.AverageMeter()}

        train_loop(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            adversarial_hyp=adv_hyp,
            device=device,
            mask=mask,
            meters=meters,
            ite_optimizer_step=ite_optimizer_step,
            checkpoint_save_location=checkpoint_save_location,
            epoch=epoch,
            ite_start=ite_start
        )
        if verbose:
            current_loss = meters["loss"].avg
            current_acc = meters["acc"].avg
            print(f"# Ep. {epoch}/{num_epochs} - Loss: {current_loss:.4f} - Acc: {current_acc:.4f}")

def train_loop(
    model,
    trainloader,
    optimizer,
    loss_fn,
    lr_scheduler,
    adversarial_hyp,
    device,
    mask:Mask,
    meters,
    ite_optimizer_step,
    checkpoint_save_location,
    no_grad,
    epoch,
    ite_start=0,
    adv_eval=False
):
    if (not no_grad) and loss_fn is None:
        raise AttributeError("Gradient backprop requires loss_fn to be ≠ None")
    skipped_ites = (ite_start > 0)
    with torch.set_grad_enabled(not no_grad):
        for i, (data, labels) in enumerate(trainloader):
            if i < ite_start:
                continue
            if skipped_ites:
                print(f"Training for this epoch starts at iteration {i+1}")
            optimizer_step = (i % ite_optimizer_step == 0) if (not no_grad) else False
            batch_loop(
                model=model,
                data=data,
                labels=labels,
                loss_fn=loss_fn,
                optimizer=optimizer,
                optimizer_step=optimizer_step,
                lr_scheduler=lr_scheduler,
                adversarial_hyp=adversarial_hyp,
                device=device,
                mask=mask,
                meters=meters,
                adv_eval=adv_eval,
                checkpoint_save_location=checkpoint_save_location,
                epoch=epoch,
                iteration=i
            )



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
    mask:Mask,
    meters,
    checkpoint_save_location,
    no_grad,
    adv_eval,
    epoch=None,
    iteration=None
):
    data = data.to(device)
    labels = labels.to(device)
    if adversarial_hyp is not None:
        if not adv_eval:
            perturb = adversarial.init_perturbation(adversarial_hyp, device)
            prediction = model(data + perturb[:data.shape[0]])
            loss = loss_fn(prediction, labels)
            loss.backward()
            if mask is not None:
                mask.apply_mask(to_grad=True)
            adversarial.apply_perturbation(perturb, data, adversarial_hyp)
        else:
            perturb = adversarial.attack_pgd(model, data, labels, adversarial_hyp["eps"], adversarial_hyp["alpha"], adversarial_hyp["ite"], adversarial_hyp["restarts"])
        prediction = model(data + perturb[:data.shape[0]])
    else:
        prediction = model(data)

    if loss_fn is not None:
        loss = loss_fn(prediction, labels)
        
        if (not no_grad) and (not adv_eval):
            if optimizer_step:
                optimizer.zero_grad()

            loss.backward()
            if mask is not None:
                mask.apply_mask(to_grad=True)

            if optimizer_step:
                optimizer.step()
    
        meters["loss"].update(loss.item(), labels.shape[0])

        
    acc = utils.accuracy(prediction, labels).item()
    meters["acc"].update(acc, labels.shape[0])
    if lr_scheduler is not None:
        lr_scheduler.step()

    if checkpoint_save_location is not None:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                location=checkpoint_save_location,
                loss=loss.item(),
                accuracy=acc,
                epoch=epoch,
                iteration=iteration
            )

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    location: str,
    loss: float,
    accuracy: float,
    epoch: int,
    iteration: int,
):
    torch.save({
        "state_dict": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
        "epoch": epoch,
        "iteration": iteration

    }, location)

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    location: str,
    device
):
    checkpoint = torch.load(location, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optim"])
    lr_scheduler.load_state_dict(checkpoint["scheduler"])
    
    return checkpoint["epoch"], checkpoint["iteration"]

def calculate_start_epoch_and_iteration(checkpoint_epoch, checkpoint_ite, num_epoch, num_ite_batch):
    epoch_start = checkpoint_epoch
    ite_start = checkpoint_ite

    if checkpoint_epoch > (num_epoch - 1):
        raise RuntimeError(f"The epoch in the checkpoint {checkpoint_epoch + 1} is larger than the number of epochs for training {num_epoch}")

    if checkpoint_ite > (num_ite_batch - 1):
        raise RuntimeError(f"The iteration number in the checkpoint {num_ite_batch + 1} is larger than the number of itrerations for this dataloader {num_ite_batch}, probably the batch size for this training is different than the one from the checkpoint")

    if checkpoint_epoch == (num_epoch - 1) and checkpoint_ite >= (num_ite_batch - 1):
        raise RuntimeError(f"Loading a fully-trained model. Maximum number of epochs for training {num_epoch} - maximum number of iterations for this dataloader {num_ite_batch}")
    
    if checkpoint_ite == (num_ite_batch - 1):
        # finished current epoch, start from next one
        epoch_start += 1
        ite_start = 0
    
    return epoch_start, ite_start



def test(
    model: torch.nn.Module,
    testloader: torch.utils.data.DataLoader,
    loss_fn=None,
    adv_eval=True,
    eps_adv:int=8,
    alpha_adv:float=2,
    pgd_ite=40,
    pgd_restarts=10,
    dataset_mean:torch.Tensor=None,
    dataset_std:torch.Tensor=None,
    device=None,
    mask=None,
):
    meters = {"acc": utils.AverageMeter()}
    if loss_fn is not None:
        meters["loss"] = utils.AverageMeter()
    
    model = model.eval()

    adv_hyp = adversarial.init_adv_hyperparams(eps=eps_adv, alpha=alpha_adv, dataset_mean=dataset_mean, dataset_std=dataset_std, pgd_ite=pgd_ite, restarts=pgd_restarts)\
        if adv_eval else None
    
    train_loop(
        model=model,
        trainloader=testloader,
        optimizer=None,
        loss_fn=loss_fn,
        lr_scheduler=None,
        adversarial_hyp=adv_hyp,
        device=device,
        mask=mask,
        meters=meters,
        ite_optimizer_step=None,
        no_grad=(not adv_eval),
        adv_eval=adv_eval
    )













