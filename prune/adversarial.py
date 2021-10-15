import torch
import torch.nn.functional as F

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = perturb_clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = perturb_clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = perturb_clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def init_adv_hyperparams(eps, alpha, dataset_std, dataset_mean, pgd_ite=None, restarts=None):
    epsilon_adv = (eps / 255) / dataset_std
    alpha_adv = (alpha / 255) / dataset_std
    pgd_alpha = (2 / 255) / dataset_std
    upper = (1 - dataset_mean) / dataset_std
    lower = (0 - dataset_mean) / dataset_std
    return {"eps": epsilon_adv, "alpha": alpha_adv, "pgd_alpha": pgd_alpha, "upper": upper, "lower": lower, "ite": pgd_ite, "restarts": restarts}

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