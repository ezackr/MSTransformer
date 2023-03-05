import torch


def sdr(target, estimate):
    source = torch.sum(torch.square(target), dim=(1, 2)) + 1e-10
    error = torch.sum(torch.square(target - estimate), dim=(1, 2)) + 1e-10
    score = 10 * torch.log10(source / error)
    return torch.mean(score)


def get_number_of_parameters(model):
    num_param = 0
    for name, param in model.named_parameters():
        num_param += param.numel()
    return num_param
