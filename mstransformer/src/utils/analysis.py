def get_number_of_parameters(model):
    num_param = 0
    for name, param in model.named_parameters():
        print(name)
        num_param += param.numel()
    return num_param
