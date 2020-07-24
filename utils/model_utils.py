'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''


def get_num_parameters(module):
    params = [p.nelement() for p in module.parameters() if p.requires_grad]
    num = sum(params)
    return num
