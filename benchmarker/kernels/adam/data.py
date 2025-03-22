import torch


def get_data(params):
    # if isinstance(params["problem"]["size"], int):
    #     params["problem"]["size"] = (params["problem"]["size"], 128)
    # assert params["problem"]["size"][0] % params["batch_size"] == 0
    # params["problem"]["len_sequence"] = params["problem"]["size"][1]

    return [{"dummy": torch.tensor(0)}]
