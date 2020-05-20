import argparse
import torch.nn as nn

def get_kernel(params, unparsed_args=None):
    assert params["mode"] == "inference"
    parser = argparse.ArgumentParser(description='Benchmark gru kernel')
    parser.add_argument('--cnt_units', type=int, default=512)
    parser.add_argument('--cnt_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=bool, default=False)
    args = parser.parse_args(unparsed_args)
    params["problem"].update(vars(args))
    Net = nn.GRU(input_size=params["problem"]["size"][2],
                  hidden_size=args.cnt_units,
                  num_layers=args.cnt_layers,
                  bias=True,
                  bidirectional=params["problem"]["bidirectional"])
    return Net
