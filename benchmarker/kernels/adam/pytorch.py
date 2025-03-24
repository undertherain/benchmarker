import torch
from torch.optim import AdamW


class Optim:
    def __init__(self, model_size):
        data = torch.rand((model_size,))
        self.model = torch.nn.Linear(model_size, 1, bias=None)
        self.model.weight.grad = torch.ones_like(self.model.weight)
        #self.params = torch.nn.parameter.Parameter(data, requires_grad=True)
        #self.params.grad = torch.rand((model_size,))
        self.optim =AdamW(self.model.parameters(), lr=0.00001)

    def __call__(self, dummy):
        # print("simulating forward in KV")
        self.optim.step()
        # print("output shape:", output.shape)

    def half(self):
        raise NotImplementedError()

    def to(self, device):
        print("MOVIN ADAM to ", device)
        self.model.to(device)
        for state in self.optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def eval(self):
        pass

def get_kernel(params):
    return Optim(params["problem"]["model_size"])
