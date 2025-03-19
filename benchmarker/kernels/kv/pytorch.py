import torch


class KV_Attention:
    def __call__(self, k, q):
        print("simulating forward in KV")
        scores = torch.matmul(q, k.transpose(-2, -1)) 

        print("attention shape:", scores.shape)

    def half(self):
        pass

    def to(self, device):
        pass

    def eval(self):
        pass

def get_kernel(params):
    return KV_Attention()
