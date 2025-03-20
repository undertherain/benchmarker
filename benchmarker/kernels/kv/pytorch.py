import torch
import torch.nn.functional as F


class KV_Attention:
    def __call__(self, k, q, v):
        # print("simulating forward in KV")
        scores = torch.matmul(q, k.transpose(-2, -1)) 
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        # print("output shape:", output.shape)


    def half(self):
        pass

    def to(self, device):
        pass

    def eval(self):
        pass

def get_kernel(params):
    return KV_Attention()
