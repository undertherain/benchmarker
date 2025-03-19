class KV_Attention:
    def __call__(self, kv):
        print("simulating forward in KV")

    def half(self):
        pass

    def to(self, device):
        pass

    def eval(self):
        pass

def get_kernel(params):
    return KV_Attention()
