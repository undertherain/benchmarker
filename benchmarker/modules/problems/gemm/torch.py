class Kernel:
    def __call__(self, data):
        x, y, c = data
        c = x @ y  # + c
        return c


def get_kernel(params):
    return Kernel()
