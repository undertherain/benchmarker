class Net:
    def __call__(self, data):
        x, y = data
        result = x @ y
        return result


def get_kernel(params):
    return Net()
