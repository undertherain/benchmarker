from mxnet.gluon.model_zoo import vision

def get_kernel(params):
    return vision.vgg16(pretrained=False)
