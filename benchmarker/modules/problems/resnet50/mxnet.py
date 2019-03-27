from mxnet.gluon.model_zoo import vision

#Net = vision.resnet50_v1(pretrained=False)
Net = vision.get_model("resnet50_v1")
