from fastai.vision import *
from pretrainedmodels.models.senet import se_resnext50_32x4d, se_resnext101_32x4d, se_resnet50
from torchvision.models.densenet import densenet121, densenet169
from pretrainedmodels.models.xception import xception

def create_densenet(type, pretrained):
    if type == 'densenet121':
        model = densenet121(pretrained=pretrained)
    elif type == 'densenet169':
        model = densenet169(pretrained=pretrained)
    conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

    densenet_layer0_children = list(model.features.children())
    conv1.weight.data[:, 0:3, :, :] = densenet_layer0_children[0].weight.data
    conv1.weight.data[:, 3, :, :] = densenet_layer0_children[0].weight.data[:, 0, :, :].clone()
    model = nn.Sequential(*([conv1] + densenet_layer0_children[1:]))

    return model

def create_xception(pretrained):
    model = xception(pretrained='imagenet' if pretrained else None)

    conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)
    conv1.weight.data[:, 0:3, :, :] = model.conv1.weight.data
    conv1.weight.data[:, 3, :, :] = model.conv1.weight.data[:, 0, :, :].clone()
    model.conv1 = conv1
    return model

def create_resnet(type, pretrained):
    if type == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif type == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif type == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif type == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    else:
        raise Exception('Unsupported model type: "{}"'.format(type))

    conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv1.weight.data[:, 0:3, :, :] = model.conv1.weight.data
    conv1.weight.data[:, 3, :, :] = model.conv1.weight.data[:, 0, :, :].clone()
    model.conv1 = conv1
    return model

def create_senet(type, pretrained):
    if type == 'seresnext50':
        model = se_resnext50_32x4d(pretrained='imagenet' if pretrained else None)
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif type == 'se_resnet50':
        model = se_resnet50(pretrained='imagenet' if pretrained else None)
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif type == 'seresnext101':
        model = se_resnext101_32x4d(pretrained='imagenet' if pretrained else None)
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        raise Exception('Unsupported model type: ''{}'''.format(type))

    senet_layer0_children = list(model.layer0.children())
    conv1.weight.data[:, 0:3, :, :] = senet_layer0_children[0].weight.data
    conv1.weight.data[:, 3, :, :] = senet_layer0_children[0].weight.data[:, 0, :, :].clone()
    model.layer0 = nn.Sequential(*([conv1] + senet_layer0_children[1:]))
    return model

def resnet18(pretrained):
    return create_resnet('resnet18', pretrained)

def resnet34(pretrained):
    return create_resnet('resnet34', pretrained)

def resnet50(pretrained):
    return create_resnet('resnet50', pretrained)

def resnet101(pretrained):
    return create_resnet('resnet101', pretrained)

def seresnext50(pretrained):
    return create_senet('seresnext50', pretrained)

def seresnet50(pretrained):
    return create_senet('se_resnet50', pretrained)

def seresnext101(pretrained):
    return create_senet('seresnext101', pretrained)

def Xception(pretrained):
    return create_xception(pretrained)

def Densenet121(pretrained):
    return create_densenet('densenet121', pretrained)

def Densenet169(pretrained):
    return create_densenet('densenet169', pretrained)