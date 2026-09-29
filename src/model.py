from torch import nn
from torchvision.models import mobilenet_v3_small

from src.dataset import CLASSES
from src.preprocessing import WEIGHTS


def build_model(pretrained=True):
    model = mobilenet_v3_small(weights=WEIGHTS if pretrained else None)

    # Replace the ImageNet output layer with our four classes
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))

    # Keep the learned image features; only train the classifier.
    for parameter in model.features.parameters():
        parameter.requires_grad = False
        
    return model
