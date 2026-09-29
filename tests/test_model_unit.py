import torch

from src.model import build_model


def test_build_model_freezes_features_and_classifier_shape():
    """Verify feature weights are frozen and the classifier has four outputs."""
    model = build_model(pretrained=False)
    assert all(not p.requires_grad for p in model.features.parameters())
    output_layer = model.classifier[-1]
    assert isinstance(output_layer, torch.nn.Linear)
    assert output_layer.out_features == 4
