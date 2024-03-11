import os
os.environ["CUDA_VISIABLE_DEVICES"] = "7"

import torch
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
print(BASE_DIR)
sys.path.append(BASE_DIR)

from Classifier3D.models import SWINUNETR_CLS

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



def deploy(weights, output_pt):
    device = torch.device("cuda")
    model = SWINUNETR_CLS(
            in_channels=1,
            num_classes=11,
            feature_size=12,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.3,
            use_checkpoint=False,
            mode="deploy"
            )

    checkpoint = torch.load(weights, map_location=device)
    state = model.state_dict()
    state.update(checkpoint['state_dict'])
    model.load_state_dict(state, strict=True)  # , strict=False
    print("model successfully loaded")
    set_requires_grad(model)
    model.eval().to(device)
    inputs = torch.rand((1, 1, 144, 144, 144)).float().to(device)
    
    with torch.no_grad():
        traced_model = torch.jit.trace(model, (inputs))
        print(traced_model.graph)
        traced_model.save(output_pt)


if __name__ == "__main__":
    deploy(weights="./epoch_82.pth", output_pt="epoch_82.pt")

