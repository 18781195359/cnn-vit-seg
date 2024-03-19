import torch
import torch.nn as nn
from pvtv2 import pvt_v2_b2
from pvtv2 import pvt_v2_b5
import json
from decoder import pixel_decoder
from FastFusion import vit_cnn_fusion
def create_fastvit():
    model = pvt_v2_b2()
    checkpoint = torch.load('weight/pvt_v2_b2.pth')
    model_state_dict = model.state_dict()
    pretrained_weights = {k: v for k, v in checkpoint.items() if k in model_state_dict}
    model_state_dict.update(pretrained_weights)
    model.load_state_dict(model_state_dict)

    return model

def create_decoder():
    with open("configs/twin.json", 'r') as fp:
        cfg_model = json.load(fp)
    backbone = cfg_model.pop("backbone")
    cfg_model['n_layers'] = 2
    decoder = pixel_decoder(**cfg_model)
    return decoder

def create_model():
    encoder_rgb = create_fastvit()
    encoder_tir = create_fastvit()
    decoder = create_decoder()

    with open("configs/twin.json", 'r') as fp:
        cfg_model = json.load(fp)
    backbone = cfg_model.pop("backbone")
    cfg_model["image_size"] = list(cfg_model["image_size"].split(" "))
    cfg_model["image_size"][0] = int(cfg_model["image_size"][0])
    cfg_model["image_size"][1] = int(cfg_model["image_size"][1])

    img_size = cfg_model['image_size']
    d_model = cfg_model['d_model']
    model = vit_cnn_fusion(img_size, d_model, encoder_rgb, encoder_tir, decoder)
    return model

