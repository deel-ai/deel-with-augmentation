import sys
import os
import json

sys.path.append("../")
from models import gan_networks
from options.train_options import TrainOptions
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import argparse


def get_z_random(batch_size=1, nz=8, random_type="gauss"):
    if random_type == "uni":
        z = torch.rand(batch_size, nz) * 2.0 - 1.0
    elif random_type == "gauss":
        z = torch.randn(batch_size, nz)
    return z.detach()


def load_model(modelpath, model_in_file, cpu, gpuid):
    train_json_path = modelpath + "/train_config.json"
    with open(train_json_path, "r") as jsonf:
        train_json = json.load(jsonf)
    opt = TrainOptions().parse_json(train_json, set_device=False)
    if opt.model_multimodal:
        opt.model_input_nc += opt.train_mm_nz
    opt.jg_dir = "../"

    if not cpu:
        device = torch.device("cuda:" + str(gpuid))
    else:
        device = torch.device("cpu")

    model = gan_networks.define_G(**vars(opt))
    model.eval()
    model.load_state_dict(
        torch.load(modelpath + "/" + model_in_file, map_location=device)
    )

    model = model.to(device)
    return model, opt, device

def main(model_in_file, img_in, img_out, cpu, gpuid):
    # loading model
    modelpath = model_in_file.replace(os.path.basename(model_in_file), "")
    #print("modelpath=", modelpath)

    model, opt, device = load_model(
        modelpath, os.path.basename(model_in_file), cpu, gpuid
    )

    # reading image
    img = cv2.imread(img_in)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # preprocessing
    tranlist = [
        transforms.ToTensor(),
        transforms.Resize((256,256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    tran = transforms.Compose(tranlist)
    img_tensor = tran(img)
    if not cpu:
        img_tensor = img_tensor.to(device)

    if opt.model_multimodal:
        z_random = get_z_random(batch_size=1, nz=opt.train_mm_nz)
        z_random = z_random.to(device)
        # print('z_random shape=', self.z_random.shape)
        z_real = z_random.view(z_random.size(0), z_random.size(1), 1, 1).expand(
            z_random.size(0),
            z_random.size(1),
            img_tensor.size(1),
            img_tensor.size(2),
        )
        img_tensor = torch.cat([img_tensor.unsqueeze(0), z_real], 1)
    else:
        img_tensor = img_tensor.unsqueeze(0)

    # run through model
    out_tensor = model(img_tensor)[0].detach()

    # post-processing
    out_img = out_tensor.data.cpu().float().numpy()
    out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    # print(out_img)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_out, out_img)
    print("Successfully generated image ", img_out)
