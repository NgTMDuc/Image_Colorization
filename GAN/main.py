from model.Gan import MainModel
from dataset.dataset import make_dataloaders
from utils.utils import lab_to_rgb, create_loss_meters, update_losses, log_results, visualize
import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import tqdm
import numpy as np
from torch import nn, optim
from model.Generator import build_res_unet
from model.pretrainedGenerator import pretrain_generator

def save_checkpoint(checkpoint, filename = "GAN_Train_On_ImageWar.pth"):
    print("Saving model")
    torch.save(checkpoint, filename)


def train_model(model, train_dl, val_dl, epochs, display_every=200):
    # getting a batch for visualizing the model output after fixed intrvals
    data = next(iter(val_dl)) 
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() 
        i = 0                                 
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()
             # updating the log losses
            update_losses(model, loss_meter_dict, count=data['L'].size(0))
            i += 1
            
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                # print out the losses
                log_results(loss_meter_dict)
                # display the model's outputs
                visualize(model, data, save=False)

if __name__ == "__main__":
    all_images = "" #List of path to images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(123)
    train_range = int(0.8 * len(all_images))
    rand_idxs = np.random.permutation(len(all_images))
    train_idxs = rand_idxs[:train_range]
    val_idxs = rand_idxs[train_range:] 
    train_paths = [all_images[x] for x in train_idxs]
    val_paths = [all_images[x] for x in val_idxs]

    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')
    
    # Get the pretrained-generator (ResNet-18)
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    # Pretrain generator on dataset
    pretrain_generator(net_G, train_dl, opt, criterion, 20)

    model = MainModel(net_G = net_G)
    train_model(model= model, train_dl= train_dl, val_dl= val_dl,epochs= 20)

    # Uncomment below line of codes to save checkpoint of model
    # checkpoint = {"state_dict": model.state_dict(), "optimizerG": model.opt_G.state_dict(),"optimizerD": model.opt_D.state_dict() }
    # save_checkpoint(checkpoint)

