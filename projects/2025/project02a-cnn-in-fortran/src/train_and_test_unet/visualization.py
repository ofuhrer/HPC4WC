import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from models import UNet
from dataloader import get_dataloaders

import os
import torch

def gen_animation(seq, save_path):

    print("generating and saving animation...", flush=True)
    fig = plt.figure()
    im = plt.imshow(seq[0,...], cmap="magma", animated=True)
    txt = plt.text(50, 50, f'frame = 0', animated=True)
    cbar = plt.colorbar(im)
    i = 0

    def updatefig(frame):

        if (frame<seq.shape[0]-1):
            frame += 1
        else:
            frame=0
        im.set_array(seq[frame,...])
        txt.set_text(f'frame = {frame}')
        
        return (im, txt)

    ani = animation.FuncAnimation(fig, updatefig,  blit=True, interval=500, frames=seq.shape[0])
    writer = animation.PillowWriter(fps=1,
                                bitrate=1000)
    ani.save(save_path, writer=writer)
    print("done.", flush=True)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_len=5
    out_len=1

    with torch.no_grad():

        test_loader, _ = get_dataloaders(1,
                                    in_len=in_len,
                                    out_len=out_len,
                                    )

        test_input, test_target = next(iter(test_loader))
        test_input, test_target = test_input.to(device), test_target.to(device).float()
        
        for depth in [3,5,7]:
            
            model_name = f"unet_d{depth}_out1_gpu_L1"
            model = UNet(in_channels=in_len, 
                 num_classes=out_len, 
                 depth=depth, merge_mode='concat').to(device)
    
            model.load_state_dict(torch.load(f'./checkpoint/{model_name}.pt', weights_only=True))
            model.eval()

            test_output = model(test_input)
    
            test_input_out = test_input.squeeze().cpu().numpy()
            test_target_out = test_target.squeeze().cpu().numpy()
            test_output_out = test_output.squeeze().cpu().numpy()
            test_input_out = np.flip(test_input_out, axis=1)
            test_target_out = np.flip(test_target_out, axis=0)
            test_output_out = np.flip(test_output_out, axis=0)
    
            if not os.path.exists("./results"):
                os.mkdir("results")
            
            gen_animation(np.concat([test_input_out, [test_output_out]], axis=0), save_path=f"./results/{model_name}_o.gif")
            gen_animation(np.concat([test_input_out, [test_target_out]], axis=0), save_path=f"./results/{model_name}_gt.gif")

