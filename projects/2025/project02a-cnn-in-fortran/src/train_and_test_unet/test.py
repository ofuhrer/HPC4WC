import numpy as np

from models_custom import CustomUNet
from dataloader import get_dataloaders

import os
import torch

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_len=5
    out_len=1
    depth = 5
    with torch.no_grad():

        test_loader, _ = get_dataloaders(1,
                                    in_len=in_len,
                                    out_len=out_len,
                                    )

        test_input, test_target = next(iter(test_loader))
        test_input, test_target = test_input.to(device), test_target.to(device).float()
            
        # model_name = f"unet_d{depth}_out1_gpu_L1"
        model = CustomUNet(in_channels=in_len, 
             num_classes=out_len, 
             depth=depth, merge_mode='concat').to(device)

        # model.load_state_dict(torch.load(f'./checkpoint/{model_name}.pt', weights_only=True))
        model.eval()
        test_output = model(test_input)