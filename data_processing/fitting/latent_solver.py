import logging
from pathlib import Path
import torch
import torch.utils.data as data

from latent_mlp import LatentMLP
from latent_controller import LatentController

from torch_render import TorchRender


class LatentModelSolver(object):
    '''
    LatentModelSolver is a model solver used to train and optimize lighting parameters with the Adam optimizer.

    Methods: 
        __init__:  Initializes the LatentModelSolver with a model, data loader, and various parameters.
        train:     Performs fitting, calculates losses, and saves final texture results.
    '''
    def __init__(
        self,
        model: LatentMLP,
        data_loader: data.DataLoader,
        **kwargs
    ) -> None:
        self.model = model
        self.data_loader = data_loader

        self.lr = kwargs.pop("lr", 0.0005)
        self.num_iters = kwargs.pop("num_iters", 1000000)
        self.device = kwargs.pop("device", "cuda:0")

        self.step = 0   # current training step
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
        )

    def train(self, latent_controller: LatentController, fitting_dir: str, output_dir: str):

        torch_render = TorchRender(latent_controller.setup)
        lp = self.data_loader.dataset.get_light_pattern().to(self.device)

        for train_data in self.data_loader:

            self.model.train()

            self.step += 1
            self.model.step += 1
            assert(self.step == self.model.step)
            if self.step > self.num_iters:
                break

            batch_uvs = train_data['uv'].to(self.device)
            batch_lumi = train_data['lumitexel'].to(self.device)
            batch_m = train_data['measurements'].to(self.device)
            loss, meta = self.model.loss(batch_uvs, batch_lumi, lp, batch_m)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.step % 100 == 0:
                log_info = "[%d] Loss: %.6f" % (self.step, loss.item())
                for key in meta:
                    if "loss" in key:
                        log_info = log_info + " %s: %.6f" % (key, meta[key])
                print(log_info)
        
        if output_dir is not None :
            output_dir = Path(output_dir)
            # export texture map
            self.model.eval()
            textures = self.model.export_texture(str(output_dir), str(output_dir))


