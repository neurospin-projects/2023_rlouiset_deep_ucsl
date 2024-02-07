import torch
import pytorch_lightning as pl
class StackInitializer(pl.callbacks.Callback):

    def on_fit_start(self, trainer, pl_module):
        train_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, train=True).train

        # train representations
        latent_representations = [pl_module.forward(batch.cuda())[1].detach() for _, batch, _, _ in train_loader]
        pl_module.stack = torch.cat(latent_representations, dim=0)[-pl_module.stack_size:]

def norm(x):
    return torch.nn.functional.normalize(x, dim=1, p=2)

