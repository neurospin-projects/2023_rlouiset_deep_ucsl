import numpy as np
from pytorch_lightning.core.lightning import LightningModule

from DeepUCSL.neuropsy_xps.architectures.vae import BrainVAE


class LitVAE(LightningModule):
    def __init__(self, model_type, name, loss, loss_params, lr,
                 step_size_scheduler, gamma_scheduler, fold):
        super().__init__()
        self.model_type = model_type
        self.loss = loss
        self.loss_params = loss_params
        self.lr = lr
        self.step_size_scheduler = step_size_scheduler
        self.gamma_scheduler = gamma_scheduler
        self.model = BrainVAE()
        self.data_manager = None
        self.with_validation = False
        self.fold_index = fold

    def forward(self, x):
        return self.model.forward(x)

    def set_data_manager(self, data_manager, with_validation=False, fold_index=0):
        self.data_manager = data_manager
        self.with_validation = with_validation
        self.fold_index = fold_index

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer], []

    def compute_elbo_loss(self, x_reconstructed, x, z_mean, z_log_var):
        # elbo loss
        reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
        kl_div_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return reconstruction_loss, kl_div_loss

    def training_step(self, train_batch, batch_idx):
        x, _, y, _ = train_batch
        reconstructed_x, z_mean, z_log_var = self.forward(x)
        reconstruction_loss, kl_div_loss = self.compute_elbo_loss(reconstructed_x, x, z_mean, z_log_var)
        loss = reconstruction_loss + self.loss_params["beta"] * kl_div_loss

        # logs training loss in a dictionary
        self.log('train_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_div_loss": kl_div_loss,
            "metadata": y
        }
        return batch_dictionary

    def validation_step(self, val_batch, batch_idx):
        x, _, y, _ = val_batch
        reconstructed_x, z_mean, z_log_var = self.forward(x)
        reconstruction_loss, kl_div_loss = self.compute_elbo_loss(reconstructed_x, x, z_mean, z_log_var)
        loss = reconstruction_loss + kl_div_loss

        # logs training loss in a dictionary
        self.log('val_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_div_loss": kl_div_loss,
            "metadata": y
        }
        return batch_dictionary

    def test_step(self, test_batch, batch_idx):
        x, _, y, _ = test_batch
        reconstructed_x, z_mean, z_log_var = self.forward(x)
        reconstruction_loss, kl_div_loss = self.compute_elbo_loss(reconstructed_x, x, z_mean, z_log_var)
        loss = reconstruction_loss + kl_div_loss

        # logs training loss in a dictionary
        self.log('test_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_div_loss": kl_div_loss,
            "metadata": y
        }
        return batch_dictionary

    def validation_epoch_end(self, outputs):
        reconstruction_loss = torch.stack([x['reconstruction_loss'] for x in outputs]).view(-1).detach()
        kl_div_loss = torch.stack([x['kl_div_loss'] for x in outputs]).view(-1).detach()

        self.logger.experiment.add_scalar("Reconstruction Loss/Val",
                                          reconstruction_loss.mean().item(),
                                          self.current_epoch)
        self.logger.experiment.add_scalar("KL Div Loss/Val",
                                          kl_div_loss.mean().item(),
                                          self.current_epoch)

    def training_epoch_end(self, outputs):
        reconstruction_loss = torch.stack([x['reconstruction_loss'] for x in outputs]).view(-1).detach()
        kl_div_loss = torch.stack([x['kl_div_loss'] for x in outputs]).view(-1).detach()

        self.logger.experiment.add_scalar("Reconstruction Loss/Train",
                                          reconstruction_loss.mean().item(),
                                          self.current_epoch)
        self.logger.experiment.add_scalar("KL Div Loss/Train",
                                          kl_div_loss.mean().item(),
                                          self.current_epoch)

    def test_epoch_end(self, outputs):
        reconstruction_loss = torch.stack([x['reconstruction_loss'] for x in outputs]).view(-1).detach()
        kl_div_loss = torch.stack([x['kl_div_loss'] for x in outputs]).view(-1).detach()

        self.logger.experiment.add_scalar("Reconstruction Loss/Test",
                                          reconstruction_loss.mean().item(),
                                          self.current_epoch)
        self.logger.experiment.add_scalar("KL Div Loss/Test",
                                          kl_div_loss.mean().item(),
                                          self.current_epoch)


    def train_dataloader(self):
        return self.data_manager.get_dataloader(train=True, fold_index=self.fold_index).train

    def val_dataloader(self):
        return self.data_manager.get_dataloader(validation=self.with_validation, fold_index=self.fold_index).validation

    def test_dataloader(self):
        return self.data_manager.get_dataloader(test=True, fold_index=self.fold_index).test
