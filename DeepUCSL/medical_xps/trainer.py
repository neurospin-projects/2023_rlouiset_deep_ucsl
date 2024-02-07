import shutil
from math import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.dataset import *
from pl_models_getter import *
from pl_callbacks_getter import *


class Trainer:
    def __init__(self, args, cfg_file):
        self.manager = build_data_manager(args)
        self.saving_folder = None
        self.cfg_file = cfg_file
        self.args = args
        self.net = None

    def run(self):
        # create saving folder
        self.saving_folder = create_checkpoint_dir(self.args['checkpoint_dir'], self.args['xp_name'])
        shutil.copy2(self.cfg_file, self.saving_folder)

        folds = range(self.manager.number_of_folds)
        for fold in folds:
            # save metrics with TensorBoard Logger
            tb_logger = TensorBoardLogger(self.saving_folder + str(fold) + '/', name="tensorboard", )

            # instantiate network and training loss and data manager
            self.net = get_pl_model(self.args["model_type"],
                                   self.args["n_classes"], self.args["n_clusters"],
                                   self.args["loss"], self.args["loss_params"],
                                   self.args["lr"], fold)
            self.net.set_data_manager(self.manager, fold_index=fold)

            # load pretrained weights if necessary
            if self.args["pretrained_path"] is not None:
                snapshot_model = torch.load(self.args["pretrained_path"]).state_dict()
                self.net.model.load_state_dict(snapshot_model, strict=True)

            # instantiate callbacks
            callbacks = get_callbacks_getter(self.args["model_type"], self.saving_folder, fold)

            # instantiate Trainer and fit it
            trainer = pl.Trainer(min_epochs=self.args["min_epochs"], max_epochs=self.args["max_epochs"],
                                 gpus=self.args["gpus"], logger=tb_logger,
                                 reload_dataloaders_every_n_epochs=self.args["reload_dataloader"],
                                 callbacks=callbacks, precision=32)
            trainer.fit(self.net)

            # run test set
            test_result = trainer.test(self.net)

        return test_result

def create_checkpoint_dir(checkpoint_dir, xp_name):
    done = False
    attempt = 0
    while not done:
        if os.path.isdir(checkpoint_dir + xp_name + '_' + str(attempt)):
            attempt += 1
        else:
            saving_folder_path = checkpoint_dir + xp_name + '_' + str(attempt) + '/'
            print("SAVING FOLDER CREATED : ", saving_folder_path)
            os.mkdir(saving_folder_path)
            return saving_folder_path