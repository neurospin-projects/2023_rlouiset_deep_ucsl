from numpy.random import seed
from trainer import Trainer
import argparse
import logging
import yaml

# set seed for reproducibility
seed(2)

if __name__ == "__main__":
    # parse config file argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, dest="cfg_file")
    args = parser.parse_args()

    # .yaml file safe load
    cfg_file = args.cfg_file
    with open(cfg_file, 'r') as input_cfg:
        try:
            XP_CONFIGS = yaml.safe_load(input_cfg)
        except yaml.YAMLError as error:
            logging.error(error)

    # define logger
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    logging.info("Let us launch the training of the experiment ")

    # load the args of the config file
    args = {}
    for CONFIG_NAME in XP_CONFIGS.keys():
        try:
            args.update(XP_CONFIGS[CONFIG_NAME])
        except:
            print(XP_CONFIGS[CONFIG_NAME])

    # initialize and run the trainer
    trainer = Trainer(args, cfg_file)
    trainer.run()
