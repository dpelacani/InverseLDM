from invldm.utils.setup import setup_train
from invldm.runners.trainer import Trainer

if __name__ == "__main__":
    # Gather arguments from CL and from YAML config file,
    # adjust and check for arguemnts
    # create necessary folders for experiment
    args = setup_train()

    # With given args, set up the training model with
    # datasets, dataloaders, model and optimiser instan
    # ciations, and data conditioning
    trainer = Trainer(args)

    # Train autoencoder (first stage model) and diffusion
    # in sequence
    trainer.train()
