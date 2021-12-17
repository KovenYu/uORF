import pytorch_lightning as pl
import torch
from util.options import parse_custom_options
from models.test_model import uorfTestGanModel

from data import MultiscenesDataModule

if __name__=='__main__':
    print('Parsing options...')
    opt = parse_custom_options()

    pl.seed_everything(opt.seed)

    print('Creating dataset...')
    dataset = MultiscenesDataModule(opt)  # create a dataset given opt.dataset_mode and other options

    print('Creating module...')
    module = uorfTestGanModel(opt) # .load_from_checkpoint(opt.checkpoint)
    
    ckpt = torch.load(opt.checkpoint)
    module.load_state_dict(ckpt["state_dict"], strict=False)

    trainer = pl.Trainer(
        gpus=opt.gpus,
        max_epochs=1)

    print('Start testing...')
    trainer.test(module, dataset)