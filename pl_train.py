import pytorch_lightning as pl

from data import create_dataset
from options.train_options import TrainOptions

from custom.model import uorfGanModel
from custom.data import MultiscenesDataModule



if __name__=='__main__':
    print('Parsing options...')
    opt = TrainOptions().parse()   # get training options

    pl.seed_everything(opt.seed)

    print('Creating dataset...')
    dataset = MultiscenesDataModule(opt)  # create a dataset given opt.dataset_mode and other options

    print('Creating module...')
    module = uorfGanModel(opt)
    trainer = pl.Trainer(
        gpus=2,
        strategy="ddp_spawn", # ddp_spawn
        max_epochs=opt.niter + opt.niter_decay + 1)

    print('Start training...')
    trainer.fit(module, dataset)