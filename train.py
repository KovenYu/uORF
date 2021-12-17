import pytorch_lightning as pl
from util.options import parse_custom_options

from pytorch_lightning.callbacks import LearningRateMonitor

from models.train_model import uorfGanModel
from data import MultiscenesDataModule

if __name__=='__main__':
    print('Parsing options...')
    opt = parse_custom_options()

    pl.seed_everything(opt.seed)

    print('Creating dataset...')
    dataset = MultiscenesDataModule(opt)  # create a dataset given opt.dataset_mode and other options

    print('Creating module...')
    module = uorfGanModel(opt)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        gpus=opt.gpus,
        strategy="ddp", # ddp_spawn
        max_epochs=opt.niter + opt.niter_decay + 1,
        callbacks=[lr_monitor])

    print('Start training...')
    trainer.fit(module, dataset)