import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from data.multiscenes_dataset import MultiscenesDataset, collate_fn


class MultiscenesDataModule(pl.LightningDataModule):
    def __init__(self, opt):
        self.opt = opt
    
    def setup(self, stage = None):
        if stage in (None, 'fit'):
            print('Fetching train dataset ...')
            self.train_dataset = MultiscenesDataset(self.opt, is_train=True)

        if stage in (None, 'test'):
            print('Fetching test dataset...')
            self.test_dataset = MultiscenesDataset(self.opt, is_train=False)
    
    @staticmethod
    def collate_fn_batches(data):
        """collate_fn from data.multiscenes_dataset combines data from all batches to one large
        tensor, respectively."""

        B = len(data) # n_images_each_scene
        N = len(data[0])

        C, H, W = data[0][0]['img_data'].shape
        
        images = torch.empty((B, N, C, H, W))
        cam2world = torch.empty((B, N, 4, 4))
        azi_rot = torch.empty((B, N, 3, 3))

        for batch_idx in range(B):
            for img_idx in range(N):
                images[batch_idx][img_idx] = data[batch_idx][img_idx]['img_data'] # img_data: C×H×W
                cam2world[batch_idx][img_idx] = data[batch_idx][img_idx]['cam2world'] # cam2world: 4×4
                azi_rot[batch_idx][img_idx] = data[batch_idx][img_idx]['azi_rot']  # azi_rot: 3×3

        return images, cam2world, azi_rot
           

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.opt.batch_size,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.num_threads),
            collate_fn = self.collate_fn_batches,
            persistent_workers=self.opt.num_threads > 0
        )

    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.num_threads),
            collate_fn = self.collate_fn_batches,
            persistent_workers=self.opt.num_threads > 0
        )
