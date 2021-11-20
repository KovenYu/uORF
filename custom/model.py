from itertools import chain
import time
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn, optim
from torchvision.transforms.transforms import Normalize

from models.model import Decoder, Discriminator, Encoder, SlotAttention, d_logistic_loss, d_r1_loss, g_nonsaturating_loss, get_perceptual_net, raw2outputs, toggle_grad
from models.networks import get_scheduler, init_weights
from models.projection import Projection


class uorfGanModel(pl.LightningModule):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # pytorch lightning setting
        # We need manual optimization
        self.automatic_optimization = False

        # Net for perceptual loss
        self.perceptual_net = get_perceptual_net()
        self.vgg_norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Projections for coarse training
        # Frustum
        render_size = (opt.render_size, opt.render_size)
        frustum_size = [opt.frustum_size, opt.frustum_size, opt.n_samp]
        self.projection = Projection(
            device=self.device, nss_scale=opt.nss_scale, frustum_size=frustum_size, 
            near=opt.near_plane, far=opt.far_plane, render_size=render_size)

        # Projections for fine training on small section of original image
        frustum_size_fine = [opt.frustum_size_fine, opt.frustum_size_fine, opt.n_samp]
        self.projection_fine = Projection(
            device=self.device, nss_scale=opt.nss_scale, frustum_size=frustum_size_fine, 
            near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        
        # Slot attention noise dim and number of slots
        z_dim = opt.z_dim
        self.num_slots = opt.num_slots

        # Initialize encoder (U-Net)
        self.netEncoder = Encoder(3, z_dim=z_dim, bottom=opt.bottom)
        init_weights(self.netEncoder, init_type='normal', init_gain=0.02)

        # Initialize slot attention
        self.netSlotAttention = SlotAttention(
            num_slots=opt.num_slots, in_dim=z_dim, slot_dim=z_dim, iters=opt.attn_iter)
        init_weights(self.netSlotAttention, init_type='normal', init_gain=0.02)

        # Initialize nerf decoder network
        self.netDecoder = Decoder(
            n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=opt.z_dim, 
            n_layers=opt.n_layer, locality_ratio=opt.obj_scale/opt.nss_scale, 
            fixed_locality=opt.fixed_locality)
        init_weights(self.netDecoder, init_type='xavier', init_gain=0.02)

        # Initialize discriminator for adverserial loss
        self.netDisc = Discriminator(size=opt.supervision_size, ndf=opt.ndf)
        init_weights(self.netDisc, init_type='normal', init_gain=0.02)

        # Loss module for reconstruction and perceptual loss
        self.L2_loss = nn.MSELoss()


    def forward(self, input):
        imgs, cam2world, cam2world_azi = input  # B×S×C×H×W, B×S×4×4, B×S×3×3
        
        # S images per scene
        B, S, C, H, W = imgs.shape

        if self.opt.fixed_locality:
            nss2cam0 = cam2world[:, 0:1].inverse()
        else:
            nss2cam0 = cam2world_azi[:, 0:1].inverse()

        # Run encoder on first images, then flatten H×W to F
        first_imgs = imgs[:, 0, ...]  # B×C×H×W

        feature_map = self.netEncoder(F.interpolate(
            first_imgs, size=self.opt.input_size, mode='bilinear', align_corners=False))  # BxCxHxW
        feat = feature_map.flatten(start_dim=2).permute([0, 2, 1])  # BxFxC
        
        z_slots, attn = self.netSlotAttention(feat)  # B×K×C, B×K×F
        K = z_slots.shape[1]

        # combine batches and number of imgs per scene
        N = B*S
        cam2world = cam2world.view(N, 4, 4)  # N×4×4
        imgs = imgs.view(N, C, H, W)

        # Use full image for training, resize to supervision size
        if self.opt.stage == 'coarse':
            frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3

            imgs = F.interpolate(
                imgs, size=self.opt.supervision_size, mode='bilinear', align_corners=False)

            # Bring back batch dimension
            W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp
            frus_nss_coor = frus_nss_coor.view([B, S*D*H*W, 3])
            z_vals, ray_dir =  z_vals.view([B, S*H*W, D]), ray_dir.view([B, S*H*W, 3])
            imgs = imgs.view(B, S, C, H, W)

        # Use part of image for finer training
        else:
            W, H, D = self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp  
            frus_nss_coor, z_vals, ray_dir = self.projection_fine.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3

            # Bring back batch dimension
            frus_nss_coor = frus_nss_coor.view([B, S*D*H*W, 3])
            z_vals, ray_dir = z_vals.view([B, S*H*W, D]), ray_dir.view([B, S*H*W, 3])

            # Cut out part of image
            start_range = self.opt.frustum_size_fine - self.opt.render_size
            H_idx = torch.randint(low=0, high=start_range, size=(1,), device=self.device)
            W_idx = torch.randint(low=0, high=start_range, size=(1,), device=self.device)

            rs = self.opt.render_size
            frus_nss_coor_= frus_nss_coor[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            z_vals_  = z_vals[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            ray_dir_ = ray_dir[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            imgs = imgs[..., H_idx:H_idx + rs, W_idx:W_idx + rs]

            z_vals, ray_dir = z_vals_.flatten(1, 3), ray_dir_.flatten(1, 3)
            frus_nss_coor = frus_nss_coor_.flatten(1, 4)

        # Repeat sampling coordinates for each object (K-1 objects, one background), P = S*D*H*W
        sampling_coor_fg = frus_nss_coor[:, None, ...].expand(-1, K - 1, -1, -1)  # B×(K-1)xPx3
        sampling_coor_bg = frus_nss_coor  # B×Px3

        """
        print('Before decoder shapes:', 
            "sampling_coor_fg", sampling_coor_fg.shape, "\n", 
            "sampling_coor_bg", sampling_coor_bg.shape, "\n",
            "z_slots", z_slots.shape, "\n",
            "nss2cam0", nss2cam0.shape, "\n")
        """
        
        # Run decoder
        W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp
        raws, masked_raws, unmasked_raws, _ = self.netDecoder(
            sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0)  
        # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4,  (Kx(NxDxHxW)x1 <- masks, not needed)

        raws = raws.view([B, N, D, H, W, 4]).permute([0, 1, 3, 4, 2, 5]).flatten(start_dim=1, end_dim=3)  # B×(NxHxW)xDx4
        # masked_raws = masked_raws.view([B, K, N, D, H, W, 4])
        # unmasked_raws = unmasked_raws.view([B, K, N, D, H, W, 4])
        
        rgb_maps = []
        for i in range(B):
            rgb_map, _, _ = raw2outputs(raws[i], z_vals[i], ray_dir[i])
            rgb_maps.append(rgb_map)
            # (NxHxW)x3, (NxHxW)
        rgb_map_out = torch.stack(rgb_maps)

        img_rendered = rgb_map_out.view(B, N, H, W, 3).permute([0, 1, 4, 2, 3])  # B×Nx3xHxW
        return imgs, img_rendered


    def on_epoch_start(self) -> None:
        self.opt.stage = 'coarse' if self.current_epoch < self.opt.coarse_epoch else 'fine'
        self.netDecoder.locality = self.current_epoch < self.opt.no_locality_epoch
        self.weight_percept = self.opt.weight_percept if self.current_epoch >= self.opt.percept_in else 0
        self.weight_gan = self.opt.weight_gan if self.current_epoch >= self.opt.gan_train_epoch + self.opt.gan_in else 0


    def on_epoch_end(self) -> None:
        if self.opt.custom_lr:
            uorf_scheduler, _ = self.lr_schedulers()
            uorf_scheduler.step()


    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.opt.gan_train_epoch:
            self.optimize_uorf(batch)

            if self.opt.custom_lr and self.opt.stage == 'coarse':
                uorf_scheduler, _ = self.lr_schedulers()
                uorf_scheduler.step()
        
        else:
            if batch_idx % 2 == 0:
                if self.current_epoch < self.opt.gan_train_epoch + self.opt.gan_in:
                    return
                
                self.optimize_uorf(batch)

                if self.opt.custom_lr and self.opt.stage == 'coarse':
                    uorf_scheduler, _ = self.lr_schedulers()
                    uorf_scheduler.step()


    def optimize_uorf(self, batch):
        # Forward batch

        with torch.autograd.set_detect_anomaly(True):
            imgs, imgs_rendered = self(batch)
            imgs_reconstructed = imgs_rendered * 2 - 1

            # Combine batches and number of imgs in scene 
            B, S, C, H, W = imgs.shape
            imgs = imgs.view(B*S, C, H, W)
            imgs_rendered = imgs_rendered.view(B*S, C, H, W)
            imgs_reconstructed = imgs_reconstructed.view(B*S, C, H, W)

            # Adverserial loss
            d_fake = self.netDisc(imgs_reconstructed)
            loss_gan = self.weight_gan * g_nonsaturating_loss(d_fake)

            # Reconstruction loss
            loss_recon = self.L2_loss(imgs_reconstructed, imgs)

            # Perceptual loss
            x_norm, rendered_norm = self.vgg_norm((imgs + 1) / 2), self.vgg_norm(imgs_rendered)
            rendered_feat, x_feat = self.perceptual_net(rendered_norm), self.perceptual_net(x_norm)
            loss_perc = self.weight_percept * self.L2_loss(rendered_feat, x_feat)

            loss = loss_gan + loss_recon + loss_perc

        uorf_optimizer, _ = self.optimizers()
        uorf_optimizer.zero_grad()
        self.manual_backward(loss)
        uorf_optimizer.step()

        """
        self.log('losses', {
            'loss': loss.cpu().item(), 
            'loss_gan': loss_gan.cpu().item(), 
            'loss_recon': loss_recon.cpu().item(),
            'loss_percept': loss_perc.cpu().item(),
            }, prog_bar=True)
            """


    def optimize_discriminator(self, batch, batch_idx):
        # Forward batch
        imgs, imgs_rendered = self(batch)
        imgs_reconstructed = imgs_rendered * 2 - 1

        toggle_grad(self.netDisc, True)
        fake_pred = self.netDisc(imgs_reconstructed)
        real_pred = self.netDisc(imgs)

        d_loss_real, d_loss_fake = d_logistic_loss(real_pred, fake_pred)
        d_loss = d_loss_real + d_loss_fake

        _, disc_optimizer = self.optimizers()
        disc_optimizer.zero_grad()
        self.manual_backward(d_loss)
        disc_optimizer.step()

        if (batch_idx + 1) % 32 == 0:
            imgs.requires_grad = True
            real_pred = self.netDisc(imgs)
            r1_loss = d_r1_loss(real_pred, imgs)

            disc_optimizer.zero_grad()
            self.manual_backward(self.opt.weight_r1 * r1_loss)
            disc_optimizer.step()

        toggle_grad(self.netDisc, False)


    def configure_optimizers(self):
        # uORF = encoder -> slot attention -> decoder
        uorf_optimizer = optim.Adam(chain(
            self.netEncoder.parameters(), 
            self.netSlotAttention.parameters(), 
            self.netDecoder.parameters()
            ), lr=self.opt.lr)
        
        # discriminator for adverserial loss
        disc_optimizer = optim.Adam(self.netDisc.parameters(), lr=self.opt.d_lr, betas=(0., 0.9))

        # Set up schedulers
        uorf_scheduler = get_scheduler(uorf_optimizer, self.opt)
        disc_scheduler = get_scheduler(disc_optimizer, self.opt)

        return [uorf_optimizer, disc_optimizer], [uorf_scheduler, disc_scheduler]
        

