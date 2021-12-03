from itertools import chain
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn, optim
from torchvision.transforms.transforms import Normalize
from custom.visualization import make_average_gradient_plot

from models.model import Decoder, Discriminator, Encoder, SlotAttention, d_logistic_loss, d_r1_loss, g_nonsaturating_loss, get_perceptual_net, raw2outputs, toggle_grad
from models.networks import get_scheduler, init_weights
from models.projection import Projection

from pytorch_lightning.loggers import TensorBoardLogger

from util.util import tensor2im

class uorfGanModel(pl.LightningModule):

    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters() # Save hyperparameters
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

        # Use part of image for finer training
        else:
            W, H, D = self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp  
            frus_nss_coor, z_vals, ray_dir = self.projection_fine.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3

            # Bring back batch dimension
            frus_nss_coor = frus_nss_coor.view([B, S, D, H, W, 3])
            z_vals, ray_dir = z_vals.view([B, S, H, W, D]), ray_dir.view([B, S, H, W, 3])

            # Cut out part of image
            start_range = self.opt.frustum_size_fine - self.opt.render_size
            H_idx = torch.randint(low=0, high=start_range, size=(1,), device=self.device)
            W_idx = torch.randint(low=0, high=start_range, size=(1,), device=self.device)

            rs = self.opt.render_size
            frus_nss_coor_= frus_nss_coor[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            z_vals_  = z_vals[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            ray_dir_ = ray_dir[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            imgs = imgs[..., H_idx:H_idx + rs, W_idx:W_idx + rs]

            frus_nss_coor = frus_nss_coor_.flatten(1, 4)
            z_vals, ray_dir = z_vals_.flatten(1, 3), ray_dir_.flatten(1, 3)

        # Repeat sampling coordinates for each object (K-1 objects, one background), P = S*D*H*W
        sampling_coor_fg = frus_nss_coor[:, None, ...].expand(-1, K - 1, -1, -1)  # B×(K-1)xPx3
        sampling_coor_bg = frus_nss_coor  # B×Px3

        # Run decoder
        raws, masked_raws, unmasked_raws, maks = self.netDecoder(
            sampling_coor_bg, sampling_coor_fg, z_slots, nss2cam0)  
        # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4,  (Kx(NxDxHxW)x1 <- masks, not needed)

        # Reshape for further processing
        W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp
        imgs = imgs.view(B, S, C, H, W)
        raws = raws.view([B, N, D, H, W, 4]).permute([0, 1, 3, 4, 2, 5]).flatten(start_dim=1, end_dim=3)  # B×(NxHxW)xDx4
        masked_raws = masked_raws.view([B, K, N, D, H, W, 4])
        unmasked_raws = unmasked_raws.view([B, K, N, D, H, W, 4])
        
        rgb_maps = []
        for i in range(B):
            rgb_map, _, _ = raw2outputs(raws[i], z_vals[i], ray_dir[i])
            rgb_maps.append(rgb_map)
            # (NxHxW)x3, (NxHxW)
        rgb_map_out = torch.stack(rgb_maps)

        img_rendered = rgb_map_out.view(B, N, H, W, 3).permute([0, 1, 4, 2, 3])  # B×Nx3xHxW
        return imgs, img_rendered, \
            (masked_raws.detach(), unmasked_raws.detach(), z_vals, ray_dir, attn.detach())


    def on_epoch_start(self) -> None:
        self.opt.stage = 'coarse' if self.current_epoch < self.opt.coarse_epoch else 'fine'
        self.netDecoder.locality = self.current_epoch < self.opt.no_locality_epoch
        self.weight_percept = self.opt.weight_percept if self.current_epoch >= self.opt.percept_in else 0
        self.weight_gan = self.opt.weight_gan if self.current_epoch >= self.opt.gan_train_epoch + self.opt.gan_in else 0

        if self.trainer.is_global_zero:
            self.log('training_schedule', {
                'weight_gan': self.weight_gan, 
                'weight_percept': self.weight_percept, 
                'only_uorf_training': 0 if self.current_epoch < self.opt.gan_train_epoch else 1,
                'gan_training': 1 if self.current_epoch < self.opt.gan_train_epoch + self.opt.gan_in else 0,
                'coarse_training': 1 if self.current_epoch < self.opt.coarse_epoch else 0,
                'decoder_locality': 1 if self.current_epoch < self.opt.no_locality_epoch else 0,
                }, prog_bar=False, rank_zero_only=True)


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
            # Train generator (uORF)
            if batch_idx % 2 == 0:
                if self.current_epoch < self.opt.gan_train_epoch + self.opt.gan_in:
                    return
                
                self.optimize_uorf(batch)

                if self.opt.custom_lr and self.opt.stage == 'coarse':
                    uorf_scheduler, _ = self.lr_schedulers()
                    uorf_scheduler.step()
            # Train discriminator
            else:
                self.optimize_discriminator(batch, batch_idx)

                _, disc_scheduler = self.lr_schedulers()
                disc_scheduler.step()


    def optimize_uorf(self, batch):
        # Forward batch
        imgs, imgs_rendered, raw_data = self(batch)
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

        if self.trainer.is_global_zero:
            self.log('losses', {
                'loss': loss.cpu().item(),
                'loss_gan': loss_gan.cpu().item(),
                'loss_recon': loss_recon.cpu().item(),
                'loss_percept': loss_perc.cpu().item(),
                }, prog_bar=True, rank_zero_only=True)

            if (self.global_step) % self.opt.display_freq == 0:
                self.log_visualizations(imgs, imgs_reconstructed, raw_data=raw_data)


    def optimize_discriminator(self, batch, batch_idx):
        # Forward batch
        imgs, imgs_rendered, raw_data = self(batch)
        imgs_reconstructed = imgs_rendered * 2 - 1

        # Combine batches and number of imgs in scene 
        B, S, C, H, W = imgs.shape
        imgs = imgs.view(B*S, C, H, W)
        imgs_rendered = imgs_rendered.view(B*S, C, H, W)
        imgs_reconstructed = imgs_reconstructed.view(B*S, C, H, W)

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

        if self.trainer.is_global_zero:
            self.log('losses_disc', {
                'loss_real': d_loss_real.cpu().item(),
                'loss_fake': d_loss_fake.cpu().item(),
                }, prog_bar=False, rank_zero_only=True)

            if (self.global_step) % self.opt.display_freq == 0:
                self.log_visualizations(imgs, imgs_reconstructed, raw_data=raw_data)


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
        

    def log_visualizations(self, imgs: torch.Tensor, imgs_reconstructed: torch.Tensor, raw_data) -> None:
        # only tensorboard supported
        if self.trainer.is_global_zero and isinstance(self.logger, TensorBoardLogger):
            tensorboard = self.logger.experiment

            # Average gradients
            avg_grad_plot = make_average_gradient_plot(chain(
                self.netEncoder.named_parameters(), 
                self.netSlotAttention.named_parameters(), 
                self.netDecoder.named_parameters())
                )

            tensorboard.add_figure(
                'gradients',
                avg_grad_plot
            )

            with torch.no_grad():
                # Compute visuals from batched raw data
                b_masked_raws, b_unmasked_raws, b_z_vals, b_ray_dir, b_attn = raw_data
                B, K, N, D, H, W, _ = b_masked_raws.shape
                
                # iterate slots, only display first batch
                for k in range(K):
                    # Render images from masked raws
                    raws = b_masked_raws[0][k]
                    raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                    z_vals, ray_dir = b_z_vals[0], b_ray_dir[0]
                    rgb_map, depth_map, _ = raw2outputs(raws, z_vals, ray_dir)
                    rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                    imgs_recon = rendered * 2 - 1

                    for i in range(N):
                        tensorboard.add_image(f"masked/{k}_{i}", tensor2im(imgs_recon[i]).transpose(2, 0, 1))

                    # Render images from unmasked raws
                    raws = b_unmasked_raws[0][k]
                    raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                    z_vals, ray_dir = b_z_vals[0], b_ray_dir[0]
                    rgb_map, depth_map, _ = raw2outputs(raws, z_vals, ray_dir)
                    rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                    imgs_recon = rendered * 2 - 1

                    for i in range(N):
                        tensorboard.add_image(f"unmasked/{k}_{i}", tensor2im(imgs_recon[i]).transpose(2, 0, 1))

                        # Render reconstructed images (whole scene)
                        tensorboard.add_image(f"recon/{k}_{i}", tensor2im(imgs_recon[i]).transpose(2, 0, 1))

                    # Render attention
                    b_attn = b_attn.view(B, K, 1, H, W)
                    tensorboard.add_image(f"attn/{k}", tensor2im(b_attn[0][k]*2 - 1 ).transpose(2, 0, 1))

                # iterate scenes
                for s in range(N):
                    # Images from forward pass
                    tensorboard.add_image(f"out_imgs/{s}", tensor2im(imgs[s]).transpose(2, 0, 1))
                    tensorboard.add_image(f"out_imgs_recon/{s}", tensor2im(imgs_reconstructed[s]).transpose(2, 0, 1))
