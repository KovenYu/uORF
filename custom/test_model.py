import lpips
from piq import psnr, ssim
import torch
import torch.nn.functional as F

from custom.model import uorfGanModel
from models.model import raw2outputs
from util.util import AverageMeter


class uorfTestGanModel(uorfGanModel):

    def __init__(self, opt):
        # Initialize train model
        super().__init__(opt)

        self.LPIPS_loss = lpips.LPIPS()

        self.loss_recon_avg = AverageMeter()
        self.loss_lpips_avg = AverageMeter()
        self.loss_psnr_avg = AverageMeter()
        self.loss_ssim_avg = AverageMeter()


    def forward(self, input):
        imgs = input['img_data']
        gt_masks = input['masks']
        mask_idx = input['mask_idx']
        fg_idx = input['fg_idx']
        obj_idxs = input['obj_idxs']

        cam2world = input['cam2world']
        if self.opt.fixed_locality:
            nss2cam0 = cam2world[0:1].inverse()
        else:
            cam2world_azi = input['azi_rot']
            nss2cam0 = cam2world_azi[0:1].inverse()

        # Encoding images
        feature_map = self.netEncoder(F.interpolate(imgs[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False))  # BxCxHxW
        feat = feature_map.flatten(start_dim=2).permute([0, 2, 1])  # BxNxC

        # Slot Attention
        z_slots, attn = self.netSlotAttention(feat)  # 1xKxC, 1xKxN
        z_slots, attn = z_slots.squeeze(0), attn.squeeze(0)  # KxC, KxN
        K = attn.shape[0]

        N = cam2world.shape[0]

        W, H, D = self.projection.frustum_size
        scale = H // self.opt.render_size
        frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world, partitioned=True)
        # 4x(NxDx(H/2)x(W/2))x3, 4x(Nx(H/2)x(W/2))xD, 4x(Nx(H/2)x(W/2))x3
        
        x_recon = torch.zeros([N, 3, H, W]).type_as(imgs)
        rendered = torch.zeros([N, 3, H, W]).type_as(imgs)
        masked_raws = torch.zeros([K, N, D, H, W, 4]).type_as(imgs)
        unmasked_raws = torch.zeros([K, N, D, H, W, 4]).type_as(imgs)
        
        for (j, (frus_nss_coor_, z_vals_, ray_dir_)) in enumerate(zip(frus_nss_coor, z_vals, ray_dir)):
            h, w = divmod(j, scale)
            H_, W_ = H // scale, W // scale
            sampling_coor_fg_ = frus_nss_coor_[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
            sampling_coor_bg_ = frus_nss_coor_  # Px3

            raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(
                sampling_coor_bg_[None, ...], sampling_coor_fg_[None, ...], z_slots[None, ...], nss2cam0[None, ...])  
            # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1

            # batch size must be 1!
            raws_ = raws_.view([N, D, H_, W_, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
            masked_raws_ = masked_raws_.view([K, N, D, H_, W_, 4])
            unmasked_raws_ = unmasked_raws_.view([K, N, D, H_, W_, 4])
            masked_raws[..., h::scale, w::scale, :] = masked_raws_
            unmasked_raws[..., h::scale, w::scale, :] = unmasked_raws_
            rgb_map_, depth_map_, _ = raw2outputs(raws_, z_vals_, ray_dir_)
            # (NxHxW)x3, (NxHxW)
            rendered_ = rgb_map_.view(N, H_, W_, 3).permute([0, 3, 1, 2])  # Nx3xHxW
            rendered[..., h::scale, w::scale] = rendered_
            x_recon_ = rendered_ * 2 - 1
            x_recon[..., h::scale, w::scale] = x_recon_

        # x_recon_novel, x_novel = x_recon[1:], imgs[1:]
        return x_recon, imgs, (masked_raws.detach(), unmasked_raws.detach())


    def test_step(self, batch, batch_idx):
        x_recon, imgs, (masked_raws, unmasked_raws) = self(batch)  # B×S×C×H×W

        x_recon_novel, x_novel = x_recon[1:], imgs[1:]
        loss_recon = self.L2_loss(x_recon_novel, x_novel)
        loss_lpips = self.LPIPS_loss(x_recon_novel, x_novel).mean()
        loss_psnr = psnr(x_recon_novel/2+0.5, x_novel/2+0.5, data_range=1.)
        loss_ssim = ssim(x_recon_novel/2+0.5, x_novel/2+0.5, data_range=1.)

        self.log('test_losses', {
            'loss_recon': loss_recon.cpu().item(),
            'loss_lpips': loss_lpips.cpu().item(),
            'loss_psnr': loss_psnr.cpu().item(),
            'loss_ssim': loss_ssim.cpu().item(),
            }, prog_bar=True)

        print('loss_lpips', loss_lpips.cpu().item())
        print('loss_psnr', loss_psnr.cpu().item())
        print('loss_ssim', loss_ssim.cpu().item())

        self.loss_recon_avg.update(loss_recon)
        self.loss_lpips_avg.update(loss_lpips)
        self.loss_psnr_avg.update(loss_psnr)
        self.loss_ssim_avg.update(loss_ssim)


    def on_test_end(self) -> None:
        print("--- Results ---")
        print("Reconstruction", self.loss_recon_avg.avg)
        print("LPIPS", self.loss_lpips_avg.avg)
        print("PSNR", self.loss_psnr_avg.avg)
        print("SSIM", self.loss_ssim_avg.avg)