from itertools import chain

import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import os
import time
from .projection import Projection
from .model import Encoder, Decoder, SlotAttention, raw2outputs
import lpips
from piq import ssim as compute_ssim
from piq import psnr as compute_psnr


class uorfManipModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--num_slots', metavar='K', type=int, default=8, help='Number of supported slots')
        parser.add_argument('--z_dim', type=int, default=64, help='Dimension of individual z latent per slot')
        parser.add_argument('--attn_iter', type=int, default=3, help='Number of refine iteration in slot attention')
        parser.add_argument('--nss_scale', type=float, default=7, help='Scale of the scene, related to camera matrix')
        parser.add_argument('--render_size', type=int, default=64, help='Shape of patch to render each forward process. Must be Frustum_size/(2^N) where N=0,1,..., Smaller values cost longer time but require less GPU memory.')
        parser.add_argument('--obj_scale', type=float, default=4.5, help='Scale for locality on foreground objects')
        parser.add_argument('--n_freq', type=int, default=5, help='how many increased freq?')
        parser.add_argument('--n_samp', type=int, default=64, help='num of samp per ray')
        parser.add_argument('--n_layer', type=int, default=3, help='num of layers bef/aft skip link in decoder')
        parser.add_argument('--bottom', action='store_true', help='one more encoder layer on bottom')
        parser.add_argument('--input_size', type=int, default=64)
        parser.add_argument('--frustum_size', type=int, default=128, help='Size of rendered images')
        parser.add_argument('--near_plane', type=float, default=6)
        parser.add_argument('--far_plane', type=float, default=20)
        parser.add_argument('--fixed_locality', action='store_true', help='enforce locality in world space instead of transformed view space')

        parser.set_defaults(batch_size=1, lr=3e-4, niter_decay=0,
                            dataset_mode='multiscenes', niter=1200, custom_lr=True, lr_policy='warmup')

        parser.set_defaults(exp_id='run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S')))

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.loss_names = ['psnr_moving', 'ssim_moving', 'lpips_moving', 'psnr_changing', 'ssim_changing', 'lpips_changing']
        n = opt.n_img_each_scene
        self.visual_names = ['input_view', 'our_recon', 'input_bg', 'placeholder'] + \
                            ['gt_moved{}'.format(i) for i in range(n)] + \
                            ['ours_moved{}'.format(i) for i in range(n)] + \
                            ['gt_changed{}'.format(i) for i in range(n)] + \
                            ['ours_changed{}'.format(i) for i in range(n)]
        self.model_names = ['Encoder', 'SlotAttention', 'Decoder']
        render_size = (opt.render_size, opt.render_size)
        frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
        self.projection = Projection(device=self.device, nss_scale=opt.nss_scale,
                                     frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        z_dim = opt.z_dim
        self.num_slots = opt.num_slots
        self.netEncoder = networks.init_net(Encoder(3, z_dim=z_dim, bottom=opt.bottom),
                                            gpu_ids=self.gpu_ids, init_type='normal')
        self.netSlotAttention = networks.init_net(
            SlotAttention(num_slots=opt.num_slots, in_dim=z_dim, slot_dim=z_dim, iters=opt.attn_iter), gpu_ids=self.gpu_ids, init_type='normal')
        self.netDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=opt.z_dim, n_layers=opt.n_layer, locality=False,
                                                    locality_ratio=opt.obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality), gpu_ids=self.gpu_ids, init_type='xavier')
        self.L2_loss = torch.nn.MSELoss()
        self.LPIPS_loss = lpips.LPIPS().to(self.device)

    def setup(self, opt):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_{}'.format(opt.load_iter) if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.x = input['img_data'].to(self.device)
        self.cam2world = input['cam2world'].to(self.device)
        if not self.opt.fixed_locality:
            self.cam2world_azi = input['azi_rot'].to(self.device)
        self.image_paths = input['paths']
        if 'masks' in input:
            self.gt_masks = input['masks']
            self.mask_idx = input['mask_idx']
            self.fg_idx = input['fg_idx']
        self.movement = input['movement'].to(self.device)  # Nx3
        self.gt_moved = input['img_data_moved'].to(self.device)  # Nx3xHxW
        self.gt_changed = input['img_data_changed'].to(self.device)  # Nx3xHxW
        self.bg_img = input['img_data_bg'].to(self.device)  # Nx3xHxW

    def forward(self, epoch=0):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        dev = self.x[0:1].device
        nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()

        # Encoding images
        feature_map = self.netEncoder(F.interpolate(self.x[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False))  # BxCxHxW
        feat = feature_map.flatten(start_dim=2).permute([0, 2, 1])  # BxNxC

        # Slot Attention
        z_slots, attn = self.netSlotAttention(feat)  # 1xKxC, 1xKxN
        z_slots, attn = z_slots.squeeze(0), attn.squeeze(0)  # KxC, KxN
        K = attn.shape[0]

        cam2world = self.cam2world
        N = cam2world.shape[0]

        W, H, D = self.projection.frustum_size
        scale = H // self.opt.render_size
        frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world, partitioned=True)
        # 4x(NxDx(H/2)x(W/2))x3, 4x(Nx(H/2)x(W/2))xD, 4x(Nx(H/2)x(W/2))x3

        """ reconstruct """
        x_recon, rendered, masked_raws, unmasked_raws = \
            torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, D, H, W, 4], device=dev), torch.zeros([K, N, D, H, W, 4], device=dev)
        for (j, (frus_nss_coor_, z_vals_, ray_dir_)) in enumerate(zip(frus_nss_coor, z_vals, ray_dir)):
            h, w = divmod(j, scale)
            H_, W_ = H // scale, W // scale
            sampling_coor_fg_ = frus_nss_coor_[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
            sampling_coor_bg_ = frus_nss_coor_  # Px3

            raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(sampling_coor_bg_, sampling_coor_fg_, z_slots, nss2cam0)  # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
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

        setattr(self, 'our_recon', x_recon[0])

        """ moving object """
        mask_maps = []
        for k in range(self.num_slots):
            raws = masked_raws[k]  # NxDxHxWx4
            _, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
            raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
            rgb_map, depth_map, _, mask_map = raw2outputs(raws, z_vals, ray_dir, render_mask=True)
            mask_maps.append(mask_map.view(N, H, W))

        mask_maps = torch.stack(mask_maps)  # KxNxHxW
        mask_idx = mask_maps.cpu().argmax(dim=0)  # NxHxW, i.e., 1xHxW
        predefined_colors = torch.tensor([[0.2510, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.7373],
                                          [0.2510, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.7373],
                                          [0.2510, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 1.0000, 0.7373]])
        mask_visuals = predefined_colors[:, mask_idx]  # 3xNxHxW
        mask_visuals = mask_visuals * 2 - 1

        setattr(self, 'input_view', self.x[0])

        ious = []
        fg_idx = self.fg_idx[0]
        for k in range(1, self.num_slots):
            mask_this_slot = mask_idx[0:1] == k
            mask_this_slot_visual = mask_this_slot.type(torch.float32)  # 1xHxW, {0,1}
            mask_this_slot_visaul = mask_this_slot_visual * 2 - 1
            setattr(self, 'individual_mask{}'.format(k - 1), mask_this_slot_visaul)
            setattr(self, 'fg_idx', fg_idx.type(torch.float32) * 2 - 1)
            iou = (fg_idx & mask_this_slot).type(torch.float).sum() / (fg_idx | mask_this_slot).type(torch.float).sum()
            print('{}-th slot IoU: {}'.format(k, iou))
            ious.append(iou)
        move_slot_idx = torch.tensor(ious).argmax()
        print('to move: {}'.format(move_slot_idx))

        x_recon, rendered, masked_raws, unmasked_raws = \
            torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, D, H, W, 4], device=dev), torch.zeros([K, N, D, H, W, 4], device=dev)
        for (j, (frus_nss_coor_, z_vals_, ray_dir_)) in enumerate(zip(frus_nss_coor, z_vals, ray_dir)):
            h, w = divmod(j, scale)
            H_, W_ = H // scale, W // scale
            sampling_coor_fg_ = frus_nss_coor_[None, ...].expand(K - 1, -1, -1).clone()  # (K-1)xPx3
            sampling_coor_fg_[move_slot_idx] = sampling_coor_fg_[move_slot_idx] - self.movement / self.opt.nss_scale
            sampling_coor_bg_ = frus_nss_coor_  # Px3

            raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(sampling_coor_bg_, sampling_coor_fg_, z_slots, nss2cam0)  # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
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

        x_recon_novel, x_novel = x_recon, self.gt_moved
        self.loss_recon_moving = self.L2_loss(x_recon_novel, x_novel)
        self.loss_lpips_moving = self.LPIPS_loss(x_recon_novel, x_novel).mean()
        self.loss_psnr_moving = compute_psnr(x_recon_novel / 2 + 0.5, x_novel / 2 + 0.5, data_range=1.)
        self.loss_ssim_moving = compute_ssim(x_recon_novel / 2 + 0.5, x_novel / 2 + 0.5, data_range=1.)

        for i in range(self.opt.n_img_each_scene):
            setattr(self, 'gt_moved{}'.format(i), self.gt_moved[i])
            setattr(self, 'ours_moved{}'.format(i), x_recon[i])

        """ change bg """

        setattr(self, 'input_bg', self.bg_img[0])
        setattr(self, 'placeholder', torch.zeros_like(self.bg_img[0]))

        # Encoding bg images
        feature_map = self.netEncoder(F.interpolate(self.bg_img[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False))  # BxCxHxW
        feat = feature_map.flatten(start_dim=2).permute([0, 2, 1])  # BxNxC

        # Slot Attention
        z_slots_, attn_ = self.netSlotAttention(feat)  # 1xKxC, 1xKxN
        z_slots_, attn_ = z_slots_.squeeze(0), attn_.squeeze(0)  # KxC, KxN

        z_slots_bg = z_slots_[0:1]
        z_slots_fg = z_slots[1:]
        z_slots = torch.cat([z_slots_bg, z_slots_fg], dim=0)

        x_recon, rendered, masked_raws, unmasked_raws = \
            torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, D, H, W, 4], device=dev), torch.zeros([K, N, D, H, W, 4], device=dev)
        for (j, (frus_nss_coor_, z_vals_, ray_dir_)) in enumerate(zip(frus_nss_coor, z_vals, ray_dir)):
            h, w = divmod(j, scale)
            H_, W_ = H // scale, W // scale
            sampling_coor_fg_ = frus_nss_coor_[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
            sampling_coor_bg_ = frus_nss_coor_  # Px3

            raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(sampling_coor_bg_, sampling_coor_fg_, z_slots, nss2cam0)  # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
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

        x_recon_novel, x_novel = x_recon, self.gt_changed
        self.loss_recon_changing = self.L2_loss(x_recon_novel, x_novel)
        self.loss_lpips_changing = self.LPIPS_loss(x_recon_novel, x_novel).mean()
        self.loss_psnr_changing = compute_psnr(x_recon_novel/2+0.5, x_novel/2+0.5, data_range=1.)
        self.loss_ssim_changing = compute_ssim(x_recon_novel/2+0.5, x_novel/2+0.5, data_range=1.)

        for i in range(self.opt.n_img_each_scene):
            setattr(self, 'gt_changed{}'.format(i), self.gt_changed[i])
            setattr(self, 'ours_changed{}'.format(i), x_recon[i])

    def compute_visuals(self):
        pass

    def backward(self):
        pass

    def optimize_parameters(self, ret_grad=False, epoch=0):
        """Update network weights; it will be called in every training iteration."""
        self.forward(epoch)
        for opm in self.optimizers:
            opm.zero_grad()
        self.backward()
        avg_grads = []
        layers = []
        if ret_grad:
            for n, p in chain(self.netEncoder.named_parameters(), self.netSlotAttention.named_parameters(), self.netDecoder.named_parameters()):
                if p.grad is not None and "bias" not in n:
                    with torch.no_grad():
                        layers.append(n)
                        avg_grads.append(p.grad.abs().mean().cpu().item())
        for opm in self.optimizers:
            opm.step()
        return layers, avg_grads

    def save_networks(self, surfix):
        """Save all the networks to the disk.

        Parameters:
            surfix (int or str) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        super().save_networks(surfix)
        for i, opm in enumerate(self.optimizers):
            save_filename = '{}_optimizer_{}.pth'.format(surfix, i)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(opm.state_dict(), save_path)

        for i, sch in enumerate(self.schedulers):
            save_filename = '{}_lr_scheduler_{}.pth'.format(surfix, i)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(sch.state_dict(), save_path)

    def load_networks(self, surfix):
        """Load all the networks from the disk.

        Parameters:
            surfix (int or str) -- current epoch; used in he file name '%s_net_%s.pth' % (epoch, name)
        """
        super().load_networks(surfix)

        if self.isTrain:
            for i, opm in enumerate(self.optimizers):
                load_filename = '{}_optimizer_{}.pth'.format(surfix, i)
                load_path = os.path.join(self.save_dir, load_filename)
                print('loading the optimizer from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                opm.load_state_dict(state_dict)

            for i, sch in enumerate(self.schedulers):
                load_filename = '{}_lr_scheduler_{}.pth'.format(surfix, i)
                load_path = os.path.join(self.save_dir, load_filename)
                print('loading the lr scheduler from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                sch.load_state_dict(state_dict)


if __name__ == '__main__':
    pass