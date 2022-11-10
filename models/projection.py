import torch
import numpy as np
import torch.nn.functional as F


class Projection(object):
    def __init__(self, focal_ratio=(350. / 320., 350. / 240.),
                 near=5, far=16, frustum_size=[128, 128, 128], device='cpu',
                 nss_scale=7, render_size=(64, 64)):
        self.render_size = render_size
        self.device = device
        self.focal_ratio = focal_ratio
        self.near = near
        self.far = far
        self.frustum_size = frustum_size

        self.nss_scale = nss_scale
        self.world2nss = torch.tensor([[1/nss_scale, 0, 0, 0],
                                        [0, 1/nss_scale, 0, 0],
                                        [0, 0, 1/nss_scale, 0],
                                        [0, 0, 0, 1]]).unsqueeze(0).to(device)
        focal_x = self.focal_ratio[0] * self.frustum_size[0]
        focal_y = self.focal_ratio[1] * self.frustum_size[1]
        bias_x = (self.frustum_size[0] - 1.) / 2.
        bias_y = (self.frustum_size[1] - 1.) / 2.
        intrinsic_mat = torch.tensor([[focal_x, 0, bias_x, 0],
                                      [0, focal_y, bias_y, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        self.cam2spixel = intrinsic_mat.to(self.device)
        self.spixel2cam = intrinsic_mat.inverse().to(self.device)

    def construct_frus_coor(self):
        x = torch.arange(self.frustum_size[0])
        y = torch.arange(self.frustum_size[1])
        z = torch.arange(self.frustum_size[2])
        x, y, z = torch.meshgrid([x, y, z])
        x_frus = x.flatten().to(self.device)
        y_frus = y.flatten().to(self.device)
        z_frus = z.flatten().to(self.device)
        # project frustum points to vol coord
        depth_range = torch.linspace(self.near, self.far, self.frustum_size[2]).to(self.device)
        z_cam = depth_range[z_frus].to(self.device)

        x_unnorm_pix = x_frus * z_cam
        y_unnorm_pix = y_frus * z_cam
        z_unnorm_pix = z_cam
        pixel_coor = torch.stack([x_unnorm_pix, y_unnorm_pix, z_unnorm_pix, torch.ones_like(x_unnorm_pix)])
        return pixel_coor

    def construct_sampling_coor(self, cam2world, partitioned=False):
        """
        construct a sampling frustum coor in NSS space, and generate z_vals/ray_dir
        input:
            cam2world: Nx4x4, N: #images to render
        output:
            frus_nss_coor: (NxDxHxW)x3
            z_vals: (NxHxW)xD
            ray_dir: (NxHxW)x3
        """
        N = cam2world.shape[0]
        W, H, D = self.frustum_size
        pixel_coor = self.construct_frus_coor()
        frus_cam_coor = torch.matmul(self.spixel2cam, pixel_coor.float())  # 4x(WxHxD)

        frus_world_coor = torch.matmul(cam2world, frus_cam_coor)  # Nx4x(WxHxD)
        frus_nss_coor = torch.matmul(self.world2nss, frus_world_coor)  # Nx4x(WxHxD)
        frus_nss_coor = frus_nss_coor.view(N, 4, W, H, D).permute([0, 4, 3, 2, 1])  # NxDxHxWx4
        frus_nss_coor = frus_nss_coor[..., :3]  # NxDxHxWx3
        scale = H // self.render_size[0]
        if partitioned:
            frus_nss_coor_ = []
            for i in range(scale**2):
                h, w = divmod(i, scale)
                frus_nss_coor_.append(frus_nss_coor[:, :, h::scale, w::scale, :])
            frus_nss_coor = torch.stack(frus_nss_coor_, dim=0)  # 4xNxDx(H/s)x(W/s)x3
            frus_nss_coor = frus_nss_coor.flatten(start_dim=1, end_dim=4)  # 4x(NxDx(H/s)x(W/s))x3
        else:
            frus_nss_coor = frus_nss_coor.flatten(start_dim=0, end_dim=3)  # (NxDxHxW)x3

        z_vals = (frus_cam_coor[2] - self.near) / (self.far - self.near)  # (WxHxD) range=[0,1]
        z_vals = z_vals.expand(N, W * H * D)  # Nx(WxHxD)
        if partitioned:
            z_vals = z_vals.view(N, W, H, D).permute([0, 2, 1, 3])  # NxHxWxD
            z_vals_ = []
            for i in range(scale**2):
                h, w = divmod(i, scale)
                z_vals_.append(z_vals[:, h::scale, w::scale, :])
            z_vals = torch.stack(z_vals_, dim=0)  # 4xNx(H/s)x(W/s)xD
            z_vals = z_vals.flatten(start_dim=1, end_dim=3)  # 4x(Nx(H/s)x(W/s))xD
        else:
            z_vals = z_vals.view(N, W, H, D).permute([0, 2, 1, 3]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xD

        # construct cam coord for ray_dir
        x = torch.arange(self.frustum_size[0])
        y = torch.arange(self.frustum_size[1])
        X, Y = torch.meshgrid([x, y])
        Z = torch.ones_like(X)
        pix_coor = torch.stack([Y, X, Z]).to(self.device)  # 3xHxW, 3=xyz
        cam_coor = torch.matmul(self.spixel2cam[:3, :3], pix_coor.flatten(start_dim=1).float())  # 3x(HxW)
        ray_dir = cam_coor.permute([1, 0])  # (HxW)x3
        ray_dir = ray_dir.view(H, W, 3)
        if partitioned:
            ray_dir = ray_dir.expand(N, H, W, 3)
            ray_dir_ = []
            for i in range(scale ** 2):
                h, w = divmod(i, scale)
                ray_dir_.append(ray_dir[:, h::scale, w::scale, :])
            ray_dir = torch.stack(ray_dir_, dim=0)  # 4xNx(H/s)x(W/s)x3
            ray_dir = ray_dir.flatten(start_dim=1, end_dim=3)  # 4x(Nx(H/s)x(W/s))x3
        else:
            ray_dir = ray_dir.expand(N, H, W, 3).flatten(start_dim=0, end_dim=2)  # (NxHxW)x3
        return frus_nss_coor, z_vals, ray_dir

if __name__ == '__main__':
    pass