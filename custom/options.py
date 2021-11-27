import argparse

def parse_custom_options():
    """Removed unnecessary arguments from original implementation"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_dataroot', required=True, help='path to room diverse dataset')
    parser.add_argument('--test_dataroot', required=True, help='path to room diverse test dataset')

    parser.add_argument('--gpus', type=str, default='-1', help='gpu parameter eg. 3 for three gpus; 0,1 for gpu 0 and 1; -1 for all gpus; and 0 for CPU')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for the experiment')
    # dataset parameters
    parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--load_size', type=int, default=128, help='scale images to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    # additional parameters
    parser.add_argument('--custom_lr', action='store_true', help='Custom lr(per step) scheduler for slot model. Currently hack.')


    ##########
    # MultiscenesDataset options
    ##########

    parser.add_argument('--n_scenes', type=int, default=5000, help='dataset length is #scenes')
    parser.add_argument('--n_img_each_scene', type=int, default=4, help='for each scene, how many images to load in a batch')
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--mask_size', type=int, default=128)


    ##########
    # Train options
    ##########

    # visdom and HTML visualization parameters
    parser.add_argument('--display_grad', action='store_true')
    parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing visuals on screen')

    
    # training parameters
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

    ###########
    # Test options
    #########

    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    # Dropout and Batchnorm has different behavioir during training and test.
    parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
    parser.add_argument('--testset_name', type=str, default='testset')


    ##########
    # GAN Model
    #########

    parser.add_argument('--num_slots', metavar='K', type=int, default=8, help='Number of supported slots')
    parser.add_argument('--z_dim', type=int, default=64, help='Dimension of individual z latent per slot')
    parser.add_argument('--attn_iter', type=int, default=3, help='Number of refine iteration in slot attention')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--nss_scale', type=float, default=7, help='Scale of the scene, related to camera matrix')
    parser.add_argument('--render_size', type=int, default=64, help='Shape of patch to render each forward process. Must be Frustum_size/(2^N) where N=0,1,..., Smaller values cost longer time but require less GPU memory.')
    parser.add_argument('--supervision_size', type=int, default=64)
    parser.add_argument('--obj_scale', type=float, default=4.5, help='Scale for locality on foreground objects')
    parser.add_argument('--n_freq', type=int, default=5, help='how many increased freq?')
    parser.add_argument('--n_samp', type=int, default=64, help='num of samp per ray')
    parser.add_argument('--n_layer', type=int, default=3, help='num of layers bef/aft skip link in decoder')
    parser.add_argument('--weight_percept', type=float, default=0.006)
    parser.add_argument('--percept_in', type=int, default=20)
    parser.add_argument('--no_locality_epoch', type=int, default=60)
    parser.add_argument('--bottom', action='store_true', help='one more encoder layer on bottom')
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--frustum_size', type=int, default=64)
    parser.add_argument('--frustum_size_fine', type=int, default=128)
    parser.add_argument('--attn_decay_steps', type=int, default=2e5)
    parser.add_argument('--coarse_epoch', type=int, default=120)
    parser.add_argument('--near_plane', type=float, default=6)
    parser.add_argument('--far_plane', type=float, default=20)
    parser.add_argument('--d_lr', type=float, default=0.001)
    parser.add_argument('--weight_gan', type=float, default=0.01)
    parser.add_argument('--gan_in', type=int, default=5)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--weight_r1', type=float, default=10)
    parser.add_argument('--gan_train_epoch', type=int, default=20)
    parser.add_argument('--fixed_locality', action='store_true', help='enforce locality in world space instead of transformed view space')

    parser.set_defaults(batch_size=1, lr=3e-4, niter_decay=0,
                        dataset_mode='multiscenes', niter=240, custom_lr=True, lr_policy='warmup')


    return parser.parse_args()
