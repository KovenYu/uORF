import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import AverageMeter, set_seed


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    # Set random seed for this experiment
    set_seed(opt.seed)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        print('Dataset size:', len(dataset))
        meters_trn = {stat: AverageMeter() for stat in model.loss_names}
        opt.stage = 'coarse' if epoch < opt.coarse_epoch else 'fine'
        model.netDecoder.locality = True if epoch < opt.no_locality_epoch else False
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            if epoch < opt.gan_train_epoch:
                # print('iter {}, training attn'.format(i))
                layers, avg_grad = model.optimize_parameters(opt.display_grad, epoch)   # calculate loss functions, get gradients, update network weights
                if opt.display_grad and total_iters % opt.display_freq == 0:
                    visualizer.display_grad(layers, avg_grad)
                if opt.custom_lr and opt.stage == 'coarse':
                    model.update_learning_rate()    # update learning rates at the beginning of every step
            else:  # adv loss in
                if i % 2 == 0:  # generator
                    if epoch < opt.gan_train_epoch + opt.gan_in:
                        # print('iter {}, do not train generator'.format(i))
                        continue
                    # print('iter {}, training generator'.format(i))
                    layers, avg_grad = model.optimize_parameters(opt.display_grad, epoch)  # calculate loss functions, get gradients, update network weights
                    if opt.custom_lr and opt.stage == 'coarse':
                        model.update_learning_rate()  # update learning rates at the beginning of every step
                else:
                    # print('iter {}, training disc'.format(i))
                    model.forward_disc(i + 1)
                    model.d_scheduler.step()
                if epoch < opt.gan_train_epoch + opt.gan_in:
                    # print('iter {}, no intermediate results to show'.format(i))
                    continue

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            losses = model.get_current_losses()
            for loss_name in model.loss_names:
                meters_trn[loss_name].update(float(losses[loss_name]))
                losses[loss_name] = meters_trn[loss_name].avg
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                print('learning rate:', model.optimizers[0].param_groups[0]['lr'])

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        if not opt.custom_lr:
            model.update_learning_rate()  # update learning rates at the end of every epoch.


