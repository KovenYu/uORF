from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, save_images
from util.html import HTML
import os
from util.util import AverageMeter, set_seed

if __name__ == '__main__':

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    meters_tst = {stat: AverageMeter() for stat in model.loss_names}

    set_seed(opt.seed)

    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.testset_name, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    for i, data in enumerate(dataset):
        visualizer.reset()
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference: forward + compute_visuals

        losses = model.get_current_losses()
        visualizer.print_test_losses(i, losses)
        for loss_name in model.loss_names:
            meters_tst[loss_name].update(float(losses[loss_name]))

        visuals = model.get_current_visuals()
        visualizer.display_current_results(visuals, epoch=None, save_result=False)
        img_path = model.get_image_paths()
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size)
        losses = {}
        for loss_name in model.loss_names:
            losses[loss_name] = meters_tst[loss_name].avg
        visualizer.print_test_losses('average', losses)

    webpage.save()