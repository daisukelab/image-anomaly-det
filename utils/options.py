import argparse
import torch

class Options():
    """This class defines options used during both training, test time and run time.

    Args:
        argstr: (option) command line argument string for manually feeding in your program.
    """

    def __init__(self, argstr=None):
        self.argstr = argstr

    def add_options(self, parser):
        """Define the common options that are used all the time."""
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--project', type=str, default='your_project', help='name of the project/experiment. It is used as name of sub folders where files are stored.')
        parser.add_argument('--work', type=str, default='work', help='working folder name.')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # model parameters
        parser.add_argument('--model', type=str, default='arc_face', help='so far, no choice. TBD: chooses which model to use. [arc_face]')
        parser.add_argument('--backbone', type=str, default='resnet34', help='backbone model: chooses which model to use. [resnet18|resnet34|resnet50|resnet101]')
        parser.add_argument('--weights', type=str, default='', help='model weight file, or imagenet pretrained weights by default.')
        # dataset parameters
        parser.add_argument('--no_norm', action='store_true', help='no normalization/augmentation will be applied when loading images.')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data (workers)')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--suffix', default='.png', type=str, help='image file suffix [.png|.PNG|.jpg|.JPG|.gif|.GIF]')
        # additional parameters
        parser.add_argument('--anomaly_gray', action='store_true', help='anomaly part is color by default, set this to make it grayscale')
        parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs with the initial learning rate')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
        parser.add_argument('--lr_step', type=int, default=10, help='step LR steps')
        parser.add_argument('--weight_decay', type=float, default=0.95, help='weight decay')
        # app specific
        parser.add_argument('--app_dataset_class', type=str, default='default', help='(app option) dataset class')
        parser.add_argument('--app_album_tfm', type=str, default='none', help='(app option) albumentation transform')

        return parser

    def print_options(self, opt):
        message = '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-----------------------------------------'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.add_options(parser)

        # save and return the parser
        self.parser = parser
        opt = parser.parse_args(args=self.argstr.split() if self.argstr else None)

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
