import os
import glob
import argparse
import builtins
import math
import random
import shutil
import time
import warnings


import mindspore as ms
from mindspore import nn, Tensor, Model, ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size

from MoCo_ms import MoCo
from dataset_ms import create_dataset
from ResNet_ms import resnet18, resnet50


parser = argparse.ArgumentParser(description='MindSpore ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='./ckpt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: ./ckpt)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use MindSpore for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true', help='use mlp head')
parser.add_argument('--aug-plus', action='store_true', help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

parser.add_argument('--hidden', default=-1, type=int, help='hidden layer')


def get_last_checkpoint(checkpoint_dir):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_moco_*.ckpt'))
    if all_ckpt:
        all_ckpt = sorted(all_ckpt)
        return all_ckpt[-1]
    else: return ''


def get_lr(steps_per_epoch, args):
    """ generate learning rate array """
    lr_each_step = []
    for epoch in range(args.epochs):
        lr = args.lr
        if args.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        for step in range(steps_per_epoch):
            lr_each_step.append(lr)

    lr_each_step = Tensor(lr_each_step, ms.float32)
    return lr_each_step
 

class NetWithLossCell(nn.Cell):
    def __init__(self, network, loss):
        super(NetWithLossCell, self).__init__()
        self.network = network
        self.loss = loss

    def construct(self, data_x, data_y):
        logits, labels = self.network(data_x, data_y)
        return self.loss(logits, labels)


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
    args = parser.parse_args()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.distributed: # GPU target
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
        init("nccl")
        ms.set_auto_parallel_context(device_num=get_group_size(), parameter_broadcast=True,
                                     dataset_strategy="data_parallel",
                                     parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    else: ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU", device_id=args.gpu)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    train_dataset = create_dataset(traindir, args.batch_size, args.aug_plus, args.distributed)
    steps_per_epoch = train_dataset.get_dataset_size()
    # print(steps_per_epoch)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet50': base_model = resnet50
    model = MoCo(base_model, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args)

    # # optionally resume from a checkpoint
    # if args.resume:
    #     checkpoint_path = get_last_checkpoint(args.resume)
    #     if os.path.isfile(checkpoint_path):
    #         print("=> loading checkpoint '{}'".format(checkpoint_path))
    #         param_dict = ms.load_checkpoint(checkpoint_path)
    #         ms.load_param_into_net(model, param_dict)
    #         args.start_epoch = int(checkpoint_path[-8:-5])
    #         print("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, args.start_epoch))
    #     else: print("=> no checkpoint found at '{}'".format(args.resume))
    
    # # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = nn.SGD(model.trainable_params(), learning_rate= get_lr(steps_per_epoch, args), 
    #                    momentum=args.momentum, weight_decay=args.weight_decay)

    # net_loss = NetWithLossCell(model, criterion)
    # train_net = nn.TrainOneStepCell(net_loss, optimizer)
    # # model = Model(train_net)

    ''''''
    for item in train_dataset.create_tuple_iterator():
        im_q, im_k = item[0], item[1]
        # print(im_q.shape)
        logits, labels = model(im_q, im_k)
        break
    ''''''

    # config_ck = ms.CheckpointConfig(save_checkpoint_steps=steps_per_epoch)
    # ckpoint_cb = ms.ModelCheckpoint(prefix="checkpoint_moco", directory=args.resume, config=config_ck)
    # model.train(args.epochs-args.start_epoch, train_dataset, 
    #             callbacks=[ms.TimeMonitor(steps_per_epoch), ckpoint_cb, ms.LossMonitor(args.print_freq)], dataset_sink_mode=True)
    # print(train_dataset.get_dataset_size())
