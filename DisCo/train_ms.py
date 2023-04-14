import os
import argparse

import mindspore as ms
from mindspore import nn, ParallelMode
from mindspore.communication.management import init, get_rank

from dataset_ms import create_dataset
from ResNet_ms import resnet18, resnet50
from moco.builder_kq_mse_largeembedding_2048 import MoCo

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
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-ckpt', action='store_true',
                    help='whether or not to resume from ./ckpt')
parser.add_argument('--teacher_arch', default='resnet50', type=str,
                    help='teacher architecture')
parser.add_argument('--teacher', default='', type=str, metavar='PATH',
                    help='path to teacher checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
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

parser.add_argument('--nmb_prototypes', default=0, type=int, help='num prototype')
parser.add_argument('--only-mse', action='store_true', help='only use mse loss')


if __name__ == '__main__':
    args = parser.parse_args()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.distributed:
    # GPU target
        ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
        init()
        ms.set_auto_parallel_context(device_num=args.device_num, 
                                     parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU", device_id=args.gpu)

    


    # Data loading code
    traindir = os.path.join(args.data, 'train')
    train_dataset = create_dataset(traindir, args.batch_size, args.aug_plus, args.distributed)

    # create model
    print("=> creating teacher model '{}'".format(args.teacher_arch))
    if args.teacher_arch == 'resnet50':
        teacher_model = resnet50(class_num=args.moco_dim)
    print("=> creating student model '{}'".format(args.arch))
    if args.arch == "resnet18":
        model = resnet18(class_num=args.moco_dim)



    model = MoCo(model, teacher_model, args.moco_dim, 
                 args.moco_k, args.moco_m, args.moco_t, args.mlp, args)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss(reduction='sum')

    optimizer = nn.SGD(model.parameters(), learning_rate=args.lr,
                       momentum=args.momentum, weight_decay=args.weight_decay)