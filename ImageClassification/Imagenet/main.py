import argparse
import os
import random
import shutil
import time
import warnings
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from resnet_GN import ResNet18_GN
from properties_checker import compute_linear_approx, compute_smoothness, compute_difference, compute_inner_product, compute_L2_norm
# from properties_checker import compute_linear_approx, compute_smoothness,compute_L2_norm,compute_L1_norm, compute_PL,compute_inner_product, cosine_similarity
# import hacks

from SGDHess import SGDHess
from STORM import STORM
from sgd import sgd
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--path', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--opt', default='', type=str, metavar='PATH',
                    help='optimizer')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1235', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument(
        "--id",
         type=str,
        default="1"
    )
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--save', default=False, type=bool, help='save param')
parser.add_argument('--name', default='experiment', type=str, help='experiment name')

best_acc1 = 0
# def compute_linear_approx(model, prev_param):
#     #this function compute the following inner product: <\nabla f(x_t), x_t - x_{t-1}>
#     # It takes 2 arguments:
#     # 1. model: the model that we are running. We need this to access the current parameters and their gradients. 
#     # 2. prev_param: the model parameters in the previous iteration (x_{t-1})
#     linear_approx = 0
#     i = 0
#     with torch.no_grad():
#         for p in model.parameters():
#             linear_approx+= torch.dot(torch.flatten(p.grad), torch.flatten(p.add(-prev_param[i]))).item()
#             i+=1
#     return linear_approx

# def compute_smoothness(model, prev_param, prev_grad):
#     # This function compute the smoothness constant which is L = max(\|\nabla f(x_t) -\nabla f(x_{t-1})\| /  \|f(x_t) -f(x_{t-1})\|)
#     # It takes 4 arguments:
#     # 1. model: the model that we are running. We need this to access the current parameters and their gradients. 
#     # 2. prev_param: the model parameters in the previous iteration (x_{t-1})
#     # 3. prev_grad: the gradient of the model parameters in the previous iteration (\nabla f(x_{t-1}))
#     # 4. L: the current smoothness constant (since we want to find the largest L that satisfies the smoothness condition)
#     sum_num = 0
#     sum_denom = 0
#     i=0
#     with torch.no_grad():
#         for p in model.parameters():
#             sum_num += torch.norm(p.grad - prev_grad[i])
#             sum_denom += torch.norm(p - prev_param[i])
#             i+=1
#     return sum_num/sum_denom
# ####

def main():
    # hacks.apply()
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    # model = ResNet18_GN()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        rank = int(args.gpu)
        print(f"RANK: {rank}")
        time.sleep(rank)
        print("reach distributed!")
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    wandb.init(project='test_convexity', config=args, name=args.name)
    wandb.watch(model)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    # Data loading code
    traindir = os.path.join(args.data, 'train_processed')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    acc = []
    filename = args.path + "0.pth.tar"
    torch.save({'state_dict':optimizer.state_dict(), 'model_dict': model.state_dict() }, filename)
    # dict to save last iterate of each epoch
    for epoch in range(args.start_epoch, args.epochs):
        print("reach training!")
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        # save param
        # if args.save:
        #     save_param[epoch] = {'state_dict': optimizer.state_dict(), 'loss':loss}
        # train for one epoch
        # name = 'checkpoint/'+ str(epoch+1) + ".pth.tar"
        # saved_checkpoint = torch.load(name)
        # optimizer.save_param(saved_checkpoint['state_dict'])
        # prev_loss = saved_checkpoint['current_loss']
        prev_grad, prev_param, current_grad_L, current_grad_linear, current_param, prev_loss, current_loss, convexity_gap, smoothness,ratio,num, denom,exp_avg_L_1, \
        exp_avg_L_2, exp_avg_gap_1, exp_avg_gap_2, linear_actual,num_actual,current_convexity_gap,  \
        smoothness_ratio_prev, smoothness_ratio_actual, update_product  = train(train_loader, model, criterion, optimizer, epoch, args)
        # prev_grad: \nabla f(w_T,x_T)
        # current_grad: \nabla f(w_{T+1},x_T)
        # prev_param: w_T
        # current_param : w_{T+1}
        # prev_loss: f(w_T,x_T)
        # current_loss: f(w_{T+1},x_T)
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        wandb.log(
        {
            "train_loss": current_loss,
            "convexity_gap": convexity_gap,
            "smoothness": smoothness,
            "linear/loss_gap": ratio,
            "numerator" : num,
            "denominator": denom,
            'exp_avg_L_.99': exp_avg_L_1,
            'exp_avg_L_.9999': exp_avg_L_2, 
            "exp_avg_gap_.99":  exp_avg_gap_1, 
            "exp_avg_gap_.9999":  exp_avg_gap_2, 
            'linear_actual': linear_actual, 
            'num_actual': num_actual, 
            'current_convexity_gap': current_convexity_gap, 
            'smoothness_ratio_prev': smoothness_ratio_prev,
            'smoothness_ratio_actual': smoothness_ratio_actual,
            'update_product': update_product, 
            'accuracy': best_acc1,
        }
    )
        # print("best_acc1", best_acc1)
        # print("data type", type(best_acc1))
        acc.append(best_acc1)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, opt_save = args.save_path)
            if args.save:
                filename = args.path + str(epoch+1) + ".pth.tar"
                torch.save({'state_dict':optimizer.state_dict(),'prev_grad':prev_grad, 
                            'prev_param': prev_param, 'current_grad_L':current_grad_L, 'current_grad_linear':current_grad_linear,'current_param': current_param
                            , 'prev_loss':prev_loss , 'current_loss': current_loss
                            ,'best_acc1': best_acc1, 'arch': args.arch, 'model_dict': model.state_dict() }, filename)
        # wandb.log({'best_acc1': best_acc1, 'convexity_gap': convexity_gap, 'smoothness_constant': L, 'innerProd/gap': ratio})
    # np.savetxt('test_sgdhess.csv', acc, delimiter=',')

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    convexity_gap = 0
    L = 0
    num = 0
    denom = 0
    prev_loss = 0
    current_loss = 0
    train_loss = 0
    exp_avg_L_1 = 0
    exp_avg_L_2 = 0
    exp_avg_gap_1 = 0
    exp_avg_gap_2 = 0
    step = 0
    linear_actual = 0
    num_actual = 0
    sum_dif = 0
    prev_update = [torch.zeros_like(p) for p in model.parameters()]
    dif = [torch.zeros_like(p) for p in model.parameters()]
    update_product = 0
    smoothness_ratio_prev = 0
    smoothness_ratio_actual = 0
    current_convexity_gap = 0
    prev_grad = [torch.zeros_like(p) for p in model.parameters()]
    prev_param = [torch.zeros_like(p) for p in model.parameters()] 
    current_grad_L = [torch.zeros_like(p) for p in model.parameters()]
    current_grad_linear = [torch.zeros_like(p) for p in model.parameters()]
    current_param = [torch.zeros_like(p) for p in model.parameters()] 
    iterator = iter(train_loader)
    prev_batch = next(iterator)
    # current_loss = 0
    for i, (images, target) in enumerate(iterator):
        # measure data loading time
        if i%100 == 0:
            print("currently in iteration", i)
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
        #compute \nabla f(w_t,x_{t-1})
        prev_batch_image = prev_batch[0].cuda(args.gpu, non_blocking=True)
        prev_batch_target = prev_batch[1].cuda(args.gpu, non_blocking=True)
        prev_batch_outputs = model(prev_batch_image) 
        prev_batch_loss = criterion(prev_batch_outputs, prev_batch_target) #f(w_t,x_{t-1})
        current_loss = prev_batch_loss.item() 
        prev_batch_loss.backward()
        i = 0
        with torch.no_grad():
            for p in model.parameters():
                current_grad_L[i].copy_(p.grad) #\nabla f(w_t,x_{t-1})
                current_param[i].copy_(p) #w_t
                i+=1
        # zero grad to do the actual update
        optimizer.zero_grad()
        
        # compute output
        output = model(images)
        loss = criterion(output, target) #f(w_t,x_t)
        # total_loss +=loss.item()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        i = 0
        with torch.no_grad():
            for p in model.parameters():
                current_grad_linear[i].copy_(p.grad) # curent_grad = \nabla f(w_t,x_{t-1})
                i+=1
        if step >0:
            # get the inner product
            dif = compute_difference(current_param, prev_param)
            sum_dif += (compute_L2_norm(dif))**2
            update_product = compute_inner_product(dif, prev_update)
            prev_update = dif
            linear_approx_prev = compute_linear_approx(current_param, current_grad_L, prev_param)
            linear_actual = compute_linear_approx(current_param, current_grad_linear, prev_param)
            # get the smoothness constant, small L means function is relatively smooth
            current_L = compute_smoothness(current_param, current_grad_L, prev_param, prev_grad)
            L = max(L,current_L)
            # L = max(L,compute_smoothness(model, current_param, current_grad))
            # this is another quantity that we want to check: linear_approx / loss_gap. The ratio is positive is good
            num+= linear_approx_prev
            num_actual += linear_actual
            denom+= current_loss - prev_loss # f(w_t,x_{t-1}) - f(w_{t-1},x_{t-1})
            smoothness_ratio_actual = (-denom + num_actual)/(1/2*sum_dif)
            smoothness_ratio_prev = (-denom + num)/(1/2*sum_dif)
            current_convexity_gap = current_loss - prev_loss - linear_approx_prev 
            exp_avg_gap_1 = 0.99*exp_avg_gap_1 + (1-0.99)*current_convexity_gap
            exp_avg_gap_2 = 0.9999*exp_avg_gap_2 + (1-0.9999)*current_convexity_gap
            exp_avg_L_1 = 0.99*exp_avg_L_1+ (1-0.99)*current_L
            exp_avg_L_2 = 0.9999*exp_avg_L_2+ (1-0.9999)*current_L
            convexity_gap+= current_convexity_gap
        # optimizer.save_param()
        i = 0
        with torch.no_grad():
            for p in model.parameters():
                prev_grad[i].copy_(p.grad) #hold \nabla f(w_{t-1},x_{t-1}) for next iteration
                prev_param[i].copy_(p) # hold w_{t-1 } for next iteration
                i+=1
        optimizer.step()
        prev_loss = loss.item() 
        prev_batch = (images, target)
        # current_loss = loss.item()
        step+=1
        # print("convexity gap", convexity_gap, "iteration", i )
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
    # optimizer.save_param()
    prev_batch_image = prev_batch[0].cuda(args.gpu, non_blocking=True)
    prev_batch_target = prev_batch[1].cuda(args.gpu, non_blocking=True)
    prev_batch_outputs = model(prev_batch_image) 
    prev_batch_loss = criterion(prev_batch_outputs, prev_batch_target) #f(w_t,x_{t-1})
    current_loss = prev_batch_loss.item() 
    prev_batch_loss.backward()
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            current_grad_L[i].copy_(p.grad) #\nabla f(w_t,x_{t-1})
            current_param[i].copy_(p) #w_t
            i+=1
    # zero grad to do the actual update
    optimizer.zero_grad()
    return prev_grad, prev_param, current_grad_L, current_grad_linear, current_param, prev_loss, current_loss, convexity_gap/(step-1), L,num/denom, num, denom,exp_avg_L_1,\
        exp_avg_L_2, exp_avg_gap_1, exp_avg_gap_2, linear_actual, num_actual,current_convexity_gap,  smoothness_ratio_prev, smoothness_ratio_actual, update_product


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, opt_save):
    filename = opt_save+ "_main_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        save_name = opt_save +'_model_best.pth.tar'
        shutil.copyfile(filename, save_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
