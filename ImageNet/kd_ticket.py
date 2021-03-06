
from __future__ import print_function

import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101, resnet152

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.misc import get_conv_zero_param


# ############################### Parameters ###############################
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# ******************* Datasets *******************
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_dir', default='/data/imagenet/raw-data', type=str, help='imagenet location')

# ******************* Optimization options *******************
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=512, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=512, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 80],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# ******************* Checkpoints *******************
# Checkpoints (for model_ref, which is the unpruned model)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# path to the Lottery Ticket initialization !!!
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the initialization checkpoint (default: none)')

# path to the teacher model !!!
parser.add_argument('--teacher', default='', type=str, metavar='PATH', required=True,
                    help='path to the teacher model checkpoint, which is the unpruned model')

# ******************* Architecture *******************
parser.add_argument('--depth', type=int, default=50, help='Model depth.')


# ******************* Miscs *******************
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='results/', type=str)

# ******************* Device options *******************
parser.add_argument('--gpu-id', default=[0, 1, 2, 3], type=int, nargs='+',
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--apex', default=True, type=bool, help="use APEX or not")

# ******************* Knowledge distillation parameters *******************
parser.add_argument('--temperature', default=4, type=float, help='temperature of KD')
parser.add_argument('--alpha', default=0.9, type=float, help='ratio for KL loss')
parser.add_argument('--eskd', default=80, type=int, help='early stop of KD')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# ############################### GPU ###############################
use_cuda = torch.cuda.is_available()
device_ids = args.gpu_id
torch.cuda.set_device(device_ids[0])

# ############################### Random seed ###############################
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 100000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


# ##################### APEX #####################

if args.apex:
    try:
        import apex
        USE_APEX = True
        print('Use APEX !!!')
    except ImportError:
        USE_APEX = False
else:
    USE_APEX = False


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    os.makedirs(args.save_dir, exist_ok=True)

    # ######################################### Dataset ################################################
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(0.8, 1.25)),  # according to official open LTH

        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # #################### train / valid dataset ####################
    train_dir = os.path.join(args.img_dir, 'train')
    valid_dir = os.path.join(args.img_dir, 'val')

    trainset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    devset = datasets.ImageFolder(root=valid_dir, transform=val_transforms)

    print('Total images in train, ', len(trainset))
    print('Total images in valid, ', len(devset))

    # #################### data loader ####################
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch,
                                  shuffle=True, num_workers=args.workers)

    devloader = data.DataLoader(devset, batch_size=args.test_batch,
                                shuffle=False, num_workers=args.workers)

    # ######################################### Model ##################################################
    print("==> creating model ResNet={}".format(args.depth))
    if args.depth == 18:
        model = resnet18(pretrained=False)
        model_ref = resnet18(pretrained=False)
        teacher_model = resnet18(pretrained=False)

    elif args.depth == 34:
        model = resnet34(pretrained=False)
        model_ref = resnet34(pretrained=False)
        teacher_model = resnet34(pretrained=False)

    elif args.depth == 50:
        model = resnet50(pretrained=False)
        model_ref = resnet50(pretrained=False)
        teacher_model = resnet50(pretrained=False)

    elif args.depth == 101:
        model = resnet101(pretrained=False)
        model_ref = resnet101(pretrained=False)
        teacher_model = resnet101(pretrained=False)

    elif args.depth == 152:
        model = resnet152(pretrained=False)
        model_ref = resnet152(pretrained=False)
        teacher_model = resnet152(pretrained=False)
    else:
        model = resnet50(pretrained=False)  # default Res-50
        model_ref = resnet50(pretrained=False)  # default Res-50
        teacher_model = resnet50(pretrained=False)  # default Res-50

    model.cuda(device_ids[0])                           # model to train (student model)
    model_ref.cuda(device_ids[0])                       # pruned model
    teacher_model.cuda(device_ids[0])  # teacher model, the last epoch of unpruned training model

    # ############################### Optimizer and Loss ###############################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    # ************* USE APEX *************
    if USE_APEX:
        print('Use APEX !!! Initialize Model with APEX')
        model, optimizer = apex.amp.initialize(model, optimizer, loss_scale='dynamic', verbosity=0)

    # ****************** multi-GPU ******************
    model = nn.DataParallel(model, device_ids=device_ids)
    model_ref = nn.DataParallel(model_ref, device_ids=device_ids)
    teacher_model = nn.DataParallel(teacher_model, device_ids=device_ids)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # ############################### Resume ###############################
    # load pruned model (model_ref), use it to mute some weights of model
    title = 'ImageNet'
    if args.resume:
        # Load checkpoint.
        print('==> Getting reference model from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        best_acc = checkpoint['best_acc']
        start_epoch = args.start_epoch
        model_ref.load_state_dict(checkpoint['state_dict'])

    logger = Logger(os.path.join(args.save_dir, 'log_scratch.txt'), title=title)
    logger.set_names(['EPOCH', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # set some weights to zero, according to model_ref ---------------------------------
    # ############## load Lottery Ticket (initialization parameters of un pruned model) ##############
    if args.model:
        print('==> Loading init model (Lottery Ticket) from %s' % args.model)
        checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if 'init' in args.model:
            start_epoch = 0
        else:
            start_epoch = checkpoint['epoch']
        print('Start Epoch ', start_epoch)
    for m, m_ref in zip(model.modules(), model_ref.modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m_ref.weight.data.abs().clone()
            mask = weight_copy.gt(0).float().cuda()
            m.weight.data.mul_(mask)

    # ############## load parameters of teacher model ##############
    print('==> Loading teacher model (un-pruned) from %s' % args.teacher)
    checkpoint = torch.load(args.teacher, map_location=lambda storage, loc: storage)
    teacher_model.load_state_dict(checkpoint['state_dict'])
    teacher_model.eval()

    # ############################### Train and val ###############################
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        num_parameters = get_conv_zero_param(model)
        print('Zero parameters: {}'.format(num_parameters))
        num_parameters = sum([param.nelement() for param in model.parameters()])
        print('Parameters: {}'.format(num_parameters))

        # train model
        train_loss, train_acc = train(trainloader, model, teacher_model, optimizer, epoch, use_cuda)

        # ######## acc on validation data each epoch ########
        dev_loss, dev_acc = test(devloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([ epoch, state['lr'], train_loss, dev_loss, train_acc, dev_acc])

        # save model after one epoch
        # Note: save all models after one epoch, to help find the best rewind
        is_best = dev_acc > best_acc
        best_acc = max(dev_acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': dev_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.save_dir, filename=str(epoch + 1)+'_checkpoint.pth.tar')

    print('Best val acc:')
    print(best_acc)

    logger.close()


def train(trainloader, model, tch_model, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    tch_model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    print(args)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # get teacher model outputs
        with torch.no_grad():
            tch_logit = tch_model(inputs)

        # compute output
        outputs = model(inputs)

        # KL div loss (tch_logit, outputs) + CE loss (target, output)
        loss = loss_fn_kd(outputs, targets, tch_logit, epoch)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # ************* USE APEX *************
        if USE_APEX:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        for k, m in enumerate(model.modules()):
            # print(k, m)
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        info = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        print(info)
        bar.suffix = info
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    print('VALID! ')
    end = time.time()
    bar = Bar('Processing', max=len(testloader))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            info = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            print(info)
            bar.suffix = info
            bar.next()
        bar.finish()

    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    
    if epoch < 5:  # Learning Rate warm up
        state['lr'] = args.lr * (epoch + 1) / 5
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def loss_fn_kd(outputs, labels, teacher_outputs, epoch):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    if epoch < args.eskd:
        alpha = args.alpha
        T = args.temperature
        KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                                 F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
                  F.cross_entropy(outputs, labels) * (1.0 - alpha)
    else:
        KD_loss = F.cross_entropy(outputs, labels)
    return KD_loss


if __name__ == '__main__':
    main()



