import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
#from data_loader import *
from PIL import Image
import sys
sys.path.append('..')
from losses.contrastive import ContrastiveLoss
from data_utils.ModelNetContrastiveDataLoader import ModelNetContrastiveDataLoader
from math import ceil, floor
import argparse
from models import pointnet2_cls_ssg_contrastive
from numpy import inf

INFINITE = inf
DATA_PATH = './data/modelnet40_normal_resampled'

def save_checkpoint(v_loss, epoch, model, optimizer, args):
    # save the best loss checkpoint
    print('\nWriting model checkpoint')
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss': v_loss
    }
    file_name = f'{args.log_dir}/{args.checkpoint}'

    torch.save(state, file_name)

def compute_grad(model):
    param_count = 0
    grad_ = 0.0
    for f in model.parameters():
        param_count += 1
        if f.grad is None:
            continue
        grad_ += torch.sum(torch.abs(f.grad))

    grad_ /= param_count

    return grad_


def train_validate(simclr, loss_func, optimizer, data, args, is_train):
    # if is_train set the model to be trainable and
    # else to only eval data
    if is_train:
        simclr.train()
        simclr.zero_grad()
    else:
        simclr.eval()

    total_loss = 0.0
    grad_ = 0.0

    data_iterator = iter(data)

    # keep iterating over data loader until StopIteration exception
    i = 0
    while True:
        try:
            xi, xj, _ = data_iterator.next()

            xi = xi.cuda() if args.use_cuda else xi
            xj = xj.cuda() if args.use_cuda else xj

            xi = xi.transpose(2, 1)
            xj = xj.transpose(2, 1)
            # get z(h(x))
            _, zi = simclr(xi)
            _, zj = simclr(xj)

            # compute contrastive loss
            loss = loss_func(zi, zj)

            # if is_train backpropagate
            if is_train:
                loss.backward()

            if (i + 1) % args.accumulation_steps == 0 and is_train:
                optimizer.step()
                grad_ = compute_grad(simclr)
                simclr.zero_grad()

            # accumulate losses
            total_loss += loss.item()
            i += 1

            print(f'\t- Loss: {loss.item()}\tGrad: {grad_}', end='\r')
        except StopIteration as si:
            break

    # return the epoch mean loss
    return total_loss / len(data)


def run_trainer_evaluator(simclr, loss_func, optimizer, args):
    data_train = ModelNetContrastiveDataLoader(root=DATA_PATH, npoint=1024, split='train',
                                                     normal_channel=False)
    data_test = ModelNetContrastiveDataLoader(root=DATA_PATH, npoint=1024, split='test',
                                                    normal_channel=False)
    best_vloss = INFINITE

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(cifar_train), eta_min=0, last_epoch=-1)

    for epoch in range(args.epochs):
        print(f'Starting epoch [{epoch}/{args.epochs}]')

        # create the data loader for train and validation data
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)

        # train and retrieve training loss
        t_loss = train_validate(simclr, loss_func, optimizer, train_loader, args, is_train=True)

        # retrieve validation loss
        v_loss = train_validate(simclr, loss_func, optimizer, test_loader, args, is_train=False)

        # adjust learning rate
        # scheduler.step()

        print(f'\nTotal epoch losses:\ttrain: {t_loss}\tvalidation: {v_loss}\n', end='\r')

        # if the current loss is the new best, update checkpoint
        if v_loss < best_vloss:
            best_vloss = v_loss
            save_checkpoint(v_loss, epoch, simclr, optimizer, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SIMCLR')

    parser.add_argument('--dataset-name', type=str, default='CIFAR10',
                        help='Name of dataset (default: CIFAR10')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to dataset (default: data')
    parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of training epochs (default: 150)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3')
    parser.add_argument("--decay-lr", default=1e-6, action="store", type=float,
                        help='Learning rate decay (default: 1e-6')
    parser.add_argument('--accumulation-steps', type=int, default=4, metavar='N',
                        help='Gradient accumulation steps (default: 4')
    parser.add_argument('--tau', default=0.5, type=float,
                        help='Tau temperature smoothing (default 0.5)')
    parser.add_argument('--log-dir', type=str, default='checkpoint',
                        help='logging directory (default: checkpoint)')
    parser.add_argument('--checkpoint', type=str, default='bestcheckpoint.pt',
                        help='model checkpoint (default: bestcheckpoint.pt)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='using cuda (default: True')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Load model to resume training for (default None)')
    parser.add_argument('--feature_model', type=str, default='resnet50',
                        help='Load feature extractor model (default resnet50)')
    parser.add_argument('--feature-size', type=int, default=128,
                        help='Feature output size (default: 128')
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU device id (default: 0')

    args = parser.parse_args()

    if args.use_cuda:
        dtype = torch.cuda.FloatTensor
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        print('GPU')
    else:
        dtype = torch.FloatTensor
        device = torch.device("cpu")

    simclr = pointnet2_cls_ssg_contrastive.get_model(num_class=40, normal_channel=False)#SimCLR(args).type(dtype)
    loss_func = ContrastiveLoss(args.tau)
    
    optimizer = torch.optim.Adam(simclr.parameters(), lr=args.lr, weight_decay=args.decay_lr)

    run_trainer_evaluator(simclr, loss_func, optimizer, args)
