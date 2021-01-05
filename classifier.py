import torch
from PIL import Image
import torchvision
import os
from math import ceil, floor
from losses.contrastive import ContrastiveLoss
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
from models import pointnet2_cls_ssg_contrastive
from numpy import inf

INFINITE = inf
DATA_PATH = './data/modelnet40_normal_resampled'

def train_validate(simclr, classifier, optimizer, data, args, is_train):
    # if is_train set the model to be trainable and
    # else to only eval data
    if is_train:
        classifier.train()
    else:
        classifier.eval()

    loss_func = torch.nn.CrossEntropyLoss()
    initial_loss = None
    total_loss = 0.0
    total_acc = 0.0

    data_iterator = iter(data)

    # keep iterating over data loader until StopIteration exception
    while True:
        try:
            # zero gradients
            classifier.zero_grad()
            x, y = data_iterator.next()

            x = x.cuda() if args.use_cuda else x
            y = y.cuda() if args.use_cuda else y

            y = y[:, 0]

            x = x.transpose(2, 1)
            # get h(x)
            h, _ = simclr(x)
            # get classification
            z = classifier(h)

            # compute contrastive loss
            loss = loss_func(z, y.long())

            # if is_train backpropagate
            if is_train:
                loss.backward()
                optimizer.step()

            # accumulate losses
            total_loss += loss.item()

            # accumulate accuracy
            pred = z.max(dim=1)[1]
            correct = pred.eq(y).sum().item()
            correct /= y.size(0)
            batch_acc = (correct * 100)
            total_acc += batch_acc

            print(f'\t- Loss: {loss.item()}\tAcc: {batch_acc}', end='\r')
        except StopIteration as si:
            break

    # return the epoch mean loss
    return total_loss / len(data), total_acc / len(data)


def run_trainer_evaluator(simclr, classifier, optimizer, args):
    data_train = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='train',
                                                     normal_channel=False)
    data_test = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='test',
                                                    normal_channel=False)
    best_vloss = INFINITE

    for epoch in range(args.epochs):
        print(f'Starting epoch [{epoch}/{args.epochs}]')
        # create the data loader for train and validation data
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

        # train and retrieve training loss
        t_loss, t_acc = train_validate(simclr, classifier, optimizer, train_loader, args, is_train=True)

        # retrieve validation loss
        v_loss, v_acc = train_validate(simclr, classifier, optimizer, test_loader, args, is_train=False)

        print(f'\nTotal epoch losses: train: {round(t_loss,4)} - validation: {round(v_loss,4)}')
        print(f'\nTotal epoch acc: train: {round(t_acc,4)} - validation: {round(v_acc,4)}\n', end='\r')

        # if the current loss is the new best, update checkpoint
        if v_loss < best_vloss:
            best_vloss = v_loss
            # save_checkpoint(v_loss, epoch, simclr, optimizer, scheduler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SIMCLR')

    parser.add_argument('--dataset-name', type=str, default='CIFAR10',
                        help='Name of dataset (default: CIFAR10')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to dataset (default: data')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3')
    parser.add_argument("--decay-lr", default=1e-6, action="store", type=float,
                        help='Learning rate decay (default: 1e-6')
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

    simclr = pointnet2_cls_ssg_contrastive.get_model(num_class=40, normal_channel=False).type(dtype)
    # do not train it anymore
    simclr.eval()

    if os.path.isfile(f'{args.log_dir}/{args.checkpoint}'):
       checkpoint = torch.load(f'{args.log_dir}/{args.checkpoint}')
       simclr.load_state_dict(checkpoint['model'])
       epoch = checkpoint['epoch']
       print(f'Loading model: {args.checkpoint}, from epoch: {epoch}')
    else:
       print('Trained model not found!')

    classifier = pointnet2_cls_ssg_contrastive.ClassifierHead(args).type(dtype)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.decay_lr)

    run_trainer_evaluator(simclr, classifier, optimizer, args)
