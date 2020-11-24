from __future__ import division

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import os
import time

from train.build import build_dataset, build_config
from train import resblock, resblock_256
from train.utils import AverageMeter, initialize_logger, save_checkpoint, record_loss
from train.loss import rrmse_loss


def main():
    cudnn.benchmark = True

    # Config
    cfg_path = 'config.ini'
    cfg = build_config(cfg_path)

    # Dataset
    # train_data = DatasetFromHdf5('./train.h5')
    train_data = build_dataset(cfg, type="train", kfold_th=4, kfold=4)
    print(len(train_data))
    # val_data = DatasetFromHdf5('./test.h5')
    val_data = build_dataset(cfg, type="test", kfold_th=4, kfold=4)
    print(len(val_data))

    # Data Loader (Input Pipeline)
    train_data_loader = DataLoader(dataset=train_data,
                                   num_workers=1,
                                   batch_size=cfg.getint('Train', 'batch_size'),
                                   shuffle=True,
                                   pin_memory=True)
    val_loader = DataLoader(dataset=val_data,
                            num_workers=1,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True)

    # Model
    net_type = cfg.get('Train', 'net_type')
    if net_type == 'n16_64':
        model = resblock.resblock(resblock.conv_relu_res_relu_block, 16, 3, 31)
    elif net_type == 'n16_256':
        model = resblock_256.resblock(resblock_256.conv_relu_res_relu_block, 16, 3, 31)
    elif net_type == 'n14':
        model = resblock.resblock(resblock.conv_relu_res_relu_block, 14, 3, 31)
    else:
        raise RuntimeError('unsupported net type:%s' % net_type)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()

    # Parameters, Loss and Optimizer
    start_epoch = 0
    end_epoch = cfg.getint('Train', 'end_epoch')
    init_lr = cfg.getfloat('Train', 'init_lr')
    iteration = 0
    record_test_loss = 1000
    criterion = rrmse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    model_path = cfg.get('Train', 'model_path')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    loss_csv = open(os.path.join(model_path, 'loss.csv'), 'w+')

    log_dir = os.path.join(model_path, 'train.log')
    logger = initialize_logger(log_dir)
    # Print config
    with open(cfg_path, "r") as f:
        logger.info('\n====================content of config file====================\n'
                    + f.read()
                    + '\n==============================================================')

    # Resume
    resume_file = ''
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch + 1, end_epoch):

        start_time = time.time()
        train_loss, iteration, lr = train(train_data_loader, model, criterion, optimizer, iteration, init_lr, end_epoch)
        test_loss = validate(val_loader, model, criterion)

        # Save model
        if test_loss < record_test_loss:
            record_test_loss = test_loss
            save_checkpoint(model_path, epoch, iteration, model, optimizer)

        # print loss
        end_time = time.time()
        epoch_time = end_time - start_time
        print("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " % (
            epoch, iteration, epoch_time, lr, train_loss, test_loss))
        # save loss
        record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss)
        logger.info("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " % (
            epoch, iteration, epoch_time, lr, train_loss, test_loss))


# Training
def train(train_data_loader, model, criterion, optimizer, iteration, init_lr, end_epoch):
    losses = AverageMeter()
    model.train()
    for i, (images, labels) in enumerate(train_data_loader):
        labels = labels.cuda()
        images = images.cuda()

        # Decaying Learning Rate

        lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=968000, power=1.5)
        iteration = iteration + 1
        # Forward + Backward + Optimize
        output = model(images)

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        #  record loss
        losses.update(loss.item())

    return losses.avg, iteration, lr


# Validate
def validate(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            #  record loss
            losses.update(loss.item())

    return losses.avg


# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr * (1 - iteraion / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    main()
