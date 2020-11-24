import os
import time

import torch
from tqdm import tqdm

from train.build import build_logger, build_config, build_hscnnr, build_dataset
from test import test
from train import train


def exists_file(file):
    return os.path.isfile(file)


def exists_dir(dir):
    return os.path.exists(dir)


def zfill(s, width=6):
    return str(s).zfill(width)


def bfill(s, width=6):
    return f'{str(s):>{width}}'


if __name__ == '__main__':
    cfg_path = 'config.ini'
    cfg = build_config(cfg_path)
    logger = build_logger(cfg)
    with open(cfg_path, "r") as f:
        logger.info('\n====================content of config file====================\n'
                    + f.read()
                    + '\n==============================================================')
    TRAIN_SECTION = 'Train'
    SETTING_SECTION = 'Setting'

    n_fold = 5
    save_path = cfg.get(TRAIN_SECTION, 'net_path')
    logger.info(f'using {n_fold}-fold cross validation')
    epochs = cfg.getint(TRAIN_SECTION, 'epochs')
    log_epoch_interval = cfg.getint(SETTING_SECTION, 'log_epoch_interval')

    final_rmse, final_rmse_g, final_rrmse, final_rrmse_g = 0.0, 0.0, 0.0, 0.0
    for i_fold in range(5, n_fold + 1):
        logger.info(f'current fold：{zfill(i_fold)}/{zfill(n_fold)}')
        net = build_hscnnr(cfg)

        trained = False
        i_net_path = f'{save_path}\\fold_{zfill(i_fold)}_net.hscnnr'
        # resuming model from file
        if cfg.getboolean(SETTING_SECTION, 'resume'):
            if exists_file(i_net_path):
                logger.info(f'resuming net model from file:{i_net_path}, training skipped')
                net.load(i_net_path)
                trained = True
            else:
                logger.info(f'net model was not found in file:{i_net_path}, and training will be taken')
        # build test data
        test_data = build_dataset(cfg, "test", i_fold, n_fold)
        test_loader = test_data.get_dataloader(1, False)
        # train
        if not trained:
            iteration = 0
            # build train data
            train_data = build_dataset(cfg, "train", i_fold, n_fold)
            train_loader = train_data.get_dataloader(cfg.getint(TRAIN_SECTION, 'batch_size'), False)
            # loss_func = torch.nn.MSELoss()
            def rrmse_loss(outputs, label):
                """Computes the rrmse value"""
                error = torch.abs(outputs - label) / (label + 0.01)
                rrmse = torch.mean(error.view(-1))
                return rrmse

            start_time = time.time()
            logger.info('start training')
            for epoch in tqdm(range(1, epochs + 1)):
                avg_loss, iteration, lr = train(net, train_loader, rrmse_loss, iteration)
                if epoch % log_epoch_interval == 0:
                    rmse, rmse_g, rrmse, rrmse_g = test(net, test_loader)
                    logger.info(
                        f'epoch of {zfill(epoch)}/{zfill(epochs)}: \n{bfill("iter")}:{iteration:9}, {bfill("loss")}:{avg_loss:9.4f}, {bfill("lr")}:{lr}\n'
                        f'{bfill("rmse")}:{rmse:9.4f}, {bfill("rmse_g")}:{rmse_g:9.4f}, {bfill("rrmse")}:{rrmse:9.4f}, {bfill("rrmse_g")}:{rrmse_g:9.4f}')
            logger.info(f'training ended, time consuming : {time.time() - start_time}s')
            # save model to file
            if cfg.getboolean(SETTING_SECTION, 'auto_save'):
                logger.info(f'saving net model to file:{i_net_path}')
                if not exists_dir(save_path):
                    os.makedirs(save_path)
                net.save(i_net_path)
        rmse, rmse_g, rrmse, rrmse_g = test(net, test_loader)
        final_rmse += rmse / n_fold
        final_rmse_g += rmse_g / n_fold
        final_rrmse += rrmse / n_fold
        final_rrmse_g += rrmse_g / n_fold
        logger.info(f'final epoch : {bfill("loss")}:{avg_loss:9.4f},  rmse:{rmse}, rmse_g:{rmse_g}, rrmse:{rrmse}, rrmse_g:{rrmse_g}')
    logger.info(f'final test result : rmse:{rmse}, rmse_g:{rmse_g}, rrmse:{rrmse}, rrmse_g:{rrmse_g}')
