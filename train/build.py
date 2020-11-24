import logging
import os
import time
from configparser import ConfigParser

from train.dataset import dataset


def build_config(config_path):
    """
    读取配置文件
    :param config_path:
    :return:
    """
    assert os.path.isfile(config_path)
    config = ConfigParser()
    config.read(config_path)
    return config


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def build_logger(config):
    SECTION = 'Log'
    dir = mkdir(config.get(SECTION, 'out_path'))
    cur_time = time.strftime("%Y%m%d-%H%M%S"),
    cur_time = cur_time[0]
    format_str = '%(asctime)s - %(filename)s[line:%(lineno)d]-[%(levelname)s]: %(message)s'
    # 配置基本信息
    logging.basicConfig(filename=('%s/%s.log' % (dir, cur_time)),
                        format=format_str,
                        level=logging.INFO)
    # 定义一个Handler打印INFO及以上级别的日志到sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 设置日志打印格式
    formatter = logging.Formatter(format_str)
    console.setFormatter(formatter)
    # 将定义好的console日志handler添加到root logger
    logging.getLogger('').addHandler(console)
    return logging

def build_dataset(cfg, type, kfold_th=None, kfold=None):
    SECTION = 'Train'
    dataroot = cfg.get(SECTION, 'data_path')
    patch_interval = cfg.getint(SECTION, 'patch_interval')
    patch_size = cfg.getint(SECTION, 'patch_size')
    if type == "train":
        return dataset(dataroot, type, kfold_th, kfold, argument=True,
                       patch_interval=patch_interval,
                       patch_size=patch_size)
    elif type == "test":
        return dataset(dataroot, type, kfold_th, kfold, argument=False)
    elif type == "all":
        return dataset(dataroot, type, argument=False)
    pass
