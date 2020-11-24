import torch
from train.build import build_dataset, build_config
import numpy as np
from PIL import Image
from train import resblock, resblock_256
import os
from tqdm import tqdm
from test import test


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def bfill(s, width=6):
    return f'{str(s):>{width}}'


def last_path_name(name):
    delimiter_index = name.rfind('/')
    if (delimiter_index == -1):
        delimiter_index = name.rfind('\\')
    if (delimiter_index != -1):
        name = name[delimiter_index + 1:len(name)]
    return name


def main():
    model_path = './models/hscnn_5layer_dim10_765.pkl'
    cfg = build_config('config.ini')
    result_path = './out'

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
    save_point = torch.load(model_path)
    model_param = save_point['state_dict']
    model.load_state_dict(model_param)

    test_data = build_dataset(cfg, type="test", kfold_th=5, kfold=5)
    model = model.cuda()
    model.eval()

    # test
    rmse, rmse_g, rrmse, rrmse_g = test(model, test_data)
    print(
        f'{bfill("rmse")}:{rmse:9.4f}, {bfill("rmse_g")}:{rmse_g:9.4f}, {bfill("rrmse")}:{rrmse:9.4f}, {bfill("rrmse_g")}:{rrmse_g:9.4f}')

    # generate spe imgs
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_data.get_dataloader(1, False)):
            out = model(images.cuda())
            img_res = out.cpu().numpy() * 65535
            # shape from [1,C,H,W] to [C,H,W]
            img_res = np.squeeze(img_res, axis=0)
            # format right to  image data
            img_res_limits = np.minimum(img_res, 65535)
            img_res_limits = np.maximum(img_res_limits, 0)
            # shape from [C,H,W] to [H,W,C]
            arr = img_res_limits.transpose(1, 2, 0)
            rgb_name = last_path_name(test_data.rgb_names[i])
            rgb_name = rgb_name[0:rgb_name.rfind('_')]
            mkdir(f'{result_path}/{rgb_name}')
            for ch in tqdm(range(31), desc=f'generating {rgb_name} spe files'):
                img_arr = arr[:, :, ch]
                img = Image.fromarray(img_arr.astype(np.uint16), mode="I;16")
                img.save(f'{result_path}/{rgb_name}/{rgb_name}_ms_{str(ch + 1).zfill(2)}.png', 'png')
                # img.show()


if __name__ == '__main__':
    main()
