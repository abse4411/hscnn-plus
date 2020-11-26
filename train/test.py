import torch
from train.evalute import compare_rmse_g, compare_rrmse_g, compare_rrmse, compare_rmse


def bfill(s, width=6):
    return f'{str(s):>{width}}'


def test(model, dataset):
    model.eval()
    cnt = len(dataset)
    rmse_avg, rmse_g_avg, rrmse_avg, rrmse_g_avg = 0.0, 0.0, 0.0, 0.0
    for i, (rgb, spe) in enumerate(dataset.get_dataloader(1, False)):
        rgb = rgb.cuda()
        spe = spe.cuda()
        with torch.no_grad():
            pred = model(rgb)
            pred = pred * 255
            spe = spe * 255

            rmse = compare_rmse(pred, spe)
            rmse_g = compare_rmse_g(pred, spe)
            rrmse = compare_rrmse(pred, spe)
            rrmse_g = compare_rrmse_g(pred, spe)
            print(
                f'{bfill("rmse")}:{rmse:9.4f}, {bfill("rmse_g")}:{rmse_g:9.4f}, {bfill("rrmse")}:{rrmse:9.4f}, {bfill("rrmse_g")}:{rrmse_g:9.4f}')
            rmse_avg += rmse / cnt
            rmse_g_avg += rmse_g / cnt
            rrmse_avg += rrmse / cnt
            rrmse_g_avg += rrmse_g / cnt
    return rmse_avg, rmse_g_avg, rrmse_avg, rrmse_g_avg
