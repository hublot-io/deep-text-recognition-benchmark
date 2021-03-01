import os
import sys
import time
import random
import string
import argparse
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import CTCLabelConverter, AttnLabelConverter, Averager
# from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from dataset import BatchBalancedDataModule, Batch_Balanced_Dataset, AlignCollate, hierarchical_dataset
from model import Model
from test import validation
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torchvision


def run_banchmark(opt):
    dm = BatchBalancedDataModule(opt)
    print(
        f"Lets init the dataloader, batchsize = {opt.batch_size}, workers: {opt.workers}")
    seed = opt.manualSeed
    seed_everything(seed)
    model = Model(opt)

    logger = TensorBoardLogger("lightning_logs", name="ocr")
    # logger.experiment.add_hparams(vars(model.hparams), {})
    trainer = Trainer.from_argparse_args(
        opt,
        gpus=1,
        logger=logger,
        profiler="simple",
        # limit_train_batches=10,
        max_epochs=1,
        # doubles the batch size, without doubling it's VRAM footprint
        # to fake a bigger batch_size -> dired_batch_size / current_batch_size
        accumulate_grad_batches=1,
        precision=16,  # uses 16bit floats instead of 32, less RAM usage, faster, without real performance decrease
        distributed_backend='ddp',
        deterministic=True,
        val_check_interval=opt.valInterval


    )

    # trainer.tune(model, datamodule=dm)
    trainer.fit(model, dm)
    accuracy = model.last_accuracy
    shown_params = {
        "lr": opt["lr"],
        "num_fiducial": opt["num_fiducial"],
        "output_channel": opt["output_channel"],
        "hidden_size": opt["hidden_size"],
        "manualSeed": opt["manualSeed"]
    }

    logger.experiment.add_hparams(shown_params, {"hparam/accuracy": accuracy})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True,
                        help='path to training dataset')
    parser.add_argument('--valid_data', required=True,
                        help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int,
                        default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)

    # parser.add_argument('--batch_size', type=int,
    #                     default=192, help='input batch size')

    parser.add_argument('--num_iter', type=int, default=300000,
                        help='number of iterations to train for')
    parser.add_argument('--valInterval', type=float, default=0.25,
                        help='Validate every x% of the training set')
    parser.add_argument('--saved_model', default='',
                        help="path to model to continue training")
    parser.add_argument('--FT', action='store_true',
                        help='whether to do fine-tuning')
    parser.add_argument('--ckpt', type=str, default=None,
                        help="[If FT] path to load the ckpt model")
    parser.add_argument('--adam', action='store_true',
                        help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1,
                        help='learning rate, default=1.0 for Adadelta')

    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 for adam. default=0.9')

    parser.add_argument('--rho', type=float, default=0.95,
                        help='decay rate rho for Adadelta. default=0.95')

    parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps for Adadelta. default=1e-8')

    parser.add_argument('--grad_clip', type=float, default=5,
                        help='gradient clipping value. default=5')

    parser = Model.add_model_specific_args(parser)
    parser = BatchBalancedDataModule.add_argparse_args(parser)
    opt = parser.parse_args()

    # load the right converter for the model
    opt.character = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    # fix: from readable arg to usable data
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    dm = BatchBalancedDataModule(opt)
    print(
        f"Lets init the dataloader, batchsize = {opt.batch_size}, workers: {opt.workers}")
    dt = DataLoader(dm, batch_size=opt.batch_size,
                    num_workers=opt.workers)

    # ---------- visualize input data
    _AlignCollate = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root="data_lmdb_release/training/FR",
        opt=opt,
    )
    dl = DataLoader(valid_dataset, batch_size=6, shuffle=True,
                    num_workers=opt.workers, collate_fn=_AlignCollate)
    batch = next(iter(dl))
    imgs = [r for r in batch[0]]
    grid = torchvision.utils.make_grid(imgs)

    seed = opt.manualSeed

    print(f"FT  {opt.FT}, ckpt {opt.ckpt}")
    resume = None
    if (opt.FT is True and opt.ckpt is not None):
        print(f"Loaded model from {opt.ckpt}")
        resume = opt.ckpt
        model = Model.load_from_checkpoint(opt.ckpt)
    else:
        # seed_everything(seed)
        model = Model(opt)
    logger = TensorBoardLogger("lightning_logs", name="ocr")

    # sampleImg = batch[0][0]
    # sampleTxt = batch[1][0]
    sampleImg = torch.rand(1, 1, 32, 100, device=device)
    sampleTxt = torch.full((1, 26), 0, device=device)
    custom_opts = opt
    # custom_opts.transformation = None
    logger.experiment.add_image("images", grid, 0)

    r = model(sampleImg, sampleTxt)

    # from torchviz import make_dot
    # print(f"model result {r} -> mean = {r.mean()}")
    # dot = make_dot(r.mean(), params=dict(model.named_parameters()))
    # dot.save()
    # print("done rendering")
    logger.experiment.add_graph(
        Model(custom_opts).to(device), (sampleImg, sampleTxt))

    # BENCHMARK DIFFERENT HYPERPARAMETERS

    # clean hparams

    logger.experiment.add_hparams(vars(model.hparams), {})

    # torch_profiler = PyTorchProfiler(emit_nvtx=True)
    trainer = Trainer.from_argparse_args(
        opt,
        gpus=1,
        resume_from_checkpoint=resume,
        logger=logger,

        # profiler="pytorch",
        # profiler=torch_profiler,

        # limit_train_batches=10,
        # max_epochs=1,
        # doubles the batch size, without doubling it's VRAM footprint
        # to fake a bigger batch_size -> dired_batch_size / current_batch_size
        accumulate_grad_batches=4,
        precision=16,  # uses 16bit floats instead of 32, less RAM usage, faster, without real performance decrease
        # auto_scale_batch_size='power'
        distributed_backend='ddp',
        # benchmark=True,
        #  deterministic=True,
        auto_scale_batch_size='binsearch',
        # auto_lr_find=True,
        gradient_clip_val=opt.grad_clip,
        val_check_interval=opt.valInterval
    )

    # lr_finder = trainer.tuner.lr_find(model, datamodule=dm)
    # # fig = lr_finder.plot(suggest=True)
    # # fig.show()
    # new_lr = lr_finder.suggestion()
    # print(f"Found a good starting learning rate: {new_lr}")
    # model.hparams.lr = new_lr
    # model.opt.lr = new_lr

    # trainer.tune(model, datamodule=dm)
   # enable cudnn benchmark
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    trainer.fit(model, dm)

    # torch.jit.save(model.to_torchscript(), "model.pt")

    # if not opt.exp_name:
    #     opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
    #     opt.exp_name += f'-Seed{opt.manualSeed}'
    #     # print(opt.exp_name)

    # os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    # """ vocab / character number configuration """
    # if opt.sensitive:
    #     # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    #     # same with ASTER setting (use 94 char).
    #     opt.character = string.printable[:-6]
    #     # opt.character = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
    # """ Seed and GPU setting """
    # # print("Random Seed: ", opt.manualSeed)
    # random.seed(opt.manualSeed)
    # np.random.seed(opt.manualSeed)
    # torch.manual_seed(opt.manualSeed)
    # torch.cuda.manual_seed(opt.manualSeed)
    # print("MODEL SEEDED")
    # cudnn.benchmark = True
    # cudnn.deterministic = True
    # opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    # if opt.num_gpu > 1:
    #     print('------ Use multi-GPU setting ------')
    #     print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
    #     # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
    #     opt.workers = opt.workers * opt.num_gpu
    #     opt.batch_size = opt.batch_size * opt.num_gpu

    #     """ previous version
    #     print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
    #     opt.batch_size = opt.batch_size * opt.num_gpu
    #     print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
    #     If you dont care about it, just commnet out these line.)
    #     opt.num_iter = int(opt.num_iter / opt.num_gpu)
    #     """
    # print("started training, with opts: {}".format(opt))
    # train(opt)
