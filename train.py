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

# Set the default device to k80
# torch.cuda.device(1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True,
                        help='path to training dataset')
    parser.add_argument('--valid_data', required=True,
                        help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int,
                        default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)
    parser.add_argument('--valInterval', type=float, default=0.25,
                        help='Validate every x% of the training set')
    parser.add_argument('--numEpoch', type=int, default=10,
                        help='Max num of epochs')
    parser.add_argument('--FT', action='store_true',
                        help='whether to do fine-tuning')
    parser.add_argument('--ckpt', type=str, default=None,
                        help="[If FT] path to load the ckpt model")
    parser.add_argument('--lr', type=float, default=1,
                        help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 for adam. default=0.9')
    parser.add_argument('--grad_clip', type=float, default=5,
                        help='gradient clipping value. default=5')

    parser = Model.add_model_specific_args(parser)
    parser = BatchBalancedDataModule.add_argparse_args(parser)
    opt = parser.parse_args()

    # load the right converter for the model
    # opt.character = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
    opt.character = "0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"

    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    # fix: from readable arg to usable data
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
 
    opt.select_valid = opt.select_valid.split('-')
    opt.batch_valid = opt.batch_valid.split('-')
    
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

    sampleImg = torch.rand(1, 1, 32, 100, device=device)
    sampleTxt = torch.full((1, 26), 0, device=device)
    custom_opts = opt

    logger.experiment.add_image("images", grid, 0)

    r = model(sampleImg, sampleTxt)

    logger.experiment.add_graph(
        Model(custom_opts).to(device), (sampleImg, sampleTxt))

    # BENCHMARK DIFFERENT HYPERPARAMETERS

    # clean hparams

    logger.experiment.add_hparams(vars(model.hparams), {})

    # torch_profiler = PyTorchProfiler(emit_nvtx=True)

    trainer = Trainer.from_argparse_args(
        opt,
        gpus=[0],
        fast_dev_run=True,
        resume_from_checkpoint=resume,
        logger=logger,
        # profiler=True,
        # profiler=torch_profiler,
        # auto_lr_find=True,
        # limit_train_batches=10,
        max_epochs=opt.numEpoch,
        # doubles the batch size, without doubling it's VRAM footprint
        # to fake a bigger batch_size -> dired_batch_size / current_batch_size
        # accumulate_grad_batches=2,
        precision=16,  # uses 16bit floats instead of 32, less RAM usage, faster, without real performance decrease
        # auto_scale_batch_size='power'
        accelerator='ddp',
        distributed_backend='ddp',

        #  deterministic=True,
        auto_scale_batch_size='binsearch',
        # auto_lr_find=True,
        gradient_clip_val=opt.grad_clip,
        val_check_interval=opt.valInterval,

    )

    # lr_finder = trainer.tuner.lr_find(model, dm)
    # print(f"Best lr value: {lr_finder.suggestion()} ")
    # print(lr_finder.results)
    # fig = lr_finder.plot()
    # fig.show()
   # enable cudnn benchmark
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    trainer.fit(model, dm)
