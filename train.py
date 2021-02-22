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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from pytorch_lightning.accelerators.ddp_accelerator import DDPAccelerator
import torchvision
# def train(opt):
#     """ dataset preparation """
#     if not opt.data_filtering_off:
#         print('Filtering the images containing characters which are not in opt.character')
#         print('Filtering the images whose label is longer than opt.batch_max_length')
#         # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

#     opt.select_data = opt.select_data.split('-')
#     opt.batch_ratio = opt.batch_ratio.split('-')
#     train_dataset = Batch_Balanced_Dataset(opt)

#     log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
#     print("opened log file")
#     AlignCollate_valid = AlignCollate(
#         imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
#     valid_dataset, valid_dataset_log = hierarchical_dataset(
#         root=opt.valid_data, opt=opt)
#     valid_loader = torch.utils.data.DataLoader(
#         valid_dataset, batch_size=opt.batch_size,
#         # 'True' to check training progress with validation function.
#         shuffle=True,
#         num_workers=int(opt.workers),
#         collate_fn=AlignCollate_valid, pin_memory=True)
#     log.write(valid_dataset_log)
#     print('-' * 80)
#     log.write('-' * 80 + '\n')
#     log.close()

#     """ model configuration """
#     if 'CTC' in opt.Prediction:
#         converter = CTCLabelConverter(opt.character)
#     else:
#         converter = AttnLabelConverter(opt.character)
#     opt.num_class = len(converter.character)

#     if opt.rgb:
#         opt.input_channel = 3
#     model = Model(opt)
#     print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
#           opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
#           opt.SequenceModeling, opt.Prediction)

#     # weight initialization
#     for name, param in model.named_parameters():
#         if 'localization_fc2' in name:
#             print(f'Skip {name} as it is already initialized')
#             continue
#         try:
#             if 'bias' in name:
#                 init.constant_(param, 0.0)
#             elif 'weight' in name:
#                 init.kaiming_normal_(param)
#         except Exception as e:  # for batchnorm.
#             if 'weight' in name:
#                 param.data.fill_(1)
#             continue

#     # data parallel for multi-GPU
#     model = torch.nn.DataParallel(model).to(device)
#     model.train()
#     if opt.saved_model != '':
#         print(f'loading pretrained model from {opt.saved_model}')
#         if opt.FT:
#             model.load_state_dict(torch.load(
#                 opt.saved_model, map_location=torch.device('cpu')), strict=False)
#         else:
#             model.load_state_dict(torch.load(opt.saved_model))
#     print("Model:")
#     print(model)

#     """ setup loss """
#     if 'CTC' in opt.Prediction:
#         criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
#     else:
#         criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(
#             device)  # ignore [GO] token = ignore index 0
#     # loss averager
#     loss_avg = Averager()

#     # filter that only require gradient decent
#     filtered_parameters = []
#     params_num = []
#     for p in filter(lambda p: p.requires_grad, model.parameters()):
#         filtered_parameters.append(p)
#         params_num.append(np.prod(p.size()))
#     print('Trainable params num : ', sum(params_num))
#     # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

#     # setup optimizer
#     if opt.adam:
#         optimizer = optim.Adam(filtered_parameters,
#                                lr=opt.lr, betas=(opt.beta1, 0.999))
#     else:
#         optimizer = optim.Adadelta(
#             filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
#     print("Optimizer:")
#     print(optimizer)

#     """ final options """
#     # print(opt)
#     with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
#         opt_log = '------------ Options -------------\n'
#         args = vars(opt)
#         for k, v in args.items():
#             opt_log += f'{str(k)}: {str(v)}\n'
#         opt_log += '---------------------------------------\n'
#         print(opt_log)
#         opt_file.write(opt_log)

#     """ start training """
#     start_iter = 0
#     if opt.saved_model != '':
#         try:
#             start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
#             print(f'continue to train, start_iter: {start_iter}')
#         except:
#             pass

#     start_time = time.time()
#     best_accuracy = -1
#     best_norm_ED = -1
#     iteration = start_iter
#     pbar = tqdm(total=opt.valInterval)
#     while(True):
#         # train part
#         pbar.update(1)
#         image_tensors, labels = train_dataset.get_batch()
#         image = image_tensors.to(device)
#         text, length = converter.encode(
#             labels, batch_max_length=opt.batch_max_length)
#         batch_size = image.size(0)

#         if 'CTC' in opt.Prediction:
#             preds = model(image, text)
#             preds_size = torch.IntTensor([preds.size(1)] * batch_size)
#             preds = preds.log_softmax(2).permute(1, 0, 2)
#             cost = criterion(preds, text, preds_size, length)

#         else:
#             preds = model(image, text[:, :-1])  # align with Attention.forward
#             target = text[:, 1:]  # without [GO] Symbol
#             cost = criterion(
#                 preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

#         model.zero_grad()
#         cost.backward()
#         # gradient clipping with 5 (Default)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
#         optimizer.step()

#         loss_avg.add(cost)

#         # validation part
#         # To see training progress, we also conduct validation when 'iteration == 0'
#         if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
#             pbar = tqdm(total=opt.valInterval)
#             elapsed_time = time.time() - start_time
#             # for log
#             with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
#                 model.eval()
#                 with torch.no_grad():
#                     valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
#                         model, criterion, valid_loader, converter, opt)
#                 model.train()

#                 # training loss and validation loss
#                 loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
#                 loss_avg.reset()

#                 current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'
#                 writer.add_scalar("Loss/train", cost, iteration)
#                 writer.add_scalar("Accuracy/train",
#                                   current_accuracy, iteration)

#                 # keep best accuracy model (on valid dataset)
#                 if current_accuracy > best_accuracy:
#                     best_accuracy = current_accuracy
#                     torch.save(
#                         model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
#                 if current_norm_ED > best_norm_ED:
#                     best_norm_ED = current_norm_ED
#                     torch.save(model.state_dict(),
#                                f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
#                 best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

#                 loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
#                 print(loss_model_log)
#                 log.write(loss_model_log + '\n')

#                 # show some predicted results
#                 dashed_line = '-' * 80
#                 head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
#                 predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
#                 for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
#                     if 'Attn' in opt.Prediction:
#                         gt = gt[:gt.find('[s]')]
#                         pred = pred[:pred.find('[s]')]

#                     predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
#                 predicted_result_log += f'{dashed_line}'
#                 print(predicted_result_log)
#                 log.write(predicted_result_log + '\n')

#         # save model per 1e+5 iter.
#         if (iteration + 1) % 2e+4 == 0:
#             torch.save(
#                 model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

#         if (iteration + 1) == opt.num_iter:
#             print('end the training')
#             sys.exit()
#         iteration += 1


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
        root="lmdb",
        opt=opt,
    )
    dl = DataLoader(valid_dataset, batch_size=6,
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
    sampleImg = torch.rand(1, 1, 32, 100).to(device)
    sampleTxt = torch.full((1, 26), 0).to(device)
    custom_opts = opt
    # custom_opts.transformation = None
    logger.experiment.add_image("images", grid, 0)

    logger.experiment.add_graph(
        Model(custom_opts).to(device), (sampleImg, sampleTxt))

    # BENCHMARK DIFFERENT HYPERPARAMETERS

    # clean hparams

    logger.experiment.add_hparams(vars(model.hparams), {})
    trainer = Trainer.from_argparse_args(
        opt,
        gpus=1,
        resume_from_checkpoint=resume,
        logger=logger,
        profiler="simple",
        # limit_train_batches=10,
        # max_epochs=10,
        # doubles the batch size, without doubling it's VRAM footprint
        # to fake a bigger batch_size -> dired_batch_size / current_batch_size
        accumulate_grad_batches=4,
        precision=16,  # uses 16bit floats instead of 32, less RAM usage, faster, without real performance decrease
        # auto_scale_batch_size='power'
        distributed_backend='ddp',
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
    torch.backends.cudnn.benchmark = True
    trainer.fit(model, dm)

    torch.jit.save(model.to_torchscript(), "model.pt")

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
