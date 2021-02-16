"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from pytorch_lightning import LightningModule
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from argparse import ArgumentParser
import torch
from utils import CTCLabelConverter, AttnLabelConverter, Averager
from torch.nn import functional as F
import re
from os import environ
if environ.get("TRAIN") is "True":
    from test import validation
from torch.utils.tensorboard import SummaryWriter
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def matplotlib_imshow(img, one_channel=False):
    transform = torchvision.transforms.ToPILImage()
    pil = transform(img.cpu().squeeze(1))
    npimg = np.asarray(pil)
    # if one_channel:
    #     img = img.mean(dim=0)
    # npimg = npimg / 2 + 0.5
    # npimg = img.cpu().numpy()

    plt.imshow(npimg)


class Model(LightningModule):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.last_accuracy = 0
        self.training_losses = []
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(
                opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(
                opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(
                opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        # int(imgH/16-1) * 512
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output,
                                  opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(
                self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(
                self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')
        self.converter = AttnLabelConverter(opt.character)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.lr = self.opt.lr
        self.save_hyperparameters()

        # self.logger.experiment.add_hparams(self.hparams)

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(
            visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            # for convenience. this is NOT contextually modeled by BiLSTM
            contextual_feature = visual_feature

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(
            ), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction

    def training_step(self, batch, batch_idx):
        opt = self.opt
        converter = self.converter
        # we are working on several dataset, so our batch is an array of batch
        image_tensors, labels = batch
        image = image_tensors.to(device)
        text, length = converter.encode(
            labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)
        # align with Attention.forward
        preds = self(image, text[:, :-1]).to(device)
        target = text[:, 1:].to(device)  # without [GO] Symbol
        total = len(labels)
        length_for_pred = torch.IntTensor(
            [opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(
            batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = self.converter.encode(
            labels,
            batch_max_length=opt.batch_max_length)

        preds = preds[:, :text_for_loss.shape[1] - 1, :]
        target = text_for_loss[:, 1:].to(device)  # without [GO] Symbol

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, length_for_pred)
        correct = len(
            list(filter(lambda q: q[0].replace("[s]", "") == q[1], zip(preds_str, list(labels)))))
        cost = self.criterion(
            preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        self.logger.experiment.add_scalar(
            "Train/Loss", cost, global_step=self.global_step)
        batch_dict = {
            "loss": cost,
            "correct": correct,
            "total": total
        }
        return batch_dict

    def training_epoch_end(self, batch_outputs):
        avg_loss = torch.stack([x['loss'].float()
                                for x in batch_outputs]).mean()
        correct = sum([x["correct"] for x in batch_outputs])
        total = sum([x["total"] for x in batch_outputs])
        self.logger.experiment.add_scalar(
            "Train/Loss", avg_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar(
            "Train/Valid", correct / total, global_step=self.global_step)

    def validation_step(self, batch, batch_idx):
        opt = self.opt
        converter = self.converter
        image_tensors, labels = batch
        image = image_tensors.to(device)

        return {
            'batch': batch,
            'transformed': (image, labels)
        }

    def validation_epoch_end(self, preds_output):
        if (environ.get("TRAIN") is "False"):
            pass
        else:
            opt = self.opt
            transformeds = [p['transformed'] for p in preds_output]
            batches = [p['batch'] for p in preds_output]
            with torch.no_grad():
                valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                    self, self.criterion, transformeds, self.converter, self.opt)
            # show some predicted results in the terminal
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'

            fig = plt.figure(figsize=(12, 48))
            i = 0
            for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                if 'Attn' in opt.Prediction:
                    gt = gt[:gt.find('[s]')]
                    pred = pred[:pred.find('[s]')]
                predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                # if (i < len(transformeds)):
                #     (image, label) = transformeds[i]
                #     ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
                #     matplotlib_imshow(image, one_channel=True)
                #     ax.set_title(
                #         f"pred={pred:25s}\nconf={confidence:0.4f}\n gt={gt:25s}", color=("green" if pred == gt else "red"))
                #     i += 1
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)
            # we can also show them in the Tensorboard UI
            # self.logger.experiment.add_figure(
            #     "Prediction vs GT", fig, global_step=self.global_step)
            self.logger.experiment.add_scalar(
                'Valid/Loss', valid_loss, global_step=self.global_step)
            self.logger.experiment.add_scalar(
                'Valid/Accuracy', current_accuracy, global_step=self.global_step)
            self.last_accuracy = current_accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=(self.lr or self.opt.lr))
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--Transformation', type=str,
                            required=True, help='Transformation stage. None|TPS')
        parser.add_argument('--FeatureExtraction', type=str, required=True,
                            help='FeatureExtraction stage. VGG|RCNN|ResNet')
        parser.add_argument('--SequenceModeling', type=str,
                            required=True, help='SequenceModeling stage. None|BiLSTM')
        parser.add_argument('--Prediction', type=str,
                            required=True, help='Prediction stage. CTC|Attn')
        parser.add_argument('--num_fiducial', type=int, default=20,
                            help='number of fiducial points of TPS-STN')
        parser.add_argument('--input_channel', type=int, default=1,
                            help='the number of input channel of Feature extractor')
        parser.add_argument('--output_channel', type=int, default=512,
                            help='the number of output channel of Feature extractor')
        parser.add_argument('--hidden_size', type=int, default=256,
                            help='the size of the LSTM hidden state')
        return parser
