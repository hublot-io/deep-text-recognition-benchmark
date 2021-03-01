import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(LightningModule):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.attention_cell = AttentionCell(
            input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):

        input_char = input_char.unsqueeze(1)  # .to(device)
        batch_size = input_char.size(0)
        one_hot = torch.empty(
            batch_size, onehot_dim, device=self.device).zero_()  # .to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.zeros(
            batch_size, num_steps, self.hidden_size, device=self.device, dtype=torch.float)  # .to(device)
        hidden = (torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=torch.float),  # .to(device),
                  torch.zeros(batch_size, self.hidden_size, device=self.device, dtype=torch.float))  # .to(device))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(
                    text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(
                    hidden, batch_H, char_onehots)
                # LSTM hidden index (0: hidden, 1: Cell)
                output_hiddens[:, i, :] = hidden[0]
            probs = self.generator(output_hiddens)

        else:
            # .to(device)  # [GO] token
            targets = torch.zeros(
                batch_size, device=self.device, dtype=torch.long)
            probs = torch.zeros(
                batch_size, num_steps, self.num_classes, device=self.device, dtype=torch.float)  # .to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(
                    hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(LightningModule):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        # either i2i or h2h should have bias
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        # batch_size x num_encoder_step * 1
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(
            1)  # batch_size x num_channel
        # batch_size x (num_channel + num_embedding)
        concat_context = torch.cat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
