import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    
def one_hot(tensors, num_classes):
    onehot = []
    tensors = tensors.cuda()
    for tensor in tensors:
        tensor = tensor.unsqueeze(1)
        t = torch.zeros(tensor.shape[0], num_classes).cuda().scatter_(1, tensor, 1)
        onehot.append(t)
    onehot = torch.stack(onehot)
    return onehot
class ACE_Loss(nn.Module):
    def __init__(self, alpha=0, size_average=False):
        super(ACE_Loss, self).__init__()
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, logits, targets, input_lengths, target_lengths):
        T_, B, C = logits.size()

        tagets_split = list(torch.split(targets, target_lengths.tolist()))
        targets_padded = torch.nn.utils.rnn.pad_sequence(tagets_split, batch_first=True, padding_value=0)
        targets_padded = one_hot(targets_padded.long(), num_classes=C)

        targets_padded = (targets_padded * (1-self.alpha)) + (self.alpha/C)
        targets_padded = torch.sum(targets_padded, 1).float().cuda()
        targets_padded[:,0] = T_ - target_lengths
        #print(logits)
        #print(type(logits))
        probs = torch.softmax(logits, dim=2)
        probs = torch.sum(probs, 0)
        probs = probs / T_
        targets_padded = targets_padded / T_

        #targets_padded = F.normalize(targets_padded, p=1, dim=1)
        #loss = F.kl_div(torch.log(probs), targets_padded, reduction='sum')

        #print(-torch.sum(torch.log(probs[0]) * targets_padded[0])) , (-torch.sum(torch.log(probs[1:]) * targets_padded[1:]))
        loss = -torch.sum(torch.log(probs) * targets_padded) / B

        return loss

    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        #self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()
        #here add grad-cam
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        #self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            #inputs = x.reshape(batch * temp, channel, height, width)
            #framewise = self.masked_bn(inputs, len_x)
            #framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            #"framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            
            if k == 'ConvCTC':
                c_loss=self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int())
                w = torch.exp(-c_loss)
                w = 1.0 - w
                w = 0.25*torch.square(w)
                f_loss = c_loss * w
                loss += weight * (f_loss.mean())           #print(self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),label.cpu().int(), ret_dict["feat_len"].cpu().int(),label_lgt.cpu().int()).mean())
                #print(self.loss['ACE'](ret_dict["conv_logits"].log_softmax(-1),label.cpu().int(), ret_dict["feat_len"].cpu().int(),label_lgt.cpu().int()).mean())
            elif k == 'SeqCTC':
                c_loss=self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int())
                w = torch.exp(-c_loss)
                w = 1.0 - w
                w = 0.25*torch.square(w)
                f_loss = c_loss * w
                loss += weight * (f_loss.mean())
                #print(self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),label.cpu().int(), ret_dict["feat_len"].cpu().int(),label_lgt.cpu().int()).mean())
                #print(self.loss['ACE'](ret_dict["conv_logits"].log_softmax(-1),label.cpu().int(), ret_dict["feat_len"].cpu().int(),label_lgt.cpu().int()).mean())
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['ACE']=ACE_Loss()
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
