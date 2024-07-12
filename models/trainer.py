# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co
 
 from typing import Optional

import torch
import torch.nn as nn

from timm.models import register_model

import copy

class Trainer(nn.Module):

    def __init__(self, generator: nn.Module, discriminators: nn.ModuleDict = None, supp_attr=[], pres_attr=[], eval_attr=[]):
        """
        """
        super().__init__()
        self.generator = generator
        self.discriminators = discriminators

        tmp = {}
        for attr, m in self.discriminators.items():
            if attr != 'infonce':
                tmp[attr] = copy.deepcopy(m)
            else:
                tmp[attr] = m
        self.discriminators_xp = nn.ModuleDict(tmp)

        self.supp_attr = supp_attr
        self.pres_attr = pres_attr
        self.eval_attr = eval_attr

        # preparation
        self.freeze_discriminators()
        self.train_and_unfreeze()

    def forward(self, x, training_dis):

        if not training_dis:
            # step 1: anonymized the x by the generator -> anonymized_x
            anonymized_x = self.anonymize(x)

            # step 2: feed anonymized_x to all attributes, e.g. id and gender
            _, outs, outs_feat = self.run_discriminators(x)
            _, outsp, outs_featp = self.run_discriminators_xp(
                anonymized_x)
            return anonymized_x, outs, outs_feat, None, None, outsp, outs_featp

        else:
            _, outsp, outs_featp = self.run_discriminators_xp(x)
            return outsp, outs_featp

    def anonymize(self, x):
        anonymized_x = self.generator(x)
        return anonymized_x

    def freeze_discriminators(self):
        for attr, m in self.discriminators.items():
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def freeze_discriminators_xp(self):
        for attr, m in self.discriminators_xp.items():
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_discriminators_xp(self):
        for attr, m in self.discriminators_xp.items():
            m.train()
            for p in m.parameters():
                p.requires_grad = True

    def freeze_infonce_discriminators(self):
        for attr, m in self.discriminators.items():
            if attr == 'infonce':
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def unfreeze_infonce_discriminators(self):
        for a, m in self.discriminators.items():
            if a == 'infonce':
                m.train()
                for p in m.parameters():
                    p.requires_grad = True

    def freeze_generator(self):
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False

    def unfreeze_generator(self):
        self.generator.train()
        for p in self.generator.parameters():
            p.requires_grad = True

    def train_and_unfreeze(self):
        self.unfreeze_discriminators_xp()
        self.unfreeze_infonce_discriminators()
        self.unfreeze_generator()

    def eval_and_freeze(self):
        self.freeze_discriminators_xp()
        self.freeze_infonce_discriminators()
        self.freeze_generator()

    def run_discriminators(self, x):
        outs = {}
        outs_feat = {}
        for attr, net in self.discriminators.items():
            outs[attr], outs_feat[attr] = net(x, feature=True)
        return x, outs, outs_feat

    def run_discriminators_xp(self, x):
        outs = {}
        outs_feat = {}
        for attr, net in self.discriminators_xp.items():
            outs[attr], outs_feat[attr] = net(x, feature=True)
        return x, outs, outs_feat

    def get_optimizer_params(self):
        return list(self.generator.parameters())+list(self.discriminators['infonce'].parameters())


@register_model
def gan_trainer(**kwargs):
    model = Trainer(kwargs['generator'], kwargs.pop(
        'discriminators'), kwargs['supp_attr'], kwargs['pres_attr'], 
        kwargs['eval_attr'])
    return model
