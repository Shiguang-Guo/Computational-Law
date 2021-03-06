#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/30 22:24
# @Author : gsg
# @Site : 
# @File : bet-fimodelune.py
# @Software: PyCharm
import argparse
import copy
import json
import re
import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
import torch.nn.functional as F

form_label_list = ["其他", "网上电子汇款", "现金", "未知或模糊", "网络贷款平台", "授权支配特定资金账户", "未出借", "银行转账", "票据"]
borrower_attr_label_list = ["其他组织", "自然人", "法人"]
use_label_list = ["个人生活", "其他", "夫妻共同生活", "违法犯罪", "企业生产经营"]
proof_label_list = ["其他", "未知或模糊", "还款承诺", "收据、收条", "欠条", "担保", "微信、短信、电话等聊天记录", "借款合同、借条、借据"]
lender_attr_label_list = ["其他组织", "自然人", "法人"]
intention_label_list = ["转贷牟利", "其他", "正常出借"]
type_label_list = ["无担保", "保证", "抵押", "质押"]
rate_label_list = ["其他", "24%（不含）-36%（含）", "24%（含）以下", "36%（不含）以上"]
calculation_method_label_list = ["其他", "无利息", "单利", "复利", "约定不明"]
repayment_label_list = ["网上电子汇款", "其他", "未知或模糊", "现金", "银行转账", "未还款", "部分还款", "票据"]

name_dict = {
    'form': "借款交付形式",
    'borrower_attr': "借款人基本属性",
    'use': "借款用途",
    'proof': "借贷合意的凭据",
    'lender_attr': "出借人基本属性",
    'intention': "出借意图",
    'type': "担保类型",
    'rate': "约定期内利率（换算成年利率）",
    'calculation_method': "约定计息方式",
    'repayment': "还款交付形式"
}


class Predictor:
    def __init__(self):
        pass

    def predict(self, fact):
        attr = {
            "借款交付形式": [],
            "借款人基本属性": [],
            "借款用途": [],
            "借贷合意的凭据": [],
            "出借人基本属性": [],
            "出借意图": [],
            "担保类型": [],
            "约定期内利率（换算成年利率）": [],
            "约定计息方式": [],
            "还款交付形式": []
        }
        return attr


class TaskDataset(Dataset):
    # name是要处理的任务的名字
    def __init__(self, raw_data, config):
        self.raw_data = raw_data
        self.name = config['name']
        self.chinesename = name_dict[self.name]
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.labels = np.array([self.onehot(item['attr'][self.chinesename], self.name) for item in self.raw_data])

        input_ids = []
        bert_attention_masks = []
        tokens = [data['fact'] for data in self.raw_data]
        for sens in tokens:
            encoded_dict = self.tokenizer.encode_plus(
                re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", sens),  # 输入文本
                add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                max_length=min(config['trunc_lenth'], 512),  # 填充 & 截断长度
                pad_to_max_length=True,
                return_attention_mask=True,  # 返回 attn. masks.
                return_tensors='pt',  # 返回 pytorch tensors 格式的数据
                truncation=True
            )
            input_ids.append(encoded_dict['input_ids'])
            bert_attention_masks.append(encoded_dict['attention_mask'])
        self.input_ids = torch.cat(input_ids, dim=0)
        self.bert_attention_masks = torch.cat(bert_attention_masks, dim=0)

    def onehot(self, label, name):
        label_list = eval(name + '_label_list')
        vector = np.zeros(len(label_list))
        for i in label:
            vector[label_list.index(i)] = 1
        return vector

    def __getitem__(self, item):
        return self.input_ids[item], self.bert_attention_masks[item], self.labels[item]

    def __len__(self):
        return self.labels.shape[0]


class Bert_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_nums = min(config['trunc_lenth'], 512)
        self.output_nums = len(eval(config['name'] + '_label_list'))
        self.bert = BertModel.from_pretrained(config['bert_path'])
        self.fc1 = nn.Linear(768, self.output_nums)
        self.LSTM = nn.LSTM(input_size=768, hidden_size=512, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(512 * 2, self.output_nums)

    def forward(self, input_id, mask):
        out = self.bert(input_id, attention_mask=mask)
        out, hidden = self.LSTM(out[0])
        return torch.mean(torch.sigmoid(self.linear(out)), dim=1)


if __name__ == '__main__':
    paresr = argparse.ArgumentParser()
    paresr.add_argument('--task', default='use', type=str)
    args = paresr.parse_args()
    config = {
        'seed': 1,
        'name': 'form',
        'batch_size': 32,
        'lr': 5e-5,
        'wd': 0.0005,
        'epoches': 1000,
        'trunc_lenth': 700,
        'bert_path': 'bert-base-chinese',
        'patience': 20,
        'log_dir': './log/'
    }
    config['name'] = args.task

    tb_writer = SummaryWriter(config['log_dir'] + config['name'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    with open('train.json', 'r', encoding='utf-8')as f:
        train_data = json.load(f)
    with open('dev.json', 'r', encoding='utf-8')as f:
        test_data = json.load(f)

    traindataset = TaskDataset(train_data, config)
    testdataset = TaskDataset(test_data, config)

    train_size = int(len(traindataset) * 0.8)
    valid_size = len(traindataset) - train_size
    traindataset, validdataset = torch.utils.data.random_split(traindataset, [train_size, valid_size])

    trainData = DataLoader(traindataset, batch_size=config['batch_size'], shuffle=True)
    validData = DataLoader(validdataset, batch_size=config['batch_size'], shuffle=True)
    testData = DataLoader(testdataset, batch_size=config['batch_size'], shuffle=False)

    model = Bert_Model(config).cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=config['lr'], weight_decay=config['wd'])

    best_loss = 1000000
    patience = 0
    best_model = None
    epoch_iter = tqdm(range(config['epoches']))
    for epoch in epoch_iter:
        patience += 1
        if patience == config['patience']:
            break
        model.train()
        train_losses, valid_losses = [], []
        for input_ids, attention_masks, labels in trainData:
            input_ids, attention_masks, labels = input_ids.cuda(), attention_masks.cuda(), labels.float().cuda()
            model.zero_grad()
            score = model(input_ids, attention_masks)
            train_loss = criterion(score, labels)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.cpu().data)

            # 模型评估
        model.eval()
        with torch.no_grad():
            for input_ids, attention_masks, labels in testData:
                input_ids, attention_masks, labels = input_ids.cuda(), attention_masks.cuda(), labels.float().cuda()
                valid_score = model(input_ids, attention_masks)
                valid_loss = criterion(valid_score, labels)
                valid_losses.append(valid_loss.cpu().data)
                label_pred = torch.zeros(size=labels.size())
                v = valid_score.cpu().data
                max_ = torch.max(v, dim=1)[0].unsqueeze(dim=1).expand(-1, labels.size(1))
                label_pred = torch.where(max_ - v <= 0.1, 1, 0)

                f1 = f1_score(labels.cpu().data, label_pred, average='macro')

        epoch_iter.set_description('epoch: %d, train loss: %.4f, test loss: %.4f, f1: %.2f,patience: %d' %
                                   (epoch, np.mean(train_losses), np.mean(valid_losses), f1,
                                    patience))

        tb_writer.add_scalar('train_loss', np.mean(train_losses), global_step=epoch)
        tb_writer.add_scalar('valid_loss', np.mean(valid_losses), global_step=epoch)
        tb_writer.add_scalar('f1_macro', f1, global_step=epoch)
        if best_loss > np.mean(valid_losses):
            best_model = copy.deepcopy(model)
            best_loss = np.mean(valid_losses)
            patience = 0

    torch.save(best_model, './model/' + args.task)
    model = best_model
    model.eval()
    test_losses = []
    with torch.no_grad():
        for input_ids, attention_masks, labels in testData:
            input_ids, attention_masks, labels = input_ids.cuda(), attention_masks.cuda(), labels.float().cuda()
            test_score = model(input_ids, attention_masks)
            test_loss = criterion(test_score, labels)
            test_losses.append(test_loss.cpu().data)
            label_pred = torch.zeros(size=labels.size())
            v = test_score.cpu().data
            max_ = torch.max(v, dim=1)[0].unsqueeze(dim=1).expand(-1, labels.size(1))
            label_pred = torch.where(max_ - v <= 0.1, 1, 0)

            f1 = f1_score(labels.cpu().data, label_pred, average='macro')

    print('loss: {},f1: {}'.format(np.mean(test_losses), f1))
