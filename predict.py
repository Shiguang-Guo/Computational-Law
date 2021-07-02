#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/7/2 18:57
# @Author : gsg
# @Site :
# @File : predict.py
# @Software: PyCharm
import re

import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel

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


class Predictor:
    def __init__(self, config):
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.config = config

    def onehot(self, label, name):
        label_list = eval(name + '_label_list')
        vector = np.zeros(len(label_list))
        for i in label:
            vector[label_list.index(i)] = 1
        return vector

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

        input_ids = []
        attention_masks = []
        encoded_dict = self.tokenizer.encode_plus(
            re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", fact),  # 输入文本
            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
            max_length=min(config['trunc_lenth'], 512),  # 填充 & 截断长度
            pad_to_max_length=True,
            return_attention_mask=True,  # 返回 attn. masks.
            return_tensors='pt',  # 返回 pytorch tensors 格式的数据
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids * 32, dim=0)
        attention_masks = torch.cat(attention_masks * 32, dim=0)

        for task, name in name_dict.items():
            model = torch.load(config['model'] + task)

            model.eval()
            with torch.no_grad():
                input_ids, attention_masks = input_ids.cuda(), attention_masks.cuda()
                valid_score = model(input_ids, attention_masks)
                lenth = len(eval(task + '_label_list'))
                v = valid_score.cpu().data
                max_ = torch.max(v, dim=1)[0].unsqueeze(dim=1).expand(-1, lenth)
                label_pred = torch.where(max_ - v <= 0.1, 1, 0)[0]
                for i in range(lenth):
                    if label_pred[i] == 1:
                        attr[name].append(eval(task + '_label_list')[i])

        return attr


if __name__ == '__main__':
    config = {
        'seed': 1,
        'name': 'form',
        'batch_size': 32,
        'lr': 5e-5,
        'wd': 0.0005,
        'epoches': 1000,
        'trunc_lenth': 700,
        'bert_path': 'bert-base-chinese',
        'patience': 50,
        'model': './model/'
    }
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    # fact = input("Wait for input")
    fact = '原告：赵香娟，女，1968年5月31日出生，汉族，万荣县。 被告：张青霞，女，1963年4月30日出生，汉族，万荣县。 被告：黄建业，男，1962年5月10日出生，汉族，万荣县。\n\n 原告向本院提出诉讼请求：1、二被告立即偿还我200000元本金及其利息（利息按借条注明的利率从借款之日算至还清之日）；2、二被告承担诉讼费用。事实与理由：二被告系夫妻关系。其二人因家庭经营需要资金为由，先后分四次向我借款200000元。借款到期后，我多次催要，二被告至今未予归还，请求法院判如所请。 二被告未到庭参加诉讼，亦未提交答辩意见。 原告为证实其主张，提交了四张借条。1、被告张青霞于2013年4月24日借走原告现金20000元，借条未注明利息；2、被告张青霞于2013年5月24日借走原告现金30000元，借条注明月息为1.5分；3、被告张青霞、黄建业于2015年11月24日借走原告现金90000元，借条注明月息为1.5分；4、被告张青霞、黄建业于2016年4月1日借走原告现金60000元，借条注明月息为1.8分。上述借条，拟证明二被告共向原告借款200000元之事实。 原告称，上述被告张青霞于2013年4月24日所借的20000元，借条虽未注明利息，但二被告实际按月息2.5分向其结息，原告未就此进行举证。 原告称，2016年6月30，经结算二被告尚欠原告利息4000元。休庭后，原告考虑到她与二被告系亲戚关系，故向本院书面声明愿放弃4000元欠息，并主张四笔借款的利息应自2016年7月1日起，统一按月息1.5分计算。 二被告未提交任何证据。 \n'

    pre = Predictor(config)
    pre.predict(fact)
