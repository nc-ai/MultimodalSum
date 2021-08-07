import os
import glob
import time
import argparse
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from apex.parallel import DistributedDataParallel as DDP

from utils import str2bool, reduce_tensor, LabelSmoothingLoss, AverageMeter
from data_utils import tokenize
from data_utils import yelp_table_processing, amazon_table_processing
from train_utils import set_environments, get_dataloader, get_optimizer, get_scheduler, train_model
from transformer.modeling_multimodalsum import BartForEncConditionalGeneration


class TableDataset(Dataset):
    def __init__(self, tokenizer, mode='train', dataset='yelp'):
        self.tokenizer = tokenizer
        self.mode = mode
        self.dataset = dataset

        # Text
        self.tokenize = tokenize(tokenizer, 130)

        print('# Load csv')
        filelist = glob.glob('data/%s/5.text/%s/*.csv' % (dataset, mode))
        all_lines = []
        for file in filelist:
            all_lines.extend(self.read_csv(file))
        self.all_csv = pd.DataFrame(all_lines)

        # Table
        self.meta_csv = pd.read_csv('data/%s/meta.csv' % dataset, sep='\t', na_filter=False, dtype=str)
        mode_group_id = sorted(set(self.all_csv.group_id) & set(self.meta_csv.group_id))
        self.meta_csv.index = self.meta_csv.group_id
        self.meta_csv = self.meta_csv.loc[mode_group_id].reset_index(drop=True)

        if dataset == 'yelp':
            self.field = torch.tensor(tokenizer(list(self.meta_csv.columns)[1:], add_special_tokens=False, add_prefix_space=True,
                                                max_length=6, padding='max_length')['input_ids'])
        elif dataset == 'amazon':
            self.field = torch.tensor(tokenizer(list(self.meta_csv.columns)[1:], add_special_tokens=False, add_prefix_space=True)['input_ids'])
        self.set_epoch()

    def read_csv(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]
            lines = [x.strip().split('\t') for x in lines]
        return [{'group_id' : x[0], 'review_text' : x[-3]} for x in lines]

    def set_epoch(self):
        print('# Set epoch')
        if self.mode == 'train':
            self.df = self.all_csv.groupby('group_id', as_index=False, sort=False, group_keys=False).apply(lambda x : x.sample(9))
        else:
            self.df = self.all_csv.groupby('group_id', as_index=False, sort=False, group_keys=False).apply(lambda x : x.sample(9, random_state=7))
        self.df = self.df.merge(self.meta_csv, on='group_id')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # Label
        review_text = self.tokenize(item.review_text)
        label = torch.tensor(review_text['input_ids'][1:-1])

        # Table
        if self.dataset == 'yelp':
            name, category, str_categorical, str_boolean, rating, hours = yelp_table_processing(item, self.tokenizer)
            return (name, category, str_categorical, str_boolean, rating, hours, label)

        elif self.dataset == 'amazon':
            price, rating, brand, name, category, description = amazon_table_processing(item, self.tokenizer)
            return (price, rating, brand, name, category, description, label)


class TableSupervised(nn.Module):
    def __init__(self, bart_pretrained, TableEncoder):
        super().__init__()
        self.bart_model = BartForEncConditionalGeneration.from_pretrained(bart_pretrained, config='cfg/bart-large.json')
        self.table_encoder = TableEncoder(self.bart_model.model.shared)

    def forward(self,
                field,
                field_value,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_past_key_values=None,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
                **unused,
               ):

        encoder_hiddens, encoder_attention_mask = self.table_encoder(field, field_value)
        rating_diff = torch.zeros([encoder_hiddens.size(0), 1], device=encoder_hiddens.device)
        decoder_outputs = self.bart_model(
            encoder_hiddens.unsqueeze(1),
            rating_diff,
            encoder_attention_mask.unsqueeze(1),
            decoder_input_ids,
            decoder_attention_mask,
            decoder_past_key_values,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict
        )

        lm_logits = decoder_outputs[0]

        lm_loss = None
        if labels is not None:
            if args.label_smoothing != None:
                loss_fct = LabelSmoothingLoss(self.bart_model.config.vocab_size, smoothing=args.label_smoothing)
            else:
                loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(lm_logits.view(-1, self.bart_model.config.vocab_size), labels.view(-1))
        return (lm_loss,)


class yelp_data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            name, category, str_categorical, str_boolean, rating, hours, label = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.name = name.cuda(non_blocking=True)
                self.category = category.cuda(non_blocking=True)
                self.str_categorical = str_categorical.cuda(non_blocking=True)
                self.str_boolean = str_boolean.cuda(non_blocking=True)
                self.rating = rating.cuda(non_blocking=True)
                self.hours = hours.cuda(non_blocking=True)
                self.label = label.cuda(non_blocking=True)
        except StopIteration:
            self.name = None
            self.category = None
            self.str_categorical = None
            self.str_boolean = None 
            self.rating = None
            self.hours = None
            self.label = None

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        name = self.name
        category = self.category
        str_categorical = self.str_categorical
        str_boolean = self.str_boolean
        rating = self.rating
        hours = self.hours
        label = self.label

        if name is not None:
            name.record_stream(torch.cuda.current_stream())
        if category is not None:
            category.record_stream(torch.cuda.current_stream())
        if str_categorical is not None:
            str_categorical.record_stream(torch.cuda.current_stream())
        if str_boolean is not None:
            str_boolean.record_stream(torch.cuda.current_stream())
        if rating is not None:
            rating.record_stream(torch.cuda.current_stream())
        if hours is not None:
            hours.record_stream(torch.cuda.current_stream())
        if label is not None:
            label.record_stream(torch.cuda.current_stream())
        self.preload()
        return [name, category, str_categorical, str_boolean, rating, hours], label


class amazon_data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            price, rating, brand, name, category, description, label = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.price = price.cuda(non_blocking=True)
                self.rating = rating.cuda(non_blocking=True)
                self.brand = brand.cuda(non_blocking=True)
                self.name = name.cuda(non_blocking=True)
                self.category = category.cuda(non_blocking=True)
                self.description = description.cuda(non_blocking=True)
                self.label = label.cuda(non_blocking=True)
        except StopIteration:
            self.price = None
            self.rating = None
            self.brand = None
            self.name = None
            self.category = None
            self.description = None
            self.label = None

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        price = self.price
        rating = self.rating
        brand = self.brand
        name = self.name
        category = self.category
        description = self.description
        label = self.label

        if price is not None:
            price.record_stream(torch.cuda.current_stream())
        if rating is not None:
            rating.record_stream(torch.cuda.current_stream())
        if brand is not None:
            brand.record_stream(torch.cuda.current_stream())
        if name is not None:
            name.record_stream(torch.cuda.current_stream())
        if category is not None:
            category.record_stream(torch.cuda.current_stream())
        if description is not None:
            description.record_stream(torch.cuda.current_stream())
        if label is not None:
            label.record_stream(torch.cuda.current_stream())
        self.preload()
        return [price, rating, brand, name, category, description], label


def train(start_time, train_dataloader, model, optimizer, scheduler, e, t_epoch):
    model.train()

    if args.dataset == 'yelp':
        prefetcher = yelp_data_prefetcher(train_dataloader)
    elif args.dataset == 'amazon':
        prefetcher = amazon_data_prefetcher(train_dataloader)
    field_value, label = prefetcher.next()
    i = 0

    while field_value[0] is not None:
        loss = model(field, field_value, labels=label)[0]

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm != None:
            if args.distributed:
                torch.nn.utils.clip_grad_norm_([p for n, p in model.module.table_encoder.named_parameters() if not n.startswith('bart')], args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_([p for n, p in model.table_encoder.named_parameters() if not n.startswith('bart')], args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if i % args.log_interval == 0:
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()
            if args.local_rank == 0:
                end_time = time.time()
                timedelta = str(datetime.timedelta(seconds=int(end_time - start_time)))
                print("{} epoch {} batch id {}/{} loss {}".format(timedelta, e+1, i+1, t_epoch, reduced_loss.item()))

        field_value, label = prefetcher.next()
        i += 1


def validate(val_dataloader, model, e):
    model.eval()
    losses = AverageMeter()

    if args.dataset == 'yelp':
        prefetcher = yelp_data_prefetcher(val_dataloader)
    elif args.dataset == 'amazon':
        prefetcher = amazon_data_prefetcher(val_dataloader)
    field_value, label = prefetcher.next()
    i = 0

    while field_value[0] is not None:
        with torch.no_grad():
            loss = model(field, field_value, labels=label)[0]

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
        else:
            reduced_loss = loss.data
        losses.update(reduced_loss.item(), field_value[0].size(0))

        field_value, label = prefetcher.next()
        i += 1

    torch.cuda.synchronize()
    if args.local_rank == 0:
        print("{} epoch valid loss {}".format(e+1, losses.avg))
    return losses.avg


def parse():
    parser = argparse.ArgumentParser()
    # Device
    parser.add_argument('--gpus', type=int, nargs='+', default=[2,3])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=0)

    # Data
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)

    # Training
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--early_stopping', type=str2bool, default=False)

    # Pretrained
    parser.add_argument('--bart_tokenizer', type=str, default='facebook/bart-large')
    parser.add_argument('--bart_pretrained', type=str, default='ckpt/text_pretrained')
    args = parser.parse_args()

    gpu_list = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpu_list])
    return args

def main():
    global args
    args = parse()

    # Setting
    args.ckpt = "ckpt/table_pretrained_%s" % args.dataset
    args.bart_pretrained = '%s_%s' % (args.bart_pretrained, args.dataset)
    args = set_environments(args)

    if args.dataset == 'yelp':
        from table_encoder import YelpTableEncoder as TableEncoder
    elif args.dataset == 'amazon':
        from table_encoder import AmazonTableEncoder as TableEncoder

    # Model
    model = TableSupervised(args.bart_pretrained, TableEncoder)
    model.cuda()

    # Optimizer
    no_decay = ['bias']
    optimizer = get_optimizer(args.learning_rate, no_decay, model.table_encoder.named_parameters(), lambda n : not n.startswith('bart'))

    # Dataset
    data_train, train_sampler, train_dataloader, val_dataloader = get_dataloader(args, TableDataset)

    # Field
    global field
    field = data_train.field.cuda()

    # Distributed model
    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    # Scheduler
    t_epoch = len(train_dataloader)
    args.log_interval = int(t_epoch * args.warmup_ratio)
    scheduler = get_scheduler(args, t_epoch, optimizer)

    # Train
    train_model(args, model, train_sampler, train_dataloader, val_dataloader, train, validate, optimizer, scheduler, t_epoch, 'table')

if __name__ == '__main__':
    main()  