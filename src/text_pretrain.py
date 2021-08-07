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

from utils import str2bool, reduce_tensor, AverageMeter
from data_utils import tokenize, text_processing
from train_utils import set_environments, get_dataloader, get_optimizer, get_scheduler, train_model
from transformer.modeling_multimodalsum import BartForEncConditionalGeneration


class TextDataset(Dataset):
    def __init__(self, tokenizer, mode='train', dataset='yelp'):
        self.mode = mode

        # Text
        self.tokenize = tokenize(tokenizer, 130)

        print('# Load csv')
        filelist = glob.glob('data/%s/5.text/%s/*.csv' % (dataset, mode))
        all_lines = []
        for file in filelist:
            all_lines.extend(self.read_csv(file))

        self.all_csv = pd.DataFrame(all_lines)
        self.set_epoch()

    def read_csv(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]
            lines = [x.strip().split('\t') for x in lines]
        return [{'group_id' : x[0], 'review_text' : x[-3], 'review_rating' : x[-2]} for x in lines]

    def set_epoch(self):
        print('# Set epoch')
        if self.mode == 'train':
            self.df = self.all_csv.groupby('group_id', as_index=False, sort=False, group_keys=False).apply(lambda x : x.sample(9))
        else:
            self.df = self.all_csv.groupby('group_id', as_index=False, sort=False, group_keys=False).apply(lambda x : x.sample(9, random_state=7))
 
        self.df = self.df.groupby('group_id', sort=False).apply(lambda x : pd.Series([tuple(x.review_text), tuple(x.review_rating)], 
                                                                                     index=['review_text', 'review_rating'])).reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # Text
        reviews, reviews_mask = text_processing(item.review_text, self.tokenize)

        # Rating
        reviews_rating = torch.tensor([float(x) for x in item.review_rating])
        return (reviews, reviews_mask, reviews_rating)


class TextSupervised(nn.Module):
    def __init__(self, bart_pretrained):
        super().__init__()
        self.bart_model = BartForEncConditionalGeneration.from_pretrained(bart_pretrained, config='cfg/bart-large.json')

    def forward(self,
               reviews, # [bsz, 8, 128]
               reviews_mask, # [bsz, 8, 128]
               reviews_rating, # [bsz, 8]
               decoder_input_ids=None,
               decoder_attention_mask=None,
               decoder_past_key_values=None,
               use_cache=None,
               output_attentions=False,
               output_hidden_states=False,
               return_dict=False,
               **unused,
              ):

        # Text
        bsz, n_reviews, seq_len = reviews.size()
        reviews_ = reviews.view([bsz*n_reviews, seq_len])
        reviews_mask_ = reviews_mask.view([bsz*n_reviews, seq_len])
        encoder_hiddens = self.bart_model.model.encoder(input_ids=reviews_, attention_mask=reviews_mask_)[0]

        encoder_hiddens = encoder_hiddens.view([bsz, n_reviews, seq_len, -1])
        encoder_attention_mask = reviews_mask

        if args.label_smoothing != None:
            loss_fct = LabelSmoothingLoss(self.bart_model.config.vocab_size, smoothing=args.label_smoothing)
        else:
            loss_fct = nn.CrossEntropyLoss()

        review_indices = list(np.arange(n_reviews))
        lm_loss_list = []
        for i in range(n_reviews):
            encode_indices = review_indices[:i] + review_indices[i+1:] # all but i index
            source_ratings = torch.mean(reviews_rating[:, encode_indices], dim=1)
            target_ratings = reviews_rating[:, i]
            rating_diff = target_ratings - source_ratings
            decoder_outputs = self.bart_model(encoder_hiddens[:, encode_indices, :, :],
                                              rating_diff.unsqueeze(1),
                                              encoder_attention_mask[:, encode_indices, :],
                                              labels=reviews[:, i, :])
            lm_logits = decoder_outputs[0] # [bsz, seq_len, vocab_size]
            lm_loss_list.append(loss_fct(lm_logits.view(-1, self.bart_model.config.vocab_size), reviews[:, i, :].reshape(-1)))
        lm_loss = torch.mean(torch.stack(lm_loss_list))
        return (lm_loss,)


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            reviews, reviews_mask, reviews_rating = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.reviews = reviews.cuda(non_blocking=True)
                self.reviews_mask = reviews_mask.cuda(non_blocking=True)
                self.reviews_rating = reviews_rating.cuda(non_blocking=True)
        except StopIteration:
            self.reviews = None
            self.reviews_mask = None
            self.reviews_rating = None

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        reviews = self.reviews
        reviews_mask = self.reviews_mask
        reviews_rating = self.reviews_rating

        if reviews is not None:
            reviews.record_stream(torch.cuda.current_stream())
        if reviews_mask is not None:
            reviews_mask.record_stream(torch.cuda.current_stream())
        if reviews_rating is not None:
            reviews_rating.record_stream(torch.cuda.current_stream())

        self.preload()
        return reviews, reviews_mask, reviews_rating


def train(start_time, train_dataloader, model, optimizer, scheduler, e, t_epoch):
    model.train()

    prefetcher = data_prefetcher(train_dataloader)
    reviews, reviews_mask, reviews_rating = prefetcher.next()
    i = 0

    while reviews is not None:
        loss = model(reviews, reviews_mask, reviews_rating)[0]
        
        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm != None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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

        reviews, reviews_mask, reviews_rating = prefetcher.next()
        i += 1


def validate(val_dataloader, model, e):
    model.eval()
    losses = AverageMeter()

    prefetcher = data_prefetcher(val_dataloader)
    reviews, reviews_mask, reviews_rating = prefetcher.next()
    i = 0

    while reviews is not None:
        with torch.no_grad():
            loss = model(reviews, reviews_mask, reviews_rating)[0]

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
        else:
            reduced_loss = loss.data
        losses.update(reduced_loss.item(), reviews.size(0))

        reviews, reviews_mask, reviews_rating = prefetcher.next()
        i += 1

    torch.cuda.synchronize()
    if args.local_rank == 0:
        print("{} epoch valid loss {}".format(e+1, losses.avg))
    return losses.avg


def parse():
    parser = argparse.ArgumentParser()
    # Device
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1,2,3])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=0)

    # Data
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=5)

    # Training
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--label_smoothing', type=float, default=None)
    parser.add_argument('--early_stopping', type=str2bool, default=False)

    # Pretrained
    parser.add_argument('--bart_tokenizer', type=str, default='facebook/bart-large')
    parser.add_argument('--bart_pretrained', type=str, default='ckpt/bart-review')
    args = parser.parse_args()

    gpu_list = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpu_list])
    return args


def main():
    global args
    args = parse()

    # Setting
    args.ckpt = "ckpt/text_pretrained_%s" % args.dataset
    args.bart_pretrained = '%s_%s' % (args.bart_pretrained, args.dataset)
    args = set_environments(args)

    # Model
    model = TextSupervised(args.bart_pretrained)
    model.cuda()

    # Optimizer
    no_decay = ['bias', 'layer_norm.weight', 'layernorm_embedding.weight']
    optimizer = get_optimizer(args.learning_rate, no_decay, model.named_parameters(), None)

    # Dataset
    data_train, train_sampler, train_dataloader, val_dataloader = get_dataloader(args, TextDataset)

    # Distributed model
    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    # Scheduler
    t_epoch = len(train_dataloader)
    args.log_interval = int(t_epoch * args.warmup_ratio)
    scheduler = get_scheduler(args, t_epoch, optimizer)

    # Train
    train_model(args, model, train_sampler, train_dataloader, val_dataloader, train, validate, optimizer, scheduler, t_epoch, 'text')

if __name__ == '__main__':
    main()