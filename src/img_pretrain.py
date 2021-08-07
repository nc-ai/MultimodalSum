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
from data_utils import image_loader, append_photo, train_img_transforms, test_img_transforms, img_processing
from train_utils import set_environments, get_dataloader, get_optimizer, get_scheduler, train_model
from img_encoder import Resnet
from transformer.modeling_multimodalsum import BartForEncConditionalGeneration


class ImgDataset(Dataset):
    def __init__(self, tokenizer, mode='train', dataset='yelp'):
        self.mode = mode
        self.dataset = dataset

        # Text
        self.tokenize = tokenize(tokenizer, 130)

        # Image
        self.image_loader = lambda x : image_loader(dataset, x)
        if self.mode == 'train':
            self.img_transforms = train_img_transforms
        else:
            self.img_transforms = test_img_transforms
        photo_business_list = pd.read_pickle('data/%s/photo_business_dict.pickle' % dataset)[mode]
        if dataset == 'yelp':
            photo_dict = pd.read_pickle('data/yelp/photo_dict.pickle')
        elif dataset == 'amazon':
            photo_dict = None

        print('# Load csv')
        filelist = glob.glob('data/%s/5.text/%s/*.csv' % (dataset, mode))
        photo_filelist = [x for x in filelist if x.split('/')[-1][:-4] in photo_business_list]
        all_lines = []
        for file in photo_filelist:
            all_lines.extend(self.read_csv(file))
        all_lines = list(map(lambda x : append_photo(x, dataset, photo_business_list, photo_dict), all_lines))

        self.all_csv = pd.DataFrame(all_lines)
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # Label
        review_text = self.tokenize(item.review_text)
        label = torch.tensor(review_text['input_ids'][1:-1])

        # Img
        if self.dataset == 'yelp':
            max_imgs = 10
        elif self.dataset == 'amazon':
            max_imgs = 1
        img, img_mask = img_processing(item.photo_id, self.image_loader, self.img_transforms, max_imgs)
        return (img, img_mask, label)


class ImgSupervised(nn.Module):
    def __init__(self, bart_pretrained):
        super().__init__()
        self.bart_model = BartForEncConditionalGeneration.from_pretrained(bart_pretrained, config='cfg/bart-large.json')
        self.img_encoder = Resnet(self.bart_model.config.d_model)

    def forward(self,
                input_imgs,
                input_imgs_mask=None,
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
        r'''
            input_imgs : [bsz, max_examples, C, H, W]
            input_imgs_mask : [bsz, max_examples]
        '''

        bsz, max_examples = input_imgs.size()[:2]
        encoder_hiddens = self.img_encoder(input_imgs.reshape([-1, 3, 224, 224])).reshape(bsz, max_examples, -1, self.bart_model.config.d_model) # [bsz, max_examples, seq_len, hidden_size]
        if input_imgs_mask is not None:
            seq_len = encoder_hiddens.size(2)
            encoder_attention_mask = input_imgs_mask.unsqueeze(-1).repeat([1, 1, seq_len])
        else:
            encoder_attention_mask=None

        rating_diff = torch.zeros([bsz, 1], device=encoder_hiddens.device)
        decoder_outputs = self.bart_model(
            encoder_hiddens,
            rating_diff,
            encoder_attention_mask,
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


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            img, img_mask, label = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.input_imgs = img.cuda(non_blocking=True)
                self.input_imgs_mask = img_mask.cuda(non_blocking=True)
                self.labels = label.cuda(non_blocking=True)
        except StopIteration:
            self.input_imgs = None
            self.input_imgs_mask = None
            self.labels = None

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input_imgs = self.input_imgs
        input_imgs_mask = self.input_imgs_mask
        labels = self.labels

        if input_imgs is not None:
            input_imgs.record_stream(torch.cuda.current_stream())
        if input_imgs_mask is not None:
            input_imgs_mask.record_stream(torch.cuda.current_stream())
        if labels is not None:
            labels.record_stream(torch.cuda.current_stream())
        self.preload()
        return input_imgs, input_imgs_mask, labels


def train(start_time, train_dataloader, model, optimizer, scheduler, e, t_epoch):
    model.train()

    prefetcher = data_prefetcher(train_dataloader)
    input_imgs, input_imgs_mask, labels = prefetcher.next()
    i = 0

    while input_imgs is not None:
        loss = model(input_imgs, input_imgs_mask, labels=labels)[0]

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm != None:
            if args.distributed:
                torch.nn.utils.clip_grad_norm_([p for n, p in model.module.named_parameters() if n.startswith('img_encoder')], args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if n.startswith('img_encoder')], args.max_grad_norm)
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

        input_imgs, input_imgs_mask, labels = prefetcher.next()
        i += 1


def validate(val_dataloader, model, e):
    model.eval()
    losses = AverageMeter()

    prefetcher = data_prefetcher(val_dataloader)
    input_imgs, input_imgs_mask, labels = prefetcher.next()
    i = 0

    while input_imgs is not None:
        with torch.no_grad():
            loss = model(input_imgs, input_imgs_mask, labels=labels)[0]

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
        else:
            reduced_loss = loss.data
        losses.update(reduced_loss.item(), input_imgs.size(0))

        input_imgs, input_imgs_mask, labels = prefetcher.next()
        i += 1

    torch.cuda.synchronize()
    if args.local_rank == 0:
        print("{} epoch valid loss {}".format(e+1, losses.avg))
    return losses.avg


def parse():
    parser = argparse.ArgumentParser()
    # Device
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
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
    args.ckpt = "ckpt/img_pretrained_%s" % args.dataset
    args.bart_pretrained = '%s_%s' % (args.bart_pretrained, args.dataset)
    args = set_environments(args)

    # Model
    model = ImgSupervised(args.bart_pretrained)
    model.cuda()

    # Optimizer
    no_decay = ['bias', 'bn1.weight', 'bn2.weight', 'bn3.weight']
    optimizer = get_optimizer(args.learning_rate, no_decay, model.img_encoder.named_parameters(), None)

    # Dataset
    data_train, train_sampler, train_dataloader, val_dataloader = get_dataloader(args, ImgDataset)

    # Distributed model
    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    # Scheduler
    t_epoch = len(train_dataloader)
    args.log_interval = int(t_epoch * args.warmup_ratio)
    scheduler = get_scheduler(args, t_epoch, optimizer)

    # Train
    train_model(args, model, train_sampler, train_dataloader, val_dataloader, train, validate, optimizer, scheduler, t_epoch, 'img')

if __name__ == '__main__':
    main()