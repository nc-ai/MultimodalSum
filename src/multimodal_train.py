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
from data_utils import tokenize, text_processing
from data_utils import image_loader, append_photo, train_img_transforms, test_img_transforms, img_processing
from data_utils import yelp_table_processing, amazon_table_processing
from train_utils import set_environments, get_dataloader, get_optimizer, get_scheduler, train_model
from img_encoder import Resnet
from transformer.modeling_multimodalsum import BartForMultiEncConditionalGeneration


class MultimodalDataset(Dataset):
    def __init__(self, tokenizer, mode='train', dataset='yelp'):
        self.tokenizer = tokenizer
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
        all_lines = []
        for file in filelist:
            all_lines.extend(self.read_csv(file))
        all_lines = list(map(lambda x : append_photo(x, dataset, photo_business_list, photo_dict), all_lines))
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
        return [{'group_id' : x[0], 'review_text' : x[-3], 'review_rating' : x[-2]} for x in lines]

    def set_epoch(self):
        print('# Set epoch')
        if self.mode == 'train':
            self.df = self.all_csv.groupby('group_id', as_index=False, sort=False, group_keys=False).apply(lambda x : x.sample(9))
        else:
            self.df = self.all_csv.groupby('group_id', as_index=False, sort=False, group_keys=False).apply(lambda x : x.sample(9, random_state=7))
 
        self.df = self.df.groupby('group_id', sort=False).apply(lambda x : pd.Series([tuple(x.review_text), tuple(x.review_rating), x.photo_id.iloc[0]], 
                                                                                     index=['review_text', 'review_rating', 'photo_id'])).reset_index()
        self.df = self.df.merge(self.meta_csv, on='group_id')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # Text
        reviews, reviews_mask = text_processing(item.review_text, self.tokenize)

        # Rating
        reviews_rating = torch.tensor([float(x) for x in item.review_rating])

        # Img
        if self.dataset == 'yelp':
            max_imgs = 10
        elif self.dataset == 'amazon':
            max_imgs = 1
        img, img_mask = img_processing(item.photo_id, self.image_loader, self.img_transforms, max_imgs)

        # Table
        if self.dataset == 'yelp':
            name, category, str_categorical, str_boolean, rating, hours = yelp_table_processing(item, self.tokenizer)
            return (reviews, reviews_mask, reviews_rating, name, category, str_categorical, str_boolean, rating, hours, img, img_mask)

        elif self.dataset == 'amazon':
            price, rating, brand, name, category, description = amazon_table_processing(item, self.tokenizer)
            return (reviews, reviews_mask, reviews_rating, price, rating, brand, name, category, description, img, img_mask)


class MultimodalSum(nn.Module):
    def __init__(self, bart_pretrained, table_pretrained, img_pretrained, TableEncoder):
        super().__init__()
        if bart_pretrained == None:
            bart_pretrained = 'facebook/bart-large'
        self.bart_model = BartForMultiEncConditionalGeneration.from_pretrained(bart_pretrained, config='cfg/bart-large.json')
        self.table_encoder = TableEncoder(self.bart_model.model.shared)
        if table_pretrained != None:
            self.table_encoder.load_state_dict(torch.load('%s/pytorch_model.bin' % table_pretrained, map_location='cpu'))
        self.img_encoder = Resnet(self.bart_model.config.d_model)
        if img_pretrained != None:
            self.img_encoder.load_state_dict(torch.load('%s/pytorch_model.bin' % img_pretrained, map_location='cpu'))

    def forward(self,
               reviews, # [bsz, 8, 128]
               reviews_mask, # [bsz, 8, 128]
               reviews_rating, # [bsz, 8]
               field,
               field_value,
               img,
               img_mask,
               decoder_input_ids=None,
               decoder_attention_mask=None,
               decoder_past_key_values=None,
               use_cache=None,
               output_attentions=False,
               output_hidden_states=False,
               return_dict=False,
               **unused,
              ):

        multimodal_outputs = self.get_multimodal_outputs(reviews, reviews_mask, field, field_value, img, img_mask)
        n_reviews, text_hiddens, text_attention_mask, table_hiddens, table_attention_mask, img_hiddens, img_attention_mask = multimodal_outputs

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
            decoder_outputs = self.bart_model(text_hiddens[:, encode_indices, :, :], text_attention_mask[:, encode_indices, :], 
                                              table_hiddens, table_attention_mask, img_hiddens, img_attention_mask,
                                              rating_diff=rating_diff.unsqueeze(1), labels=reviews[:, i, :])
            lm_logits = decoder_outputs[0] # [bsz, seq_len, vocab_size]
            lm_loss_list.append(loss_fct(lm_logits.view(-1, self.bart_model.config.vocab_size), reviews[:, i, :].reshape(-1)))
        lm_loss = torch.mean(torch.stack(lm_loss_list))
        return (lm_loss,)

    def get_multimodal_outputs(self,
                               reviews, # [bsz, 8, 128]
                               reviews_mask, # [bsz, 8, 128]
                               field,
                               field_value,
                               img,
                               img_mask
                              ):
        # Text
        bsz, n_reviews, seq_len = reviews.size()
        reviews_ = reviews.view([bsz*n_reviews, seq_len])
        reviews_mask_ = reviews_mask.view([bsz*n_reviews, seq_len])
        text_hiddens = self.bart_model.model.encoder(input_ids=reviews_, attention_mask=reviews_mask_)[0]

        text_hiddens = text_hiddens.view([bsz, n_reviews, seq_len, -1])
        text_attention_mask = reviews_mask

        # Table
        table_hiddens, table_attention_mask = self.table_encoder(field, field_value)
        table_hiddens = table_hiddens.unsqueeze(1)
        table_attention_mask = table_attention_mask.unsqueeze(1)

        # Img
        max_imgs = img.size(1)
        img_hiddens = self.img_encoder(img.reshape([-1, 3, 224, 224])).reshape(bsz, max_imgs, -1, self.bart_model.config.d_model)
        if img_mask is not None:
            pix_len = img_hiddens.size(2)
            img_attention_mask = img_mask.unsqueeze(-1).repeat([1, 1, pix_len])
        return n_reviews, text_hiddens, text_attention_mask, table_hiddens, table_attention_mask, img_hiddens, img_attention_mask


class yelp_data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            reviews, reviews_mask, reviews_rating, name, category, str_categorical, str_boolean, rating, hours, img, img_mask = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.reviews = reviews.cuda(non_blocking=True)
                self.reviews_mask = reviews_mask.cuda(non_blocking=True)
                self.reviews_rating = reviews_rating.cuda(non_blocking=True)
                self.name = name.cuda(non_blocking=True)
                self.category = category.cuda(non_blocking=True)
                self.str_categorical = str_categorical.cuda(non_blocking=True)
                self.str_boolean = str_boolean.cuda(non_blocking=True)
                self.rating = rating.cuda(non_blocking=True)
                self.hours = hours.cuda(non_blocking=True)
                self.img = img.cuda(non_blocking=True)
                self.img_mask = img_mask.cuda(non_blocking=True)
        except StopIteration:
            self.reviews = None
            self.reviews_mask = None
            self.reviews_rating = None
            self.name = None
            self.category = None
            self.str_categorical = None
            self.str_boolean = None 
            self.rating = None
            self.hours = None
            self.img = None
            self.img_mask = None

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        reviews = self.reviews
        reviews_mask = self.reviews_mask
        reviews_rating = self.reviews_rating
        name = self.name
        category = self.category
        str_categorical = self.str_categorical
        str_boolean = self.str_boolean
        rating = self.rating
        hours = self.hours
        img = self.img
        img_mask = self.img_mask

        if reviews is not None:
            reviews.record_stream(torch.cuda.current_stream())
        if reviews_mask is not None:
            reviews_mask.record_stream(torch.cuda.current_stream())
        if reviews_rating is not None:
            reviews_rating.record_stream(torch.cuda.current_stream())
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
        if img is not None:
            img.record_stream(torch.cuda.current_stream())
        if img_mask is not None:
            img_mask.record_stream(torch.cuda.current_stream())

        self.preload()
        return reviews, reviews_mask, reviews_rating, [name, category, str_categorical, str_boolean, rating, hours], img, img_mask


class amazon_data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            reviews, reviews_mask, reviews_rating, price, rating, brand, name, category, description, img, img_mask = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.reviews = reviews.cuda(non_blocking=True)
                self.reviews_mask = reviews_mask.cuda(non_blocking=True)
                self.reviews_rating = reviews_rating.cuda(non_blocking=True)
                self.price = price.cuda(non_blocking=True)
                self.rating = rating.cuda(non_blocking=True)
                self.brand = brand.cuda(non_blocking=True)
                self.name = name.cuda(non_blocking=True)
                self.category = category.cuda(non_blocking=True)
                self.description = description.cuda(non_blocking=True)
                self.img = img.cuda(non_blocking=True)
                self.img_mask = img_mask.cuda(non_blocking=True)
        except StopIteration:
            self.reviews = None
            self.reviews_mask = None
            self.reviews_rating = None
            self.price = None
            self.rating = None
            self.brand = None
            self.name = None
            self.category = None
            self.description = None
            self.img = None
            self.img_mask = None

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        reviews = self.reviews
        reviews_mask = self.reviews_mask
        reviews_rating = self.reviews_rating
        price = self.price
        rating = self.rating
        brand = self.brand
        name = self.name
        category = self.category
        description = self.description
        img = self.img
        img_mask = self.img_mask

        if reviews is not None:
            reviews.record_stream(torch.cuda.current_stream())
        if reviews_mask is not None:
            reviews_mask.record_stream(torch.cuda.current_stream())
        if reviews_rating is not None:
            reviews_rating.record_stream(torch.cuda.current_stream())
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
        if img is not None:
            img.record_stream(torch.cuda.current_stream())
        if img_mask is not None:
            img_mask.record_stream(torch.cuda.current_stream())

        self.preload()
        return reviews, reviews_mask, reviews_rating, [price, rating, brand, name, category, description], img, img_mask


def train(start_time, train_dataloader, model, optimizer, scheduler, e, t_epoch):
    model.train()

    if args.dataset == 'yelp':
        prefetcher = yelp_data_prefetcher(train_dataloader)
    elif args.dataset == 'amazon':
        prefetcher = amazon_data_prefetcher(train_dataloader)
    reviews, reviews_mask, reviews_rating, field_value, img, img_mask = prefetcher.next()
    i = 0

    while reviews is not None:
        loss = model(reviews, reviews_mask, reviews_rating, field, field_value, img, img_mask)[0]

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

        reviews, reviews_mask, reviews_rating, field_value, img, img_mask = prefetcher.next()
        i += 1

def validate(val_dataloader, model, e):
    model.eval()
    losses = AverageMeter()

    if args.dataset == 'yelp':
        prefetcher = yelp_data_prefetcher(val_dataloader)
    elif args.dataset == 'amazon':
        prefetcher = amazon_data_prefetcher(val_dataloader)
    reviews, reviews_mask, reviews_rating, field_value, img, img_mask, = prefetcher.next()
    i = 0

    while reviews is not None:
        with torch.no_grad():
            loss = model(reviews, reviews_mask, reviews_rating, field, field_value, img, img_mask)[0]

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
        else:
            reduced_loss = loss.data
        losses.update(reduced_loss.item(), reviews.size(0))

        reviews, reviews_mask, reviews_rating, field_value, img, img_mask, = prefetcher.next()
        i += 1

    torch.cuda.synchronize()
    if args.local_rank == 0:
        print("{} epoch valid loss {}".format(e+1, losses.avg))
    return losses.avg


def parse():
    parser = argparse.ArgumentParser()
    # Device
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=0)

    # Data
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=5)

    # Training
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--early_stopping', type=str2bool, default=False)

    # Pretrained
    parser.add_argument('--bart_tokenizer', type=str, default='facebook/bart-large')
    parser.add_argument('--bart_pretrained', type=str, default='ckpt/text_pretrained')
    parser.add_argument('--table_pretrained', type=str, default='ckpt/table_pretrained')
    parser.add_argument('--img_pretrained', type=str, default='ckpt/img_pretrained')
    args = parser.parse_args()

    gpu_list = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpu_list])
    return args

def main():
    global args
    args = parse()

    # Setting
    args.ckpt = "ckpt/multimodal_trained_%s" % args.dataset
    args.bart_pretrained = '%s_%s' % (args.bart_pretrained, args.dataset)
    args.table_pretrained = '%s_%s' % (args.table_pretrained, args.dataset)
    args.img_pretrained = '%s_%s' % (args.img_pretrained, args.dataset)
    args = set_environments(args)

    if args.dataset == 'yelp':
        from table_encoder import YelpTableEncoder as TableEncoder
    elif args.dataset == 'amazon':
        from table_encoder import AmazonTableEncoder as TableEncoder

    # Model
    model = MultimodalSum(args.bart_pretrained, args.table_pretrained, args.img_pretrained, TableEncoder)
    model.cuda()

    # Optimizer
    no_decay = ['bias', 'bn1.weight', 'bn2.weight', 'bn3.weight', 'layer_norm.weight', 'layernorm_embedding.weight']
    optimizer = get_optimizer(args.learning_rate, no_decay, model.named_parameters(), None)

    # Dataset
    data_train, train_sampler, train_dataloader, val_dataloader = get_dataloader(args, MultimodalDataset)

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
    train_model(args, model, train_sampler, train_dataloader, val_dataloader, train, validate, optimizer, scheduler, t_epoch, 'whole')

if __name__ == '__main__':
    main()