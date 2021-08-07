import os
import glob
import argparse
import rouge
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer

from data_utils import tokenize, text_processing
from data_utils import image_loader, test_img_transforms, img_processing
from data_utils import yelp_table_processing, amazon_table_processing
from multimodal_train import MultimodalSum, yelp_data_prefetcher, amazon_data_prefetcher


def rouge_preprocess(text):
    ''' preprocessing function for rouge calculation from PlanSum [https://github.com/rktamplayo/PlanSum] '''
    text = rouge.Rouge.REMOVE_CHAR_PATTERN.sub(' ', text.lower()).strip()
    tokens = rouge.Rouge.tokenize_text(rouge.Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', text))
    rouge.Rouge.stem_tokens(tokens)
    preprocessed_text = rouge.Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(tokens))
    num_tokens = len(tokens)
    return preprocessed_text

def calc_rouge(generated_reviews, reference_reviews, tokenizer, use_stemmer=False):
    ''' rouge calculation function based on PlanSum [https://github.com/rktamplayo/PlanSum] '''
    rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2, limit_length=False, apply_avg=True, apply_best=False, alpha=0.5, stemming=use_stemmer)
    generated_reviews = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_reviews]

    pred_sums = [rouge_preprocess(g) for g in generated_reviews]
    gold_sums = [rouge_preprocess(g) for g in reference_reviews]

    scores = rouge_eval.get_scores(pred_sums, gold_sums)
    rouge_1 = scores['rouge-1']['f'] * 100
    rouge_2 = scores['rouge-2']['f'] * 100
    rouge_l = scores['rouge-l']['f'] * 100

    incomplete = [x for x in generated_reviews if not (x.endswith('.') or x.endswith('!'))]
    return {'rouge1' : rouge_1, 'rouge2' : rouge_2, 'rougeL' : rouge_l, 'incomplete': len(incomplete)}


class MultimodalTestDataset(Dataset):
    def __init__(self, tokenizer, mode='dev', dataset='yelp'):
        self.tokenizer = tokenizer
        self.dataset = dataset

        # Text
        if dataset == 'yelp':
            max_length = 160
        elif dataset == 'amazon':
            max_length = 120
        self.tokenize = tokenize(tokenizer, max_length)

        # Image
        self.image_loader = lambda x : image_loader(dataset, x)
        self.img_transforms = test_img_transforms

        print('# Load csv')
        if dataset == 'yelp':
            if mode == 'dev':
                all_csv = pd.read_csv('data/yelp/test/summaries_0-200_cleaned.csv', encoding='utf-8').iloc[:100]
            elif mode == 'test':
                all_csv = pd.read_csv('data/yelp/test/summaries_0-200_cleaned.csv', encoding='utf-8').iloc[100:]
            group_id_col = 'Input.business_id'
            review_text_cols = ['Input.original_review_%d' % i for i in range(8)]
            summary_cols = ['Answer.summary']
        elif dataset == 'amazon':
            all_csv = pd.read_csv('data/amazon/test/%s.csv' % mode, sep='\t', encoding='utf-8')
            group_id_col = 'prod_id'
            review_text_cols = ['rev%d' % i for i in range(1, 9)]
            summary_cols = ['summ%d' % i for i in range(1, 4)]

        review_process = lambda x : " ".join(x.encode('ascii', 'ignore').decode().split())
        summary_process = lambda x : " ".join(x.split())

        group_id = all_csv.loc[:, group_id_col]
        review_text = all_csv.loc[:, review_text_cols].apply(lambda x : list(map(review_process, x)), axis=1)
        summary = all_csv.loc[:, summary_cols].apply(lambda x : list(map(summary_process, x)), axis=1)
        self.df = pd.concat([group_id, review_text, summary], axis=1)
        self.df.columns = ['group_id', 'review_text', 'summary']

        # Append Img
        if dataset == 'yelp':
            photo_dict = pd.read_pickle('data/yelp/photo_dict.pickle')
            self.df['photo_id'] = group_id.apply(lambda x : photo_dict[x] if x in photo_dict else [])
        elif dataset == 'amazon':
            photo_business_list = [x.split('/')[-1][:-4] for x in glob.glob('data/amazon/raw_others/photos/*.jpg')]
            self.df['photo_id'] = group_id.apply(lambda x : [x] if x in photo_business_list else [])

        # Append table
        self.meta_csv = pd.read_csv('data/%s/meta.csv' % dataset, sep='\t', na_filter=False, dtype=str)
        self.df = self.df.merge(self.meta_csv, on='group_id', how='left')

        if dataset == 'yelp':
            self.field = torch.tensor(tokenizer(list(self.meta_csv.columns)[1:], add_special_tokens=False, add_prefix_space=True,
                                                max_length=6, padding='max_length')['input_ids'])
        elif dataset == 'amazon':
            self.field = torch.tensor(tokenizer(list(self.meta_csv.columns)[1:], add_special_tokens=False, add_prefix_space=True)['input_ids'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # Text
        reviews, reviews_mask = text_processing(item.review_text, self.tokenize)

        # Rating
        reviews_rating = torch.zeros([8])

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


def test(test_dataloader, model):
    model.eval()

    if args.dataset == 'yelp':
        prefetcher = yelp_data_prefetcher(test_dataloader)
    elif args.dataset == 'amazon':
        prefetcher = amazon_data_prefetcher(test_dataloader)

    reviews, reviews_mask, reviews_rating, field_value, img, img_mask = prefetcher.next()
    i = 0

    generated_list = []
    while reviews is not None:
        print('%d / %d' % (i, len(test_dataloader)))

        with torch.no_grad():
            _, text_hiddens, text_attention_mask, table_hiddens, table_attention_mask, img_hiddens, img_attention_mask = \
            model.get_multimodal_outputs(reviews, reviews_mask, field, field_value, img, img_mask)
            rating_diff = torch.zeros([text_hiddens.size(0), 1], device=text_hiddens.device)
            generated = model.bart_model.generate(text_hiddens, text_attention_mask, table_hiddens, table_attention_mask, img_hiddens, img_attention_mask, 
                                                  rating_diff=rating_diff, num_beams=args.num_beams, length_penalty=args.length_penalty, max_length=args.max_length,
                                                  no_repeat_ngram_size=3, early_stopping=True)
        generated_list.extend(generated)
        reviews, reviews_mask, reviews_rating, field_value, img, img_mask = prefetcher.next()
        i += 1
    return generated_list


def parse():
    parser = argparse.ArgumentParser()
    # Device
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--workers', type=int, default=4)

    # Data
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=8)

    # Generation
    parser.add_argument('--num_beams', type=int)
    parser.add_argument('--length_penalty', type=float)
    parser.add_argument('--max_length', type=int)

    # Pretrained
    parser.add_argument('--bart_tokenizer', type=str, default='facebook/bart-large')
    parser.add_argument('--multimodal_trained', type=str, default='ckpt/multimodal_trained')
    args = parser.parse_args()

    gpu_list = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpu_list])
    return args


def main():
    global args
    args = parse()
    args.multimodal_trained = '%s_%s' % (args.multimodal_trained, args.dataset)

    if args.dataset == 'yelp':
        from table_encoder import YelpTableEncoder as TableEncoder
    elif args.dataset == 'amazon':
        from table_encoder import AmazonTableEncoder as TableEncoder

    # Model
    model = MultimodalSum(None, None, None, TableEncoder)
    model.load_state_dict(torch.load('%s/pytorch_model.bin' % args.multimodal_trained, map_location='cpu'))
    model.cuda()

    # Dataset
    tokenizer = BartTokenizer.from_pretrained(args.bart_tokenizer)
    data = MultimodalTestDataset(tokenizer=tokenizer, mode=args.mode, dataset=args.dataset)
    dataloader = DataLoader(data, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    # Field
    global field
    field = data.field.cuda()

    # Test
    print ('# Test')
    generated_list = test(dataloader, model)
    summary_list = list(data.df.summary)

    rouge_list = []
    for i in range(len(summary_list[0])):
        rouge_list.append(calc_rouge(generated_list, [x[i] for x in list(data.df.summary)], tokenizer))

    key_list = ['rouge1', 'rouge2', 'rougeL', 'incomplete']
    value_list = []
    for key in key_list:
        value = np.mean([x[key] for x in rouge_list])
        value_list.append(value)

    print('# Results')
    print(" | ".join(list(map(lambda x : "%s : %.2f" % (x[0], x[1]), list(zip(key_list, value_list))))))


if __name__ == '__main__':
    main()