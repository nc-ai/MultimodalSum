import os
import re
import glob
import pandas as pd
from tqdm import tqdm

amazon_category_dict = {
    'electronics' : 'Electronics',
    'home_and_kitchen' : 'Home_and_Kitchen',
    'health_and_personal_care' : 'Health_and_Personal_Care',
    'clothing_shoes_and_jewelry' : 'Clothing_Shoes_and_Jewelry'
}

def _text_preprocessing(text):
    ''' Basic text preprocessing for raw text '''
    return " ".join(text.encode('ascii', 'ignore').decode().split())

def _yelp_text_preprocessing(filename):
    group_id = filename.split('/')[-1][:-4]
    data = pd.read_csv(filename, sep='\t', encoding='utf-8')

    try:
        raw_data = pd.read_csv('data/yelp/1.prep/reviews/%s.csv' % group_id, 
                               sep='\t', encoding='utf-8')
    except:
        with open('data/yelp/1.prep/reviews/%s.csv' % group_id, 
                  encoding='utf-8', mode='r') as f:
            lines = f.readlines()[1:]
        raw_data = pd.DataFrame([x.strip().split('\t') for x in lines],
                                columns=['group_id', 'review_id', 'review_text', 'rating', 'category'])

    raw_data.review_text = raw_data.review_text.apply(_text_preprocessing)
    merged_data = data.merge(raw_data[['review_id', 'review_text']], how='left', on='review_id')
    merged_data = merged_data[['group_id', 'review_id', 'review_text_y', 'rating', 'category']]
    merged_data.columns = ['group_id', 'review_id', 'review_text', 'rating', 'category']
    return merged_data

def _amazon_text_preprocessing(filename, category_dict=amazon_category_dict):
    group_id = filename.split('/')[-1][:-4]
    data = pd.read_csv(filename, sep='\t', encoding='utf-8')

    data = data[data.rating.apply(lambda x : x != 'None')]
    category = category_dict[data.category.iloc[0]]

    try:
        raw_data = pd.read_csv('data/amazon/1.prep/%s/%s.csv' % (category, group_id),
                               sep='\t', encoding='utf-8')
    except:
        with open('data/amazon/1.prep/%s/%s.csv' % (category, group_id),
                  encoding='utf-8', mode='r') as f:
            lines = f.readlines()[1:]
        raw_data = pd.DataFrame([x.strip().split('\t') for x in lines], 
                                columns=['group_id', 'review_text', 'rating', 'category'])

    # Use review text for making review_id
    raw_data.review_text.fillna('', inplace=True)
    data['review_id'] = data.review_text.apply(lambda x : "".join(re.sub('[\x00-\x1f\x7f-\x9f]', '', x).split()))
    raw_data['review_id'] = raw_data.review_text.apply(lambda x : "".join(re.sub('[\x00-\x1f\x7f-\x9f]', '', x).split()))
    data = data.loc[data.review_id.drop_duplicates().index]
    raw_data = raw_data.loc[raw_data.review_id.drop_duplicates().index]

    raw_data.review_text = raw_data.review_text.apply(_text_preprocessing)
    merged_data = data.merge(raw_data[['review_id', 'review_text']], how='left', on='review_id')
    merged_data = merged_data[['group_id', 'review_text_y', 'rating', 'category']]
    merged_data.columns = ['group_id', 'review_text', 'rating', 'category']
    return merged_data

def preprocessing(dataset):
    """ Convert word tokenized text to raw text """
    if dataset == 'yelp':
        text_preprocessing_func = _yelp_text_preprocessing
        min_threshold = 25
    elif dataset == 'amazon':
        text_preprocessing_func = _amazon_text_preprocessing
        min_threshold = 30

    path = 'data/%s' % dataset
    if not os.path.exists('%s/5.text' % path):
        os.mkdir('%s/5.text' % path)
        os.mkdir('%s/5.text/train' % path)
        os.mkdir('%s/5.text/val' % path)

    for mode in ['train', 'val']:
        groups = glob.glob('%s/4.part/%s/*' % (path, mode))
        print('Text modality preprocessing: %s_%s' % (dataset, mode))
        for filename in tqdm(groups):
            merged_data = text_preprocessing_func(filename)
            merged_data = merged_data[merged_data.review_text.apply(lambda x : len(x.split()) >= min_threshold)]
            merged_data.to_csv(filename.replace('4.part', '5.text'),
                               sep='\t', encoding='utf-8', index=False)