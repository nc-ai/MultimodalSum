import os
import json
import glob
import pickle
import tarfile
import itertools
import numpy as np
import pandas as pd
import urllib.request
from tqdm import tqdm

def _yelp_img_preprocessing():
    # Extract photo meta
    photos_tar = tarfile.TarFile('data/yelp/raw_others/yelp_photos.tar')
    photos_tar.extract('photos.json', 'data/yelp/raw_others')

    # Load photo meta
    json_list = []
    f = open('data/yelp/raw_others/photos.json', 'r')
    for line in f.readlines():
        json_ = json.loads(line)
        json_list.append(json_)
    df = pd.DataFrame(json_list)
    df.sort_values('business_id', inplace=True)

    # Set max photos (90th percentile)
    business2photo = dict(df.groupby('business_id').apply(lambda x : list(x.photo_id)))
    length_distribution = [len(x) for x in list(business2photo.values())]
    threshold = np.percentile(length_distribution, 90)

    # Discard popular businesses (over max photos)
    net_business2photo = dict()
    for business in business2photo:
        if len(business2photo[business]) < threshold:
            net_business2photo[x] = business2photo[x]

    # Save photo dictionary
    with open('data/yelp/photo_dict.pickle', 'wb') as f:
        pickle.dump(net_business2photo, f)

    # Extract photo files
    photo_files = [x.name for x in photos_tar.getmembers() if x.name.startswith('photos/')]
    net_photos = list(itertools.chain(*net_business2photo.values()))
    print('--extract photo files')
    for file in tqdm(photo_files):
        if file[7:-4] in net_photos:
            photos_tar.extract(file, 'data/yelp/raw_others')

    # Save business list having photos
    train_business = [x.split('/')[-1][:-4] for x in glob.glob('data/yelp/5.text/train/*.csv')]
    val_business = [x.split('/')[-1][:-4] for x in glob.glob('data/yelp/5.text/val/*.csv')]

    net_business_train = [x for x in train_business if x in net_business2photo]
    net_business_val = [x for x in val_business if x in net_business2photo]
    photo_business_dict = {'train' : net_business_train, 'val' : net_business_val}

    with open('data/yelp/photo_business_dict.pickle', 'wb') as f:
        pickle.dump(photo_business_dict, f)

def _amazon_img_preprocessing():
    # Load meta
    meta = pd.read_csv('data/amazon/meta.csv', encoding='utf-8', sep='\t', na_filter=False, dtype=str)

    # Prepare img preprocessing
    PATH = 'data/amazon/raw_others/photos'
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    # Save photo files
    print('--save photo files')
    for i in tqdm(range(len(meta))):
        group = meta.loc[i]
        group_id = group['group_id']
        group_path = '%s/%s.jpg' % (PATH, group_id)
        if not os.path.exists(group_path):
            url = group['imUrl']
            if url != '':
                try:
                    urllib.request.urlretrieve(url, group_path)
                except:
                    # Not Found img
                    meta.loc[i, 'imUrl'] = ''

    # Update meta
    meta.to_csv('data/amazon/meta.csv', sep='\t', encoding='utf-8', index=False)

    # Save business list having photos
    photo_business = [x.split('/')[-1][:-4] for x in glob.glob('%s/*.jpg' % PATH)]

    train_business = [x.split('/')[-1][:-4] for x in glob.glob('data/amazon/5.text/train/*.csv')]
    val_business = [x.split('/')[-1][:-4] for x in glob.glob('data/amazon/5.text/val/*.csv')]

    net_business_train = [x for x in train_business if x in photo_business]
    net_business_val = [x for x in val_business if x in photo_business]
    photo_business_dict = {'train' : net_business_train, 'val' : net_business_val}

    with open('data/amazon/photo_business_dict.pickle', 'wb') as f:
        pickle.dump(photo_business_dict, f)

def preprocessing(dataset):
    print('Img modality preprocessing: %s' % dataset)
    if dataset == 'yelp':
        _yelp_img_preprocessing()
    elif dataset == 'amazon':
        _amazon_img_preprocessing()
