import re
import json
import glob
import math
import html
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

import lxml.html
import lxml.html.clean

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

YELP_BIN_COLS = ['BikeParking', 'BusinessAcceptsCreditCards', 'ByAppointmentOnly', 'Caters', 'GoodForKids', 'HasTV', 'OutdoorSeating',
                 'RestaurantsDelivery', 'RestaurantsGoodForGroups', 'RestaurantsReservations', 'RestaurantsTakeOut', 'WheelchairAccessible',
                 'Ambience_casual', 'Ambience_classy', 'Ambience_divey', 'Ambience_hipster', 'Ambience_intimate', 
                 'Ambience_romantic', 'Ambience_touristy', 'Ambience_trendy', 'Ambience_upscale', 
                 'BusinessParking_garage', 'BusinessParking_lot', 'BusinessParking_street', 'BusinessParking_valet', 'BusinessParking_validated', 
                 'GoodForMeal_breakfast', 'GoodForMeal_brunch', 'GoodForMeal_dessert', 'GoodForMeal_dinner', 'GoodForMeal_latenight', 'GoodForMeal_lunch']
YELP_STR_COLS = ['business_id', 'name', 'categories', 'NoiseLevel', 'Alcohol', 'RestaurantsAttire', 'WiFi', 'RestaurantsPriceRange2']

YELP_FIELDS = ['group_id', 'name', 'category', 'noise level', 'alcohol', 'restaurants attire', 'wifi', 'restaurants price range', 'bike parking', 
               'accept credit cards', 'by appointment only', 'cater', 'good for kids', 'has tv', 'outdoor seating', 'restaurants delivery', 
               'restaurants good for group', 'restaurants reservations', 'restaurants take out', 'wheelchair accessible', 'ambience casual', 
               'ambience classy', 'ambience divey', 'ambience hipster', 'ambience intimate', 'ambience romantic', 'ambience touristy', 'ambience trendy',
               'ambience upscale', 'parking garage', 'parking lot', 'parking street', 'parking valet', 'parking validated', 'good for meal breakfast',
               'good for meal brunch', 'good for meal dessert', 'good for meal dinner', 'good for meal latenight', 'good for meal lunch', 
               'ratings', 'hours monday', 'hours tuesday', 'hours wednesday', 'hours thursday', 'hours friday', 'hours saturday', 'hours sunday']

AMAZON_CATEGORIES = ['Clothing_Shoes_and_Jewelry', 'Electronics', 'Health_and_Personal_Care', 'Home_and_Kitchen']

def _basic_str_preprocessing(x):
    if type(x) == str:
        if x.startswith("u'") and x.endswith("'"):
            x = x[1:]
        if x.startswith("'") and x.endswith("'"):
            x = x[1:-1]
        if x == 'None':
            x = None
    return x

def _to_binary(x, max_len, rounding=True):
    if rounding:
        x = round(x * 2.0) / 2.0
    p_float, p_integer = math.modf(x)
    b_integer = bin(int(p_integer))[2:]
    b_float = bin(math.ceil(p_float))[2:]
    binary = b_integer + b_float
    return '0' * (max_len-len(binary)) + binary

def _to_coordinate(hours):
    s, e = hours.split('-')
    s_h, s_m = [int(x) for x in s.split(':')]
    e_h, e_m = [int(x) for x in e.split(':')]

    s = float(s_h + s_m / 60.)
    e = float(e_h + e_m / 60.)

    if s >= e:
        e = e + 24.
    return [s, e]

def _clean_html(html):
    html = re.sub('\[if gte mso 9\][\s\S]+\[endif\]', ' ', html)
    html = re.sub('&[amp;]+lt;[\s\S]+&[amp;]+gt;', ' ', html)
    html = re.sub('.caption \{[\s\S]+\}', ' ', html)
    html = re.sub('#jl_box\{[\s\S]+\}', ' ', html)
    html = re.sub('#review[\s\S]+\}', ' ', html)
    html = re.sub('.productDescriptionWrapper table\{[\s\S]+\}', ' ', html)
    html = re.sub('#productDescription \{[\s\S]+\}', ' ', html)
    html = re.sub('DOCTYPE html PUBLIC[\s\S]+\}', ' ', html)
    html = re.sub('A\+[\s\S]+\}', ' ', html)
    html = re.sub('[\S]+[\s]?\{[\s\S]+\}', ' ', html)
    html = re.sub('\\xa0', ' ', html)
    html = re.sub('\s+', ' ', html)
    if html in ['', ' ']:
        return ''
    else:
        doc = lxml.html.fromstring(html)
        cleaner = lxml.html.clean.Cleaner(style=True)
        doc = cleaner.clean_html(doc)
        text = doc.text_content()
        text = text.replace('\xa0', ' ')
        text = re.sub('<.*?>', ' ', text)
        text = re.sub('\s+', ' ', text)
        if text == ' ':
            return ''
        else:
            return text

def _make_field(x, raw_field, new_field, null_value, processing_function):
    if raw_field not in x:
        x[new_field] = null_value
    else:
        x[new_field] = processing_function(x.pop(raw_field))
    return x

def _yelp_table_preprocessing():
    # Load meta
    json_list = []
    f = open('data/yelp/raw_others/business.json', 'r', encoding='utf-8')
    for line in f.readlines():
        json_ = json.loads(line)
        json_list.append(json_)

    # Process meta
    all_attributes = sorted(set(itertools.chain(*[list(x['attributes'].keys()) 
                                                  if x['attributes'] != None else [] for x in json_list])))
    all_hours = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    processed_list = []
    for raw in json_list:
        processed = {
            'business_id' : raw['business_id'],
            'name' : raw['name'],
            'stars' : raw['stars'],
            'categories' : raw['categories']
        }
        for attributes in all_attributes:
            processed[attributes] = None
            if raw['attributes']:
                if attributes in raw['attributes']:
                    processed[attributes] = raw['attributes'][attributes]
        for hours in all_hours:
            processed['hours_%s' % hours] = None
            if raw['hours']:
                if hours in raw['hours']:
                    processed['hours_%s' % hours] = raw['hours'][hours]
        processed_list.append(processed)

    df = pd.DataFrame(processed_list)

    # Basic str preprocessing
    for col in df.columns:
        temp = df[col]
        df[col] = temp.apply(_basic_str_preprocessing)

    # Feature selection based on null ratio
    col_stats = df.notnull().sum(axis=0) / len(df)
    net_cols = col_stats[col_stats >= 0.1].keys()
    df = df[net_cols]

    # Hierarchical feature processing
    hier_keys = dict()
    for col in df.columns[2:]:
        temp = df[col]
        dict_temp = temp[temp.apply(lambda x : type(x) == str and '{' in x)]
        if len(dict_temp) != 0:
            hier_keys[col] = sorted(set(list(itertools.chain(*dict_temp.apply(lambda x : list(eval(x).keys()))))))

    for key in hier_keys.keys():
        def _dict_to_list(x, values):
            result = []
            x = eval(x)
            for val in values:
                if val in x:
                    result.append(x[val])
                else:
                    result.append(None)
            return result

        temp = df[key]
        values = hier_keys[key]

        temp = temp.apply(lambda x : [None] * len(values) if x == None else _dict_to_list(x, values))
        df_temp = pd.DataFrame(list(temp), columns=['%s_%s' % (key, val) for val in values])

        df.pop(key)
        df = pd.concat([df, df_temp], axis=1)

    # Fill null
    df.fillna('', inplace=True)

    # Erase _ in string
    df['NoiseLevel'] = df['NoiseLevel'].apply(lambda x : x.replace('_', ' ') if '_' in x else x)
    df['Alcohol'] = df['Alcohol'].apply(lambda x : x.replace('_', ' ') if '_' in x else x)

    # Categories processing
    categories_len = df.categories.apply(lambda x : len(x.split(', ')))
    categories_len_threshold = np.percentile(categories_len, 90)
    df['categories'] = df.categories.apply(lambda x : ', '.join(x.split(', ')[:int(categories_len_threshold-1)]))

    # Hours processing
    hours_columns = [x for x in df.columns if 'hours' in x]
    all_hours = pd.Series(list(itertools.chain(*[list(df[x][df[x] != '']) for x in hours_columns])))
    all_hours_val = all_hours.value_counts()

    for i in range(len(all_hours_val)):
        set_hours_val = all_hours_val[all_hours_val >= i]
        ratio = sum(set_hours_val) / len(all_hours)
        if ratio < 0.9:
            break

    set_hours_val = all_hours_val[all_hours_val >= i]
    set_hours = pd.Series(list(set_hours_val.index))
    X = np.array(list(set_hours.apply(_to_coordinate)))

    cluster_list = [3, 4, 5, 7, 10]
    score_list = []
    for cluster in cluster_list:
        kmeans = KMeans(n_clusters=cluster, random_state=0).fit(X)
        y = kmeans.fit_predict(X)
        score = silhouette_score(X, y)
        score_list.append(score)

    best_cluster = cluster_list[np.argmax(score_list)]
    kmeans = KMeans(n_clusters=best_cluster, random_state=0).fit(X)

    # Binary type processing (to str type)
    for col in YELP_BIN_COLS:
        temp = df[col]
        temp = temp.apply(lambda x : 'true' if x in [1.0, 1, True, 'True'] else x)
        temp = temp.apply(lambda x : 'false' if x in [0.0, 0, False, 'False'] else x)
        df[col] = temp

    # Ordinary type processing (to str type)
    price_dict = {
        '1' : 'cheap',
        '2' : 'average',
        '3' : 'expensive',
        '4' : 'very expensive',
    }
    df['RestaurantsPriceRange2'] = df.RestaurantsPriceRange2.apply(lambda x : price_dict[x] if x != '' else x)

    # String type features
    df_str = df[YELP_STR_COLS + YELP_BIN_COLS]

    # Numeric type processing
    df_num = pd.DataFrame(df['stars'].apply(lambda x : _to_binary(x, 4, False)))

    # Categorical type processing
    df_categorical = []
    for col in hours_columns:
        temp = df[col].apply(lambda x : _to_coordinate(x) if x != '' else x)
        temp_notnull = temp[temp != '']
        temp[temp != ''] = pd.Series([str(x) for x in list(kmeans.predict(np.array(list(temp_notnull))))], index=temp_notnull.index)
        df_categorical.append(temp)
    df_categorical = pd.concat(df_categorical, axis=1)

    # Get meta
    meta = pd.concat([df_str, df_num, df_categorical], axis=1)
    meta.columns = YELP_FIELDS

    # Select essential meta
    train_file_list = glob.glob('data/yelp/5.text/train/*')
    val_file_list = glob.glob('data/yelp/5.text/val/*')
    test_file = pd.read_csv('data/yelp/test/summaries_0-200_cleaned.csv', encoding='utf-8')

    train_group_id_list = [x.split('/')[-1][:-4] for x in train_file_list]    
    val_group_id_list = [x.split('/')[-1][:-4] for x in val_file_list]
    test_group_id_list = [x for x in test_file['Input.business_id'] if type(x) == str]

    meta.index = meta.group_id
    set_meta = meta.loc[sorted(set(train_group_id_list + val_group_id_list + test_group_id_list))]

    # Save meta
    set_meta.to_csv('data/yelp/meta.csv', sep='\t', encoding='utf-8', index=False)


def _amazon_table_preprocessing():
    # Load meta
    json_list = []
    len_list = []
    print('--load meta')
    for category in tqdm(AMAZON_CATEGORIES):
        f = open('data/amazon/raw_others/meta_%s.json' % category, 'r')
        lines = f.readlines()
        len_list.append(len(lines))
        for line in lines:
            json_list.append(eval(line))

    # Set max categories
    categories_threshold = int(np.percentile(pd.Series([len(x['categories']) for x in json_list]), 90))

    # Select essential meta
    train_group_ids = [x.split('/')[-1][:-4] for x in glob.glob('data/amazon/5.text/train/*.csv')]
    val_group_ids = [x.split('/')[-1][:-4] for x in glob.glob('data/amazon/5.text/val/*.csv')]
    test_group_ids = list(pd.read_csv('data/amazon/test/all.csv', encoding='utf-8', sep='\t').prod_id)

    asin = pd.Series([x['asin'] for x in json_list])
    asin = asin[asin.drop_duplicates().index]
    asin = asin.reset_index()
    asin.index = asin[0]
    asin = asin.loc[train_group_ids + val_group_ids + test_group_ids]

    net_index_list = list(asin['index'])
    net_json_list = [json_list[i] for i in net_index_list]

    # Process meta
    print('--process meta')
    for x in tqdm(net_json_list):
        # asin
        x['group_id'] = x.pop('asin')
        # categories
        x['category'] = "||".join(["|".join(y) for y in x.pop('categories')[:categories_threshold]])

        # brand
        x = _make_field(x, 'brand', 'brand', '', lambda y : html.unescape(y))
        # description
        x = _make_field(x, 'description', 'description', '', lambda y : _clean_html(y))
        # imUrl
        x = _make_field(x, 'imUrl', 'imUrl', '', lambda y : y if y.startswith('http://ecx') else '')
        # price
        x = _make_field(x, 'price', 'price', _to_binary(0.0, 11), lambda y : _to_binary(float(y), 11))
        # title
        x = _make_field(x, 'title', 'name', '', lambda y : html.unescape(y))

        # related & salesRank
        if 'related' in x:
            x.pop('related')
        if 'salesRank' in x:
            x.pop('salesRank')

        # ratings
        df = None
        for category in AMAZON_CATEGORIES:
            try:
                with open('data/amazon/1.prep/%s/%s.csv' % (category, x['group_id']), 
                          encoding='utf-8', mode='r') as f:
                    lines = f.readlines()[1:]
                    df = pd.DataFrame([x.strip().split('\t') for x in lines], 
                                      columns=['group_id', 'review_text', 'rating', 'category'])
                break
            except FileNotFoundError:
                df = None

        if df is not None:
            x['ratings'] = _to_binary(df.rating.apply(float).mean(), 4)
        else:
            x['ratings'] = ''

    # Save meta
    set_meta = pd.DataFrame(net_json_list)[['group_id', 'price', ' ratings', 'brand', 'name', 'category', 'description']]
    set_meta.to_csv('data/amazon/meta.csv', sep='\t', encoding='utf-8', index=False)

def preprocessing(dataset):
    print('Table modality preprocessing: %s' % dataset)
    if dataset == 'yelp':
        _yelp_table_preprocessing()
    elif dataset == 'amazon':
        _amazon_table_preprocessing()
