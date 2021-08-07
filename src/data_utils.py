import numpy as np
from PIL import Image

import torch
from torchvision import transforms

train_img_transforms = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomRotation(5),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                     std = [0.229, 0.224, 0.225])
                            ])

test_img_transforms = transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                 std = [0.229, 0.224, 0.225])
                            ])


def tokenize(tokenizer, max_length):
    return lambda x : tokenizer(x, add_special_tokens=True, padding='max_length', truncation=True, 
                                max_length=max_length, add_prefix_space=True)

def image_loader(dataset, name):
    with open('data/%s/raw_others/photos/%s.jpg' % (dataset, name), 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def append_photo(json, dataset, photo_business_list, photo_dict):
    group_id = json['group_id']
    if group_id in photo_business_list:
        if dataset == 'yelp':
            photo_list = photo_dict[json['group_id']]
        elif dataset == 'amazon':
            photo_list = [json['group_id']]
    else:
        photo_list = []
    json['photo_id'] = photo_list
    return json


def text_processing(text, tokenize):
    reviews_text = [tokenize(x) for x in text]
    reviews = torch.tensor([x['input_ids'][1:-1] for x in reviews_text])
    reviews_mask = torch.tensor([x['attention_mask'][1:-1] for x in reviews_text])
    return reviews, reviews_mask

def img_processing(img, image_loader, img_transforms, max_imgs):
    img_list = [image_loader(x) for x in img]
    if img_list:
        img = torch.stack([img_transforms(x) for x in img_list])
        valid_img = img.size(0)

        img = torch.cat([img, torch.zeros([max_imgs-valid_img, 3, 224, 224])], dim=0)
        img_mask = torch.cat([torch.ones([valid_img]), torch.zeros([max_imgs-valid_img])]).bool()
    else:
        img = torch.zeros([max_imgs, 3, 224, 224])
        img_mask = torch.zeros([max_imgs]).bool()
    return img, img_mask

def yelp_table_processing(item, tokenizer):
    if type(item.group_id) != float:
        i_s = list(item.index).index('name')
        name = torch.tensor(tokenizer(item[i_s], add_special_tokens=False, max_length=24, truncation=True, 
                                           padding='max_length', add_prefix_space=True)['input_ids'])
        category = tokenizer(item[i_s+1].split(', '), add_special_tokens=False, max_length=12, 
                                  padding='max_length', add_prefix_space=True)['input_ids']
        category = torch.tensor(category + [[1]*12] * (6 - len(category)))
        str_categorical = torch.tensor(tokenizer(list(item[i_s+2:i_s+7]), add_special_tokens=False, max_length=3, 
                                                      padding='max_length', add_prefix_space=True)['input_ids'])
        str_boolean = torch.tensor(tokenizer(list(item[i_s+7:i_s+39]), add_special_tokens=False, max_length=1, 
                                                  padding='max_length', add_prefix_space=True)['input_ids'])
        rating = torch.tensor([int(x) for x in list(item[i_s+39])])
        hours = torch.tensor(list(item[i_s+40:].apply(lambda x : list(np.eye(4, dtype=int)[int(x)]) if x != '' else [0, 0, 0, 0])))
    else:
        name = torch.ones([24], dtype=torch.int64)
        category = torch.ones([6, 12], dtype=torch.int64)
        str_categorical = torch.ones([5, 3], dtype=torch.int64)
        str_boolean = torch.ones([32, 1], dtype=torch.int64)
        rating = torch.zeros([4], dtype=torch.int64)
        hours = torch.zeros([7, 4], dtype=torch.int64)
    return name, category, str_categorical, str_boolean, rating, hours

def amazon_table_processing(item, tokenizer):
    price = torch.tensor([int(x) for x in item['price']])
    rating = torch.tensor([int(x) for x in item['ratings']])

    if item['brand'] == '':
        brand = torch.ones([12], dtype=int)
    else:
        brand = torch.tensor(tokenizer(item['brand'], add_special_tokens=False, max_length=12, truncation=True, 
                                       padding='max_length', add_prefix_space=True)['input_ids'])
    if item['name'] == '':
        name = torch.ones([32], dtype=int)
    else:
        name = torch.tensor(tokenizer(item['name'], add_special_tokens=False, max_length=32, truncation=True, 
                                      padding='max_length', add_prefix_space=True)['input_ids'])

    category = [[tokenizer.encode(y, add_special_tokens=False, max_length=12, truncation=True, padding='max_length', add_prefix_space=True) 
                 for y in x.split('|')] for x in item['category'].split('||')]
    category_pad = []
    for category_ in category:
        category_pad.append(category_ + [[1]*12] * (8-len(category_)))
    category = torch.tensor(category_pad + (3-len(category_pad)) * [[[1]*12]*8])

    if item['description'] != '':
        description = torch.tensor(tokenizer(item['description'], add_special_tokens=False, max_length=128, truncation=True, 
                                             padding='max_length', add_prefix_space=True)['input_ids'])
    else:
        description = torch.ones([128], dtype=int)
    return price, rating, brand, name, category, description