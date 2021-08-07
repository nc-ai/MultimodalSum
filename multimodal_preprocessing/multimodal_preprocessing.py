from text_modality_preprocessing import preprocessing as text_preprocessing
from img_modality_preprocessing import preprocessing as img_preprocessing
from table_modality_preprocessing import preprocessing as table_preprocessing
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    text_preprocessing(args.dataset)
    if args.dataset == 'yelp':
        img_preprocessing(args.dataset)
        table_preprocessing(args.dataset)
    elif args.dataset == 'amazon':
        table_preprocessing(args.dataset)
        img_preprocessing(args.dataset)