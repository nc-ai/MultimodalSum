import os
import glob
import math
import argparse
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import sent_tokenize

from transformers import PreTrainedTokenizer, BartTokenizer

from transformer.trainer import Trainer
from transformer.training_args import TrainingArguments
from transformer.modeling_multimodalsum import BartForConditionalGeneration


class ReviewDataset(Dataset):
    def __init__(self, dataset: str, tokenizer: PreTrainedTokenizer, block_size: int, mode='train'):
        assert dataset == 'yelp' or dataset == 'amazon'
        self.tokenizer = tokenizer
        file_path = os.path.join('data', dataset, '5.text', mode)
        all_csv = glob.glob(f'{file_path}/*.csv')
        reviews = []
        for csv in all_csv:
            df = pd.read_csv(csv, sep='\t')
            reviews += [row['review_text'] for _, row in df.iterrows() if len(row['review_text']) > 5 and not row['review_text'].isspace()]

        batch_encoding = self.tokenizer(reviews, add_special_tokens=False, truncation=True, max_length=block_size,
                                        padding='max_length')
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


def create_decoder_input_ids(labels, bos_token_id):
    prev_output_tokens = labels.clone()
    prev_output_tokens[:, 1:] = labels[:, :-1:]
    prev_output_tokens[:, 0] = bos_token_id

    return prev_output_tokens


@dataclass
class DataCollatorForBartModeling:
    tokenizer: PreTrainedTokenizer
    permute_sentence_ratio: float
    mask_ratio: float
    block_size: int
    mask_span_distribution: torch.distributions.categorical.Categorical

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        labels = batch.clone()

        if self.permute_sentence_ratio > 0.:
            inputs = self._permute_sentences(batch)

        if self.mask_ratio > 0.:
            inputs = self._infill_text(inputs)

        decoder_input_ids = create_decoder_input_ids(labels, self.tokenizer.bos_token_id)

        has_pad = (labels[:, -1:] == self.tokenizer.pad_token_id).squeeze()
        first_pad_idx = labels[has_pad].ne(self.tokenizer.pad_token_id).sum(dim=1)

        labels[has_pad, first_pad_idx] = self.tokenizer.eos_token_id

        return {'input_ids': inputs, 'labels': labels, 'decoder_input_ids': decoder_input_ids}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def _permute_sentences(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []

        for idx, source in enumerate(inputs):
            source = source[~(source == self.tokenizer.pad_token_id)]
            sentence_list = np.asarray(sent_tokenize(self.tokenizer.decode(source)))
            num_sentences = len(sentence_list)

            num_to_permute = math.ceil((num_sentences * 2 * self.permute_sentence_ratio) / 2.)
            substitutions = torch.randperm(num_sentences)[:num_to_permute]
            ordering = torch.arange(0, num_sentences)
            ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

            result = ' '.join(sentence_list[ordering])
            result = self.tokenizer.encode(result, max_length=self.block_size, padding='max_length', truncation=True,
                                           add_special_tokens=False, return_tensors='pt').squeeze(0)
            has_pad = result[-1] == self.tokenizer.pad_token_id
            if has_pad:
                first_pad_idx = (result == self.tokenizer.pad_token_id).nonzero()[0]
                result[first_pad_idx] = self.tokenizer.eos_token_id

            outputs.append(result)

        outputs = torch.stack(outputs, dim=0)

        return outputs

    def _infill_text(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        dim = self.block_size

        for idx, source in enumerate(inputs):
            assert source.size(0) == dim
            source = source[~(source == self.tokenizer.pad_token_id)]
            is_word_start = torch.ones(source.size())
            is_word_start[-1] = 0

            num_to_mask = int(
                math.ceil(
                    is_word_start.float().sum() * self.mask_ratio))
            if num_to_mask == 0:
                if source.size(0) < dim:
                    pads = torch.ones(dim - source.size(0)).long()
                    source = torch.cat((source, pads))

                outputs.append(source)
                continue

            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat([lengths, self.mask_span_distribution.sample(sample_shape=(num_to_mask,))], dim=0)
                cum_length = torch.cumsum(lengths, 0)

            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                source = self._add_insertion_noise(source, num_inserts / source.size(0), dim)
                if source.size(0) < dim:
                    pads = torch.ones(dim - source.size(0)).long()
                    source = torch.cat((source, pads))

                outputs.append(source)
                continue

            assert (lengths > 0).all()
            assert is_word_start[-1] == 0

            word_starts = is_word_start.nonzero()
            indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)

            source_length = source.size(0)
            assert source_length - 1 not in indices
            to_keep = torch.ones(source_length, dtype=torch.bool)
            is_word_start[-1] = 255

            source[indices] = self.tokenizer.mask_token_id

            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                lengths = lengths[uncompleted]
                to_keep[indices] = 0

            source = source[to_keep]

            if num_inserts > 0:
                source = self._add_insertion_noise(source, num_inserts / source.size(0), dim)

            assert source.size(0) <= dim

            if source.size(0) < dim:
                pads = torch.ones(dim - source.size(0)).long()
                source = torch.cat((source, pads))

            outputs.append(source)

        outputs = torch.stack(outputs, dim=0)

        return outputs

    def _add_insertion_noise(self, tokens, p, dim):
        if p == 0.:
            return tokens

        num_tokens = len(tokens)
        if num_tokens == dim:
            return tokens

        n = int(math.ceil(num_tokens * p))
        if num_tokens + n > dim:
            n = dim - num_tokens

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        result[noise_indices] = self.tokenizer.mask_token_id
        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result


def make_poisson_distribution(_lambda):
    lambda_to_the_k = 1
    e_to_the_minus_lambda = math.exp(-_lambda)
    k_factorial = 1
    ps = []
    for k in range(0, 128):
        ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
        lambda_to_the_k *= _lambda
        k_factorial *= (k + 1)
        if ps[-1] < 0.0000001:
            break
    ps = torch.FloatTensor(ps)
    mask_span_distribution = torch.distributions.Categorical(ps)

    return mask_span_distribution


def parse():
    parser = argparse.ArgumentParser()
    # Device
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1,2,3])

    # Data
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=3)

    # Preprocess
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--dist_lambda', type=float, default=3.)
    parser.add_argument('--permute_ratio', type=float, default=1.0)
    parser.add_argument('--mask_ratio', type=float, default=0.3)

    # Training
    parser.add_argument('--save_steps', type=int, default=60000)
    parser.add_argument('--eval_steps', type=int, default=30000)
    parser.add_argument('--logging_steps', type=int, default=500)
    parser.add_argument('--logging_path', type=str, default='log/bart-review')
    parser.add_argument('--output_path', type=str, default='ckpt/bart-review')

    # Pretrained
    parser.add_argument('--bart_tokenizer', type=str, default='facebook/bart-large')
    parser.add_argument('--bart_pretrained', type=str, default='facebook/bart-large')
    args = parser.parse_args()

    gpu_list = args.gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in gpu_list])
    return args


def main():
    args = parse()
    args.logging_path = '%s_%s' % (args.logging_path, args.dataset)
    args.output_path = '%s_%s' % (args.output_path, args.dataset)

    tokenizer = BartTokenizer.from_pretrained(args.bart_tokenizer)
    model = BartForConditionalGeneration.from_pretrained(args.bart_pretrained, config='cfg/bart-large.json')

    train_dataset = ReviewDataset(args.dataset, tokenizer=tokenizer, block_size=args.block_size)
    eval_dataset = ReviewDataset(args.dataset, tokenizer=tokenizer, block_size=args.block_size, mode='val')

    mask_span_distribution = make_poisson_distribution(args.dist_lambda)
    data_collator = DataCollatorForBartModeling(tokenizer=tokenizer,
                                                permute_sentence_ratio=args.permute_ratio,
                                                mask_ratio=args.mask_ratio, block_size=args.block_size,
                                                mask_span_distribution=mask_span_distribution)

    training_args = TrainingArguments(
        output_dir=args.output_path,
        overwrite_output_dir=True,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=args.save_steps,
        evaluate_during_training=True,
        eval_steps=args.eval_steps,
        dataloader_drop_last=True,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_path,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True
    )

    print('Train')
    trainer.train()
    trainer.save_model()

    if trainer.is_world_master():
        tokenizer.save_pretrained(training_args.output_dir)

    print('Evaluation')
    eval_output = trainer.evaluate()
    perplexity = math.exp(eval_output['eval_loss'])
    result = {'perplexity': perplexity}

    output_eval_file = os.path.join(training_args.output_dir, 'eval_results_lm.txt')

    if trainer.is_world_master():
        with open(output_eval_file, 'w') as writer:
            for key in sorted(result.keys()):
                print(f'{key} = {result[key]}')
                writer.write(f'{key} = {result[key]}\n')

    print('Done')

if __name__ == '__main__':
    main()
