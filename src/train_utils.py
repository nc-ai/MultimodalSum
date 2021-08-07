import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AdamW
from transformers import BartTokenizer
from transformers.optimization import get_linear_schedule_with_warmup


def set_environments(args):
    if args.local_rank == 0:
        if not os.path.exists(args.ckpt):
            os.mkdir(args.ckpt)
        torch.save(vars(args), '%s/training_args.bin' % args.ckpt)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        print('GPU {}/{}'.format(args.gpu, args.world_size))
    return args

def get_dataloader(args, data_class):
    tokenizer = BartTokenizer.from_pretrained(args.bart_tokenizer)
    data_train = data_class(tokenizer=tokenizer, mode='train', dataset=args.dataset)
    data_val = data_class(tokenizer=tokenizer, mode='val', dataset=args.dataset)

    if args.distributed:
        train_sampler = DistributedSampler(data_train, shuffle=True)
        val_sampler = DistributedSampler(data_val, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(data_train, args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_dataloader = DataLoader(data_val, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    return data_train, train_sampler, train_dataloader, val_dataloader

def get_optimizer(lr, no_decay, named_parameters, special_condition):
    if special_condition == None:
        special_condition = lambda n : True
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_parameters if special_condition(n) and (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
        {'params': [p for n, p in named_parameters if special_condition(n) and (any(nd in n for nd in no_decay))], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer

def get_scheduler(args, t_epoch, optimizer):
    t_total = t_epoch * args.num_epochs
    warmup_step = int(t_total * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
    return scheduler

def train_model(args, model, train_sampler, train_dataloader, val_dataloader, train, validate, optimizer, scheduler, t_epoch, save_option='whole'):
    start_time = time.time()
    val_loss_list = []
    for e in range(args.num_epochs):
        print('Epoch {}'.format(e+1))
        if args.distributed:
            train_sampler.set_epoch(e)
        if e != 0:
            train_dataloader.dataset.set_epoch()

        train(start_time, train_dataloader, model, optimizer, scheduler, e, t_epoch)
        val_loss = validate(val_dataloader, model, e)
        
        torch.cuda.synchronize()
        if args.local_rank == 0:
            val_loss_list.append(val_loss)
            min_val_loss = min(val_loss_list)
            if ((args.early_stopping) and (val_loss <= min_val_loss)) or (not args.early_stopping):

                if args.distributed:
                    save_model = model.module
                else:
                    save_model = model

                if save_option == 'text':
                    save_model = model.bart_model
                elif save_option == 'img':
                    save_model = model.img_encoder
                elif save_option == 'table':
                    save_model = model.table_encoder

                torch.save(save_model.state_dict(), '%s/pytorch_model.bin' % args.ckpt)
                torch.save({'epoch':e, 'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict()}, '%s/training_state.bin' % args.ckpt)