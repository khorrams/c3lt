from args import init_parser
from datasets import init_dataloader
from torch import manual_seed
from c3lt import *

if __name__ == "__main__":

    args = init_parser()
    manual_seed(args.seed)
    init_log(args)

    dataloaders = init_dataloader(args, is_train=True), init_dataloader(args, is_train=False)

    if args.eval:
        eval_c3lt(args, dataloaders)
    else:
        train_c3lt(args, dataloaders)

