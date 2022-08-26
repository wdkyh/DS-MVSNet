import torch
from torch.utils import data
import torch.distributed as dist
from torch.utils.data import RandomSampler, SequentialSampler

from .blended import BlendedDataset
from .dtu import MVSDataset as DtuDataset
from .general_eval import MVSDataset as EvalDataset


def get_loader(args, datapath, listfile, nviews, mode="train"):
    if args.dataset_name == "dtu":
        dataset = DtuDataset(datapath, listfile, mode, nviews, args.img_size, args.numdepth, args.interval_scale)
    elif args.dataset_name == "general_eval":
        dataset = EvalDataset(datapath, listfile, mode, nviews, args.numdepth, args.interval_scale, args.inverse_depth,
                              max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res)
    elif args.dataset_name == "blended":
        dataset = BlendedDataset(datapath, listfile, mode, nviews, args.numdepth, args.interval_scale)
    else:
        raise NotImplementedError("Don't support dataset: {}".format(args.dataset_name))

    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    else:
        sampler = RandomSampler(dataset) if (mode == "train") else SequentialSampler(dataset)

    data_loader = data.DataLoader(dataset, args.batch_size, sampler=sampler, num_workers=args.num_worker, drop_last=(mode == "train"), pin_memory=True)

    return data_loader, sampler
