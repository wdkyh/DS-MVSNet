import torch
import argparse
from model import Model

parser = argparse.ArgumentParser(description="UniMVSNet args")

# network
parser.add_argument("--fea_mode", type=str, default="fpn", choices=["fpn", "unet"])
parser.add_argument("--agg_mode", type=str, default="variance", choices=["variance", "adaptive"])
parser.add_argument("--depth_mode", type=str, default="regression", choices=["regression", "classification", "unification"])
parser.add_argument("--ndepths", type=int, nargs='+', default=[48, 32, 8])
parser.add_argument("--interval_ratio", type=float, nargs='+', default=[4, 2, 1])

# dataset
parser.add_argument("--datapath", type=str)
parser.add_argument("--trainlist", type=str)
parser.add_argument("--testlist", type=str)
parser.add_argument("--dataset_name", type=str, default="dtu", choices=["dtu", "general_eval", "blended"])
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')
parser.add_argument("--nviews", type=int, default=5)
# only for train and eval
parser.add_argument("--img_size", type=int, nargs='+', default=[512, 640])
parser.add_argument("--inverse_depth", action="store_true")

# training and val
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--scheduler', type=str, default="steplr", choices=["steplr", "cosinelr"])
parser.add_argument('--warmup', type=float, default=0.2, help='warmup epochs')
parser.add_argument('--milestones', type=float, nargs='+', default=[10, 12, 14], help='lr schedule')
parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay at every milestone')
parser.add_argument('--resume', type=str, help='path to the resume model')
parser.add_argument('--log_dir', type=str, help='path to the log dir')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument("--val", action="store_true")
parser.add_argument("--sync_bn", action="store_true")
parser.add_argument("--blended_train", action="store_true")
parser.add_argument('--ssim_loss_weight', type=float, default=6.0, help='SSIM loss weight')
parser.add_argument('--smooth_loss_weight', type=float, default=0.1, help='depth smooth loss weight')
parser.add_argument('--reconstr_loss_weight', type=float, default=12.0, help='reconstruction loss weight')
parser.add_argument('--depth_consistency_weight', type=float, default=12.0, help='depth consistency loss weight')
parser.add_argument('--dlossw', type=float, nargs='+', default=[0.5, 1.0, 2.0], help='depth loss weight for different stage')

# testing & fusing
parser.add_argument('--outdir', default='./outputs', help='output dir')

# testing
parser.add_argument("--test", action="store_true")
parser.add_argument('--num_view', type=int, default=5, help='num of view')
parser.add_argument('--max_h', type=int, default=864, help='testing max h')
parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')
parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

# fusing
parser.add_argument("--fuse", action="store_true")
parser.add_argument('--points_dir', default='./points', help='output dir')
parser.add_argument('--fusepath_single_scene', help='testing data path for single scene')
parser.add_argument('--filter_method', type=str, default='gipuma', choices=["gipuma", "pcd", "dypcd"], help="filter method")
# pcd or dypcd
parser.add_argument('--conf', type=float, nargs='+', default=[0.1, 0.15, 0.9], help='prob confidence, for pcd and dypcd')
parser.add_argument('--thres_view', type=int, default=5, help='threshold of num view, only for pcd')
# dypcd
parser.add_argument('--dist_base', type=float, default=1 / 4)
parser.add_argument('--rel_diff_base', type=float, default=1 / 1300)
# gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='./fusibile')
parser.add_argument('--prob_threshold', type=float, default='0.3')
parser.add_argument('--disp_threshold', type=float, default='0.25')
parser.add_argument('--num_consistent', type=float, default='3')

# visualization
parser.add_argument("--vis", action="store_true")
parser.add_argument('--depth_path', type=str)
parser.add_argument('--depth_img_save_dir', type=str, default="./")

# device and distributed
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

# cuda memery
parser.add_argument('--mem', type=int, default=0, help="GB")

args = parser.parse_args()

# print arguments
def print_args(args):
    print("####################################  args  ####################################")
    for k, v in args.__dict__.items():
        print("{0: <20}\t{1: <35}\t{2: <20}".format(k, str(v), str(type(v))))
    print("################################################################################")
    print()

if __name__ == '__main__':
    

    model = Model(args)

    if not (args.fuse or args.vis):
        print_args(args)

    if args.mem > 0:
        mem_holder = torch.rand(
            [256, 1024, args.mem * 1024],
            dtype=torch.float32,
            device=torch.device('cuda')
        )

    model.main()
    
    if args.mem > 0: del mem_holder 
