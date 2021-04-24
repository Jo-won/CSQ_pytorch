import argparse

parser = argparse.ArgumentParser(description="Option")
# prepare
parser.add_argument('--model_root', type=str, default="MODEL_DIR",
                    help="set logging file.")
parser.add_argument('--model_dir', type=str, default="DEBUG",
                    help="set logging file.")
parser.add_argument('--gpus', type=str, default="0",
                    help="define gpu id")
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--project_name', type=str, default="deDEBUGbug",
                    help="set logging file.")

# dataset_manager
parser.add_argument('--dataset', default='UCF101', choices=['UCF101'],
                    help="path to dataset")
parser.add_argument('--data_root', type=str, default="DATA_DIR",
                    help="set dataset's path.")
parser.add_argument('--workers', default=0, type=int, 
                    help="define the workers.")
parser.add_argument('--batch_size', default=1, type=int, 
                    help="define the batch size.")
parser.add_argument('--clip_length', default=16, type=int, 
                    help="define the length of each input sample.")
parser.add_argument('--clip_interval', default=0, type=int, 
                    help="define the interval of each input sample.")


# model_manager
parser.add_argument('--pretrained_2d', type=bool, default=True,
                    help="load default 2D pretrained model.")
parser.add_argument('--pretrained_3d', type=str, 
                    default='./lib/model/mfnet_csq/pretrained/MFNet3D_Kinetics-400_72.8.pth',
                    help="load default 3D pretrained model.")
parser.add_argument('--finetune', action='store_true',
                    help="apply different learning rate for different layers")
parser.add_argument('--hash_bit', type=int, default=64, choices=[16, 32, 64],
                    help="define the length of hash bits.")
parser.add_argument('--lr_base', type=float, default=0.005,
                    help="learning rate") # in UCF101 = 0.05
parser.add_argument('--lr_scheduler', default='multi_step', choices=['multi_step', 'cosine_annealing'],
                    help="set lr scheduler")
parser.add_argument('--lr_steps', type=list, default=[int(x*10) for x in range(1, 20)],
                    help="if multi step, number of samples to pass before changing learning rate") # in UCF101 = [1,2,3,...,19,20]
parser.add_argument('--lr_factor', type=float, default=0.3,
                    help="if multi step, reduce the learning with factor")
parser.add_argument('--checkpoint', type=str, default=None,
                    help="resume train.")

# main
parser.add_argument('--train', action='store_true', 
                    help="if want to train")
parser.add_argument('--test', action='store_true', 
                    help="if want to test")
parser.add_argument('--epoch', default=10000, type=int, 
                    help="define the epoch.")
parser.add_argument('--save_frequency', type=float, default=1,
                    help="save once after N epochs")
parser.add_argument('--valid_frequency', type=float, default=0,
                    help="validate once after N epochs, if 0, it will use save_frequency")
parser.add_argument('--R', default=100, type=int,
                    help="mAP@R")
parser.add_argument('--wandb', action='store_true', 
                    help="if want to use wandb")