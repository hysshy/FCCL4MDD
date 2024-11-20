import os
import argparse
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
from utils.data_manager import DataManager, setup_seed
from baselines.finetune import Finetune
from baselines.ewc import EWC
from baselines.wsdm import WSDM
def args_parser():
    parser = argparse.ArgumentParser(description='benchmark for federated continual learning')
    # General settings
    parser.add_argument('--exp_name', type=str, default='', help='name of this experiment')
    parser.add_argument('--save_dir', type=str, default="outputs", help='save data')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--g_sigma', type=float, default=0, help='sigma of updata g dp')
    parser.add_argument('--classifer_dp', type=float, default=0, help='dp add to classifer')
    parser.add_argument('--dataset', type=str, default="Derm7pt", help='which dataset')
    parser.add_argument('--train_data', type=str, default="/home/chase/shy/dataset/Derm7pt/Train", help='where dataset train')
    parser.add_argument('--test_data', type=str, default="/home/chase/shy/dataset/Derm7pt/Test", help='where dataset test')
    parser.add_argument('--tasks', type=int, default=5, help='num of tasks')
    parser.add_argument('--method', type=str, default="wsdm", help='choose a learner')
    parser.add_argument('--net', type=str, default="resnet18", help='choose a model')
    parser.add_argument('--com_round', type=int, default=100, help='communication rounds')
    parser.add_argument('--num_users', type=int, default=3, help='num of clients')
    parser.add_argument('--local_bs', type=int, default=32, help='local batch size')
    parser.add_argument('--local_ep', type=int, default=2, help='local training epochs')
    parser.add_argument('--beta', type=float, default=0, help='control the degree of non-IID')
    parser.add_argument('--num_class', type=int, default=20, help='number of classes in dataset')
    # Target settings
    parser.add_argument('--nums', type=int, default=8000, help='the num of synthetic data')
    parser.add_argument('--w_kd', type=float, default=10., help='for kd loss')

    args = parser.parse_args()
    return args


def train(args):
    setup_seed(args["seed"])
    data_manager = DataManager(
        args["dataset"],
        True,
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["train_data"],
        args["test_data"],
    )

    syn_data_manager = None
    if args["method"] == "finetune":
        learner = Finetune(args)
    elif args["method"] == "ewc":
        learner = EWC(args)
    elif args["method"] == "wsdm":
        learner = WSDM(args)
    for _ in range(data_manager.nb_tasks):
        learner.incremental_train(data_manager, syn_data_manager) # train for one task
        learner.eval_task()
        learner.after_task()
        learner.log_metrics()


if __name__ == '__main__':
    args = args_parser()
    args.init_cls = math.ceil(args.num_class / args.tasks)
    args.increment = args.init_cls
    if args.exp_name == "":
        args.exp_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.exp_name = f"beta_{args.beta}_tasks_{args.tasks}_seed_{args.seed}_sigma_{args.g_sigma}_{args.exp_name}"
    args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, args.exp_name)
    args = vars(args)
    train(args)
