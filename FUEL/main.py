import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
# from latex_utils import latexify
from utils import setup_seed, construct_log, get_random_dir_name
from hco_model import MODEL
import time
from tensorboardX import SummaryWriter
import pandas as pd

parser = argparse.ArgumentParser()
# parser.add_argument('--other_comment', type = str, default="the_non_phd", help="the dim of the solution")
parser.add_argument('--use_saved_args', type = bool, default=False, help="the dim of the solution")
parser.add_argument('--exp-dir', type = str, default="", help="the dim of the solution")
parser.add_argument('--FedAve', type = bool, default=False, help="the dim of the solution")
parser.add_argument('--target_dir_name', type = str, default="", help="the dim of the solution")
parser.add_argument('--commandline_file', type = str, default="results/args.json", help="the dim of the solution")
parser.add_argument('--eps_g', type = float, default=0.1, help="max_epoch for unbiased_moe")
parser.add_argument('--weight_eps', type=float, default=0.5,
                    help="eps weight for specific eps")
parser.add_argument('--uniform_eps', action="store_true", help="max_epoch for unbiased_moe")
parser.add_argument('--eps_delta_l', type = float, default=1e-4, help="max_epoch for predictor")
parser.add_argument('--eps_delta_g', type = float, default=1e-4, help="iteras for printing the loss info")
parser.add_argument('--factor_delta', type = float, default=0.1, help="max_epoch for unbiased_moe")
parser.add_argument('--lr_delta', type = float, default=0.01, help="max_epoch for predictor")
parser.add_argument('--delta_l', type = float, default=0.5, help="max_epoch for predictor")
parser.add_argument('--delta_g', type = float, default=0.5, help="max_epoch for predictor")
parser.add_argument('--step_size', type = float, default=0.01, help="iteras for printing the loss info")
parser.add_argument('--max_epoch_stage1', type = int, default=800, help="iteras for printing the loss info")
parser.add_argument('--max_epoch_stage2', type = int, default=800, help="iteras for printing the loss info")
parser.add_argument('--per_epoches', type = int, default=50, help="iteras for printing the loss info")
parser.add_argument('--eval_epoch', type = int, default=20, help="iteras for printing the loss info")
parser.add_argument('--grad_tol', type = float, default=1e-4, help="iteras for printing the loss info")
parser.add_argument('--ckpt_dir', type = str, default= "results/models", help="iteras for printing the loss info")
parser.add_argument('--log_dir', type = str, default= "results", help="iteras for printing the loss info")
parser.add_argument('--log_name', type = str, default= "log", help="iteras for printing the loss info")
parser.add_argument('--board_dir', type = str, default= "results/board", help="iteras for printing the loss info")
parser.add_argument('--store_xs', type = bool, default=False, help="iteras for printing the loss info")
parser.add_argument('--seed', type = int, default=1, help="iteras for printing the loss info")
parser.add_argument('--batch_size', type = list, default=[100, 100], help="iteras for printing the loss info")
parser.add_argument('--shuffle', type = bool, default=True, help="iteras for printing the loss info")
parser.add_argument('--drop_last', type = bool, default=False, help="iteras for printing the loss info")
parser.add_argument('--data_dir', type = str, default="data", help="iteras for printing the loss info")
parser.add_argument('--dataset', type = str, default="adult", help="[adult, eicu_d, eicu_los, bank]")
parser.add_argument('--load_epoch', type = str, default=0, help="iteras for printing the loss info")
parser.add_argument('--global_epoch', type = int, default=0, help="iteras for printing the loss info")
parser.add_argument('--num_workers', type = int, default=0, help="iteras for printing the loss info")
parser.add_argument('--n_feats', type = int, default=10, help="iteras for printing the loss info")
parser.add_argument('--n_hiddens', type = int, default=40, help="iteras for printing the loss info")
parser.add_argument('--sensitive_attr', type = str, default="race", help="iteras for printing the loss info")
parser.add_argument('--policy', type = str, default="two_stage", help="[alternating, two_stage]")
parser.add_argument('--uniform', action="store_true",  help="uniform mode, without any fairness contraints")
parser.add_argument('--disparity_type', type= str, default= "DP",  help="uniform mode, without any fairness contraints")
parser.add_argument('--baseline_type', type= str, default= "none",  help="fedave_fair, individual_fair")
parser.add_argument('--weight_fair', type= float, default= 1.0,  help="weight for disparity")
parser.add_argument('--valid', action='store_true', help="iteras for printing the loss info")
parser.add_argument('--mix_dataset', type=bool, default=False, help="")
parser.add_argument('--bootstrap', type=int, default=0, help="")
parser.add_argument('--new_trial', type=int, default=0, help="")
parser.add_argument('--new_trial_train_rate', type=float, default=0.35, help="")
parser.add_argument('--new_trial_test_rate', type=float, default=0.15, help="")
parser.add_argument('--new_trial_whole_rate', type=float, default=0.5, help="")
parser.add_argument('--new_trial_method', type=str)
parser.add_argument('--new_trial_bias_rate', type=float, default=0.5)
args = parser.parse_args()


args.eps = [args.eps_g, args.eps_delta_l, args.eps_delta_g]
args.train_dir = os.path.join(args.data_dir, args.dataset, "train")
args.test_dir = os.path.join(args.data_dir, args.dataset, "test")
args.target_dir_name = os.path.join('out', args.target_dir_name)
args.ckpt_dir_ = args.ckpt_dir
args.log_dir_ = args.log_dir
args.board_dir_ = args.board_dir
args.commandline_file_ = args.commandline_file
args.done_all_dir = os.path.join(args.target_dir_name, 'doneall')
def update_args_dir():
    args.ckpt_dir = os.path.join(args.target_dir_name, args.ckpt_dir_)
    args.log_dir = os.path.join(args.target_dir_name, args.log_dir_)
    args.board_dir = os.path.join(args.target_dir_name, args.board_dir_)
    args.done_dir = os.path.join(args.target_dir_name, "done")
    args.commandline_file = os.path.join(args.target_dir_name, args.commandline_file_)
update_args_dir()

# args.step_size = 0.03 
# args.eps_g = 0.05 
# args.dataset = 'adult'
# args.max_epoch_stage1 = 800 
# args.max_epoch_stage2 = 1500 
# args.seed = 1 
# args.target_dir_name = 'test_DP_0-05' 
# args.uniform_eps = True 
# args.bootstrap = 100


if __name__ == '__main__':
    if args.new_trial != 0:
        setup_seed(seed = args.seed)
        args.target_dir_name = os.path.join(args.target_dir_name, 'xxx')
        for new_trial_round in range(args.new_trial):
            # args.new_trial_round = new_trial_round
            args.target_dir_name = os.path.join(os.path.split(args.target_dir_name)[0], str(new_trial_round))
            update_args_dir()
            writer = SummaryWriter(log_dir = args.board_dir)
            logger = construct_log(args)
            logger.info('new trial: round {}'.format(new_trial_round))
            def new_trial_valid(model: MODEL, load_epoch):
                logger.info('new trial ({}): start valid'.format(new_trial_round))
                excel_writer = pd.ExcelWriter(os.path.join(
                        args.target_dir_name, 
                        'results',
                        f'{args.new_trial_train_rate:.3f}_{args.new_trial_test_rate:.3f}_{args.new_trial_whole_rate:.3f}.xlsx'
                    ),
                    engine='openpyxl', mode='w')
                def valid_to_excel(another=False, bootstrap=False):
                    table_data = {}
                    for i in range(args.n_clients):
                        table_data["client_{}_disparity".format(i)] = []
                    for _ in range(args.bootstrap if bootstrap else 1):
                        _, _, diss, _, _ = model.valid_stage1(False, load_epoch, another=another, bootstrap=bootstrap)
                        for i, dis in enumerate(diss):
                            table_data["client_{}_disparity".format(i)].append(dis)
                    data_frame = pd.DataFrame(table_data)
                    data_frame.to_excel(excel_writer = excel_writer, index = False, 
                                        sheet_name=('another' if another else 'origin') + ('_bootstrap' if bootstrap else ''))
                valid_to_excel(False, False)
                valid_to_excel(True, False)
                if args.bootstrap != 0:
                    valid_to_excel(False, True)
                    valid_to_excel(True, True)
                excel_writer.close()
            if args.valid:
                args.load_epoch = args.max_epoch_stage1 + args.max_epoch_stage2 - 1
                model = MODEL(args, logger, writer)
                new_trial_valid(model, args.load_epoch)
            else:
                logger.info('new trial ({}): start train'.format(new_trial_round))
                while True:
                    try:
                        model = MODEL(args, logger, writer)
                        model.train()
                        break
                    except AttributeError as e:
                        print(e)
                        logger.info('new trial ({}): error occured, restart this round'.format(new_trial_round))
                logger.info('new trial ({}): finish train'.format(new_trial_round))
                new_trial_valid(model, args.max_epoch_stage1 + args.max_epoch_stage2 - 1)
            model.save_log()
        os.makedirs(args.done_all_dir, exist_ok = True)
    else:
        writer = SummaryWriter(log_dir = args.board_dir)
        if args.use_saved_args:
            with open(args.commandline_file, "r") as f:
                args.__dict__ = json.load(f)
        else:
            pass
        os.makedirs(args.log_dir, exist_ok = True)
        # linux
        # os.system("cp *.py " + args.target_dir_name)
        # windows
        # os.system("copy *.py " + args.target_dir_name)
        logger = construct_log(args)
        setup_seed(seed = args.seed)
        model = MODEL(args, logger, writer)
        if args.valid:
            if args.load_epoch == 0:
                raise "need argument load_epoch"
            if args.bootstrap != 0:
                if args.dataset == "eicu":
                    table_data = {}
                    for i in range(11):
                        table_data["client_{}_disparity".format(i)] = []
                else:
                    table_data = {"client_0_disparity":[], "client_1_disparity":[]}
                
                excel_writer = pd.ExcelWriter(os.path.join(args.log_dir, 'bootstrap.xlsx'), engine='openpyxl', mode='w')
                for _ in range(args.bootstrap):
                    losses, accs, diss, pred_diss, aucs = model.valid_stage1(False, args.load_epoch, bootstrap=True)
                    for i, (dis, pred_dis) in enumerate(zip(diss, pred_diss)):
                        table_data["client_{}_disparity".format(i)].append(dis)
                        # table_data["client_{}_pred_disparity".format(i)].append(pred_dis)
                data_frame = pd.DataFrame(table_data)
                data_frame.to_excel(excel_writer = excel_writer, index = False)
                excel_writer.close()
            else:
                losses, accs, diss, pred_diss, aucs = model.valid_stage1(False, args.load_epoch)
        else:
            model.train()
        model.save_log()
    print('done!')

