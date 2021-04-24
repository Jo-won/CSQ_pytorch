import datetime
import os
import tarfile
import time
import subprocess
import atexit
import tqdm
import wandb
import numpy as np

import torch

from opts import parser

from lib.datasets.dataset_manager import dataset_manager
from lib.model.model_manager import model_manager, hash_center_multilables
from lib.utils.utils import AverageMeter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def prepare(args):
    # Directory Setting
    KST = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(tz=KST).strftime('%Y%m%d_%H%M%S')
    args.subproject_name = now + "_" + args.model_dir
    args.model_dir = os.path.join(args.model_root, now + "_" + args.model_dir)

    if os.path.isdir(args.model_dir) is False:
        os.mkdir(args.model_dir)

    source_path = os.path.join(args.model_dir, "source")
    if os.path.isdir(source_path) is False:
        os.mkdir(source_path)

    # Ckpt Saver
    ckpt_path = os.path.join(args.model_dir, "ckpt")
    if os.path.isdir(ckpt_path) is False:
        os.mkdir(ckpt_path)

    # Code Saver
    print("-----------Save File & Folder List-----------")
    tar = tarfile.open( os.path.join(source_path, 'sources.tar'), 'w' )
    file_list = os.listdir("./")
    file_list.sort()
    for fl in file_list:
        if os.path.islink(fl) is False:
            tar.add(fl)
            print(fl)
    tar.close()
    print("----------------------------------------------")

    # Argument Saver
    args_data = vars(args)
    args_list = ["{} : {}".format(k, v) for k, v in args_data.items()]
    with open(os.path.join(source_path, 'args.txt'), 'w') as f:
        f.write('\n'.join(args_list))

    # Device Setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Frequency Setting
    args.valid_frequency = args.save_frequency

    return args

def train(args, train_loader, index_loader, val_loader, model, optimizer, scheduler, criterion, hash_center):
   
    print("\n\n\n Start Train! \n\n\n")

    valid_result = {}
    for epc in range(args.epoch):
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        start = time.time()
        pbar = tqdm.tqdm(enumerate(train_loader), desc="Epoch : %d"%epc)
        
        for batch_i, (data, label) in pbar:
            center = hash_center_multilables(label, hash_center)

            data = data.float().cuda()
            label = label.cuda()
            center = center.cuda()

            if model.training:
                input_var = torch.autograd.Variable(data, requires_grad=False)
                center_var = torch.autograd.Variable(center, requires_grad=False)
            else:
                input_var = torch.autograd.Variable(data, volatile=True)
                center_var = torch.autograd.Variable(center, volatile=True)

            data_time.update(time.time() - start)
            start = time.time()

            output = model(input_var)

            loss = criterion(0.5*(output+1), 0.5*(center_var+1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            batch_time.update(time.time() - start)
            start = time.time()
            if args.wandb:
                wandb.log({"Train/Batch Loss": losses.avg})
            
            state_msg = (
                    'Epoch: {:4d}; Loss: {:0.5f}; Data time: {:0.5f}; Batch time: {:0.5f};'.format(epc, losses.avg, data_time.avg, batch_time.avg)
                )

            pbar.set_description(state_msg)
            
        scheduler.step()
        if args.wandb:
            wandb.log({"Train/Epoch Loss": losses.avg})

        if epc%args.save_frequency==0:
            mAP = evaluation(args, index_loader, val_loader, model)
            valid_result.update({epc : mAP})
            if args.wandb:
                wandb.log({"Valid/Epoch mAP@{}".format(args.R): mAP})
            

        if epc%args.save_frequency==0:
            ckpt_path = os.path.join(args.model_dir, "ckpt", "ckpt_{:05d}.pth".format(epc))
            torch.save({'epoch' : epc,
                        'state_dict' : model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'valid_result' : valid_result},
                        ckpt_path)

def evaluation(args, index_loader, test_loader, model, T=0):
    model.eval()

    index_output=torch.tensor([])
    index_label=torch.tensor([])
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm.tqdm(index_loader, desc="Make IndexSet")):
            data = data.float().cuda()
            label = label.cuda()
            output = model(data)

            index_output = torch.cat((index_output, output.cpu()), dim=0)
            index_label = torch.cat((index_label, label.cpu()), dim=0)
    index_output = index_output.numpy() 
    index_label = index_label.numpy()

    test_output=torch.tensor([])
    test_label=torch.tensor([])
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm.tqdm(test_loader, desc="Make TestSet")):
            data = data.float().cuda()
            label = label.cuda()
            output = model(data)

            test_output = torch.cat((test_output, output.cpu()), dim=0)
            test_label = torch.cat((test_label, label.cpu()), dim=0)
    test_output = test_output.numpy() 
    test_label = test_label.numpy()
    
    index_output[index_output<T] = -1
    index_output[index_output>=T] = 1
    test_output[test_output<T] = -1
    test_output[test_output>=T] = 1

    query_num = test_output.shape[0]  # total number for testing
    sim = np.dot(index_output, test_output.T) 
    ids = np.argsort(-sim, axis=0)  

    APx = []
    Recall = []

    for i in range(query_num):  # for i=0
        label = test_label[i]  # the first test labels
        idx = ids[:, i]
        
        topk_label = index_label[idx[0:args.R]]

        # In multilabel dataset, If one is correct, the correct answer will be processed.
        imatch = np.zeros((topk_label.shape[0],))
        for tpki, tpk_l in enumerate(topk_label):
            tpk_ind = np.where(tpk_l==1)[0]
            label_ind = np.where(label==1)[0]
            intersect = np.intersect1d(tpk_ind, label_ind)
            if intersect.shape[0]!=0:
                imatch[tpki]=1

        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, args.R + 1, 1)  #

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

    return np.mean(np.array(APx))



def main():
    args = parser.parse_args()

    args = prepare(args)
    if args.wandb:
        wandb.init(project=args.project_name)
        wandb.run.name = args.subproject_name
        wandb.config.update(args)

    train_loader, index_loader, val_loader, test_loader = dataset_manager(args)
    model, optimizer, scheduler, criterion, hash_center = model_manager(args)
    if args.wandb:
        wandb.watch(model)
    if args.train:
        train(args, train_loader, index_loader, val_loader, model, optimizer, scheduler, criterion, hash_center)
    if args.test:
        mAP=evaluation(args, index_loader, test_loader, model, T=0)
        print("mAP : {:.3f}".format(mAP))



if __name__ == "__main__":
    main()