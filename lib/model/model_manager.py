import os
import torch
import torch.nn as nn

from collections import OrderedDict

from .mfnet_csq.mfnet_3d import MFNET_3D

def model_manager(args):

    model = MFNET_3D(args.hash_bit, pretrained=args.pretrained_2d)
    if (args.pretrained_3d is not None) and (args.pretrained_3d!="None"):
        assert os.path.exists(args.pretrained_3d), "cannot locate: `{}'".format(args.pretrained_3d)
        print("Initializer:: loading model states from: `{}'".format(args.pretrained_3d))
        checkpoint = torch.load(args.pretrained_3d)
        load_state(model, checkpoint['state_dict'], strict=False)

    model = model.cuda()

    param_base_layers = []
    param_new_layers = []
    name_base_layers = []
    for name, param in model.named_parameters():
        if args.finetune:
            if ('fc' in name) or ('encoder' in name):
                param_new_layers.append(param)
                print(name)
            else:
                param_base_layers.append(param)
                name_base_layers.append(name)
        else:
            param_new_layers.append(param)
    if name_base_layers:
        out = "[\'" + '\', \''.join(name_base_layers) + "\']"
        print("Optimizer:: >> recuding the learning rate of {} params: {}".format(len(name_base_layers),
                     out if len(out) < 300 else out[0:150] + " ... " + out[-150:]))
    
    optimizer = torch.optim.SGD([{'params': param_base_layers, 'lr_mult': 0.2},
                                 {'params': param_new_layers, 'lr_mult': 1.0}],
                                lr=args.lr_base,
                                momentum=0.9,
                                weight_decay=0.0001,
                                nesterov=True)
    if args.checkpoint:
        epoch_start = load_checkpoint(model, args.checkpoint, optimizer=optimizer if args.train is True else None)

    scheduler = None
    if args.lr_scheduler=='cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                        T_0=10, 
                                                                        T_mult=2, 
                                                                        eta_min=args.lr_base, 
                                                                        last_epoch=-1)
    elif args.lr_scheduler=='multi_step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=args.lr_steps,
                                                         gamma=args.lr_factor,
                                                         last_epoch=-1)
    criterion = torch.nn.BCELoss().cuda()

    
    hash_root = "lib/datasets/data/{}/raw/".format(args.dataset.upper())
    hash_path = os.listdir(hash_root)
    hash_path = [os.path.join(hash_root, i) for i in hash_path if str(args.hash_bit) in i][0]
    print("\n\n")
    print(f'-----------------------{hash_path}-----------------')
    hash_center = torch.load(f'{hash_path}')

    return model, optimizer, scheduler, criterion, hash_center


def hash_center_multilables(labels, Hash_center): # label.shape: [batch_size, num_class], Hash_center.shape: [num_class, hash_bits]
    random_center = torch.randint_like(Hash_center[0], 2)
    is_start = True
    
    for label in labels:

        one_labels = torch.nonzero((label == 1))
        one_labels = one_labels.squeeze(1)
        Center_mean = torch.mean(Hash_center[one_labels], dim=0)

        Center_mean[Center_mean<0] = -1
        
        Center_mean[Center_mean>0] = 1
        random_center[random_center==0] = -1  
        Center_mean[Center_mean == 0] = random_center[Center_mean == 0]  
        Center_mean = Center_mean.view(1, -1)

        if is_start:  # the first time
            hash_center = Center_mean
            is_start = False
        else:
            hash_center = torch.cat((hash_center, Center_mean), 0)

    return hash_center


def load_state(model, state_dict, strict=False):
    if strict:
        model.load_state_dict(state_dict=state_dict)
    else:
        # customized partialy load function
        net_state_keys = list(model.state_dict().keys())
        for name, param in state_dict.items():
            if name.split('module.')[-1] in model.state_dict().keys():
                name = name.split('module.')[-1]
            if name in model.state_dict().keys():
                dst_param_shape = model.state_dict()[name].shape
                if param.shape == dst_param_shape:
                    model.state_dict()[name].copy_(param.view(dst_param_shape))
                    net_state_keys.remove(name)
        # indicating missed keys
        if net_state_keys:
            print(">> Failed to load: {}".format(net_state_keys))
            return False
    return True

def load_checkpoint(model, checkpoint, optimizer=None):

    assert os.path.exists(checkpoint), "Failed to load: {} (file not exist)".format(checkpoint)

    checkpoint = torch.load(checkpoint)

    all_params_matched = load_state(model, checkpoint['state_dict'], strict=True)
    if optimizer:
        if 'optimizer' in checkpoint.keys() and all_params_matched:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print(">> Failed to load optimizer state from: `{}'".format(load_path))

    return checkpoint['epoch']