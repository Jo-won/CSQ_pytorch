import os

import torch
import torchvision

from . import transform, sampler, iterator

def dataset_manager(args):
    name = args.dataset.lower()

    if name == "ucf101":
        num_classes = 101
        train, index, val, test = get_ucf101(args)
        collate_fn = collate_func_custom
    else:
        assert NotImplementedError("iter {} not found".format(name))

    global classes_num 
    classes_num = num_classes
    
    train_loader = torch.utils.data.DataLoader(train,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, collate_fn=collate_fn)
    
    index_loader = torch.utils.data.DataLoader(index,
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(val,
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False, collate_fn=collate_fn)
    
    test_loader = torch.utils.data.DataLoader(test,
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False, collate_fn=collate_fn)
    
    return train_loader, index_loader, val_loader, test_loader


def get_ucf101(args,
               mean=[124 / 255, 117 / 255, 104 / 255],
               std=[1 / (.0167 * 255)] * 3):

    normalize = transform.Normalize(mean=mean, std=std, group=True)

    train_sampler = sampler.TSNSampling(num=args.clip_length,
                                        random_shift=True, 
                                        seed=(args.seed+0))
    index_sampler = sampler.TSNSampling(num=args.clip_length,
                                        random_shift=False, 
                                        seed=(args.seed+0))
    val_sampler   = sampler.TSNSampling(num=args.clip_length,
                                        random_shift=False, 
                                        seed=(args.seed+0))
    test_sampler  = sampler.TSNSampling(num=args.clip_length,
                                        random_shift=False, 
                                        seed=(args.seed+0))


    train_transform = transform.Compose([
                                            transform.GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                                            transform.RandomHorizontalFlip(group=True),
                                            transform.RandomHLS(vars=[15, 35, 25], group=True),
                                            transform.ToTensor(group=True),
                                            normalize,
                                        ], aug_seed=(args.seed+1))
    index_transform = transform.Compose([
                                            transform.Resize(256, group=True),
                                            transform.CenterCrop(224, group=True),
                                            transform.ToTensor(group=True),
                                            normalize,
                                        ], aug_seed=(args.seed+1))
    val_transform   = transform.Compose([
                                            transform.Resize(256, group=True),
                                            transform.CenterCrop(224, group=True),
                                            transform.ToTensor(group=True),
                                            normalize,
                                        ], aug_seed=(args.seed+1))
    test_transform  = transform.Compose([
                                            transform.Resize(256, group=True),
                                            transform.CenterCrop(224, group=True),
                                            transform.ToTensor(group=True),
                                            normalize,
                                        ], aug_seed=(args.seed+1))

    
    train = iterator.VideoIter( args=args,
                                video_prefix=os.path.join(args.data_root, 'video'),
                                txt_list=os.path.join(args.data_root, 'ucfTrainTestlist', 'trainlist01_ilp.txt'),
                                sampler=train_sampler,
                                video_transform=train_transform,
                                force_color=True,
                                shuffle_list_seed=(args.seed+2)
                                )
    index = iterator.VideoIter( args=args,
                                video_prefix=os.path.join(args.data_root, 'video'),
                                txt_list=os.path.join(args.data_root, 'ucfTrainTestlist', 'trainlist01_ilp.txt'),
                                sampler=index_sampler,
                                video_transform=index_transform,
                                force_color=True,
                                shuffle_list_seed=None
                                )
    val   = iterator.VideoIter( args=args,
                                video_prefix=os.path.join(args.data_root, 'video'),
                                txt_list=os.path.join(args.data_root, 'ucfTrainTestlist', 'testlist01_ilp.txt'),
                                sampler=val_sampler,
                                video_transform=val_transform,
                                force_color=True,
                                shuffle_list_seed=None
                                )
    test  = iterator.VideoIter( args=args,
                                video_prefix=os.path.join(args.data_root, 'video'),
                                txt_list=os.path.join(args.data_root, 'ucfTrainTestlist', 'testlist01_ilp.txt'),
                                sampler=test_sampler,
                                video_transform=test_transform,
                                force_color=True,
                                shuffle_list_seed=None
                                )
    
    return train, index, val, test
def collate_func_custom(batch):
    fs=torch.tensor([])
    ls=torch.zeros((len(batch), classes_num))
    for i, (f, l) in enumerate(batch):
        f = torch.transpose(f, 0, 1)
        f = f.unsqueeze(0)
        fs = torch.cat((fs, f), dim=0)
        ls[i, l] = 1
    
    return fs, ls
