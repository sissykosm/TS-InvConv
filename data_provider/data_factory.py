from data_provider.data_loader_ts import PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, \
    Dataset_pt_loader, load_and_concat_pt_data, normalize_time_series, zero_pad_sequence, calculate_padding, load_UCR, TSDataset
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch 
import os

data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'class_pt': Dataset_pt_loader,
    'UCR': load_UCR,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        if args.data == 'UCR':
            data_set, train_set, vali_set, test_set = load_UCR(args, args.root_path, args.model_id, (0.8,0.2,0.0),  batch_size, augmentation=args.augmentation, ood_flag=args.ood_test) #'Random_Offset'
            
            train_loader = DataLoader(train_set, 
                                    batch_size=batch_size, 
                                    shuffle=True,
                                    collate_fn=lambda x: collate_fn(x, max_len=args.seq_len))

            vali_loader = DataLoader(vali_set, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    collate_fn=lambda x: collate_fn(x, max_len=args.seq_len))

            test_loader = DataLoader(test_set, 
                                    batch_size=batch_size, 
                                    shuffle=False,
                                    collate_fn=lambda x: collate_fn(x, max_len=args.seq_len))

            if flag == 'TRAIN':
                return data_set, train_set, train_loader, vali_set, vali_loader
            else:
                return test_set, test_loader
        else:
            drop_last = False
            data_set = Data(
                args = args,
                root_path=args.root_path,
                flag=flag,
            )

            if flag == 'TRAIN':

                # Use the full training dataset as "validation" as well (no split)
                train_indices = np.arange(len(data_set))  # All indices used for training and validation

                # Create Subset dataset using all indices for both training and validation
                train_set = Subset(data_set, train_indices)
                vali_set = train_set  # Validation set is the same as the training set

                train_loader = DataLoader(
                    train_set,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last,
                    collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
                )

                vali_loader = DataLoader(
                    vali_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    drop_last=drop_last,
                    collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
                )

                return data_set, train_set, train_loader, vali_set, vali_loader

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
            )
            return data_set, data_loader

    elif args.task_name == 'classification_pt':
        print(flag)
        if flag=='TRAIN':
            if args.model_id == 'Fault-diagnosis':
                train_file = torch.load(os.path.join(args.root_path, f"train_a.pt"))
                vali_file = torch.load(os.path.join(args.root_path, f"val_a.pt"))
            else:
                train_file = torch.load(os.path.join(args.root_path, f"train.pt"))
                vali_file = torch.load(os.path.join(args.root_path, f"val.pt"))

            seq_len = train_file["samples"].shape[-1]

            train_dataset = Dataset_pt_loader(train_file, args)
            vali_dataset = Dataset_pt_loader(vali_file, args)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
            )

            vali_loader = torch.utils.data.DataLoader(
                vali_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=args.num_workers,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
            )
            return train_dataset, train_dataset, train_loader, vali_dataset, vali_loader

        if args.model_id == 'Fault-diagnosis':
            #test_file = torch.load(os.path.join(args.root_path, f"test_b.pt"))
            if flag=='TEST' or flag=='TESTA':
                test_file = torch.load(os.path.join(args.root_path, f"test_a.pt"))
            elif flag=='TESTB':
                test_file = torch.load(os.path.join(args.root_path, f"test_b.pt"))
            elif flag=='TESTC':
                test_file = torch.load(os.path.join(args.root_path, f"test_c.pt"))
            elif flag=='TESTD':
                test_file = torch.load(os.path.join(args.root_path, f"test_d.pt"))
        else:
            test_file = torch.load(os.path.join(args.root_path, f"test.pt"))

        test_dataset = Dataset_pt_loader(test_file, args)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return test_dataset, test_loader

    else:
        raise NotImplementedError("Only classification and anomaly detection datasets are supported.")
        return data_set, data_loader


