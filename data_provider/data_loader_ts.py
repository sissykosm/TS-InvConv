import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
from sklearn.preprocessing import StandardScaler

from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ------- pt dataset --------, har-uci, sleep-edf
def load_and_concat_pt_data(root_path, subset_names):
    all_samples = []
    all_labels = []

    for subset in subset_names:
        subset_path = os.path.join(root_path, subset)
        data_file = torch.load(subset_path)

        # Extract samples and labels (assuming 'labels' is always present)
        samples = data_file["samples"]
        labels = data_file["labels"]

        # Convert samples and labels to torch tensors if needed
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # Ensure samples are 3D (N, C, L) by adding a channel dimension if needed
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(1)

        # Collect all samples and labels
        all_samples.append(samples.float())
        all_labels.append(labels.long().squeeze())

    # Concatenate samples and labels along the first dimension
    concatenated_samples = torch.cat(all_samples, dim=0)
    concatenated_labels = torch.cat(all_labels, dim=0)

    # Return concatenated data as a dictionary
    return {"samples": concatenated_samples, "labels": concatenated_labels}

def normalize_time_series(data):
    mean = data.mean()
    std = data.std()
    normalized_data = (data - mean) / std
    return normalized_data

def zero_pad_sequence(input_tensor, pad_length):
    return torch.nn.functional.pad(input_tensor, (0, pad_length))

def calculate_padding(seq_len, patch_size):
    padding = patch_size - (seq_len % patch_size) if seq_len % patch_size != 0 else 0
    return padding


class Dataset_pt_loader(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_file, args):
        super(Dataset_pt_loader, self).__init__()
        self.data_file = data_file

        # Load samples and labels
        x_data = data_file["samples"]  # dim: [#samples, #channels, Seq_len]

        # pre_process
        if not args.no_normalize:
            print('Norm')
            x_data = normalize_time_series(x_data)
        else:
            print('No norm')

        y_data = data_file.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data).squeeze()

        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)

        self.x_data = x_data.float()
        self.y_data = y_data.long().squeeze() if y_data is not None else None

        self.max_seq_len = x_data.shape[-1]
        self.feature_df = x_data

        self.class_names = np.unique(y_data)
        # print(x_data.size())
        # print(y_data.size())

    def __getitem__(self, index):
        x = self.x_data[index].permute(1,0)
        y = self.y_data[index] if self.y_data is not None else None
        return x, y

    def __len__(self):
        return len(self.x_data)

class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        if not args.no_normalize:
            print('Norm')
            normalizer = Normalizer()
            self.feature_df = normalizer.normalize(self.feature_df)
        else:
            print('No norm')
        #print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        #print(torch.from_numpy(batch_x).size())
        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)

##### DATASET PREPROCESSING ######
# - Scripts generating the train and the test dataset and the relevant information from a specific dataset
def load_UCR(args, DIR, dataset_name, train_val_test_r, batch_size, augmentation=None, ood_flag = 0):
    np.random.seed(42)

    # load preprocessed datasets if missing values or lengths that vary
    if dataset_name in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'DodgerLoopDay',
        'DodgerLoopGame',
        'DodgerLoopWeekend',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PLAID',
        'ShakeGestureWiimoteZ'
    ]:
        DIR = os.path.join(DIR, "Missing_value_and_variable_length_datasets_adjusted")

    print('Augmentation type for UCR:', augmentation)

    train_file = os.path.join(DIR, dataset_name + "_TRAIN.tsv")
    test_file = os.path.join(DIR, dataset_name + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train_X = train_array[:, 1:].astype(np.float64)
    train_y = np.vectorize(transform.get)(train_array[:, 0])
    test_X = test_array[:, 1:].astype(np.float64)
    test_y = np.vectorize(transform.get)(test_array[:, 0])

    train_X = np.expand_dims(train_X, 1)
    test_X = np.expand_dims(test_X, 1)

    # extract length, channels
    dict_label = {}
    count = 0
    for val in set(train_y):
        dict_label[val] = count
        count += 1

    all_class_all = []
    all_label = []
    for i in range(len(train_X)):
        all_class_all.append(np.array([train_X[i]]))
        all_label.append(dict_label[train_y[i]])

    original_length = len(all_class_all[0][0][0])
    num_classes = len(set(train_y))
    original_dim = 1
    nb_instance = len(all_class_all)

    dataset_mat = TSDataset(args, train_X, train_y)

    if args.val_split:
        print('Train Val Split')
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, 
                                                                test_size= train_val_test_r[1],random_state=11081994)
    else:
        print('Use Train as Val')
        val_X = train_X
        val_y = train_y
    
    if dataset_name in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
    #    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
        mean = np.nanmean(train_X)
        std = np.nanstd(train_X)
        train_X = (train_X - mean) / std
        val_X = (val_X - mean) / std
        test_X = (test_X - mean) / std

    if augmentation == 'Random_Off':
        print(augmentation)
        if ood_flag:
            print('ood offset')
            offsets = np.random.uniform(-1, 1, size=(len(train_X)+len(val_X), 1))
            offsets_train = offsets[:len(train_X)]
            #print('offsets_train ood', offsets_train)
            offsets_val = offsets[-len(val_X):]
            #print('offsets_val ood', offsets_val)
            offsets_range1 = np.random.uniform(-2, -1, size=(len(test_X) // 2, 1))
            offsets_range2 = np.random.uniform(1, 2, size=(len(test_X) - (len(test_X) // 2), 1))

            offsets_test = np.vstack((offsets_range1, offsets_range2))
            np.random.shuffle(offsets_test)
            offsets = np.concatenate((offsets, offsets_val, offsets_test), axis=0)
        else:
            offsets = np.random.uniform(-1, 1, size=(len(train_X)+len(val_X)+len(test_X), 1))
            
        offsets = np.repeat(offsets, train_X.shape[-1], axis=1)
        offsets = offsets[:, np.newaxis, :]

        train_X += offsets[:len(train_X)]
        val_X += offsets[len(train_X):len(train_X)+len(val_X)]
        test_X += offsets[-len(test_X):]

    elif augmentation == 'Class_Off':
        class_offsets = np.linspace(-0.1, 0.1, num_classes)

        offsets_train = class_offsets[train_y].reshape(-1, 1)
        offsets_train = np.repeat(offsets_train, train_X.shape[-1], axis=1)
        offsets_train = offsets_train[:, np.newaxis, :]

        offsets_val = class_offsets[val_y].reshape(-1, 1)
        offsets_val = np.repeat(offsets_val, val_X.shape[-1], axis=1)
        offsets_val = offsets_val[:, np.newaxis, :]

        offsets_test = class_offsets[test_y].reshape(-1, 1)
        offsets_test = np.repeat(offsets_test, test_X.shape[-1], axis=1)
        offsets_test = offsets_test[:, np.newaxis, :]

        train_X += offsets_train
        val_X += offsets_val
        test_X += offsets_test

    elif augmentation == 'Class_LT':
        # Generate class-specific linear trends
        trends_train = np.zeros_like(train_X)
        trends_val = np.zeros_like(val_X)
        trends_test = np.zeros_like(test_X)
        for class_idx in range(num_classes):
            # Create a linear trend for this class
            trend = np.linspace(0, 1, train_X.shape[-1]) * (class_idx + 1)  # Trend varies with class index
            # Add the trend to all signals of this class
            trends_train[train_y.squeeze() == class_idx] = trend
            trends_val[val_y.squeeze() == class_idx] = trend
            trends_test[test_y.squeeze() == class_idx] = trend

        # Apply the class-specific linear trends to the signals
        train_X += trends_train
        val_X += trends_val
        test_X += trends_test

    elif augmentation == 'Random_LT' or augmentation == 'Random_Off_LT':
        # Generate random linear trends for each signal
        trends_train = np.zeros_like(train_X)
        trends_val = np.zeros_like(val_X)
        trends_test = np.zeros_like(test_X)

        for i in range(train_X.shape[0]):
            # Create a random slope for the linear trend
            slope = np.random.uniform(-5, 5)  # Random slope between -1 and 1
            #print('slope train', slope)
            trend = np.linspace(0, slope, train_X.shape[-1])  # Linear trend with random slope
            trends_train[i] = trend

        for i in range(val_X.shape[0]):
            # Create a random slope for the linear trend
            min_slope, max_slope = -5, 5
            slope = np.random.uniform(min_slope, max_slope)  # Random slope between -1 and 1 #(-1,1)
            trend = np.linspace(0, slope, train_X.shape[-1])  # Linear trend with random slope
            #print('slope val', slope)
            trends_val[i] = trend
        
        if not ood_flag:
            for i in range(test_X.shape[0]):
                # Create a random slope for the linear trend
                min_slope, max_slope = -5, 5
                slope = np.random.uniform(min_slope, max_slope)  # Random slope between -1 and 1
                trend = np.linspace(0, slope, train_X.shape[-1])  # Linear trend with random slope
                trends_test[i] = trend
        else:
            print('ood lt')
            slope_range1 = np.random.uniform(-10, -5, size=len(test_X)//2)
            slope_range2 = np.random.uniform(5, 10, size=len(test_X) - (len(test_X)//2))
            slopes = np.concatenate((slope_range1, slope_range2))
            np.random.shuffle(slopes)
            for i in range(len(test_X)):
                slope = slopes[i]  # Use the shuffled slope for the current sample
                trend = np.linspace(0, slope, train_X.shape[-1])  # Linear trend with the given slope
                trends_test[i] = trend

        train_X += trends_train
        val_X += trends_val
        test_X += trends_test

        if augmentation == 'Random_Off_LT':
            print(augmentation)
            if ood_flag:
                print('ood offset lt')
                offsets = np.random.uniform(-1, 1, size=(len(train_X)+len(val_X), 1))
                offsets_train = offsets[:len(train_X)]
                offsets_val = offsets[-len(val_X):]

                offsets_range1 = np.random.uniform(-2, -1, size=(len(test_X) // 2, 1))
                offsets_range2 = np.random.uniform(1, 2, size=(len(test_X) - (len(test_X) // 2), 1))

                offsets_test = np.vstack((offsets_range1, offsets_range2))
                np.random.shuffle(offsets_test)
                offsets = np.concatenate((offsets, offsets_val, offsets_test), axis=0)
            else:
                offsets = np.random.uniform(-1, 1, size=(len(train_X)+len(val_X)+len(test_X), 1))

            offsets = np.repeat(offsets, train_X.shape[-1], axis=1)
            offsets = offsets[:, np.newaxis, :]

            train_X += offsets[:len(train_X)]
            val_X += offsets[len(train_X):len(train_X)+len(val_X)]
            test_X += offsets[-len(test_X):]
        else:
            print(augmentation)

    elif augmentation == 'Random_Walk':
        print(augmentation)
        AMPLITUDE = 0.1
        #print(train_X.shape)
        #print(amplitudes.shape)
        LENGTH = train_X.shape[-1]
        SMOOTHING_LEN = LENGTH//2 #LENGTH//4
        NSIGNALS = len(train_X)+len(val_X)+len(test_X)

        randwalks = np.cumsum(AMPLITUDE*np.random.randn(NSIGNALS,LENGTH+SMOOTHING_LEN-1),axis=-1)
        randwalks += 3*(2*np.random.rand(NSIGNALS,1)-1)
        X = []
        for (x,rw) in zip(train_X,randwalks[:len(train_X)]):
            rw = np.convolve(rw,np.ones(SMOOTHING_LEN)/SMOOTHING_LEN,mode = "valid")
            X.append(x + rw[None,:])
        train_X = np.array(X)
        #print(train_X.shape)

        X = []
        for (x,rw) in zip(val_X,randwalks[len(train_X):len(train_X)+len(val_X)]):
            rw = np.convolve(rw,np.ones(SMOOTHING_LEN)/SMOOTHING_LEN,mode = "valid")
            X.append(x + rw[None,:])
        val_X = np.array(X)
        #print(val_X.shape)

        if not ood_flag:
            X = []
            for (x,rw) in zip(test_X,randwalks[-len(test_X):]):
                rw = np.convolve(rw,np.ones(SMOOTHING_LEN)/SMOOTHING_LEN,mode = "valid")
                X.append(x + rw[None,:])
            test_X = np.array(X)
        else:
            print('ood rw')
            #AMPLITUDE_ood = 0.3
            SMOOTHING_LEN = LENGTH//4
            randwalks_ood = np.cumsum(AMPLITUDE*np.random.randn(NSIGNALS,LENGTH+SMOOTHING_LEN-1),axis=-1)
            randwalks_ood += 3*(2*np.random.rand(NSIGNALS,1)-1)
            X = []
            for (x,rw) in zip(test_X,randwalks_ood[-len(test_X):]):
                rw = np.convolve(rw,np.ones(SMOOTHING_LEN)/SMOOTHING_LEN,mode = "valid")
                X.append(x + rw[None,:])
            test_X = np.array(X)
        #print(test_X.shape)

    #Generate train dataloader
    dataset_mat_train = TSDataset(args, torch.from_numpy(train_X), torch.from_numpy(train_y))

    #Generate validation dataloader
    dataset_mat_val = TSDataset(args, torch.from_numpy(val_X), torch.from_numpy(val_y))

    #Generate test dataloader
    dataset_mat_test = TSDataset(args, torch.from_numpy(test_X), torch.from_numpy(test_y))
    
    return dataset_mat, dataset_mat_train, dataset_mat_val, dataset_mat_test

class TSDataset(data.Dataset):
    def __init__(self,args,x_train,labels):
        self.samples = x_train
        self.labels = labels

        self.max_seq_len = x_train.shape[-1]
        self.class_names = np.unique(labels)
        self.feature_df = x_train
        self.val_split = args.val_split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        if self.val_split:
            return self.samples[idx].permute(1,0),self.labels[idx] #,self.samples[idx].permute(1,0)
        else:
            return self.samples[idx].permute(1,0),self.labels[idx],self.samples[idx].permute(1,0)