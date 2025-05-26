from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from utils.utils_opt import LinearWarmupCosineAnnealingLR
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

from timm.loss import LabelSmoothingCrossEntropy
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap


class WeightConstraint(object):
    def __init__(self,w1=-0.5,w2=0.5):
        self.w1=w1
        self.w2=w2
        #pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w=w.clamp(self.w1,self.w2)
            module.weight.data=w
        return

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        all_data, train_data, train_loader, _, _ = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        print(all_data.max_seq_len)
        print(test_data.max_seq_len)
        self.args.seq_len = max(all_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = all_data.feature_df.shape[1]
        self.args.num_class = len(all_data.class_names)

        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if flag=='TEST' or flag=='TESTA' or flag=='TESTB' or flag=='TESTC' or flag=='TESTD':
            data_set, data_loader = data_provider(self.args, flag)
            return data_set, data_loader
        else:
            data_set, data_set1, data_loader1, data_set2, data_loader2 = data_provider(self.args, flag)
            return data_set, data_set1, data_loader1, data_set2, data_loader2

    def _select_optimizer(self):
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu()) #+ 0.1 * torch.norm(self.model.model.layer1.weight, p=1).detach().cpu().numpy() #0.5 * torch.norm(self.model.projection[0].weight, p=1).detach().cpu().numpy() #.squeeze().cpu()
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        _, train_data, train_loader, vali_data, vali_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = LinearWarmupCosineAnnealingLR(model_optim, warmup_epochs=10, max_epochs=self.args.train_epochs, warmup_start_lr=self.args.learning_rate*0.1, eta_min=self.args.learning_rate*0.01)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
            
                loss = criterion(outputs, label.long().squeeze(-1)) #+ 0.1 * torch.norm(self.model.model.layer1.weight, p=1).detach().cpu().numpy()#0.1 * torch.norm(self.model.projection[0].weight, p=1).detach().cpu().numpy() #.squeeze()
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()


            if self.args.lradj=='cosanneal':
                scheduler.step()
                print(f'LR: {scheduler.get_last_lr()[0]:.6f}')

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj!='cosanneal':
                if (epoch + 1) % 5 == 0:
                    if self.args.lradj!='none':
                        adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        if self.args.model_id=='Fault-diagnosis':
            flags = [f'TEST{d}' for d in ['A', 'B', 'C', 'D'] if d != self.args.tl_source]
        else:
            flags=['TEST']

        print(flags)
        for flag in flags:
            test_data, test_loader = self._get_data(flag=flag)
            if test:
                print('loading model')
                path = os.path.join(self.args.checkpoints, setting)
                self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

            preds = []
            trues = []
            
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    label = label.to(self.device)

                    outputs = self.model(batch_x, padding_mask, None, None) #feature_maps
                    
                    preds.append(outputs.detach())
                    trues.append(label)

            preds = torch.cat(preds, 0)
            trues = torch.cat(trues, 0)

            probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
            predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
            trues = trues.flatten().cpu().numpy()
            accuracy = cal_accuracy(predictions, trues)

            # result save #results_old_config
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            print('accuracy:{}\n'.format(accuracy))
            file_name='result_classification.txt'
            f = open(os.path.join(folder_path,file_name), 'a')
            f.write(setting + "  \n")
            f.write('accuracy:{0:0.5f}'.format(accuracy))
            f.write('\n')
            f.write('\n')
            f.close()