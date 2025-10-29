"""Utility functions for Processing data"""

from collections import defaultdict
from einops import rearrange
from pytorch_lightning import LightningModule
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple
import os
from tqdm import tqdm
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

import sys
import logging
import torch.nn as nn
from sklearn import metrics

class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    @property
    def layer_list(self):
        return self.module.layer_list
    
    @property
    def t(self):
        return self.module.t
    
class RRL:
    def __init__(self, dim_list, device_id, use_not=False, is_rank0=False, log_file=None, writer=None, left=None,
                 right=None, save_best=False, estimated_grad=False, save_path=None, distributed=True, use_skip=False, 
                 use_nlaf=False, alpha=0.999, beta=8, gamma=1, temperature=0.01):
        super(RRL, self).__init__()
        self.dim_list = dim_list
        self.use_not = use_not
        self.use_skip = use_skip
        self.use_nlaf = use_nlaf
        self.alpha =alpha
        self.beta = beta
        self.gamma = gamma
        self.best_f1 = -1.
        self.best_loss = 1e20

        self.device_id = device_id
        self.is_rank0 = is_rank0
        self.save_best = save_best
        self.estimated_grad = estimated_grad
        self.save_path = save_path
        if self.is_rank0:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
            if log_file is None:
                logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
            else:
                logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)
        self.writer = writer

        self.net = Net(dim_list, use_not=use_not, left=left, right=right, use_nlaf=use_nlaf, estimated_grad=estimated_grad, use_skip=use_skip, alpha=alpha, beta=beta, gamma=gamma, temperature=temperature)
        self.net.cuda(self.device_id)
        if distributed:
            self.net = MyDistributedDataParallel(self.net, device_ids=[self.device_id])

    def clip(self):
        """Clip the weights into the range [0, 1]."""
        for layer in self.net.layer_list[: -1]:
            layer.clip()
    
    def edge_penalty(self):
        edge_penalty = 0.0
        for layer in self.net.layer_list[1: -1]:
            edge_penalty += layer.edge_count()
        return edge_penalty
    
    def l1_penalty(self):
        l1_penalty = 0.0
        for layer in self.net.layer_list[1: ]:
            l1_penalty += layer.l1_norm()
        return l1_penalty
    
    def l2_penalty(self):
        l2_penalty = 0.0
        for layer in self.net.layer_list[1: ]:
            l2_penalty += layer.l2_norm()
        return l2_penalty
    
    def mixed_penalty(self):
        penalty = 0.0
        for layer in self.net.layer_list[1: -1]:
            penalty += layer.l2_norm()
        penalty += self.net.layer_list[-1].l1_norm()
        return penalty

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train_model(self, data_loader=None, valid_loader=None, epoch=50, lr=0.01, lr_decay_epoch=100, 
                    lr_decay_rate=0.75, weight_decay=0.0, log_iter=50):

        if data_loader is None:
            raise Exception("Data loader is unavailable!")

        accuracy_b = []
        f1_score_b = []

        criterion = nn.CrossEntropyLoss().cuda(self.device_id)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=0.0)

        cnt = -1
        avg_batch_loss_rrl = 0.0
        epoch_histc = defaultdict(list)
        for epo in range(epoch):
            optimizer = self.exp_lr_scheduler(optimizer, epo, init_lr=lr, lr_decay_rate=lr_decay_rate,
                                              lr_decay_epoch=lr_decay_epoch)

            epoch_loss_rrl = 0.0
            abs_gradient_max = 0.0
            abs_gradient_avg = 0.0

            ba_cnt = 0
            for X, y in data_loader:
                ba_cnt += 1
                X = X.cuda(self.device_id, non_blocking=True)
                y = y.cuda(self.device_id, non_blocking=True)
                optimizer.zero_grad()  # Zero the gradient buffers.
                
                # trainable softmax temperature
                y_bar = self.net.forward(X) / torch.exp(self.net.t)
                y_arg = torch.argmax(y, dim=1)
                
                loss_rrl = criterion(y_bar, y_arg) + weight_decay * self.l2_penalty()
                
                ba_loss_rrl = loss_rrl.item()
                epoch_loss_rrl += ba_loss_rrl
                avg_batch_loss_rrl += ba_loss_rrl
                
                loss_rrl.backward()

                cnt += 1
                with torch.no_grad():
                    if self.is_rank0 and cnt % log_iter == 0 and cnt != 0 and self.writer is not None:
                        self.writer.add_scalar('Avg_Batch_Loss_GradGrafting', avg_batch_loss_rrl / log_iter, cnt)
                        edge_p = self.edge_penalty().item()
                        self.writer.add_scalar('Edge_penalty/Log', np.log(edge_p), cnt)
                        self.writer.add_scalar('Edge_penalty/Origin', edge_p, cnt)
                        avg_batch_loss_rrl = 0.0

                optimizer.step()
                
                if self.is_rank0:
                    for i, param in enumerate(self.net.parameters()):
                        abs_gradient_max = max(abs_gradient_max, abs(torch.max(param.grad)))
                        abs_gradient_avg += torch.sum(torch.abs(param.grad)) / (param.grad.numel())
                self.clip()

                if self.is_rank0 and (cnt % (TEST_CNT_MOD * (1 if self.save_best else 10)) == 0):
                    if valid_loader is not None:
                        acc_b, f1_b = self.test(test_loader=valid_loader, set_name='Validation')
                    else: # use the data_loader as the valid loader
                        acc_b, f1_b = self.test(test_loader=data_loader, set_name='Training')
                    
                    if self.save_best and (f1_b > self.best_f1 or (np.abs(f1_b - self.best_f1) < 1e-10 and self.best_loss > epoch_loss_rrl)):
                        self.best_f1 = f1_b
                        self.best_loss = epoch_loss_rrl
                        self.save_model()
                    
                    accuracy_b.append(acc_b)
                    f1_score_b.append(f1_b)
                    if self.writer is not None:
                        self.writer.add_scalar('Accuracy_RRL', acc_b, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('F1_Score_RRL', f1_b, cnt // TEST_CNT_MOD)
            if self.is_rank0:
                logging.info('epoch: {}, loss_rrl: {}'.format(epo, epoch_loss_rrl))
                if self.writer is not None:
                    self.writer.add_scalar('Training_Loss_RRL', epoch_loss_rrl, epo)
                    self.writer.add_scalar('Abs_Gradient_Max', abs_gradient_max, epo)
                    self.writer.add_scalar('Abs_Gradient_Avg', abs_gradient_avg / ba_cnt, epo)
        if self.is_rank0 and not self.save_best:
            self.save_model()
        return epoch_histc

    @torch.no_grad()
    def test(self, test_loader=None, set_name='Validation'):
        if test_loader is None:
            raise Exception("Data loader is unavailable!")
        
        y_list = []
        for X, y in test_loader:
            y_list.append(y)
        y_true = torch.cat(y_list, dim=0)
        y_true = y_true.cpu().numpy().astype(np.int)
        y_true = np.argmax(y_true, axis=1)
        data_num = y_true.shape[0]

        slice_step = data_num // 40 if data_num >= 40 else 1
        logging.debug('y_true: {} {}'.format(y_true.shape, y_true[:: slice_step]))

        y_pred_b_list = []
        for X, y in test_loader:
            X = X.cuda(self.device_id, non_blocking=True)
            output = self.net.forward(X)
            y_pred_b_list.append(output)

        y_pred_b = torch.cat(y_pred_b_list).cpu().numpy()
        y_pred_b_arg = np.argmax(y_pred_b, axis=1)
        logging.debug('y_rrl_: {} {}'.format(y_pred_b_arg.shape, y_pred_b_arg[:: slice_step]))
        logging.debug('y_rrl: {} {}'.format(y_pred_b.shape, y_pred_b[:: (slice_step)]))

        accuracy_b = metrics.accuracy_score(y_true, y_pred_b_arg)
        f1_score_b = metrics.f1_score(y_true, y_pred_b_arg, average='macro')

        logging.info('-' * 60)
        logging.info('On {} Set:\n\tAccuracy of RRL  Model: {}'
                        '\n\tF1 Score of RRL  Model: {}'.format(set_name, accuracy_b, f1_score_b))
        logging.info('On {} Set:\nPerformance of  RRL Model: \n{}\n{}'.format(
            set_name, metrics.confusion_matrix(y_true, y_pred_b_arg), metrics.classification_report(y_true, y_pred_b_arg)))
        logging.info('-' * 60)

        return accuracy_b, f1_score_b

    def save_model(self):
        rrl_args = {'dim_list': self.dim_list, 'use_not': self.use_not, 'use_skip': self.use_skip, 'estimated_grad': self.estimated_grad, 
                    'use_nlaf': self.use_nlaf, 'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma}
        torch.save({'model_state_dict': self.net.state_dict(), 'rrl_args': rrl_args}, self.save_path)

    def detect_dead_node(self, data_loader=None):
        with torch.no_grad():
            for layer in self.net.layer_list[:-1]:
                layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=self.device_id)
                layer.forward_tot = 0

            for x, y in data_loader:
                x_bar = x.cuda(self.device_id)
                self.net.bi_forward(x_bar, count=True)

    def rule_print(self, feature_name, label_name, train_loader, file=sys.stdout, mean=None, std=None, display=True):
        if self.net.layer_list[1] is None and train_loader is None:
            raise Exception("Need train_loader for the dead nodes detection.")

        # detect dead nodes first
        if self.net.layer_list[1].node_activation_cnt is None:
            self.detect_dead_node(train_loader)

        # for Binarize Layer
        self.net.layer_list[0].get_bound_name(feature_name, mean, std)  # layer_list[0].rule_name == bound_name

        # for Union Layer
        for i in range(1, len(self.net.layer_list) - 1):
            layer = self.net.layer_list[i]
            layer.get_rules(layer.conn.prev_layer, layer.conn.skip_from_layer)
            skip_rule_name = None if layer.conn.skip_from_layer is None else layer.conn.skip_from_layer.rule_name
            wrap_prev_rule = False if i == 1 else True  # do not warp the bound_name
            layer.get_rule_description((skip_rule_name, layer.conn.prev_layer.rule_name), wrap=wrap_prev_rule)

        # for LR Layr
        layer = self.net.layer_list[-1]
        layer.get_rule2weights(layer.conn.prev_layer, layer.conn.skip_from_layer)
        
        if not display:
            return layer.rule2weights
        
        print('RID', end='\t', file=file)
        for i, ln in enumerate(label_name):
            print('{}(b={:.4f})'.format(ln, layer.bl[i]), end='\t', file=file)
        print('Support\tRule', file=file)
        for rid, w in layer.rule2weights:
            print(rid, end='\t', file=file)
            for li in range(len(label_name)):
                print('{:.4f}'.format(w[li]), end='\t', file=file)
            now_layer = self.net.layer_list[-1 + rid[0]]
            print('{:.4f}'.format((now_layer.node_activation_cnt[layer.rid2dim[rid]] / now_layer.forward_tot).item()),
                  end='\t', file=file)
            print(now_layer.rule_name[rid[1]], end='\n', file=file)
        print('#' * 60, file=file)
        return layer.rule2weights

def save_activations_to_npy(model: LightningModule, dataloader: DataLoader, output_path: os.PathLike) -> os.PathLike:
    model.eval()
    first = True
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), desc="Saving Act data", total=len(dataloader)):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            loc = batch["raw_x_local"][:, -2:]
            feats = model.predict_step(batch, batch_idx)
            loc = torch.nn.functional.interpolate(loc, size=feats.shape[-2:], mode="bilinear")
            loc = rearrange(loc, 'n d h w -> (n h w) d')
            loc = loc.cpu().numpy()
            num_patch = feats.shape[-2] * feats.shape[-1]
            feats = rearrange(feats, 'n d h w -> (n h w) d')
            feats = feats.cpu().numpy()
            if first:
                shape = (len(dataloader.dataset) * num_patch,) + feats.shape[1:]
                shape_loc = (len(dataloader.dataset) * num_patch,) + loc.shape[1:]
                dtype = feats.dtype
                dtype_loc = loc.dtype
                memmap = np.lib.format.open_memmap(output_path + "-act.npy", mode='w+', dtype=dtype, shape=shape)
                memmap_loc = np.lib.format.open_memmap(output_path + "-loc.npy", mode='w+', dtype=dtype_loc, shape=shape_loc)
                first = False

            start = batch_idx * dataloader.batch_size * num_patch
            end = start + feats.shape[0]
            memmap[start:end] = feats
            memmap_loc[start:end] = loc

    return output_path


def save_codes_raw_to_npy(model: LightningModule, sae: LightningModule, dataloader: DataLoader, output_path: os.PathLike) -> os.PathLike:
    model.eval()
    sae.eval()
    first = True
    parent_dir = Path(output_path).parent.parent
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), desc="Saving Codes and Raw data", total=len(dataloader)):
            raw_data = batch["raw_x_local"]
            batch = {key: value.to(model.device) for key, value in batch.items()}
            feats = model.predict_step(batch, batch_idx)
            num_patch = feats.shape[-2] * feats.shape[-1]
            feats = rearrange(feats, 'n d h w -> (n h w) d')
            _, codes = sae.net.encode(feats)
            codes = rearrange(codes, '(n h w) d -> n d h w', n=raw_data.shape[0], h=int(num_patch**(1/2)), w=int(num_patch**(1/2)))
            raw_data = raw_data.cpu().numpy()
            codes = codes.cpu().numpy()
            if first:
                shape = (len(dataloader.dataset),) + codes.shape[1:]
                shape_raw = (len(dataloader.dataset),) + raw_data.shape[1:]
                dtype = codes.dtype
                dtype_raw = raw_data.dtype
                memmap = np.lib.format.open_memmap(output_path + "-codes.npy", mode='w+', dtype=dtype, shape=shape)
                memmap_raw = np.lib.format.open_memmap(parent_dir / (Path(output_path).name + "-raw.npy"), mode='w+', dtype=dtype_raw, shape=shape_raw)
                first = False
            start = batch_idx * dataloader.batch_size
            end = start + codes.shape[0]
            memmap[start:end] = codes
            memmap_raw[start:end] = raw_data

    return output_path


def save_labels_to_npy(dataloader: DataLoader, output_path: os.PathLike, size: Tuple[int] = None) -> os.PathLike:
    first=True
    for batch_idx, batch in tqdm(enumerate(dataloader), desc="Saving Codes and Raw data", total=len(dataloader)):
        target_data = batch["y_local"]
        if size is not None:
            # Add channel dimension for interpolation
            if target_data.dim() == 3:  # (N, H, W)
                target_data = target_data.unsqueeze(1)  # (N, 1, H, W)
            target_data = torch.nn.functional.interpolate(target_data.float(), size=size, mode="nearest")
            target_data = target_data.round().long()
            # Remove channel dimension after interpolation
            if target_data.shape[1] == 1:
                target_data = target_data.squeeze(1)  # (N, H, W)
        h, w = target_data.shape[-2], target_data.shape[-1]
        target_data = rearrange(target_data, 'n h w -> (n h w)')
        target_data = target_data.cpu().numpy()
        if first:
            shape = (len(dataloader.dataset) * h * w,)
            dtype = target_data.dtype
            memmap = np.lib.format.open_memmap(output_path + "-labels.npy", mode='w+', dtype=dtype, shape=shape)
            first = False
        start = batch_idx * dataloader.batch_size * h * w
        end = start + target_data.shape[0]
        memmap[start:end] = target_data

    return output_path


class CodesEncoder:
    """Encoder used for processing codes in RRL."""

    def __init__(self, X_fname, y_fname, y_one_hot=True):
        self.y_one_hot = y_one_hot
        self.label_enc = preprocessing.OneHotEncoder(categories='auto') if y_one_hot else preprocessing.LabelEncoder()
        self.X_fname = X_fname
        self.y_fname = y_fname
        self.discrete_flen = 0 # Placeholder
        self.continuous_flen = None
        self.mean = None
        self.std = None

    def fit(self, X, y, normalized=False):
        self.continuous_flen = X.shape[1]
        self.label_enc.fit(y.reshape(-1, 1))
        if normalized:
            means = np.mean(X, axis=0)
            stds = np.std(X, axis=0)
            self.mean = {name: float(mean) for name, mean in zip(self.X_fname, means)}
            self.std = {name: float(std) for name, std in zip(self.X_fname, stds)}


    def transform(self, X, y, eps=1e-8):
        self.continuous_flen = X.shape[1]
        y = self.label_enc.transform(y.reshape(-1, 1))
        if self.y_one_hot:
            y = y.toarray()
        if self.mean is not None or self.std is not None:
            X = (X - np.array(list(self.mean.values()))) / (np.array(list(self.std.values())) + eps)
        return X, y


def edge_count(rrl: RRL, rule2weights: Dict[int, float]) -> int:
    edge_cnt = 0
    connected_rid = defaultdict(lambda: set())
    ln = len(rrl.net.layer_list) - 1
    for rid, w in rule2weights:
        connected_rid[ln - abs(rid[0])].add(rid[1])
    while ln > 1:
        ln -= 1
        layer = rrl.net.layer_list[ln]
        for r in connected_rid[ln]:
            con_len = len(layer.rule_list[0])
            if r >= con_len:
                opt_id = 1
                r -= con_len
            else:
                opt_id = 0
            rule = layer.rule_list[opt_id][r]
            edge_cnt += len(rule)
            for rid in rule:
                connected_rid[ln - abs(rid[0])].add(rid[1])

    return edge_cnt


def stratified_sample_by_distribution(values: np.ndarray, sample_size: float, n_bins: int = 10, random_state: int = 42):

    bin_edges = np.quantile(values, np.linspace(0,1,n_bins+1)[1:-1])
    bin_ids = np.digitize(values, bin_edges)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=random_state)
    idx, _ = next(sss.split(np.zeros_like(values), bin_ids))

    return idx