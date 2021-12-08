import os
import time
import torch
import pickle
import random
import datetime
import torchvision
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms

from models import Get_Pytorch_Model


class Simple_Logger:
    def __init__(self, log_addr, version):
        self.start_time = time.time()
        self.log_addr = log_addr
        if not os.path.exists(log_addr):
            with open(log_addr, 'w') as f:
                f.write('Logger for version: {}\n'.format(version))
        print('Logger initialized at {}'.format(log_addr))
        
    def log_message(self, message):
        tosave = 'Elapsed: {}, '.format(self.process_time())
        tosave += message + '\n'
        with open(self.log_addr, 'a') as f:
            f.write(tosave)
                        
    def process_time(self):
        """
        return elapsed time in string
        """
        total_time = time.time() - self.start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        return total_time_str

def collect_data(f, nclass=100):
    """
    Collect data addrs by session-level + subject-level sorting by class  
    
    parameters
    ----------
    f: a h5py instance containing all data

    returns
    -------
    f_addrs: a nested dictionary with sub - sess - list of relative address sets (input_addr, label_addr) in f.
    class2sub: a dictionary with average sub-level class as keys and sub IDs as values
    """
    f_addrs = {}
    class2sub = {c:[] for c in range(nclass)}
    c_sess = 0
    c_epochs = 0
    for sub in tqdm(f.keys()):
        f_addrs[sub] = {}
        sub_class = []
        for sess in f[sub].keys():
            f_addrs[sub][sess] = []
            c_sess += 1
            for in_file in f[sub][sess]['inputs']:
                file_loc = os.path.join(sub, sess, 'inputs',in_file)                
                label_loc = os.path.join(sub, sess, 'labels', in_file)
                f_addrs[sub][sess].append((file_loc, label_loc))
                label = int(f[label_loc][()])
                sub_class.append(label)
                c_epochs += 1
        sub_class_mean = np.array(sub_class).mean()
        class2sub[int(sub_class_mean)].append(sub)
    print('available data: {} subjects, {} sessions, {} epochs'.format(len(f_addrs.keys()), c_sess, c_epochs))
    return f_addrs, class2sub

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def randomize_split(sub_list, seed=10, nfold=10):
    """Given the subject list, return a dictionary of n lists"""
    random_sub_seq = sub_list    
    random.seed(seed)
    random.shuffle(random_sub_seq)    
    splits = list(split(random_sub_seq, nfold))
    return splits

def balanced_heldout(class2sub, seed, nfold):
    """
    parameters
    ----------
    class2sub: a dictionary with average sub-level class as keys and sub IDs as values
    seed: random seed idx => keeps the result consistent
    nfold: number of folds for cross validation, but here, 10 means train 9 : test 1 split. 5 mean 4:1 split
    
    returns
    -------
    train: a list consisting training subject ids
    test: a list consisting testing subject ids
    """
    train, test = [], []
    nclass = len(class2sub.keys())
    for cl in range(nclass):
        cl_dependent_split = randomize_split(class2sub[cl], seed=seed, nfold=nfold)
        for i in range(nfold):
            if i < nfold-1:
                train += cl_dependent_split[i]
            else:
                test += cl_dependent_split[i]
    print('# of training / test subjects: {} / {}'.format(len(train), len(test)))
    return train, test

def plot_class_distribution(f, f_addrs, sub_list):
    """Given the subject list, plot the class distribution"""
    all_classes = []
    for sub in tqdm(sub_list):
        for sess in f_addrs[sub].keys():
            sub_data = f_addrs[sub][sess]
            for d, l in sub_data:
                cl = f[l][()]
                all_classes.append(cl)
    _ = sns.displot(all_classes)
    return 0    

def normalize_by_percentile(np_array, percentile = 95):
    """Normalize an array by 95 percentile."""
    p_time = np.percentile(np_arr, 95, keepdims=True, axis=-1)#
    out = np.divide(np_array, p_time, out=np.zeros_like(np_array), where=p_time!=0)
    return out

class Spectromer_Dataset(data.Dataset):
    def __init__(self, f, f_addrs, sub_list):
        self.f = f
        self.f_addrs = f_addrs   
        self.all_thinkers = sub_list
        self.all_pair = self.build_data()
    
    def build_data(self):
        """Collect all addresses in a list"""
        all_pair = []
        for sub in tqdm(self.all_thinkers):
            for sess in self.f_addrs[sub].keys():
                all_pair += self.f_addrs[sub][sess]                
        return all_pair
    
    def normalize_by_percentile(self, np_array, percentile = 95):
        """Normalize an array by 95 percentile."""
        p_time = np.percentile(np_array, percentile, keepdims=True, axis=-1)#
        out = np.divide(np_array, p_time, out=np.zeros_like(np_array), where=p_time!=0)
        return out
    
    def normalize_class(self, arr, numerator = 99):
        """Normalize the y value - class by a fixed number"""
        return arr / numerator
    
    def remove_empty_elec(self, np_array, elec_ids = [7, 11]):
        """
        It is preferable to keep a standard electrode space such as 10-20 standard, but in a dataset,
        some electrodes were found empty for all files. For now, remove them.
        
        parameters
        ----------
        np_array: a signal epoch in a numpy array e.g., (5, 19, time)
        elec_ids: elec ids to remove. This is from the standard elec space.
        """
        # axis 1 corresponds to electrodes
        out = numpy.delete(np_array, elec_ids, axis=1)
        return out
    
    def __getitem__(self, idx):
        data_addr, label_addr = self.all_pair[idx]
        data_np, label = self.f[data_addr][()], self.f[label_addr][()]
        label_t = torch.LongTensor([label]).squeeze()
        label_t = self.normalize_class(label_t)
        data_np = self.normalize_by_percentile(data_np, percentile = 95)
        data_t =torch.from_numpy(data_np).float().squeeze()
        return data_t, label_t
    
    def __len__(self):
        return len(self.all_pair)
    
class Sample_Dataset(data.Dataset):
    def __init__(self, train, data_dim=(5,19,3840)):
        self.train = train
        self.data_dim = data_dim
    
    def __getitem__(self):
        data_t = torch.randn(*self.data_dim)
        label_t = torch.LongTensor([random.randint(0, 4)]).squeeze()
        return data_t, label_t
    
    def __len__(self):
        return 50000 if self.train='train' else 10000
    
def Spectromer_Loader(train_dataset, test_dataset, batch_size):
    """
    Given a single train and a test dataset, return loaders and label counts.
    """   
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               drop_last=False,
                                               shuffle=True)
    print('train_loader ready')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=0,
                                              drop_last=False,
                                              shuffle=False)
    print('test_loader ready')

    return train_loader, test_loader


def main(samplerun = True):
    ### Configurations
    save_model_dir = './models'
    save_log_addr = './logs/initial.log'
    if not samplerun:
        h5_addr = '/pscratch/sd/d/dkp2129/processed/age/v2_real.hdf5'
        f = h5py.File(h5_addr, 'r') 
    best = 1e5 # Defines the maximum loss. If accuracy, must be set to 0.
    prior_model = '' # The best-performing model
    nfold = 10
    seed = 10
    lr = 0.0001
    print_iter = 10
    num_epochs = 10
    batch_size = 512
    nfold=0
    cv_fold = 0
    debug = True
    ### Configurations
    
    # Intialize logger
    Logger = Simple_Logger(save_log_addr, version='initial')      
       
    if not samplerun:
        # collect the data
        f_addrs, class2sub = collect_data(f)
        train_sub, test_sub = balanced_heldout(class2sub, seed, nfold)
        train_dataset = Spectromer_Dataset(f, f_addrs, train_sub)
        test_dataset = Spectromer_Dataset(f, f_addrs, test_sub)
    else:
        train_dataset = Sample_Dataset('train')
        test_dataset = Sample_Dataset('test')
        
    train_loader, test_loader = Spectromer_Loader(train_dataset, test_dataset, batch_size)
    total_step = len(train_loader)

    initial_lr = lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Get_Pytorch_Model(model='resnet50', pretrained=False, n_class=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    mae = nn.L1Loss()
    
    ##########################
    ##### Start training #####
    ##########################
    
    for epoch in range(num_epochs):
        model.train()
        for i, (signals, labels) in enumerate(train_loader):
            signals = signals.float().to(device)
            labels = labels.unsqueeze(1).to(device)

            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % print_iter == 0:
                log_ = "[Training] Fold [{}/{}], Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"\
                       .format(cv_fold+1, nfold, epoch+1, num_epochs, i+1, total_step, loss.item())
                print(log_)
                Logger.log_message(log_)


        # Test the model
        model.eval()
        with torch.no_grad():
            allloss = 0
            total = 0
            for signals, labels in test_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                outputs = model(signals)
                loss = mae(outputs*99, labels*99)
                total += labels.size(0)
                allloss += loss.item()*labels.size(0)

            mae_ = allloss / total
            log_ = '[Validation] Fold [{}/{}], Epoch [{}/{}], mae on test: {:.8f}'.format(cv_fold+1, nfold, 
                                                                                             epoch+1, num_epochs, 
                                                                                             mae_)
            print(log_)
            Logger.log_message(log_) 

        if best > mae_:
            best = mae_
            # Save the model checkpoint
            model_addr = os.path.join(cv_model_dir, 'resnet_epoch{}_Mae{:.8f}.ckpt'.format(epoch, mae_))

            # Remove the old model

            if os.path.exists(prior_model):
                # Removing prior model: not activated
    #                 os.remove(prior_model)
                log_ = '[Validation] Deleting the prior model in {}'.format(prior_model)
                Logger.log_message(log_)

            # updating and saving new model
            prior_model = model_addr
            log_ = '[Validation] Saving the model to {}'.format(model_addr)
            Logger.log_message(log_)            
            torch.save(model.state_dict(), model_addr)           

        # Decay learning rate        
        if (epoch+1) % 3 == 0:
            initial_lr /= 3
            update_lr(optimizer, initial_lr)
            Logger.log_message('[Validation] learning rate to {}'.format(initial_lr))

