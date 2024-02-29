import os
import random
from typing import Any
import numpy as np 
from utils import read_data
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import pdb
import json

class Federated_Dataset(Dataset):
    def __init__(self, X, Y, A):
        self.X = X
        self.Y = Y
        self.A = A

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        A = self.A[index]
        return X, Y, A 

    def __len__(self):
        return self.X.shape[0]


#### adult dataset x("51 White", "52 Asian-Pac-Islander", "53 Amer-Indian-Eskimo", "54 Other", "55 Black", "56 Female", "57 Male")
def LoadDataset(args, train_rate = 1.0, test_rate = 1.0):
    clients_name, groups, train_data, test_data = read_data(args.train_dir, args.test_dir)

    # client_name [phd, non-phd]
    client_train_loads = []
    client_test_loads = []
    args.n_clients = len(clients_name)
    # clients_name = clients_name[:1]
    if args.dataset == "adult":
        for client in clients_name:
            X = np.array(train_data[client]["x"]).astype(np.float32)

            Y = np.array(train_data[client]["y"]).astype(np.float32)

            if args.sensitive_attr == "race":
                A = X[:,51] # [1: white, 0: other]
                X = np.delete(X, [51, 52, 53, 54, 55], axis = 1)
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "sex":
                A = X[:, 56] # [1: female, 0: male]
                X = np.delete(X, [56, 57], axis = 1)
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "none-race":
                A = X[:, 51]  # [1: white, 0: other]
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "none-sex":
                A = X[:, 56]
                args.n_feats = X.shape[1]
            else:
                print("error sensitive attr")
                exit()
            dataset = Federated_Dataset(X, Y, A)
            client_train_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last))

        for client in clients_name:
            X = np.array(test_data[client]["x"]).astype(np.float32)
            Y = np.array(test_data[client]["y"]).astype(np.float32)
            if args.sensitive_attr =="race":
                A = X[:,51] # [1: white, 0: other]
                X = np.delete(X, [51, 52, 53, 54, 55],axis = 1)
            elif args.sensitive_attr == "sex":
                A = X[:, 56] # [1: female, 0: male]
                X = np.delete(X, [56, 57], axis = 1)
            elif args.sensitive_attr == "none-race":
                A = X[:, 51]  # [1: white, 0: other]
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "none-sex":
                A = X[:, 56]
                args.n_feats = X.shape[1]
            else:
                print("error sensitive attr")
                exit()

            dataset = Federated_Dataset(X, Y, A)

            client_test_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last)) 

    elif "eicu" in args.dataset:
    # elif args.dataset == "eicu_d" or args.dataset == "eicu_los":
        for client in clients_name:
            X = np.array(train_data[client]["x"]).astype(np.float32)

            Y = np.array(train_data[client]["y"]).astype(np.float32)

            if args.sensitive_attr == "race":
                A = train_data[client]["race"]
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "sex":
                A = train_data[client]["gender"]
                args.n_feats = X.shape[1]
            else:
                A = train_data[client]["race"]
                args.n_feats = X.shape[1]
            dataset = Federated_Dataset(X, Y, A)
            client_train_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last))

        for client in clients_name:
            X = np.array(test_data[client]["x"]).astype(np.float32)
            Y = np.array(test_data[client]["y"]).astype(np.float32)
            if args.sensitive_attr =="race":
                A = test_data[client]["race"]
            elif args.sensitive_attr == "sex":
                A = test_data[client]["gender"]
            else:
                A = test_data[client]["race"]

            dataset = Federated_Dataset(X, Y, A)

            client_test_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last)) 

    elif args.dataset == "health":
        for client in clients_name:
            X = np.array(train_data[client]["x"]).astype(np.float32)

            Y = np.array(train_data[client]["y"]).astype(np.float32)

            if args.sensitive_attr == "race":
                A = train_data[client]["race"]
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "sex":
                A = train_data[client]["isfemale"]
                args.n_feats = X.shape[1]
            else:
                A = train_data[client]["isfemale"]
                args.n_feats = X.shape[1]
            dataset = Federated_Dataset(X, Y, A)
            client_train_loads.append(DataLoader(dataset, X.shape[0],
                                                 shuffle=args.shuffle,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,
                                                 drop_last=args.drop_last))

        for client in clients_name:
            X = np.array(test_data[client]["x"]).astype(np.float32)
            Y = np.array(test_data[client]["y"]).astype(np.float32)
            if args.sensitive_attr == "race":
                A = test_data[client]["race"]
            elif args.sensitive_attr == "sex":
                A = test_data[client]["isfemale"]
            else:
                A = np.zeros(X.shape[0])

            dataset = Federated_Dataset(X, Y, A)

            client_test_loads.append(DataLoader(dataset, X.shape[0],
                                                shuffle=args.shuffle,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                drop_last=args.drop_last))
    elif args.dataset == "bank":
        for client in clients_name:
            def handle(data, client_loads):
                X = np.array(data[client]["x"]).astype(np.float32)
                Y = np.array(data[client]["y"]).astype(np.float32)
                if args.sensitive_attr == "married":
                    sensitive = 19
                    sensitives = [18, 19, 20]
                elif args.sensitive_attr == "noloan":
                    sensitive = 29
                    sensitives = [29, 30]
                A = X[:, sensitive]
                X = np.delete(X, sensitives, axis = 1)
                args.n_feats = X.shape[1]
                dataset = Federated_Dataset(X, Y, A)
                client_loads.append(DataLoader(dataset, X.shape[0],
                shuffle = args.shuffle,
                num_workers = args.num_workers,
                pin_memory = True,
                drop_last = args.drop_last))
            handle(train_data, client_train_loads)
            handle(test_data, client_test_loads)
    elif args.dataset == "compas":
        for client in clients_name:
            def handle(data, client_loads):
                X = np.array(data[client]["x"]).astype(np.float32)
                Y = np.array(data[client]["y"]).astype(np.float32)
                if args.sensitive_attr == "sex":
                    sensitive = 5
                    sensitives = [5, 6]
                elif args.sensitive_attr == "race":
                    sensitive = 12
                    sensitives = [12, 13, 14, 15, 16, 17]
                A = X[:, sensitive]
                X = np.delete(X, sensitives, axis = 1)
                args.n_feats = X.shape[1]
                dataset = Federated_Dataset(X, Y, A)
                client_loads.append(DataLoader(dataset, X.shape[0],
                shuffle = args.shuffle,
                num_workers = args.num_workers,
                pin_memory = True,
                drop_last = args.drop_last))
            handle(train_data, client_train_loads)
            handle(test_data, client_test_loads)

    if args.mix_dataset:
        print("mix_dataset")
        combined_loaders = []
        for loader1, loader2 in zip(client_train_loads, client_test_loads):
            combined_dataset = ConcatDataset([loader1.dataset, loader2.dataset])
            combined_loader = DataLoader(combined_dataset, 
                                         batch_size=len(combined_dataset), 
                                         shuffle = args.shuffle,
                                         num_workers = args.num_workers,
                                         pin_memory = True,
                                         drop_last = args.drop_last)
            combined_loaders.append(combined_loader)
        return client_train_loads, combined_loaders
    if args.new_trial:
        class SampleError: Exception
        def cut_dataset(dataset, rate = 0.5):
            retry_num = 0
            while True:
                retry_num += 1
                if retry_num >= 5:
                    raise SampleError()
                def judge_contain_all_sensitive(sub_dataset):
                    a_con = sum(1 for _, _, a in sub_dataset if a == 1.0)
                    na_con = sum(1 for _, _, a in sub_dataset if a == 0.0)
                    if a_con + na_con != len(sub_dataset):
                        raise "a_con + na_con != len(sub_dataset)"
                    if a_con != 0 and na_con != 0:
                        return True
                    return False
                indices_1 = np.random.choice(range(len(dataset)), int(len(dataset) * rate), replace = False)
                sub_dataset_1 = Subset(dataset, indices_1)
                if not judge_contain_all_sensitive(sub_dataset_1):
                    print("dataset only containes one sensitive feature, retry sampling")
                    continue
                indices_2 = list(set(range(len(dataset))) - set(indices_1))
                sub_dataset_2 = Subset(dataset, indices_2)
                if not judge_contain_all_sensitive(sub_dataset_2):
                    print("dataset only containes one sensitive feature, retry sampling")
                    continue
                break
            return sub_dataset_1, sub_dataset_2, indices_1.tolist()
        def create_dataloader(dataset):
            return DataLoader(dataset,
                            batch_size=len(dataset), 
                            shuffle = args.shuffle,
                            num_workers = args.num_workers,
                            pin_memory = True,
                            drop_last = args.drop_last)
        if args.new_trial_method == 'old':
            # 这里是错误的实现，client_another_loads里面全是client_train_loads和client_test_loads了
            client_train_loads = [create_dataloader(cut_dataset(loader.dataset)[0]) for loader in client_train_loads]
            client_test_loads = [create_dataloader(cut_dataset(loader.dataset)[0]) for loader in client_test_loads]
            client_another_loads = [
                create_dataloader(ConcatDataset([cut_dataset(train_loader.dataset)[1], cut_dataset(test_loader.dataset)[1]])) 
                for train_loader, test_loader in zip(client_train_loads, client_test_loads)
            ]
        elif args.new_trial_method == 'old2':
            # 同上，也是错误的实现
            client_train_results = [cut_dataset(loader.dataset) for loader in client_train_loads]
            client_train_loads = [create_dataloader(result[0]) for result in client_train_results]
            client_train_indices = [result[2] for result in client_train_results]
            if not args.valid:
                with open(os.path.join(args.target_dir_name, 'indices.json'), 'w') as f:
                    json.dump(client_train_indices, f)
            client_test_loads = [create_dataloader(cut_dataset(loader.dataset)[0]) for loader in client_test_loads]
            client_another_loads = [
                create_dataloader(cut_dataset(ConcatDataset([train_loader.dataset, test_loader.dataset]))[1]) 
                for train_loader, test_loader in zip(client_train_loads, client_test_loads)
            ]
        elif args.new_trial_method == 'old3':
            client_resample_datasets = [
                cut_dataset(ConcatDataset([train_loader.dataset, test_loader.dataset]), 0.7)[0:2]
                for train_loader, test_loader in zip(client_train_loads, client_test_loads)
            ]
            client_train_loads = [create_dataloader(dataset[0]) for dataset in client_resample_datasets]
            client_test_loads = [create_dataloader(dataset[1]) for dataset in client_resample_datasets]
            client_train_results = [cut_dataset(loader.dataset) for loader in client_train_loads]
            client_train_loads = [create_dataloader(result[0]) for result in client_train_results]
            client_train_indices = [result[2] for result in client_train_results]
            if not args.valid:
                with open(os.path.join(args.target_dir_name, 'indices.json'), 'w') as f:
                    json.dump(client_train_indices, f)
            client_test_loads = [create_dataloader(cut_dataset(loader.dataset)[0]) for loader in client_test_loads]
            client_another_loads = [
                create_dataloader(cut_dataset(ConcatDataset([train_loader.dataset, test_loader.dataset]))[1]) 
                for train_loader, test_loader in zip(client_train_loads, client_test_loads)
            ]
        elif args.new_trial_method == 'old4' or args.new_trial_method == 'old5' or args.new_trial_method == 'old6' or args.new_trial_method == 'old7':
            client_whole_datasets = [ConcatDataset([train.dataset, test.dataset]) 
                                    for train, test in zip(client_train_loads, client_test_loads)]
            def compute_indice(len_data):
                if args.new_trial_method == 'old4':
                    indices_train = np.random.choice(range(int(len_data * 0.7)), int(len_data * 0.35), replace = False)
                    indices_test = np.random.choice(range(int(len_data * 0.7), len_data), int(len_data * 0.15), replace = False)
                    indices_another = np.random.choice(range(len_data), int(len_data * 0.50), replace = False)
                elif args.new_trial_method == 'old5':
                    indices_train = np.random.choice(range(len_data), int(len_data * 0.35), replace = False)
                    indices_test = np.random.choice(list(set(range(len_data)) - set(indices_train)), int(len_data * 0.15), replace = False)
                    indices_another = np.random.choice(range(len_data), int(len_data * 0.50), replace = False)
                elif args.new_trial_method == 'old6':
                    indices_train = np.random.choice(range(int(len_data * 0.7)), int(len_data * 0.35), replace = False)
                    indices_test = np.random.choice(range(int(len_data * 0.7), len_data), int(len_data * 0.15), replace = False)
                    indices_another = list(set(range(len_data)) - set(indices_train) - set(indices_test))
                else:
                    indices_train = np.random.choice(range(int(len_data)), int(len_data * 0.35), replace = False)
                    indices_test = np.random.choice(list(set(range(int(len_data * 0.5))) - set(indices_train)), int(len_data * 0.15), replace = False)
                    indices_another = list(set(range(len_data)) - set(indices_train) - set(indices_test))
                return indices_train, indices_test, indices_another
            indice_tuples = [compute_indice(len(dataset)) for dataset in client_whole_datasets]
            # report sample
            class EqualValue:
                def __init__(self):
                    self.cnt = 0
                def __setattr__(self, __name: str, __value: Any):
                    if __name == 'value':
                        if self.cnt == 0:
                            object.__setattr__(self, __name, __value)
                        elif self.value < __value and self.value * 1.0 / __value > 0.88 or \
                            self.value > __value and __value * 1.0 / self.value > 0.88 or \
                            self.value < 0.001 and abs(self.value - __value) < 0.001 or \
                            self.value == __value:
                            object.__setattr__(self, __name, (self.value * self.cnt + __value) * 1.0 / (self.cnt+1))
                        else:
                            raise RuntimeError(f'client report value not equal: {self.value} vs {__value}')
                        self.cnt += 1
                    else:
                        object.__setattr__(self, __name, __value)
            rate_train, rate_test, rate_another = EqualValue(), EqualValue(), EqualValue()
            distrib_train, distrib_test, distrib_another = EqualValue(), EqualValue(), EqualValue()
            inter_train_test, inter_train_another, inter_test_another = EqualValue(), EqualValue(), EqualValue()
            for client_i, (train_indice, test_indice, another_indice) in enumerate(indice_tuples):
                train_set, test_set, another_set = set(train_indice), set(test_indice), set(another_indice)
                if len(train_set) != len(train_indice) or \
                    len(test_set) != len(test_indice) or \
                    len(another_set) != len(another_indice) :
                    raise RuntimeError('indice not unique')
                len_data = len(client_whole_datasets[client_i])
                def get_rate(indice):
                    return len(indice) * 1.0 / len_data
                def get_distribution(indice):
                    return sum(1 for i in indice if i < len_data * 0.7) * 1.0 / len(indice)
                rate_train.value = get_rate(train_indice)
                rate_test.value = get_rate(test_indice)
                rate_another.value = get_rate(another_indice)
                distrib_train.value = get_distribution(train_indice)
                distrib_test.value = get_distribution(test_indice)
                distrib_another.value = get_distribution(another_indice)
                def handle_inter(value, set1, set2):
                    value.value = (len(set1 & set2)) * 1.0 / len(set1)
                handle_inter(inter_train_test, train_set, test_set)
                handle_inter(inter_train_another, train_set, another_set)
                handle_inter(inter_test_another, test_set, another_set)
            with open(os.path.join(args.target_dir_name, 'indice_report'), 'w') as f:
                f.write(f'   train   rate: {rate_train.value * 100:.2f}%\n')
                f.write(f'   test    rate: {rate_test.value * 100:.2f}%\n')
                f.write(f'   another rate: {rate_another.value * 100:.2f}%\n')
                f.write(f'   train   distribution: {distrib_train.value * 100:.2f}%, {(1-distrib_train.value) * 100:.2f}%\n')
                f.write(f'   test    distribution: {distrib_test.value * 100:.2f}%, {(1-distrib_test.value) * 100:.2f}%\n')
                f.write(f'   another distribution: {distrib_another.value * 100:.2f}%, {(1-distrib_another.value) * 100:.2f}%\n')
                f.write(f'   (train, test)    intersection: {inter_train_test.value * 100:.2f}%\n')
                f.write(f'   (train, another) intersection: {inter_train_another.value * 100:.2f}%\n')
                f.write(f'   (test , another) intersection: {inter_test_another.value * 100:.2f}%\n')
            client_train_loads = [create_dataloader(Subset(dataset, indices[0])) for indices, dataset in zip(indice_tuples, client_whole_datasets) ]
            client_test_loads = [create_dataloader(Subset(dataset, indices[1])) for indices, dataset in zip(indice_tuples, client_whole_datasets) ]
            client_another_loads = [create_dataloader(Subset(dataset, indices[2])) for indices, dataset in zip(indice_tuples, client_whole_datasets) ]
        elif args.new_trial_method == 'new':
            client_combine_datasets = [
                ConcatDataset([train_loader.dataset, test_loader.dataset]) 
                for train_loader, test_loader in zip(client_train_loads, client_test_loads)
            ]
            def split_dataset(dataset):
                while True:
                    try:
                        half1, half2 = cut_dataset(dataset, 0.5)
                        train, test = cut_dataset(half2, 0.7)
                        break
                    except _ as SampleError:
                        print("dataset only containes one sensitive feature, retry sampling (outside)")
                        continue
                return train, test, half1
            client_split_datasets = [split_dataset(dataset) for dataset in client_combine_datasets]
            client_train_loads = [datasets[0] for datasets in client_split_datasets]
            client_test_loads = [datasets[1] for datasets in client_split_datasets]
            client_another_loads = [datasets[2] for datasets in client_split_datasets]
        elif args.new_trial_method == 'bias':
            client_whole_datasets = [ConcatDataset([train.dataset, test.dataset]) 
                                    for train, test in zip(client_train_loads, client_test_loads)]
            def compute_indice(len_data):
                rate = args.new_trial_bias_rate
                indices_left, indices_right = np.arange(int(len_data * 0.5)), np.arange(int(len_data * 0.5), len_data)
                indices_half = np.union1d(
                                np.random.choice(indices_left, int(len(indices_left) * rate), replace = False),
                                np.random.choice(indices_right, int(len(indices_right) * (1-rate)), replace = False)
                            )
                indices_train = np.random.choice(indices_half, int(len(indices_half) * 0.7), replace = False)
                indices_test = np.setdiff1d(indices_half, indices_train)
                indices_another = np.setdiff1d(np.arange(len_data), indices_half)
                np.random.shuffle(indices_train)
                np.random.shuffle(indices_test)
                np.random.shuffle(indices_another)
                return indices_train, indices_test, indices_another
            indice_tuples = [compute_indice(len(dataset)) for dataset in client_whole_datasets]
            if not args.valid:
                with open(os.path.join(args.target_dir_name, 'indices.json'), 'w') as f:
                    json.dump([{'train': train.tolist(), 'test': test.tolist(), 'another': another.tolist()} for train, test, another in indice_tuples], f)
            # report sample
            class EqualValue:
                def __init__(self):
                    self.cnt = 0
                def __setattr__(self, __name: str, __value: Any):
                    if __name == 'value':
                        if self.cnt == 0:
                            object.__setattr__(self, __name, __value)
                        elif self.value < __value and self.value * 1.0 / __value > 0.50 or \
                            self.value > __value and __value * 1.0 / self.value > 0.50 or \
                            self.value < 0.01 and abs(self.value - __value) < 0.01 or \
                            self.value == __value:
                            object.__setattr__(self, __name, (self.value * self.cnt + __value) * 1.0 / (self.cnt+1))
                        else:
                            raise RuntimeError(f'client report value not equal: {self.value} vs {__value}')
                        self.cnt += 1
                    else:
                        object.__setattr__(self, __name, __value)
            rate_train, rate_test, rate_another = EqualValue(), EqualValue(), EqualValue()
            distrib_train, distrib_test, distrib_another = EqualValue(), EqualValue(), EqualValue()
            inter_train_test, inter_train_another, inter_test_another = EqualValue(), EqualValue(), EqualValue()
            for client_i, (train_indice, test_indice, another_indice) in enumerate(indice_tuples):
                train_set, test_set, another_set = set(train_indice), set(test_indice), set(another_indice)
                if len(train_set) != len(train_indice) or \
                    len(test_set) != len(test_indice) or \
                    len(another_set) != len(another_indice) :
                    raise RuntimeError('indice not unique')
                len_data = len(client_whole_datasets[client_i])
                def get_rate(indice):
                    return len(indice) * 1.0 / len_data
                def get_distribution(indice):
                    return sum(1 for i in indice if i < len_data * 0.5) * 1.0 / len(indice)
                rate_train.value = get_rate(train_indice)
                rate_test.value = get_rate(test_indice)
                rate_another.value = get_rate(another_indice)
                distrib_train.value = get_distribution(train_indice)
                distrib_test.value = get_distribution(test_indice)
                distrib_another.value = get_distribution(another_indice)
                def handle_inter(value, set1, set2):
                    value.value = (len(set1 & set2)) * 1.0 / len(set1)
                handle_inter(inter_train_test, train_set, test_set)
                handle_inter(inter_train_another, train_set, another_set)
                handle_inter(inter_test_another, test_set, another_set)
            with open(os.path.join(args.target_dir_name, 'indice_report'), 'w') as f:
                f.write(f'   train   rate: {rate_train.value * 100:.2f}%\n')
                f.write(f'   test    rate: {rate_test.value * 100:.2f}%\n')
                f.write(f'   another rate: {rate_another.value * 100:.2f}%\n')
                f.write(f'   train   distribution: {distrib_train.value * 100:.2f}%, {(1-distrib_train.value) * 100:.2f}%\n')
                f.write(f'   test    distribution: {distrib_test.value * 100:.2f}%, {(1-distrib_test.value) * 100:.2f}%\n')
                f.write(f'   another distribution: {distrib_another.value * 100:.2f}%, {(1-distrib_another.value) * 100:.2f}%\n')
                f.write(f'   (train, test)    intersection: {inter_train_test.value * 100:.2f}%\n')
                f.write(f'   (train, another) intersection: {inter_train_another.value * 100:.2f}%\n')
                f.write(f'   (test , another) intersection: {inter_test_another.value * 100:.2f}%\n')
            client_train_loads = [create_dataloader(Subset(dataset, indices[0])) for indices, dataset in zip(indice_tuples, client_whole_datasets) ]
            client_test_loads = [create_dataloader(Subset(dataset, indices[1])) for indices, dataset in zip(indice_tuples, client_whole_datasets) ]
            client_another_loads = [create_dataloader(Subset(dataset, indices[2])) for indices, dataset in zip(indice_tuples, client_whole_datasets) ]
        else:
            raise RuntimeError('args.new_trial_method invalid')
        return client_train_loads, client_test_loads, client_another_loads
    elif "eicu" in args.dataset:
        def create_dataloader(dataset):
            return DataLoader(dataset,
                            batch_size=len(dataset), 
                            shuffle = args.shuffle,
                            num_workers = args.num_workers,
                            pin_memory = True,
                            drop_last = args.drop_last)
        client_whole_datasets = [ConcatDataset([train.dataset, test.dataset]) 
                                    for train, test in zip(client_train_loads, client_test_loads)]
        def compute_indice(len_data):
            indices = np.arange(len_data)
            np.random.shuffle(indices)
            return indices[:int(len(indices)* 0.7)], indices[int(len(indices)* 0.7):]
        if not args.valid:
            indice_tuples = [compute_indice(len(dataset)) for dataset in client_whole_datasets]
            with open(os.path.join(args.target_dir_name, 'indices.json'), 'w') as f:
                json.dump([{'train': train.tolist(), 'test': test.tolist()} for train, test in indice_tuples], f)
        else:
            with open(os.path.join(args.target_dir_name, 'indices.json'), 'r') as f:
                content = json.load(f)
            indice_tuples = [(np.array(client_content['train']), np.array(client_content['test'])) for client_content in content]
        client_train_loads = [create_dataloader(Subset(dataset, indices[0])) for indices, dataset in zip(indice_tuples, client_whole_datasets) ]
        client_test_loads = [create_dataloader(Subset(dataset, indices[1])) for indices, dataset in zip(indice_tuples, client_whole_datasets) ]
    return client_train_loads, client_test_loads

