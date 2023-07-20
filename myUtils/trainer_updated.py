import os
import time
from torch.autograd import Variable
from sklearn.metrics import precision_score, recall_score, auc, precision_recall_curve, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from Data.datasets_updated import GetDataset
from torchvision.transforms import transforms
from torchvision import models
os.system('')
import pandas as pd
import torch
import copy
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from modeling.resnet import cResNet, ResNet18
from modeling.models import DenseNet, VGG, MLPclassifier
from modeling.vae import LinearVAE
from torchmetrics.classification import AUROC
import torch.nn.functional as F


class Trainer_New(object):
    def __init__(self, args):
        self.args = args

        if 0 in self.args.gpu_ids:
            cuda_val = 'cuda:0'
        elif 1 in self.args.gpu_ids:
            cuda_val = 'cuda:1'
        elif not torch.cuda.is_available():
            cuda_val = 'cpu'
        else:
            cuda_val = 'cpu'
        DEVICE = torch.device(cuda_val)
        # "cuda:0" if torch.cuda.is_available() else "cpu"
        print('Current selected DEVICE: ', DEVICE)
        self.device = DEVICE
        self.pretrained = self.args.pretrained

        self.weight_eval = False
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.kwargs = kwargs
        self.nclasses = 2 # default is binary classification
        self.data_path = args.data_path

        # Define mean, std dev
        if self.args.dataset == 'RCT':
            self.mean = .1706
            self.std = .2112
        elif self.args.dataset == 'OASIS':
            self.mean = 0.1745
            self.std = 0.1518

        # Define loaders
        # OCT data has separate train, test spreadsheet
        te_path = self.args.test_spreadsheet
        tr_path = self.args.train_spreadsheet
        self.train_pool = None

        test_data = pd.read_csv(te_path)

        test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
                                                 transforms.Normalize(mean=self.mean, std=self.std)])

        test_dataset = GetDataset(df=test_data, img_dir=self.data_path,transform=test_transform, dataset=self.args.dataset)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.test_batch_size, shuffle=False)


        self.test_fixed_loader = test_loader  # entire test data

        # initialize test loader and blank dynamic loader
        self.test_loader = test_loader
        self.test_dynamic_loader = test_loader

        # init blank train loader and grad loader and unlabeled loader
        train_loader = None
        self.train_loader = None
        self.grad_loader = None
        self.cur_used = np.array([])
        self.unlabeled_loader = train_loader
        self.grad_loader_test = test_loader
        self.unlabeled_loader_test = test_loader

        self.prev_acc = np.zeros(len(test_data), dtype=int)

        # Setup model
        model = self.get_model(args.architecture)
        model.to(self.device)

        # define optimizer
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                                        weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 110], gamma=0.2)
        else:
            print('Specified optimizer not recognized. Options are: adam and sgd')

        # Define Loss
        if args.strategy == 'badge':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # init model and optimizer
        self.model, self.optimizer = model, optimizer

        # Using cuda
        # if args.cuda:
        #     # use multiple GPUs if available
        #     self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        #     # use all GPUs
        #     self.model = self.model.cuda()
        #     if self.args.train_type == 'traditional':
        #         self.criterion = self.criterion.cuda()

    def get_model(self, architecture):
        if architecture == 'resnet_18':
            model = cResNet(type=18, num_classes=self.nclasses, pretrained=self.pretrained)
            # model = ResNet18(num_classes=self.nclasses)
        elif architecture == 'resnet_34':
            model = cResNet(type=34, num_classes=self.nclasses)
        elif architecture == 'resnet_50':
            model = cResNet(type=50, num_classes=self.nclasses, pretrained=self.pretrained)
        elif architecture == 'resnet_101':
            model = cResNet(type=101, num_classes=self.nclasses)
        elif architecture == 'resnet_152':
            model = cResNet(type=152, num_classes=self.nclasses)
        elif architecture == 'densenet_121':
            if self.pretrained:
                model = models.densenet121(weights='DEFAULT')
            else:
                model = models.densenet121()
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, self.nclasses), nn.Sigmoid())
        elif architecture == 'densenet_161':
            model = DenseNet(type=161, num_classes=self.nclasses)
        elif architecture == 'densenet_169':
            model = DenseNet(type=169, num_classes=self.nclasses)
        elif architecture == 'densenet_201':
            model = DenseNet(type=201, num_classes=self.nclasses)
        elif architecture == 'vgg_11':
            model = VGG(type=11, num_classes=self.nclasses)
        elif architecture == 'vgg_13':
            model = VGG(type=13, num_classes=self.nclasses)
        elif architecture == 'vgg_16':
            model = VGG(type=16, num_classes=self.nclasses)
        elif architecture == 'vgg_19':
            model = VGG(type=19, num_classes=self.nclasses)
        elif architecture == 'mlp':
            model = MLPclassifier(dim=self.dim, num_classes=self.nclasses)
        elif architecture == 'vae':
            model = LinearVAE()
        else:
            raise NotImplementedError

        return model

    def update_loaders(self, current_idxs, batch_size, data, total_week_indices, current_idxs_test, data_test,
                       total_te_week_ind, train_pool):
        # transform
        train_transform = transforms.Compose([])
        train_transform.transforms.append(transforms.Grayscale(num_output_channels=1))
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(transforms.Normalize(mean=self.mean, std=self.std))

        test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),transforms.Normalize(mean=self.mean, std=self.std)])

        train_dataset = GetDataset(df=data.loc[current_idxs], img_dir=self.data_path, transform=train_transform,
                                   dataset=self.args.dataset, fold='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.train_pool = train_pool

        # Determine currently used indexes of samples
        if not self.args.continual:
            # Active learning
            self.cur_used = current_idxs
        else:
            # for continual learning, we need to keep track of all past used samples
            # so for BADGE, CORESET, etc. we get embeddings of UNUSED samples up to this point
            self.cur_used = np.concatenate((self.cur_used, current_idxs))

        unused = np.where(np.isin(total_week_indices, self.cur_used, invert=True))[0]

        unused_idxs = total_week_indices[unused]
        unused_data = GetDataset(df=data.loc[unused_idxs], img_dir=self.data_path, transform=test_transform,
                                 dataset=self.args.dataset, fold='train')
        grad_loader = DataLoader(dataset=unused_data, batch_size=1, shuffle=True)
        unlabeled_loader = DataLoader(dataset=unused_data, batch_size=batch_size, shuffle=True)

        # Test set (dynamic) unused idxs - needs to be modified for continual learning
        unused_te = np.where(np.isin(total_te_week_ind, current_idxs_test, invert=True))[0]

        if len(unused_te) != 0:
            # Different experimental setup where dynamic test set is composed of queried test samples (not presented in paper)
            # for dynamic test set query methods (badge and coreset)
            unused_idxs_te = total_te_week_ind[unused_te]
            unused_te_data = GetDataset(df=data_test.loc[unused_idxs_te], img_dir=self.data_path,
                                        transform=test_transform, dataset=self.args.dataset, fold='test')
            grad_loader_test = DataLoader(dataset=unused_te_data, batch_size=1, shuffle=True)
            unlabeled_loader_test = DataLoader(dataset=unused_te_data, batch_size=batch_size, shuffle=True)
            self.grad_loader_test = grad_loader_test
            self.unlabeled_loader_test = unlabeled_loader_test

        if current_idxs_test.any() != None:
            # Establish dynamic test set loader
            test_dataset = GetDataset(df=data_test.loc[current_idxs_test], img_dir=self.data_path,
                                      transform=test_transform, dataset=self.args.dataset, fold='test')
            test_loader_dynamic = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
            self.test_dynamic_loader = test_loader_dynamic

        self.train_loader = train_loader
        self.grad_loader = grad_loader
        self.unlabeled_loader = unlabeled_loader

        return

    def randomly_sample(self, number):
        # For continual learning memory set
        previous_samples = np.random.choice(self.cur_used, size=number, replace=False)
        return previous_samples

    def clear_statistics(self):
        '''Resets model; necessary if performing active learning.'''

        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        self.model = self.model.apply(weight_reset)
        # if using multiple GPUs....
        # if self.args.cuda:
        #     self.model = self.model.apply(weight_reset).cuda()
        # else:
        #     self.model = self.model.apply(weight_reset)

    def training(self, epoch):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self.model.train()
        tbar = tqdm(self.train_loader)

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0
        print('the device is: ', self.device)

        # iterate over all samples in each batch i
        for i, (image, target, idx, actual_idx) in enumerate(tbar):
            # convert target to one hot vectors
            one_hot = torch.zeros(target.shape[0], self.nclasses)

            one_hot[range(target.shape[0]), target.long()] = 1

            # assign each image and target to GPU
            if self.args.cuda:
                image, target = image.to(self.device), target.to(self.device)
                #one_hot = one_hot.cuda()
                one_hot = one_hot.to(self.device)

            # update model
            self.optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            # computes output of our model
            output = self.model(image)

            logit, pred = torch.max(output.data, 1)
            #print(pred)
            total += target.size(0)

            # Perform model update
            if self.args.strategy == 'badge':
                loss = self.criterion(output, target)
            else:
                loss = self.criterion(output, one_hot)
            # perform backpropagation
            loss.backward()

            # update params with gradient
            self.optimizer.step()

            # extract loss value as float and add to train_loss
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            correct_samples += pred.eq(target.data).cpu().sum()

        # Update optimizer step
        if self.args.optimizer == 'sgd':
            self.scheduler.step(epoch)

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('Training Accuracy: %.3f' % acc)

        return acc, self.model, self.optimizer

    def testing(self, epoch, mode, vis_mode, q, round):
        # set model to evaluation mode
        self.model.eval()
        if mode == 'normal':
            # for larger test set / different test set
            tbar = tqdm(self.test_loader)
        elif mode == 'fixed':
            # fixed test set
            tbar = tqdm(self.test_fixed_loader)
        else:
            tbar = tqdm(self.test_dynamic_loader)

        # init statistics parameters
        test_loss = 0.0

        pred_all = np.array([], dtype=int)
        gt_all = np.array([], dtype=int)
        yscore_all = np.array([], dtype=int)

        # overall test accuracy
        correct_samples = 0
        total_samples = 0
        preds_and_output = np.array([], dtype=np.int64).reshape(0, 4)

        # iterate over all sample batches
        forgets = 0
        for i, (image, target, idx, actual_idx) in enumerate(tbar):
            # convert target to one hot vectors
            one_hot = torch.zeros(target.shape[0], self.nclasses)
            one_hot[range(target.shape[0]), target.long()] = 1
            # set cuda
            if self.args.cuda:
                image, target = image.to(self.device), target.to(self.device)
                #one_hot = one_hot.cuda()
                one_hot = one_hot.to(self.device)

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():
                # get output
                output = self.model(image)

                logit, pred = torch.max(output.data, 1)
                probs = F.softmax(output.data, dim=1)[:, 1]
                yscore_all = np.concatenate((yscore_all, probs.cpu().numpy()), axis=0)

                pred_all = np.concatenate((pred_all, pred.cpu().numpy()), axis=0)
                gt_all = np.concatenate((gt_all, target.cpu().numpy()), axis=0)

                # calculate loss between output and target
                if self.args.strategy == 'badge':
                    loss = self.criterion(output, target.long())
                else:
                    loss = self.criterion(output, one_hot)

                # append loss to total loss
                test_loss += loss.item()

                t1 = np.expand_dims(idx.cpu().numpy(), axis=1)
                t3 = np.expand_dims(pred.cpu().numpy(), axis=1)
                t4 = np.expand_dims(target.cpu().numpy(), axis=1)
                t5 = np.expand_dims(logit.cpu().numpy(), axis=1)

                collect_preds_and_output = np.concatenate([t3, t4, t5, t1], axis=1)
                preds_and_output = np.vstack([preds_and_output, collect_preds_and_output])

                # forgetting events
                accuracy = pred.eq(target.data)
                # print('idxs: ', idx)
                # print('acc: ', accuracy)
                # print('prev acc: ', self.prev_acc[idx])
                delta = np.clip(self.prev_acc[idx] - accuracy.cpu().numpy(), a_min=0, a_max=1)
                # print('delta: ', delta)
                self.prev_acc[idx] = accuracy.cpu().numpy()  # cpu().
                # print('new prev: ', self.prev_acc[idx])

                forgets += np.sum(delta)
                print('FORGETS: ', forgets)

                # overall acc
                total_samples += target.size(0)
                correct_samples += pred.eq(target.data).cpu().sum()

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total_samples
        precision = precision_score(gt_all, pred_all, average=None)
        recall = recall_score(gt_all, pred_all, average=None)
        balanced_acc = balanced_accuracy_score(gt_all, pred_all)
        
        AUC = roc_auc_score(gt_all, pred_all)
        print('AUC ROC: ', AUC)

        output_struct = {}
        output_struct['acc'] = acc
        output_struct['auc'] = AUC
        output_struct['precision'] = precision
        output_struct['recall'] = recall
        output_struct['balanced acc'] = balanced_acc

        store_model_preds = {}
        store_model_preds['prediction'] = preds_and_output[:, 0]
        store_model_preds['ground truth'] = preds_and_output[:, 1]
        store_model_preds['logit'] = preds_and_output[:, 2]
        store_model_preds['idx'] = preds_and_output[:, 3]

        # Printing
        print('Testing:')
        print('Loss: %.3f' % test_loss)
        print('Test Accuracy: %.3f' % acc)
        print('Balanced test accuracy: %.3f' % balanced_acc)

        return output_struct, store_model_preds, forgets

    def get_probs(self, mode):
        '''Calculates the gradients for all elements within the test pool and ranks the highest idxs'''

        # set model to evaluation mode
        self.model.eval()

        if mode == 'tr':
            tbar = tqdm(self.grad_loader, desc='\r')
        else:
            tbar = tqdm(self.grad_loader_test, desc='\r')

        # init softmax layer
        softmax = torch.nn.Softmax(dim=1)

        # init probability array to zero maximum value of sigmoid is 1.0 therefore ignore all values larger than that
        probs = torch.full((self.train_pool, self.nclasses), 2.5, dtype=torch.float)
        indices = torch.zeros(self.train_pool)

        if self.args.cuda:
            probs, indices = probs.to(self.device), indices.to(self.device)

        with torch.no_grad():
            # iterate over all sample batches
            for i, (image, target, idxs, actual_idx) in enumerate(tbar):
                # assign each image and target to GPU
                if self.args.cuda:
                    image, target = image.to(self.device), target.to(self.device)

                # convert image to suitable dims
                image = image.float()

                # computes output of our model
                output = self.model(image)

                # get sigmoid probs
                probs_output = softmax(output)

                # insert to probs array
                probs[actual_idx.long()] = probs_output
                indices[actual_idx.long()] = 1

        # sort idxs
        output_structure = {}
        output_structure['probs'] = probs[indices == 1].cpu().numpy()
        ind_list = (indices == 1).nonzero().cpu().numpy()
        output_structure['indices'] = ind_list

        return output_structure

    def get_badge_embeddings(self, mode):
        '''Calculates the gradients for all elements within the test pool and ranks the highest idxs'''
        # update gradnorm update number
        # self.num_gradnorm_updates += 1
        # set model to evaluation mode
        self.model.eval()

        # get embed dim
        embedDim = self.model.get_penultimate_dim()

        if mode == 'tr':
            tbar = tqdm(self.grad_loader, desc='\r')
        else:
            tbar = tqdm(self.grad_loader_test, desc='\r')

        # init softmax layer
        softmax = torch.nn.Softmax(dim=1)

        # init embedding tesnors and indices for tracking
        embeddings = torch.zeros((self.train_pool, embedDim * self.nclasses), dtype=torch.float)
        indices = torch.zeros(self.train_pool)

        if self.args.cuda:
            embeddings, indices = embeddings.to(self.device), indices.to(self.device)

        with torch.no_grad():
            # iterate over all sample batches
            for i, (image, target, idxs, actual_idx) in enumerate(tbar):
                # assign each image and target to GPU
                if self.args.cuda:
                    image, target = image.to(self.device), target.to(self.device)

                # convert image to suitable dims
                image = image.float()

                # computes output of our model
                output = self.model(image)

                penultimate = self.model.penultimate_layer

                # get softmax probs
                probs_output = softmax(output)

                _, pred = torch.max(output.data, 1)

                # insert to embedding array
                for j in range(target.shape[0]):
                    for c in range(self.nclasses):
                        if c == pred[j].item():
                            embeddings[actual_idx[j], embedDim * c: embedDim * (c + 1)] = copy.deepcopy(
                                penultimate[j]) * \
                                                                                          (1 - probs_output[
                                                                                              j, c].item())
                        else:
                            embeddings[actual_idx[j], embedDim * c: embedDim * (c + 1)] = copy.deepcopy(
                                penultimate[j]) * \
                                                                                          (-1 * probs_output[
                                                                                              j, c].item())
                indices[actual_idx.long()] = 1

        # sort idxs
        output_structure = {}
        output_structure['embeddings'] = embeddings[indices == 1].cpu().numpy()
        ind_list = (indices == 1).nonzero().cpu().numpy()
        output_structure['indices'] = ind_list

        return output_structure

    def get_embeddings(self, mode, loader_type: str = 'unlabeled'):
        # set model to evaluation mode
        self.model.eval()

        # get embed dim
        embedDim = self.model.get_penultimate_dim()
            
        if loader_type == 'labeled' and mode == 'tr':
            tbar = tqdm(self.train_loader, desc='\r')
        elif loader_type == 'unlabeled' and mode == 'tr':
            tbar = tqdm(self.unlabeled_loader, desc='\r')
        elif loader_type == 'unlabeled' and mode == 'te':
            tbar = tqdm(self.unlabeled_loader_test, desc='\r')
        elif loader_type == 'labeled' and mode == 'te':
            tbar = tqdm(self.test_dynamic_loader, desc='\r')
        else:
            raise Exception('You can only load labeled and unlabeled pools!')

        # init softmax layer
        softmax = torch.nn.Softmax(dim=1)

        # init embedding tesnors and indices for tracking
        embeddings = torch.zeros((self.train_pool, embedDim * self.nclasses),
                                 dtype=torch.float)
        nongrad_embeddings = torch.zeros((self.train_pool, embedDim),
                                         dtype=torch.float)
        indices = torch.zeros(self.train_pool)

        if self.args.cuda:
            embeddings, indices = embeddings.to(self.device), indices.to(self.device)
            nongrad_embeddings = nongrad_embeddings.to(self.device)

        with torch.no_grad():
            # iterate over all sample batches
            for i, (image, target, idxs, actual_idx) in enumerate(tbar):
                # assign each image and target to GPU
                if self.args.cuda:
                    image, target = image.to(self.device), target.to(self.device)

                # convert image to suitable dims
                image = image.float()

                # computes output of our model
                output = self.model(image)

                # get penultimate embedding
                
                penultimate = self.model.penultimate_layer

                nongrad_embeddings[actual_idx] = penultimate

                # get softmax probs
                probs_output = softmax(output)

                _, pred = torch.max(output.data, 1)

                # insert to embedding array
                for j in range(target.shape[0]):
                    for c in range(self.nclasses):
                        if c == pred[j].item():
                            embeddings[actual_idx[j], embedDim * c: embedDim * (c + 1)] = copy.deepcopy(
                                penultimate[j]) * \
                                                                                          (1 - probs_output[
                                                                                              j, c].item())
                        else:
                            embeddings[actual_idx[j], embedDim * c: embedDim * (c + 1)] = copy.deepcopy(
                                penultimate[j]) * \
                                                                                          (-1 * probs_output[
                                                                                              j, c].item())
                indices[actual_idx.long()] = 1

        # sort idxs
        output_structure = {}
        output_structure['embeddings'] = embeddings[indices == 1].cpu().numpy()
        ind_list = (indices == 1).nonzero().cpu().numpy()
        output_structure['indices'] = ind_list
        output_structure['nongrad_embeddings'] = nongrad_embeddings[indices == 1].cpu().numpy()

        return output_structure

    # debugging
    def embeddings(self, mode, loader_type: str = 'unlabeled'):
        features = []
        feat = np.array([])
        iidxs = np.array([])
        self.model.eval()
        count = 0

        if loader_type == 'labeled' and mode == 'tr':
            tbar = tqdm(self.train_loader, desc='\r')
            loader = self.train_loader
        elif loader_type == 'unlabeled' and mode == 'tr':
            tbar = tqdm(self.unlabeled_loader, desc='\r')
            loader = self.unlabeled_loader

        with torch.no_grad():
            for i, (image, target, idxs, actual_idx) in enumerate(tbar):
                data = image
                target = target
                data, target = data.to(self.device), target.to(self.device)
                # convert image to suitable dims
                data = data.float()

                # computes output of our model
                output = self.model(data)
                penultimate = self.model.penultimate_layer
                count += 1
                features.append(penultimate.cpu().numpy())
                feat = np.vstack([feat, penultimate.cpu().numpy()]) if feat.size else penultimate.cpu().numpy()
                iidxs = np.concatenate((iidxs, actual_idx.cpu().numpy()))

        return feat, iidxs
