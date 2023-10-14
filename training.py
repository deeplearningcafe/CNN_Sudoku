import argparse
import datetime
import os
import random
import sys

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms



import logging

from dsets import Sudoku
import models
logging.basicConfig(level=logging.DEBUG)
import shutil


log = logging.getLogger(__name__)

seed = 42
torch.manual_seed(seed)
random.seed(seed)  # Pythonの標準の乱数シードを設定
np.random.seed(seed)  # NumPyの乱数シードを設定

# GPUが利用可能な場合、その乱数シードも設定
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TrainingApp:
    def __init__(self, sys_argv=None) -> None:
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size for training',
                            default=500,
                            type=int)
        
        parser.add_argument('--epochs',
                            help='Number of epochs to train',
                            default=16,
                            type=int)
        
        parser.add_argument('--tb-prefix',
                            default='Sudoku',
                            help="Data prefix to use for Tensorboard run.",
                            )

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='CNNModel_BaseV8_newNorm',
                            )
        
        self.args = parser.parse_args(sys_argv)    
        self.path ="E:\Data\sudoku\sudoku.csv"
        self.transform = self.init_transform()
        self.totalTrainingSamples_count = 0

        # it would be necessary to add a test dataset
        # create a list with the indexs and divide it 
        self.train_idx = random.sample(range(0, 1000000 - 2000), 1000000 - 2000)
        self.val_idx = self.train_idx[::10]
        del self.train_idx[::10]
        # then we need to transform the lists to np arrays so that we can use them as index of a narray
        self.train_idx = np.array(self.train_idx)
        self.val_idx = np.array(self.val_idx)

        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

        
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.writer_val = self.initTensorboardWriters("val")
        self.writer_trn = self.initTensorboardWriters("train")


    def initDataset(self, normalizeVal = False):
        batch_size = self.args.batch_size
        # added another way to normalize data so that is in [-0.5, 0.5]
        # this divide by 9 and substracts 0.5
        if normalizeVal:
            dataset_train = Sudoku(self.path, idx=self.train_idx, normalizeVal=True)
            dataset_val = Sudoku(self.path, idx=self.val_idx,  normalizeVal=True)
            
        else:
            # this applies the normalize transform from pytorch
            dataset_train = Sudoku(self.path, transform=self.transform, idx=self.train_idx)
            dataset_val = Sudoku(self.path, transform=self.transform, idx=self.val_idx)

        log.info(f"Lenght of the datasets {len(dataset_train)} and of the val {len(dataset_val)}")
        
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size)
        return train_loader, val_loader
    
    def initTensorboardWriters(self, phase):
        log_dir = os.path.join('runs', self.args.tb_prefix, self.time_str)
        if phase == "train":
            writer = SummaryWriter(log_dir=log_dir + '-trn-' + self.args.comment)
        elif phase == "val":
            writer = SummaryWriter(log_dir=log_dir + '-val-' + self.args.comment)
        return writer
    
    def initModel(self):
        # there is a notebook that uses a model of 10 million parameters, it gets 97 acc
        #model = CNNModel_BaseModelV3()parameters = 432.145;
        #model = CNNModel_Base() #parameters = 391.113;
        #model = CNNModel_BaseV2() parameters = 981.705;
        #model = CNNModel_BaseV3()
        #model = CNNModel_BaseV4() #parameters = 8.417.289;
        #model = CNNModel_BaseV5() #parameters = 3.918.729;
        #model = CNNModel_BaseV6() #parameters = 13.138.953;
        #model = models.CNNModel_BaseV7() #paramteters = 20.222.985;;
        model = models.CNNModel_BaseV8() #paramteters = 17.860.617;

        param_num = [p.numel() for p in model.parameters()]

        if self.use_cuda:
            log.info("Time {}, Using CUDA; {} devices.".format(datetime.datetime.now(), torch.cuda.device_count()))
            log.info("Total paramteters {}; List of parameters {}".format(sum(param_num), param_num))
            model = model.to(self.device)
            
        return model
        
    def initOptimizer(self):
        return Adam(self.model.parameters(), lr=3e-4, weight_decay=0.001)    
    
    
    def init_transform(self):
        # the mean and the std were computed using all the samples
        mean = 2.0872
        std = 2.977
        transformation = transforms.Normalize(mean, std)
        return transformation
    
    def main(self):
        train_loader, val_loader = self.initDataset(normalizeVal=True)
        #log.info("Train size {}; Val size {}".format(len(train_loader.dataset), len(val_loader.dataset)))
        
        min_loss = 5.0
        for epoch in range(1, self.args.epochs + 1):
            log.info(f"Time {datetime.datetime.now()}, Epoch {epoch}")
            
            for batch_idx, batch_tuple in enumerate(train_loader):
                self.model.train()
                loss = self.computeBatchLoss(batch_idx, batch_tuple)

                # update the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.totalTrainingSamples_count += len(train_loader.dataset)
            
            # it is not optimal to validate in all the epochs as it takes time, so select a number
            if epoch == 1 or epoch % 2 == 0:
                self.model.eval()
                for name, loader in [("train", train_loader), ("val", val_loader)]:
                    batch_metrics = torch.zeros(len(loader), 5, device=self.device) # we want to store 5 variables

                    with torch.no_grad():  # to not update the parameters as we are validating
                        for batch_idx, batch_tuple in enumerate(loader):
                            self.computeBatchLoss(batch_idx, batch_tuple, batch_metrics, evaluation=True)

                    loss_val = self.logMetrics(epoch, name, batch_metrics, self.totalTrainingSamples_count)
                    
                    # we save the model in the validation loop, so that the best model is the one with the min loss in the val set
                    if name=="val":
                        min_loss = min(min_loss, loss_val)
                        self.saveModel(epoch, min_loss == loss_val)
        
        # close the tensorboard writers
        self.writer_val.close()
        self.writer_trn.close()
                
                

    def computeBatchLoss(self, batch_idx, batch_tuple, batch_metrics=None, evaluation=False):
        # returns the mean of the batch loss by default
        # this is a k dimension case, so as there are 9 clases, the values of the target must be
        # between 0 and 8, that is why we substract 1 from the solutions dataset
        loss_func = nn.CrossEntropyLoss()

        quizzes, solutions = batch_tuple
        
        # [batch, 1, 9, 9] the quizzes and [batch, 9, 9] the solutions
        quizzes = quizzes.to(self.device)
        solutions = solutions.to(self.device)
        
        outputs = self.model(quizzes)
        loss = loss_func(outputs, solutions)

        if evaluation:
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions = 0
            
            # this metrics are somewhat not real as the model starts with several numbers, around 20, so the even if the model
            # does not predict any number, it will still have 35/81=0.43 accuracy
            for i in range(len(outputs)):
                correct_predictions += (predictions[i] == solutions[i]).sum()
                    
            accuracy = correct_predictions / (solutions.shape[0] * 81)

            
            num_classes = 9
            true_positives = torch.zeros(num_classes)
            false_positives = torch.zeros(num_classes)
            false_negatives = torch.zeros(num_classes)

            for i in range(num_classes):
                # this is comparing each class, number, of the entire sudokus
                true_positives[i] = ((outputs.argmax(dim=1) == i) & (solutions == i)).sum().item()
                false_positives[i] = ((outputs.argmax(dim=1) == i) & (solutions != i)).sum().item()
                false_negatives[i] = ((outputs.argmax(dim=1) != i) & (solutions == i)).sum().item()

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)

            # check if values are not numbers because they will be used for the f1-score
            for i in range(num_classes):
                if np.isnan(precision[i]):
                    precision[i] = 0.0
                if np.isnan(recall[i]):
                    recall[i] = 0.0


            # Store metrics in the batch_metrics tensor
            # compute precision and recall for all classes
            precision = float(precision.sum()) / precision.shape[0]
            recall = float(recall.sum()) / recall.shape[0]
            # f1 score
            if precision + recall != 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0

            batch_metrics[batch_idx, 0] = loss
            batch_metrics[batch_idx, 1] = accuracy
            batch_metrics[batch_idx, 2] = precision
            batch_metrics[batch_idx, 3] = recall
            batch_metrics[batch_idx, 4] = f1_score

        return loss
            

    def logMetrics(self, epoch, phase, batch_metrics, totalTrainingSamples_count):
        # this is the average of each epoch, as it is doing the average of the batchs metrics
        avg_loss = float(batch_metrics[:, 0].sum()) / batch_metrics.shape[0]
        avg_accuracy = float(batch_metrics[:, 1].sum()) / batch_metrics.shape[0]
        avg_precision = float(batch_metrics[:, 2].sum()) / batch_metrics.shape[0]
        avg_recall = float(batch_metrics[:, 3].sum()) / batch_metrics.shape[0]
        avg_f1_score = float(batch_metrics[:, 4].sum()) / batch_metrics.shape[0]

        time = datetime.datetime.now()
        log.info(
            "Time {} | Epoch {} | Phase {} | Loss: {:.4f} | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f}".format(
                time, epoch, phase, avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1_score
            ))
        
        # this loss is used for saving the best model
        loss_val = None
        if phase == "val":
            self.writer_val.add_scalar('Loss', avg_loss, totalTrainingSamples_count)
            self.writer_val.add_scalar('Accuracy', avg_accuracy, totalTrainingSamples_count)
            self.writer_val.add_scalar('Precision', avg_precision, totalTrainingSamples_count)
            self.writer_val.add_scalar('Recall', avg_recall, totalTrainingSamples_count)
            self.writer_val.add_scalar('F1 Score', avg_f1_score, totalTrainingSamples_count)

            loss_val = avg_loss
        elif phase == "train":
            self.writer_trn.add_scalar('Loss', avg_loss, totalTrainingSamples_count)
            self.writer_trn.add_scalar('Accuracy', avg_accuracy, totalTrainingSamples_count)
            self.writer_trn.add_scalar('Precision', avg_precision, totalTrainingSamples_count)
            self.writer_trn.add_scalar('Recall', avg_recall, totalTrainingSamples_count)
            self.writer_trn.add_scalar('F1 Score', avg_f1_score, totalTrainingSamples_count)
        
        return loss_val
        
    def saveModel(self, epoch, isBest=False):
        file_path = os.path.join(
            'models',
            self.args.tb_prefix,
            '{}_{}.{}.state'.format(
                self.time_str,
                self.args.comment,
                self.totalTrainingSamples_count,
            )
        )
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
        model = self.model

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'models',
                self.args.tb_prefix,
                '{}_{}.{}.state'.format(
                    self.time_str,
                    self.args.comment,
                    'best',
                )
            )
            shutil.copyfile(file_path, best_path)

            log.debug("Saved best model params to {}".format(best_path))
        
        
        

        
        
        
if __name__ == '__main__':
    app = TrainingApp()  # クラスのインスタンスを作成
    app.main()  # インスタンスからメソッドを呼び出す
