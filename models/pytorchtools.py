import os
import numpy as np
import torch
#####################MAKE IT SAVE THE MODEL WITH THE BEST ACCURACY
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint, patience=10, verbose=False, delta=9e-3, use_cuda=False, many_gpu=False, start_countdown=80):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """

        self.use_cuda = use_cuda ;  self.many_gpu = many_gpu
        self.checkpoint = checkpoint ; self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.start_countdown = start_countdown
        self.best_accuracy = None ; self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf ; self.val_acc_min = 0.0
        self.delta = delta

    def __call__(self, val_loss, val_acc, model, epoch=0):

        score = val_loss
        accuracy = val_acc * 100

        if self.best_accuracy is None:
            self.best_score = score
            self.best_accuracy = accuracy
            self.save_checkpoint(score, accuracy, model)

        elif accuracy <= self.best_accuracy:
            if epoch >= self.start_countdown:
                self.counter += 1
                print("EarlyStopping counter: %d out of %d" %(self.counter,self.patience))
                if self.counter >= self.patience:
                    self.early_stop = True

        elif accuracy > self.best_accuracy:
            self.best_score = score
            self.best_accuracy = accuracy
            self.save_checkpoint(score, accuracy, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_acc, model):
        '''Saves model when validation accuracy increase (and when loss decrease).'''

        if self.verbose:
            print('Validation accuracy increased ({0:.4f} --> {1:.4f}). Saving model ...'.format(self.val_acc_min,val_acc))
            # print('Validation loss decreased ({0:.6f} --> {1:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))

        if self.use_cuda and self.many_gpu:
            torch.save(model.module.state_dict(), os.getcwd()+'/DHCS_implement/Saved_models/'+self.checkpoint)
        else:
            torch.save(model.state_dict(), os.getcwd()+'/DHCS_implement/Saved_models/'+self.checkpoint)

        self.val_loss_min = val_loss
        self.val_acc_min = val_acc
