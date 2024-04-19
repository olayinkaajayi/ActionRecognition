import os
import argparse
import torch
from tqdm import tqdm
import numpy as np
from models.training_utils_py2 import load_data_from_file


def add_connection(A,indx_a,indx_b):
    """
        When a pair of nodes have a connection,
        this function adds 1 to their position in the
        (symetric) adjacency matrix.
    """
    A[indx_a,indx_b] = 1
    A[indx_b,indx_a] = 1


def kinetics_skeleton(file_name='adj_kinetics.npy' , folder='/', njoints=18):

    A = np.zeros((njoints,njoints))
    #####Create connection##############
    #neck
    add_connection(A,0,1);  add_connection(A,1,2); add_connection(A,1,5);  add_connection(A,1,8); add_connection(A,1,11)
    #head
    add_connection(A,0,15); add_connection(A,0,14); add_connection(A,14,16); add_connection(A,15,17)
    #arm
    add_connection(A,2,3); add_connection(A,3,4); add_connection(A,5,6); add_connection(A,6,7)
    #legs
    add_connection(A,8,9); add_connection(A,9,10); add_connection(A,11,12); add_connection(A,12,13)

    # Adjacency matrix and max class saving...
    adj_file = folder+file_name
    with open(adj_file,'wb') as f:
        np.save(f,A)

    max_class = 400 #Number of classes in the dataset
    max_class_file = folder+'max_class_kinetics.npy'
    with open(max_class_file,'wb') as f:
        np.save(f,max_class)

    return A, max_class



def get_labels(args_cross_, kinetics=False):

    # This is when we choose to use the Kinetics 400 dataset
    if kinetics:
        folder = '/dcs/large/u2034358/data/Kinetics/kinetics-skeleton/'
        train_file = [folder+'train_data.npy', folder+'train_label.pkl']
        test_file = [folder+'val_data.npy', folder+'val_label.pkl']
        adj_file = 'adj_kinetics.npy'
        max_class_file = 'max_class_kinetics.npy'
        duplicated_file = folder+'duplicated.txt' #Saves the truth value if the skeletons have been duplicated for single human action

        if adj_file not in os.listdir(folder):
            A, max_class = kinetics_skeleton(adj_file , folder)
        else:
            with open(folder+adj_file,'rb') as f:
                A = np.load(f,allow_pickle=True)
            with open(folder+max_class_file,'rb') as f:
                max_class = np.load(f)*1

        train_data, train_label, test_data, test_label = load_data_from_file(train_file, test_file,
                                                            kinetics=kinetics, dup_file=duplicated_file)

        return train_data, train_label, test_data, test_label, A, max_class

    adj_file_name = 'adj_matrix'
    max_class_file_name = 'max_class'
    adj_file = '/dcs/large/u2034358/'+adj_file_name+'.npy'
    max_class_file = '/dcs/large/u2034358/'+max_class_file_name+'.npy'

    if args_cross_ == 'view':
        cross_view_train_name = 'cross_view_train'
        cross_view_test_name = 'cross_view_test'
        cross_view_train_file = '/dcs/large/u2034358/'+cross_view_train_name+'.npy'
        cross_view_test_file = '/dcs/large/u2034358/'+cross_view_test_name+'.npy'

        train_data, test_data, A, num_class = load_data_from_file(cross_view_train_file, cross_view_test_file, adj_file, max_class_file)
    else:
        cross_sub_train_name = 'cross_sub_train'
        cross_sub_test_name = 'cross_sub_test'
        cross_sub_train_file = '/dcs/large/u2034358/'+cross_sub_train_name+'.npy'
        cross_sub_test_file = '/dcs/large/u2034358/'+cross_sub_test_name+'.npy'

        train_data, test_data,A, num_class = load_data_from_file(cross_sub_train_file, cross_sub_test_file, adj_file, max_class_file)


    return train_data, test_data,A, num_class




def arg_parse(notebook=False):
    parser = argparse.ArgumentParser(description='Improved Model Arguments.')

    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=16,
                        help='hidden dimension of the model')
    parser.add_argument('--checkpoint_PE', type=str, default="checkpoint_pos_encode.pt",
                        help='file name of the saved model (default: checkpoint_pos_encode.pt)')
    parser.add_argument('--many-gpu', dest='many_gpu', action='store_true', default=False,
                        help='decides whether to use many GPUs (default: False)')
    parser.add_argument('--checkpoint', type=str, default="checkpoint_improved_model.pt",
                        help='file name of the saved model (default: checkpoint_improved_model.pt)')
    parser.add_argument('--bs', dest='batch_size', type=int, default=200,
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='where our count starts from for epoch (default: 0)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--act-ES', dest='activate_early_stopping', type=int, default=120,
                        help='epoch to activate early stopping (default: 120)')
    parser.add_argument('--py_v', type=str, default="py3",
                        help='whether python 2 or 3 (default: py3)')
    parser.add_argument('--cross_', type=str, default="view",
                        help='whether cross_view or cross_sub (default: view')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed for splitting the dataset into train and test (default: 0)')
    parser.add_argument('--topk', type=int, default=2,
                        help='the second accuracy we need to report (default: 2)')
    parser.add_argument('-i','--use-cpu', dest='use_cpu', action='store_true', default=False,
                        help='overrides GPU and use CPU unstead (default: False)')
    parser.add_argument('--use-kinetics', dest='use_kinetics', action='store_true', default=False,
                        help='when we choose to switch to the Kinetics 400 dataset (default: False)')
    parser.add_argument('-u','--use-saved-model', dest='use_saved_model', action='store_true', default=False,
                        help='use existing trained model (default: False)')
    parser.add_argument('--patience', type=int, default=10,
                        help='To know when to end the training loop when a level of accuracy is reached (default: 10)')
    parser.add_argument('-g','--gather-data', dest='gather_data', action='store_true', default=False,
                        help='Decide whether to save the training & testing accuracy (default: False)')
    parser.add_argument('--train-file', dest='train_file_name', type=str, default="train_with_PE.npy",
                        help='Name of file to save training data accuracy')
    parser.add_argument('--test-file', dest='test_file_name', type=str, default="test_with_PE.npy",
                        help='Name of file to save evaluation data accuracy')
    parser.add_argument('-z', '--repeat-skeleton', dest='repeat_skeleton', action='store_false', default=True,
                        help='decide if we use skeleton0 for skeleton1 or zero padding (default: True)')
    parser.add_argument('--no-PE', dest='use_PE', action='store_false', default=True,
                        help='decides whether we use the learned position encoding or not.')
    parser.add_argument('--just-project', dest='just_project', action='store_true', default=False,
                        help='decides whether we add position encoding after projection.')
    parser.add_argument('--zeros', dest='repeat', action='store_false', default=True,
                        help='decides if we repeat the skeletons in person 1 for person 2.')
    parser.add_argument('--no-interact', dest='use_intr', action='store_false', default=True,
                        help='decides if we should use the interact module.')
    parser.add_argument('--cnt-flp', dest='count_flop', action='store_true', default=False,
                        help='activate flop counting.')
    parser.add_argument('-t','--tensorboard_name', default='multisteam',
                        help='name for our tensorboardX sub-folder')

    # infogcn data loading
    parser.add_argument('--info_data', action='store_true', default=False,
                        help='decides if we use the exact formatted dataset for the infogcn model.')
    parser.add_argument('--random_rot', action='store_true', default=True, help='')
    parser.add_argument('--balanced_sampling', action='store_true', default=False, help='')
    parser.add_argument('--num_worker', type=int, default=8, help='the number of worker for data loader')
    parser.add_argument('--num_class', type=int, default=60, help='number of actions in the dataset')
    parser.add_argument('--datacase', default='CV', help='data loader will be used')

    # Best model accuracy
    parser.add_argument('--best_acc_filename', type=str, default='best_accuracies', help='data loader will be used')
    parser.add_argument('--avg_best_acc', action='store_true', default=False,
                        help='decides if we compute the average of the best accuracies for the current dataset.')

    if notebook:
        return parser.parse_known_args()

    return parser.parse_args()
