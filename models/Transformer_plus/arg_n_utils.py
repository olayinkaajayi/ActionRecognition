import argparse
import torch
from tqdm import tqdm
import numpy as np
from training_utils import load_data_from_file

def embeddings(model, loader, device, nodes=False):
    """
        This function returns the loss of the validation/test data.
    """
    pbar = tqdm(loader, unit='batch')
    model.eval()
    ret = []
    with torch.no_grad():
        for data in pbar:
            data = data.to(device)
            g_enc = model(data,get_embed=True) if not nodes else model(data,get_embed=True,get_nodes=True)
            ret.append(g_enc.cpu().numpy())

            pbar.set_description('Embedding')
    ret = np.concatenate(ret, 0)
    return ret



def get_labels(args_cross_):

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


    # Get labels
    # labels_train = [int(ele['class']) for ele in train_data]
    # labels_test = [int(ele['class']) for ele in test_data]

    return train_data, test_data,A, num_class




def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')

    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=2,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=16,
                        help='')
    parser.add_argument('--checkpoint_gnn', type=str, default="checkpoint_unsup_DGI_GCN16_pad_batch.pt",
                        help='file name of the saved model (default: checkpoint_unsup_gnn.pt)')
    parser.add_argument('--batch_size_gnn', type=int, default=12000,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--many_gpu', type=bool, default=False,
                        help='decides whether to use many GPUs (default: False)')
    parser.add_argument('--begin_path', type=str, default="/home/olayinka/codes",
                        help='parent path that leads to DHCS_implement (default: "/home/olayinka/codes")')
    parser.add_argument('--checkpoint', type=str, default="checkpoint_gat_pyg.pt",
                        help='file name of the saved model (default: checkpoint_gcn.pt)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--py_v', type=str, default="py3",
                        help='whether python 2 or 3 (default: py3')
    parser.add_argument('--cross_', type=str, default="view",
                        help='whether cross_view or cross_sub (default: view')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed for splitting the dataset into train and test (default: 0)')
    parser.add_argument('--use_cpu', type=bool, default=False,
                        help='overrides GPU and use CPU unstead (default: False)')
    parser.add_argument('--use_saved_model', type=bool, default=False,
                        help='use existing trained model (default: False)')
    parser.add_argument('--patience', type=int, default=20,
                        help='To know when to end the training loop when a level of accuracy is reached (default: 20)')
    parser.add_argument('--use_DGI', type=bool, default=False,
                        help='Decide if we should import the InfoGraph model (default: False)')


    return parser.parse_args()
