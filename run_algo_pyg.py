import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
sys.path.append("DHCS_implement/models")
sys.path.append("DHCS_implement/models/Transformer_pyg")
from spatiotemporal_act_recog import Spatiotemp_Action_recog
from training_utils import NTURGB_view,NTURGB_sub,train_pyg,test_pyg,use_cuda,num_of_param
from pytorchtools import EarlyStopping
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch LCN plus LSTM skeletal human action recognition model')
    parser.add_argument('--begin_path', type=str, default="/home/olayinka/codes",
                        help='parent path that leads to DHCS_implement (default: "/home/olayinka/codes")')
    parser.add_argument('--checkpoint', type=str, default="checkpoint_gat_pyg.pt",
                        help='file name of the saved model (default: checkpoint_gcn.pt)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--py_v', type=str, default="py2",
                        help='whether python 2 or 3 (default: py2')
    parser.add_argument('--cross_', type=str, default="view",
                        help='whether cross_view or cross_sub (default: view')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into train and test (default: 0)')
    parser.add_argument('--use_cpu', type=bool, default=False,
                        help='overrides GPU and use CPU unstead (default: False)')
    parser.add_argument('--use_saved_model', type=bool, default=False,
                        help='use existing trained model (default: False)')
    parser.add_argument('--patience', type=int, default=20,
                        help='To know when to end the training loop when a level of accuracy is reached (default: 20)')
    parser.add_argument('--out_dim', type=int, default=8,
                        help='dimension of the output of the (first) GCN layer (default: 0)')
    args = parser.parse_args()

    #Initialize seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ########Load data################
    begin_path = args.begin_path
    print("Loading Data...")
    if args.cross_ == 'view':
        train_data = NTURGB_view(root='/dcs/large/u2034358',idx=0)
        test_data = NTURGB_view(root='/dcs/large/u2034358',idx=1)
    else:
        train_data = NTURGB_sub(root='/dcs/large/u2034358',idx=0)
        test_data = NTURGB_sub(root='/dcs/large/u2034358',idx=1)

    # number of classes
    max_class_file = begin_path+'/DHCS_implement/max_class.npy'
    with open(max_class_file,'rb') as f:
        num_class = np.load(f)*1
    print("number of classes: ",num_class)

    many_gpu = True #Decide if we use multiple GPUs or not
    device,USE_CUDA = use_cuda(args.use_cpu,many=many_gpu)
    print("Training data size:",len(train_data))
    print("Test data size:",len(test_data))

    num_nodes = 25 #This is 25 for the data we are using
    in_dim = 3 #The size of the last dimension which should be 3
    batch_size = args.batch_size
    output_dim = num_class

    max_time = 300 #max time length in data
    mlp_numbers = max_time, output_dim

    # Data loaders
    train_dataloader = DataListLoader(train_data,batch_size=batch_size)
    test_dataloader = DataListLoader(test_data,batch_size=batch_size)

    model = Spatiotemp_Action_recog(in_dim,mlp_numbers)

    if args.use_saved_model:
        # To load model
        model.load_state_dict(torch.load(begin_path+'/DHCS_implement/Saved_models/'+args.checkpoint,map_location=device))
        print("USING SAVED MODEL!")

    if USE_CUDA: #To set it up for parallel usage of both GPUs (sppeds up training)
        torch.cuda.manual_seed_all(args.seed)
        model = DataParallel(model) if many_gpu else model #use all free GPUs if needed
        model = model.cuda()
    else:
        model.to(device)

    criterion = nn.CrossEntropyLoss() #loss function
    params = list(model.parameters())
    #count number of parameters
    try:
        Num_Param = sum(p.numel() for p in params if p.requires_grad)
    except ValueError:
        Num_Param = num_of_param(params)
    print("Number of Trainable Parameters is about %d" % (Num_Param))

    optimizer = optim.Adam(params, lr= args.lr)
    # optimizer = optim.SGD(params, lr= args.lr, momentum=0.2)

    early_stopping = EarlyStopping(begin_path, args.checkpoint, patience=args.patience, verbose=True)

    for epoch in range(args.epochs):

        numbers = batch_size,epoch #This is so we can "reduce" the appearance of the parameters passed

        ave_loss_train,accuracy_train = train_pyg(train_dataloader,model,optimizer,criterion,device,numbers)

        print("%d : Average training loss: %f, Training Accuracy: %f" %(epoch+1,ave_loss_train, accuracy_train))

        # Check early stopping
        # if (epoch+1) > 180:
        #     with torch.no_grad():
        #         val_loss = validation_fn(model, test_graph, test_label, A, device, criterion, batch_size)
        #
        #     early_stopping(val_loss, model)
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break

        if (epoch==0) or (((epoch+1)%5) == 0):
            with torch.no_grad():
                test_pyg(model, test_dataloader, device)
                if USE_CUDA and many_gpu:
                    torch.save(model.module.state_dict(), begin_path+'/DHCS_implement/Saved_models/'+args.checkpoint)
                else:
                    torch.save(model.state_dict(), begin_path+'/DHCS_implement/Saved_models/'+args.checkpoint)
    test_pyg(model, test_dataloader, device)
if __name__ == '__main__':
    main()
    print("Done!!!")
