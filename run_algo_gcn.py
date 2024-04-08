import argparse
import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from sklearn.utils import shuffle
sys.path.append("DHCS_implement/models") #would be needed
sys.path.append("DHCS_implement/models/GCN_Act_recog")
from gcn_act_recog import GCN_Action_recog
from training_utils_py2 import *
from pytorchtools import EarlyStopping

def main():
    parser = argparse.ArgumentParser(
        description='PyTorch LCN plus LSTM skeletal human action recognition model')
    parser.add_argument('--begin_path', type=str, default="/home/olayinka/codes",
                        help='parent path that leads to DHCS_implement (default: /h/ola/Docs/Gith/mthSys)')
    parser.add_argument('--py_v', type=str, default="py2",
                        help='whether python 2 or 3 (default: py2')
    parser.add_argument('--cross_', type=str, default="view",
                        help='whether cross_view or cross_sub (default: view')
    parser.add_argument('--checkpoint', type=str, default="checkpoint_gcn.pt",
                        help='file name of the saved model (default: checkpoint_gcn.pt)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into train and test (default: 0)')
    parser.add_argument('--out_dim', type=int, default=0,
                        help='dimension of the output of the (first) GCN layer (default: 0)')
    parser.add_argument('--num_stacks', type=int, default=3,
                        help='number of GCN layers in the model (default: 3)')
    parser.add_argument('--average', default=True, action="store_false",
                        help='decide whether to use average or learned pooling (default: True)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in MLP (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='size of hidden dimension of the MLP (default: 16)')
    parser.add_argument('--use_cpu', type=bool, default=False,
                        help='overrides GPU and use CPU unstead (default: False)')
    parser.add_argument('--use_saved_model', type=bool, default=False,
                        help='use existing trained model (default: False)')
    parser.add_argument('--patience', type=int, default=20,
                        help='To know when to end the training loop when a level of accuracy is reached (default: 20)')
    parser.add_argument('--need_data',default=False, action="store_true",
                        help='get data to use for experiment (default: False)')
    parser.add_argument('--data_size', type=int, default=20,
                        help='Number of data to extract for experiment (default: 20)')
    args = parser.parse_args()

    #Initialize seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ########Load data################
    begin_path = args.begin_path

    adj_file_name = 'adj_matrix_py2' if args.py_v == 'py2' else 'adj_matrix'
    max_class_file_name = 'max_class_py2' if args.py_v == 'py2' else 'max_class'
    adj_file = begin_path+'/DHCS_implement/'+adj_file_name+'.npy'
    max_class_file = begin_path+'/DHCS_implement/'+max_class_file_name+'.npy'

    if args.cross_ == 'view':
        cross_view_train_name = 'cross_view_train_py2' if args.py_v == 'py2' else 'cross_view_train'
        cross_view_test_name = 'cross_view_test_py2' if args.py_v == 'py2' else 'cross_view_test'
        cross_view_train_file = begin_path+'/DHCS_implement/'+cross_view_train_name+'.npy'
        cross_view_test_file = begin_path+'/DHCS_implement/'+cross_view_test_name+'.npy'

        train_data, test_data, A, num_class = load_data_from_file(cross_view_train_file, cross_view_test_file, adj_file, max_class_file)
    else:
        cross_sub_train_name = 'cross_sub_train_py2' if args.py_v == 'py2' else 'cross_sub_train'
        cross_sub_test_name = 'cross_sub_test_py2' if args.py_v == 'py2' else 'cross_sub_test'
        cross_sub_train_file = begin_path+'/DHCS_implement/'+cross_sub_train_name+'.npy'
        cross_sub_test_file = begin_path+'/DHCS_implement/'+cross_sub_test_name+'.npy'

        train_data, test_data, A, num_class = load_data_from_file(cross_sub_train_file, cross_sub_test_file, adj_file, max_class_file)

    print("number of classes: ",num_class)
    many_gpu = True #Decide if we use multiple GPUs or not
    device,USE_CUDA = use_cuda(args.use_cpu,many=many_gpu)
    print("Train data size:",len(train_data))
    print("Test data size:",len(test_data))

    num_nodes = train_data[0]['njoints'] #This is 25 for the data we are using
    in_dim = train_data[0]['skel_body0'].shape[-1] #The size of the last dimension which should be 3
    out_dim = args.out_dim #when set to 0, it uses out_dim_gcn = in_dim
    num_stacks= args.num_stacks
    average = args.average
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    output_dim = num_class

    # Get labels
    labels_train = [int(ele['class']) for ele in train_data]
    labels_test = [int(ele['class']) for ele in test_data]

    # Get maximum time length
    max_time_train = max([ele['skel_body0'].shape[0] for ele in train_data])
    max_time_test = max([ele['skel_body0'].shape[0] for ele in test_data])
    max_time = max(max_time_test,max_time_train)

    #Pads the data to be of equal timeLength
    train_data = pad_data(train_data,max_time)
    test_data = pad_data(test_data,max_time)

    mlp_numbers = max_time, num_layers , hidden_dim , output_dim #used to populate the mlp arguments
    # Shuffle training and validation dataset
    train_graph, train_label = shuffle(train_data, labels_train, random_state=args.seed)
    test_graph, test_label = shuffle(test_data, labels_test, random_state=args.seed)

    if args.need_data:
        work_data(begin_path,train_graph[:args.data_size])
        print("Data for experiment saved!!!")

    #####NOW I NEED REAL DATA TO TEST ON :DONE!
    #####ALSO MODIFY IT TO DO TESTING AND BATCHES: DONE!
    #####CREATE CLASS FOR D-HCSF LAYER USING LCN (as done in paper): DONE!

    model = GCN_Action_recog(mlp_numbers,num_nodes,in_dim,out_dim,num_stacks,average)

    if args.use_saved_model:
        # To load model
        model.load_state_dict(torch.load(begin_path+'/DHCS_implement/Saved_models/'+args.checkpoint,map_location=device))
        print("USING SAVED MODEL!")
    if USE_CUDA: #To set it up for parallel usage of both GPUs (sppeds up training)
        torch.cuda.manual_seed_all(args.seed)
        model = torch.nn.DataParallel(model) if many_gpu else model #use all free GPUs if needed
    #     model = model.cuda()
    # else:
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())
    try:
        Num_Param = sum(p.numel() for p in params if p.requires_grad)
    except ValueError:
        Num_Param = num_of_param(params)

    print("Number of Trainable Parameters is about %d" % (Num_Param))
    optimizer = optim.Adam(params, lr= args.lr)
    # optimizer = optim.SGD(params, lr= args.lr, momentum=0.2)

    early_stopping = EarlyStopping(begin_path, args.checkpoint, patience=args.patience, verbose=True)
    batch_size = args.batch_size

    for epoch in range(args.epochs):

        numbers = batch_size,epoch #This is so we can "reduce" the appearance of the parameters passed

        ave_loss_train,accuracy_train = train(train_graph,train_label,A,model,optimizer,criterion,device,numbers)

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
                test(model, test_graph, test_label, A, device, batch_size,criterion)
                if USE_CUDA and many_gpu:
                    torch.save(model.module.state_dict(), begin_path+'/DHCS_implement/Saved_models/'+args.checkpoint)
                else:
                    torch.save(model.state_dict(), begin_path+'/DHCS_implement/Saved_models/'+args.checkpoint)
    test(model, test_graph, test_label, A, device, batch_size, criterion)

if __name__ == '__main__':
    main()
    print("Done...")
