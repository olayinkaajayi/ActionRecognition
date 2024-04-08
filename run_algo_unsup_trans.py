import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import sys
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("pyG_InfoGraph")
sys.path.append("InfoGraph")
from infograph import InfoGraph
# from vid_graph_data import NTURGB_view,NTURGB_sub

sys.path.append("DHCS_implement/models")
sys.path.append("DHCS_implement/models/Positional_encoding")############################
sys.path.append("DHCS_implement/models/Transformer_parts")
# from spatiotemporal_act_recog_wt_democracy import Spatiotemp_Action_recog
from spatiotemporal_act_recog import Spatiotemp_Action_recog
from training_utils_py2 import *
from pytorchtools import EarlyStopping #use the one in InfoGraph
from Transformer_plus.arg_n_utils import arg_parse, embeddings, get_labels
from democracy_loss_for_loop import XEnt_n_Democracy #note that there is another version without for loop
# from democracy_loss import XEnt_n_Democracy



def main():

    print()
    args = arg_parse()
    writer = None# SummaryWriter('runs/body_parts_joints')
    train_data, test_data, A, num_class = get_labels(args.cross_)
    print("number of classes: ",num_class)
    print("Train data size:",len(train_data))
    print("Test data size:",len(test_data))

    num_nodes = train_data[0]['njoints'] #This is 25 for the data we are using
    in_dim = train_data[0]['skel_body0'].shape[-1] #The size of the last dimension which should be 3
    output_dim = num_class

    # Get labels
    labels_train = [int(ele['class']) for ele in train_data]
    labels_test = [int(ele['class']) for ele in test_data]

    # file_name = 'file_name'
    # data = train_data.tolist() + test_data.tolist()
    # samples = []
    # for ele in data:
    #     name = ele[file_name]
    #     for ele2 in data:
    #         name2 = ele2[file_name]
    #         if (name[:4]+name[8:]) == (name2[:4]+name2[8:]):
    #             samples.append(ele2)
    #     break
    #
    # rotation_matrix = lambda n,N: np.array([[np.cos(2*np.pi*n/N), 0, np.sin(2*np.pi*n/N)],
    #                                     [0, 1, 0],
    #                                     [-np.sin(2*np.pi*n/N), 0, np.cos(2*np.pi*n/N)]
    #                                     ])
    #
    # def rotate(arr,n=-1,N=12):
    #
    #     arr_reshape = arr.reshape(-1,arr.shape[-1]) # time*nodes x feat_dim
    #     multiply = np.matmul(rotation_matrix(n,N), arr_reshape.transpose()).transpose()
    #     # multiply : [time*nodes x feat_dim]
    #
    #     multiply = multiply.reshape(*arr.shape) # [time x nodes x feat_dim]
    #     return multiply
    #
    # samples_pad = pad_data(samples,max_time=300)
    # results = [0]*len(samples)
    # idx = [0,0,0]
    # for i,ele in enumerate(samples):
    #     name = ele[file_name]
    #
    #     if name[7] == '1': #45 degree
    #         results[0] = rotate(samples_pad[i]) #rotate by -45 degrees
    #         idx[0]=i
    #     elif name[7] == '2': #0 degree
    #         results[1] = samples_pad[i]
    #         idx[1]=i
    #     else: #name[7] == '3': -45 degree
    #         results[2] = rotate(samples_pad[i],n=1) #rotate by -45 degrees
    #         idx[2]=i
    #
    # for i,each in enumerate(results):
    #     print("No rotation")
    #     print(f"error is {np.linalg.norm(samples_pad[idx[i]].flatten()-results[1].flatten())}")
    #     print("With rotation")
    #     print(f"error is {np.linalg.norm(each.flatten()-results[1].flatten())}")
    #     print()
    #
    #
    #
    # exit()

    # Get maximum time length
    # max_time_train = max([ele['skel_body0'].shape[0] for ele in train_data])
    # max_time_test = max([ele['skel_body0'].shape[0] for ele in test_data])
    # max_time = max(max_time_test,max_time_train)

    # Useful for analyzing the data
    # time_train = [ele['class'] for ele in train_data if ele['skel_body0'].shape[0]> 120]
    # time_test = [ele['class'] for ele in test_data if ele['skel_body0'].shape[0]> 120]
    # new_d = np.array(time_test+time_train)
    # here = np.where(new_d >= 200, 1, 0).sum()
    # print("Value is", here)
    # plt.hist(time_test+time_train, bins=239)
    # plt.title("Distribution of class for time length > 120")
    # plt.savefig("Dist_time_len_for_class_gt120.jpg")
    # exit()

    max_time = 300 #This is the chosen maximum time length
    #Pads the data to be of equal timeLength
    train_data = pad_data(train_data,max_time)#, screen=False, norm=True)
    test_data = pad_data(test_data,max_time)#, screen=False, norm=True)
    # print(f"train: {np.shape(train_data)}")
    # print(f"test: {np.shape(test_data)}")
    # exit()


    mlp_numbers = max_time, output_dim #used to populate the mlp arguments
    # Shuffle training and validation dataset
    train_graph, train_label = shuffle(train_data, labels_train, random_state=args.seed)
    test_graph, test_label = shuffle(test_data, labels_test, random_state=args.seed)


    #Initialize seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ########Load data################
    need_nodes = True # needed for parts-gnn and not pose-gnn
    begin_path = os.getcwd() #args.begin_path
    many_gpu = args.many_gpu #Decide if we use multiple GPUs or not
    device,USE_CUDA = use_cuda(args.use_cpu,many=many_gpu)
    num_nodes = 25 #try train_data.num_nodes
    batch_size = args.batch_size
    output_dim = num_class
    # max_time = 300 #max time length in data
    mlp_numbers = max_time, output_dim
    print()


    #load gnn model (to copy parameters)
    if args.use_DGI:
        gnn_model = InfoGraph(in_dim, args.hidden_dim, args.num_gc_layers)
        gnn_model.load_state_dict(torch.load(begin_path+'/InfoGraph/Saved_models/'+args.checkpoint_gnn,map_location='cpu'))
        # load spatiotemporal model
        model = Spatiotemp_Action_recog(gnn_model.dataset_num_features,gnn_model.hidden_dim, mlp_numbers, gnn_model,num_trans_layers=2)
    else:
        model = Spatiotemp_Action_recog(in_dim, args.hidden_dim, mlp_numbers, num_trans_layers=2)
    # writer.add_graph(model,(torch.rand(1,300,25,3),torch.Tensor(A)))
    # writer.close()
    # exit()
    # with SummaryWriter(comment='GNN_trans') as w:
    #     w.add_graph(model,(torch.rand(1,300,25,3),), True)
    # exit()

    if args.use_saved_model:
        # To load model
        model.load_state_dict(torch.load(begin_path+'/DHCS_implement/Saved_models/'+args.checkpoint,map_location=device))
        print("USING SAVED MODEL!")

    # Shuffle training and validation dataset
    train_graph, train_label = train_data, labels_train# shuffle(train_data, labels_train, random_state=args.seed)
    test_graph, test_label = test_data, labels_test# shuffle(test_data, labels_test, random_state=args.seed)

    if USE_CUDA: #To set it up for parallel usage of both GPUs (sppeds up training)
        torch.cuda.manual_seed_all(args.seed)
        model = torch.nn.DataParallel(model) if many_gpu else model #use all free GPUs if needed
    model.to(device)

    if not args.use_DGI:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = XEnt_n_Democracy(k1=1.0)
        if args.use_saved_model:
            criterion.load_state_dict(torch.load(begin_path+'/DHCS_implement/Saved_models/loss_fn_params.pt',map_location=device))
        criterion.to(device)

    params = list(model.parameters()) + list(criterion.parameters())
    Num_Param = sum(p.numel() for p in params if p.requires_grad)
    print("Number of Trainable Parameters is about %d" % (Num_Param))

    optimizer = optim.Adam(params, lr= args.lr)#, weight_decay=1e-3) #weight_decay=1e-3 keeps train_acc= 83.+ and test_acc= 75.7+
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)

    # early_stopping = EarlyStopping(begin_path, args.checkpoint, patience=args.patience, verbose=True)
    # A has already been passed to the model
    # A = None #No adjacency matrix needed

    print("Pre-training Accuracy:")
    test(model, test_graph, test_label, A, device, batch_size, criterion, topk=(1,2), unsupervised= args.use_DGI, writer=writer)
    for epoch in range(0,args.epochs):
        # Decay Learning Rate
        # scheduler.step()
        # Print Learning Rate
        # if (epoch>198) and (epoch < 202):
            # print('Epoch:', epoch+1,'LR:', scheduler.get_lr())

        numbers = batch_size,epoch #This is so we can "reduce" the appearance of the parameters passed

        try:
            ave_loss_train,accuracy_train = train(train_graph,train_label,A,model,optimizer,criterion,device,numbers,topk=(1,2),unsupervised=args.use_DGI,writer=writer)

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
                    test(model, test_graph, test_label, A, device, (batch_size,epoch), criterion, topk=(1,2), unsupervised= args.use_DGI, writer=writer)
                    if USE_CUDA and many_gpu:
                        torch.save(model.module.state_dict(), begin_path+'/DHCS_implement/Saved_models/'+args.checkpoint)
                        # torch.save(criterion.state_dict(), begin_path+'/DHCS_implement/Saved_models/loss_fn_params.pt')
                    else:
                        torch.save(model.state_dict(), begin_path+'/DHCS_implement/Saved_models/'+args.checkpoint)
        except KeyboardInterrupt as e:
            if writer is not None:
                writer.close()
            print(e)
            raise
    if writer is not None:
        writer.close() #close writer
    # test(model, test_graph, test_label, A, device, batch_size, criterion, unsupervised=(not args.use_DGI))

if __name__ == '__main__':
    main()
    print("Done!!!")
