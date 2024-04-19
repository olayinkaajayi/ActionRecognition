
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import sys
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os

sys.path.append("node2vec/src")
sys.path.append("ActionRecognition/models")
# sys.path.append("ActionRecognition/models/Positional_encoding")
sys.path.append("ActionRecognition/models/Learn_adjacency")
sys.path.append("ActionRecognition/models/Improved_model")

# from joints_gnn_trans_new import Joints_GNN_Trans
from pytorchtools import EarlyStopping
from st_gcn import Model
# from MMD_loss.combined_loss import Xent_n_SparseMMD # For Loss function
from training_utils_py2 import *
from arg_n_utils import arg_parse, get_labels


def savefile(data,filename):
    print("Saving file...")
    with open('/dcs/large/u2034358/'+filename,'wb') as f:
        np.save(f,np.array(data))



def main(args):

    writer = SummaryWriter(os.path.join('runs',args.tensorboard_name,args.checkpoint[:-3]))

    if args.use_kinetics:
        print("\nLoading Kinetics400 Dataset...")
        train_data, labels_train, test_data, labels_test, A, num_class = get_labels(args.cross_,kinetics=args.use_kinetics)
        _, max_time, num_nodes, in_dim = train_data.shape
        num_nodes = num_nodes//2
        d = 7 # Dimension of binary position encoding

        train_graph, train_label = train_data, labels_train
        test_graph, test_label = test_data, labels_test
    else:
        print("\nLoading NTU RGB+D Dataset...")
        train_data, test_data, A, num_class = get_labels(args.cross_)
        
    ####################################
    num_class = args.num_class
    num_nodes = 25
    d = 8
    in_dim = 3
    A = None

    # For new preprocessed dataset
    if not args.info_data:
        max_time = 300
        train_graph, train_label = load_data_align('train', xV_xS='NTU60_CV')
        test_graph, test_label = load_data_align('test', xV_xS='NTU60_CV')

    # For new preprocessed dataset that is downsampled
    max_time = 64
    the_dataset = f'NTU{args.num_class}_{args.datacase}'
    assert the_dataset in ['NTU60_CV','NTU60_CS','NTU120_CSet','NTU120_CSub'], 'num_class is 60 or 120 and datacase should be one of CV, CS, CSet, CSub'

    train_graph, test_graph = load_data_infogcn(args, window_size=max_time, xV_xS=the_dataset, num_class=num_class, repeat=args.repeat)
    train_label, test_label = None, None
    ####################################

    print("number of classes: ",num_class)
    print("Train data size:",len(train_graph.dataset))
    print("Test data size:",len(test_graph.dataset))

    del train_data, test_data #, labels_train, labels_test

    output_dim = num_class

    mlp_numbers = max_time, output_dim #used to populate the mlp arguments


    #Initialize seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ########Load data################
    begin_path = os.getcwd()
    many_gpu = args.many_gpu #Decide if we use multiple GPUs or not
    device,USE_CUDA = use_cuda(args.use_cpu,many=many_gpu)
    batch_size = args.batch_size
    output_dim = num_class
    print()

    # Load model
    graph_args = {"layout": "ntu-rgb+d", "strategy": "spatial"}
    kwargs = {"dropout": 0.5}
    model = Model(in_channels=in_dim, num_class=output_dim, graph_args=graph_args,edge_importance_weighting=False,
                d=d, PE_name=args.checkpoint_PE, use_PE=args.use_PE, just_project=args.just_project, **kwargs)

    
    if args.count_flop:
        flops_fwd = calculate_flops(model, A, T=64, M=2, V=25, C=3)
        print(f"FLOPs={flops_fwd}")
        exit()


    if args.use_saved_model:
        # To load model
        model.load_state_dict(torch.load(begin_path+'/ActionRecognition/Saved_models/'+args.checkpoint,map_location=device))
        print("USING SAVED MODEL!")


    if USE_CUDA: #To set it up for parallel usage of both GPUs (sppeds up training)
        torch.cuda.manual_seed_all(args.seed)
        model = torch.nn.DataParallel(model) if many_gpu else model #use all free GPUs if needed
    model.to(device)


    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
    # criterion = Xent_n_SparseMMD(num_class=num_class, num_corr_clas_needed=15, ls=0.2)

    params = list(model.parameters())
    Num_Param = sum(p.numel() for p in params if p.requires_grad)
    print("Number of Trainable Parameters is about %d" % (Num_Param))

    optimizer = optim.Adam(params, lr= args.lr)#, weight_decay=1e-3) #weight_decay=1e-3 keeps train_acc= 83.+ and test_acc= 75.7+
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=0.1,
                                                             patience=10,
                                                             verbose=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75,150], gamma=0.1)

    early_stopping = EarlyStopping(args.checkpoint, patience=args.patience, #delta=0.00001,
                                verbose=True, use_cuda=USE_CUDA, many_gpu=many_gpu,
                                start_countdown=args.activate_early_stopping)

    print("Pre-training Accuracy:")
    val_loss, val_acc = test(model, test_graph, test_label, A, device,
                            (batch_size,0), criterion, topk=(1,args.topk), return_accuracy=True,unsupervised=False)
    early_stopping(val_loss, val_acc, model)

    if args.gather_data:
        train_acc = []
        test_acc = []

    for epoch in range(args.epoch_start,args.epochs):

        # Decay Learning Rate
        # scheduler.step()
        # Print Learning Rate
        # if epoch in [74,75,76,149,150,151]:
            # print('Epoch:', epoch+1,'LR:', scheduler.get_lr())

        numbers = batch_size,epoch #This is so we can "reduce" the appearance of the parameters passed

        try:
            ave_loss_train,accuracy_train = train(train_graph,train_label,A,model,optimizer,
                                                criterion,device,numbers,topk=(1,args.topk),
                                                writer=writer,unsupervised=False, count_flop=args.count_flop)

            if args.gather_data:
                train_acc.append(accuracy_train)
            print("%d : Average training loss: %f" %(epoch+1,ave_loss_train))


            # Check early stopping
            # if (epoch+1) >= args.activate_early_stopping:
            #     with torch.no_grad():
            #         val_loss, val_acc = test(model, test_graph, test_label, A, device,
            #                                 (batch_size,epoch), criterion, topk=(1,),
            #                                 get_print=False, return_accuracy=True)
            #
            #     early_stopping(val_loss, val_acc, model, epoch)
            #     if early_stopping.early_stop:
            #         print("Early stopping")
            #         break
            #
            #     continue # We don't need to run the other lines left


            # if (epoch==0) or (((epoch+1)%5) == 0):

            with torch.no_grad():

                val_loss, val_acc = test(model, test_graph, test_label, A, device, (batch_size,epoch), criterion,
                            topk=(1,args.topk), writer=writer, return_accuracy=True,unsupervised=False)
                if args.gather_data:
                    test_acc.append(val_acc)
                early_stopping(val_loss, val_acc, model)

            scheduler.step(val_loss)
            min_lr=0.0001
            if optimizer.param_groups[0]['lr'] < min_lr:
                print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                break


                    # if USE_CUDA and many_gpu:
                    #     torch.save(model.module.state_dict(), begin_path+'/ActionRecognition/Saved_models/'+args.checkpoint)
                    # else:
                    #     torch.save(model.state_dict(), begin_path+'/ActionRecognition/Saved_models/'+args.checkpoint)

        except KeyboardInterrupt as e:
            if writer is not None:
                writer.close()
            print(e)
            raise

    if writer is not None:
        writer.close() #close writer

    del model
    print("\nGetting result from SAVED MODEL:")
    model = Model(in_channels=in_dim, num_class=output_dim, graph_args=graph_args,edge_importance_weighting=False,
                d=d, PE_name=args.checkpoint_PE, use_PE=args.use_PE, just_project=args.just_project, **kwargs)

    model.load_state_dict(torch.load(begin_path+'/ActionRecognition/Saved_models/'+args.checkpoint,map_location=device))
    if USE_CUDA: #To set it up for parallel usage of both GPUs (speeds up training)
        torch.cuda.manual_seed_all(args.seed)
        model = torch.nn.DataParallel(model) if many_gpu else model #use all free GPUs if needed
    model.to(device)

    test(model, test_graph, test_label, A, device, batch_size, criterion, topk=(1,args.topk),unsupervised=False)

    PE_ext = '' if args.use_PE else '_no_PE'
    repeat_ext = '' if args.repeat else '_zeros'
    intr_ext = '' if args.use_intr else '_no_interact'
    save_float_to_csv(early_stopping.best_accuracy,
                    filename=f'{args.best_acc_filename}_{the_dataset}{PE_ext}{repeat_ext}{intr_ext}.csv')

    if args.gather_data:
        savefile(train_acc,args.train_file_name)
        savefile(test_acc,args.test_file_name)


if __name__ == '__main__':
    print()
    args = arg_parse()

    if args.avg_best_acc:
        PE_ext = '' if args.use_PE else '_no_PE'
        repeat_ext = '' if args.repeat else '_zeros'
        intr_ext = '' if args.use_intr else '_no_interact'
        the_dataset = f'NTU{args.num_class}_{args.datacase}'
        scores = read_csv(filename=f'{args.best_acc_filename}_{the_dataset}{PE_ext}{repeat_ext}{intr_ext}.csv')
        avg = np.mean(scores)
        std = np.std(scores)
        print(f"Average Best Score for {the_dataset} is {avg:.1f}+/-{std:.2f}")

        with open(f'{args.best_acc_filename}_{the_dataset}{PE_ext}{repeat_ext}{intr_ext}.txt','w+') as f:
            f.write(f"Statistics for Dataset: {the_dataset}\n")
            f.write(f"\nAverage={avg:.1f}\n")
            f.write(f"\nStandard deviation={std:.2f}")
            f.write("\n")

        print("Saved average accuracy...")
    else:
        main(args)
    print("Done!!!")

