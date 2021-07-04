import time
from models import get_model
from datasets import download_data
from server import server
from clients import client
from plot import plot_loss,plot_acc
import copy
# import torch.multiprocessing as mp
# import multiprocessing
from multiprocessing import Process          # 导入多进程中的进程池
import sys
import torch
import argparse
import os


def main(dataset,get_data_way,model,batch_size,learning_rate,num_glob_iters,
         local_epochs,optimizer,global_nums,gpu):

    print('Initialize Dataset...')
    data_loader = download_data(dataset_name=dataset, batch_size=batch_size)
    m = get_model(dataset=dataset,model=model)
    # mp.set_start_method('spawn')

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    
    clients = []
    cloud = server(model=m, device=device,dataset=data_loader.get_data(get_data_way="IID"),client=global_nums)


    for i in range(global_nums):
        mid_user = client(id=i,model=m,device=device,dataset=data_loader.get_data(get_data_way=get_data_way),
                          learning_rate=learning_rate,optimizer=optimizer,local_epochs=local_epochs)
        clients.append(copy.deepcopy(mid_user))


    losses = []                         # 每个用户的平均损失
    accuracyes = [[],[],[]]                
    # 第一个元素为每个用户的平均精度，第二个为全局模型本地数据，第三个全局模型全部数据
    for epoch in range(num_glob_iters):

        losses.append(0)
        accuracyes[0].append(0)
        accuracyes[1].append(0)
        accuracyes[2].append(0)

        for j in range(global_nums):
            clients[j].local_train()
            loss,acc = clients[j].test_model()
            print("num:",j,acc)
            losses[epoch] += loss
            accuracyes[0][epoch] += acc

        losses[epoch] = losses[epoch]/global_nums
        accuracyes[0][epoch] = accuracyes[0][epoch]/global_nums

        accuracyes[2][epoch] = cloud.aggregate_model()          # 全局模型，全部数据
        for j in range(global_nums):
            clients[j].get_global_modal()
            _,acc = clients[j].test_model()
            accuracyes[1][epoch] += acc                         # 全局模型，本地数据
        accuracyes[1][epoch] = accuracyes[1][epoch]/global_nums
        print("第{}轮次：{:.3f} , {:.4f} , {:.4f} ，{:.4f} \n".format(epoch,losses[epoch],
                                                accuracyes[0][epoch],accuracyes[1][epoch],accuracyes[2][epoch]))
    plot_acc(accuracyes,get_data_way)

    return losses,accuracyes




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["MNIST", "CIFAR10", "EMNIST"])
    parser.add_argument("--get_data", type=str, default="IID", choices=["IID","nonIID","practical_nonIID"])
    parser.add_argument("--model", type=str, default="CNN", choices=["CNN", "DNN", "MCLR_Logistic"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.002, help="Local learning rate")
    parser.add_argument("--num_global_iters", type=int, default=400,help="global train epoch")
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--global_nums", type=int, default=30, help="Number of all Users")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run,-1 mean CPU, 0,1,2 for GPU")
    args = parser.parse_args()

    print("=" * 80)  # 输出80个'='
    print("Summary of training process:")
    print("Dataset: {}".format(args.dataset))        # default="MNIST", choices=["MNIST", "CIFAR10", "EMNIST"]
    print("Get data way: {}".format(args.get_data))  # default="IID", choices=["IID","nonIID","practical_nonIID"]
    print("Local Model: {}".format(args.model))      # default="CNN", choices=["CNN", "DNN", "MCLR_Logistic"]
    print("Batch size: {}".format(args.batch_size))  # default=20
    print("Learing rate: {}".format(args.learning_rate))  # default=0.01, help="Local learning rate"
    print("Number of global rounds: {}".format(args.num_global_iters))    # default=800
    print("Number of local rounds: {}".format(args.local_epochs))         # default=30
    print("Optimizer: {}".format(args.optimizer))                         # default="SGD"
    print("All users: {}".format(args.global_nums))     # default=100, help="Number of all Users"
    print("=" * 80)


    accuracy = []
    losses = []
    # pool = Pool(3)
    result = []
    get_data_way = ["IID","nonIID","practical_nonIID"]

    for j in range(3):
        r = Process(target=main, args=(args.dataset,get_data_way[j],args.model,args.batch_size,
                                        args.learning_rate,args.num_global_iters,args.local_epochs,
                                        args.optimizer,args.global_nums,j,))
        r.start()
        result.append(r)
    for j in result:
        j.join()

    for get_data_way in ["IID"]:
        loss,acc =main(dataset=args.dataset,
                        get_data_way=get_data_way,
                        model=args.model,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        num_glob_iters=args.num_global_iters,
                        local_epochs=args.local_epochs,
                        optimizer=args.optimizer,
                        global_nums=args.global_nums,
                        gpu=args.gpu)
        losses.append(loss)
        accuracy.append(acc)

        # 删除训练的文件
        for i in range(args.global_nums):
            os.remove('./cache/model_state_{}.pkl'.format(i))
        os.remove('./cache/global_model_state.pkl')
        
    sys.exit()
    # 需要获得三个数据，本地数据集的精度，模型聚合后本地数据集的精度，聚合模型的全部数据的精度
    import pandas as pd
    plot_acc(accuracy)
    plot_loss(losses)
    list1 = [accuracy[0][0],accuracy[0][1],accuracy[1][0],accuracy[1][1],accuracy[2][0],accuracy[2][1]]
    a = [[0 for i in range(len(list1))] for j in range(len(list1[0]))]
    for x in range(len(list1)):
        for y in range(len(list1[0])):
            a[y][x] = list1[x][y]

    list2 = [losses[0],losses[1],losses[2]]
    b = [[0 for i in range(len(list2))] for j in range(len(list2[0]))]
    for x in range(len(list2)):
        for y in range(len(list2[0])):
            b[y][x] = list2[x][y]

    columns = ["IID","NonIID","pNonIID"]
    dt = pd.DataFrame(b, columns=columns)
    dt.to_excel("./data2/loss_xlsx.xlsx", index=0)
    dt.to_csv("./data2/loss_csv.csv", index=0)

    columns = ["IID_(PM)","IID_(GM)","NonIID_(PM)","NonIID_(GM)","pNonIID_(PM)","pNonIID_(GM)"]
    dt = pd.DataFrame(a, columns=columns)
    dt.to_excel("./data2/acc_xlsx.xlsx", index=0)
    dt.to_csv("./data2/acc_csv.csv", index=0)


