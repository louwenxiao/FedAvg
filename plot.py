import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

losses = []           # 每个用户的平均损失
accuracyes = [[],[]]  # 第一个元素为每个用户的平均精度，第二个为全局模型的精度
#"IID","NonIID","pNonIID"

def plot_acc(acc,data):

    dt = pd.DataFrame(acc)

    dt.to_excel("./data2/acc_{}.xlsx".format(data), index=0)
    dt.to_csv("./data2/acc_{}.csv".format(data), index=0)

    plt.title('CIFAR10,30 clients')
    plt.xlabel("epoches")
    plt.ylabel("acc")
    x=np.arange(0,len(acc[0]))
    x[0]=1
    my_x_ticks = np.arange(0, 401, 50)
    plt.xticks(my_x_ticks)
    plt.plot(x,acc[0],label='IID_P')
    plt.plot(x,acc[1],label='IID_GL')
    plt.plot(x,acc[2],label='IID_GA')
    plt.legend()
    plt.savefig('./data2/acc_CIFAR10_{}.jpg'.format(data))
    plt.show()
    plt.clf()

def plot_loss(loss):

    plt.title('MNIST,30 clients,local_epoch=10')
    plt.xlabel("epoches")
    plt.ylabel("loss")
    x=np.arange(0,len(loss[0]))
    x[0]=1
    my_x_ticks = np.arange(0, 110, 20)
    plt.xticks(my_x_ticks)
    plt.plot(x,loss[0],label='IID')
    plt.plot(x,loss[1],label='NonIID')
    plt.plot(x,loss[2],label='pNonIID')
    plt.legend()
    plt.savefig('./data2/loss_1.jpg')
    plt.show()