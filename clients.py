import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import copy


class client(object):
    def __init__(self,id,model,device,dataset,learning_rate,optimizer,local_epochs):
        self.id = id
        self.model = model.to(device)
        self.device = device
        self.model_init = model
        self.train_data = dataset[0]
        self.test_data = dataset[1]
        self.lr = learning_rate
        self.optimizer = optimizer
        self.epochs = local_epochs

    # 全局模型更新，每个用户载入新的模型
    def get_global_modal(self):
        # 是否需要首先定义model=...，然后在用下面的代码
        model = copy.deepcopy(self.model_init)
        model.load_state_dict(torch.load('./cache/global_model_state.pkl'))
        self.model = copy.deepcopy(model)

    # 本地进行训练
    def local_train(self):
        self.model.train()

        if self.optimizer == "SGD":
            optimizer = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=0.5)
        else:
            optimizer = optim.Adam(params=self.model.parameters(),lr=self.lr)

        for i in range(self.epochs):
            for data, target in self.train_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                optimizer.zero_grad()
                output = self.model(data)

                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()

        torch.save(self.model.state_dict(), './cache/model_state_{}.pkl'.format(self.id))


    # 模型测试
    def test_model(self):
        test_loss = 0
        test_correct = 0
        model = copy.deepcopy(self.model)

        with torch.no_grad():   # torch.no_grad()是一个上下文管理器，用来禁止梯度的计算
            for data, target in self.test_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)

                output = model(data)

                l = nn.CrossEntropyLoss()(output, target).item()
                test_loss += l
                test_correct += (torch.sum(torch.argmax(output,dim=1)==target)).item()

        return test_loss, test_correct / len(self.test_data.dataset)
