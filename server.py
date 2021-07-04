# 需要用到的函数：获得用户的精度和损失，聚合
# 输入的参数：一批的大小，训练测试数据，用户选择个数
import copy
import torch
from torch.autograd import Variable


class server(object):
    def __init__(self,model,device,dataset,client):
        self.model = model.to(device)
        self.device = device
        self.num = client
        self.acc = []
        self.test_data = dataset[1]

    def aggregate_model(self):
        model_states = []
        for i in range(self.num):
            model_states.append(torch.load('./cache/model_state_{}.pkl'.format(i)))
        global_model_state = copy.deepcopy(model_states[0])

        for key in global_model_state.keys():
            for i in range(1, len(model_states)):
                global_model_state[key] += model_states[i][key]
            global_model_state[key] = torch.div(global_model_state[key], len(model_states))

        
        self.model.load_state_dict(global_model_state)
        test_correct = 0

        with torch.no_grad():
            for data, target in self.test_data:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_acc = test_correct / len(self.test_data.dataset)
        torch.save(global_model_state, './cache/global_model_state.pkl')
        
        return test_acc