# FedAvg

## 创作目的

  本人是一名在读研究生，学习边缘计算方向。关于联邦学习，我也是一个小白，本项目是我比葫芦画瓢写的一个基本的联邦学习。
  了解一下联邦学习的基本内容，基本的代码。这就是我的创作目的。
  
## 包含文件

  本项目包含7个Python文件，每个文件的功能详述如下
  ### 2.1 clients文件
  
    本文件只有一个client类，他的主要功能是定义一个“用户”。每个用户都是独立的个体，他们都有自己的模型和数据。
    这个类有三个函数，get_global_model 、 local_train 和 test_model 三个函数。
    
    get_global_model函数：主要功能是模型聚合后，每个用户加载自己的模型。
    local_train函数：首先定义优化函数，默认是SGD，模型利用本地数据训练后保存到cache文件中
    test_model函数：用户利用自己模型进行测试，返回两个值：损失和精度
    
    
  ### 2.2 datasets文件：
  
    本文件定义一个download_data类，下载数据并且以我们想要的方法产生数据。
    包含5个函数，__load_dataset、get_IID_data、get_nonIID_data、get_practical_nonIID_data和get_data函数。
    
    __load_dataset函数：是在类初始化额时候使用根据数据集的名称，下载相应的数据
    get_IID_data：获得IID数据，每次返回的数据比较相似
    get_nonIID_data：获得non-IID数据，每一个用户随机获得两个标签
    get_practical_nonIID_data：获得更加符合实际的数据集，一共有三个划分集合，每个用户获得集合中的一个元素，获得相同元素的用户之间数据集相似很高，不同元素之间的数据集基本不相似
    
    get_data函数：这个函数是用于外部调用的，上述四个函数并不对外调用。根据输入的数据，产生相应的数据集，输入的数据只能是三个字符串中的一个。
    
  ### 2.3 main函数：
  
    main函数是整个程序的逻辑。首先第一步是数据集的名字使用download_data函数下载数据集，产生一个data_loader的变量，用于产生数据，然后根据数据集的名字和模型的种类产生初始模型。使用模型和数据集产生一个云模型和global_nums个用户，并放在用户集合clients中。
    
    
    
    
    
    
    
    
    
    
    
