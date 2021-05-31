# quickGCNs 
## Use quickGCNs to quickly train typical GCN models
quickGCNs 是一个能够快速调用并且训练Pyg提供的GCN基础模型的多模型集成框架；  
能够保存最佳参数组合的评估结果并且画图保存，以及每次重复实验测试的结果；  
支持自定义模型；  
能够记录训练日志，并且断点恢复训练；  
能够通过网格搜索(grid search)或者随机搜索(random search)来找出模型最优参数组合。
## Installations
numpy  
pandas  
matplotlib  
skearn  
pytorch  
torch_geometric
## 目前支持的任务
* Node task
  * Node Regression: Yes
  * Node Classification: No
* Graph task
  * Graph Regression: No
  * Graph Classifation: No  

尽快更新
## 文件
quickGCNs.py：框架主要实现文件  
models.py：GCN模型存放文件  
demo.ipynb：运行demo文件  
## 更新日志
1. 2021/05/22: upload first quickGCNs files.  
2. 2021/05/24: 修复Bug，添加random search方法，添加demo文件.  
3. 2021/05/31: 修复Bug，添加Mini-batch参数，添加demo数据集，添加最佳参数组合loss画图，修改输入数据集样式（分开训练集与测试集输入，输入格式更灵活）.
