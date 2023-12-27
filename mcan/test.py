# from core.model.net import Net

# net = Net(0,1,2,3)
# print(net.state_dict().keys())
import torch
pretrained_dict = torch.load("/home/gsh/paddle_project/mcan/ckpts/ckpt_small/epoch13.pkl") #读取存储的数据
print(pretrained_dict["state_dict"].keys())