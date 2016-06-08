---------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN
--  https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
-- 
-- 
--  Jointly two models together: RGB and flow map models
--  RGB model will be a pre-trained ResNet or a wide ResNet from imagenet
--  Flow map model will be trained from UCF-101 dataset using initialization
--  from RGB model (from imagenet)
-- 
-- 	use qlua to run
-- 
--  Contact: Chih-Yao Ma at <cyma@gatech.edu>
---------------------------------------------------------------
local nn = require 'nn'
require 'cunn'
require 'nngraph'

-- ndim of input features
rgbnFeature = 2048
flownFeature = 2048
-- ndim of two full-connected layers
n1fc = (rgbnFeature + flownFeature)
n2fc = n1fc/2

-- initial the model to train two extracted feature vectors
h1 = nn.Identity()()
h2 = nn.Identity()()
concat = nn.JoinTable(1)({h1, h2})
fc1 = nn.Linear(n1fc, n1fc)(concat)
fc2 = nn.Linear(n1fc, n2fc)(fc1)
g = nn.gModule({h1, h2}, {fc2})

x1 = torch.rand(rgbnFeature)
x2 = torch.rand(flownFeature)

g:forward({x1, x2})
g:backward({x1, x2}, torch.rand(n2fc))

-- plot forward and backward graph
graph.dot(g.fg, 'Forward Graph')
graph.dot(g.bg, 'Backward Graph')