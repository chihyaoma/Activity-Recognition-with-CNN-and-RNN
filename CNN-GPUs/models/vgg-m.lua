---------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN
--  https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
-- 
-- 
--  Train a CNN on flow map of UCF-101 dataset 
-- 
-- 
--  Contact: Chih-Yao Ma at <cyma@gatech.edu>
---------------------------------------------------------------
local nn = require 'nn'
require 'cudnn'
require 'inn'
local function createModel(opt)

  local model = nn.Sequential() -- branch 1

  local iChannels = opt.nChannel
  if opt.nStacking ~= 'false' then
     iChannels = opt.nChannel * opt.nStacking
  end

  model:add(cudnn.SpatialConvolution(iChannels,96,7,7,2,2,0,0))       -- 224 -> 55
  model:add(cudnn.ReLU(true))
  model:add(inn.SpatialSameResponseNormalization(3,0.00005,0.75))
  model:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27

  model:add(cudnn.SpatialConvolution(96,256,5,5,2,2,1,1))       -- 27 ->  27
  model:add(cudnn.ReLU(true))
  model:add(inn.SpatialSameResponseNormalization(3,0.00005,0.75))
  model:add(cudnn.SpatialMaxPooling(3,3,2,2))                  -- 27 ->  13

  model:add(cudnn.SpatialConvolution(256,512,3,3,1,1,1,1))      -- 13 ->  13
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1))      -- 13 ->  13
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1))      -- 13 ->  13
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

  -- If downsample the image is required, adjust the kernel of pooling layer
  if opt.downsample ~='false' then
    assert(torch.type(model:get(#model.modules)) == 'cudnn.SpatialMaxPooling',
           'unknown network structure, is this vgg network?')
    model:add(nn.View(512*2*2))
    model:add(nn.Dropout(0.9))
    model:add(nn.Linear(512*2*2, 4096))
  else
    model:add(nn.View(512*5*5))
    model:add(nn.Dropout(0.9))
    model:add(nn.Linear(512*5*5, 4096))
  end

  model:add(nn.Threshold(0, 1e-6))
  model:add(nn.Dropout(0.8))
  model:add(nn.Linear(4096, 2048))
  model:add(nn.Threshold(0, 1e-6))
  model:add(nn.Linear(2048, 1000))
  

  model:cuda()

  model:get(1).gradInput = nil

  return model
end

return createModel