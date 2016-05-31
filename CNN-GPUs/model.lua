----------------------------------------------------------------
-- Activity-Recognition-with-CNN-and-RNN
-- https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
-- 
-- 
-- Train a CNN on flow map of UCF-101 dataset 
-- 
-- 
-- Contact: Chih-Yao Ma at <cyma@gatech.edu>

--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************
----------------------------------------------------------------
--
require 'nn'
require 'cunn'
require 'optim'

--[[
   1. Create Model
   2. Create Criterion
   3. Convert model to CUDA
]]--

-- 1. Create Network
-- 1.1 If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = loadDataParallel(opt.retrain, opt.nGPU) -- defined in util.lua
else
   paths.dofile('models/' .. opt.netType .. '.lua')
   print('=> Creating model from file: models/' .. opt.netType .. '.lua')
   model = createModel(opt.nGPU) -- for the model creation code, check the models/ folder
   if opt.backend == 'cudnn' then
      require 'cudnn'
      cudnn.convert(model, cudnn)
   elseif opt.backend ~= 'nn' then
      error'Unsupported backend'
   end
end

-- 2. Create Criterion
criterion = nn.ClassNLLCriterion()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
-- model is converted to CUDA in the init script itself
-- model = model:cuda()
criterion:cuda()

collectgarbage()
