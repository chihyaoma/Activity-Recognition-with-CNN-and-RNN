--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--
local xlua = require 'xlua'
local optim = require 'optim'
local pastalog = require 'pastalog'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

local testLogger = optim.Logger('test.log')

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      optMethod = opt.optMethod,
      lrMethod = opt.lrMethod,
      epochUpdateLR = opt.epochUpdateLR,
      learningRate = opt.LR,
      learningRateDecay = opt.LRD,
      lrDecayFactor = opt.lrDecayFactor,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   --print(self.optimState)
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader, diffTop1)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch, diffTop1)
   print(self.optimState)
   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run(self.opt) do -- sample: image & class
      local dataTime = dataTimer:time().real

      -- disp progress
      xlua.progress(n, dataloader:size())

      -- Copy input and target to the GPU
      self:copyInputs(sample)
	
      --print(self.input:size())
      local output = self.model:forward(self.input):float()

      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      if self.optimState.optMethod == 'sgd' then
      	optim.sgd(feval, self.params, self.optimState)
      elseif self.optimState.optMethod == 'adam' then
      	optim.adam(feval, self.params, self.optimState)
      end

      local top1, top5 = self:computeScore(output, sample.target, 1)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5))

      local modelName = self.opt.netType .. '-' .. self.opt.depth .. '-' .. 'LR=' .. self.optimState.learningRate
         .. '-' .. 'WD=' .. self.optimState.weightDecay
      local trainSeriesName = 'train-top1-' .. self.opt.pastalogName
      pastalog(modelName, trainSeriesName, top1, (epoch-1)*trainSize + n, 'http://ct5250-12.ece.gatech.edu:8120/data')

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run(self.opt) do
      local dataTime = dataTimer:time().real

      -- disp progress
      xlua.progress(n, dataloader:size())

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1) / nCrops
      local loss = self.criterion:forward(self.model.output, self.target)

      local top1, top5 = self:computeScore(output, sample.target, nCrops)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      N = N + batchSize

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
         epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))

      local modelName = self.opt.netType .. '-' .. self.opt.depth .. '-' .. 'LR=' .. self.optimState.learningRate
         .. '-' .. 'WD=' .. self.optimState.weightDecay
      local testSeriesName = 'test-top1-' .. self.opt.pastalogName
      pastalog(modelName, testSeriesName, top1, (epoch-1)*size + n, 'http://ct5250-12.ece.gatech.edu:8120/data')

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))


   local modelName = self.opt.netType .. '-' .. self.opt.depth .. '-' .. 'LR=' .. self.optimState.learningRate
      .. '-' .. 'WD=' .. self.optimState.weightDecay
   local epochSeriesName = 'epoch-top1-' .. self.opt.pastalogName
   pastalog(modelName, epochSeriesName, top1Sum / N, epoch, 'http://ct5250-12.ece.gatech.edu:8120/data')

   -- update log
   testLogger:add{['epoch'] = epoch, ['top-1 error'] = top1Sum / N, ['top-5 error'] = top5Sum / N}

   return top1Sum / N, top5Sum / N
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch, diffTop1)
   -- Training schedule
   local decayPower = 0
   if self.opt.dataset == 'imagenet' or self.opt.dataset == 'ucf101' then
      decayPower = math.floor((epoch - 1) / 30)
      return self.opt.LR * math.pow(self.optimState.lrDecayFactor, decayPower)
   elseif self.opt.dataset == 'cifar10' then
      decayPower = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
      return self.opt.LR * math.pow(self.optimState.lrDecayFactor, decayPower)
   elseif self.opt.dataset == 'ucf101-flow' then
      if self.optimState.lrMethod == 'manual' then
      	decayPower = decayPower
      elseif self.optimState.lrMethod == 'fixed' then
      	decayPower = math.floor((epoch - 1) / self.optimState.epochUpdateLR)
      elseif self.optimState.lrMethod == 'adaptive' then
      	if epoch > 1 and diffTop1 < 5e-3 then
      		decayPower = decayPower + 0.1
      	end
      end	

      return self.opt.LR * math.pow(self.optimState.lrDecayFactor, decayPower)
   end
   --return self.opt.LR * math.pow(0.1, decayPower)
end

return M.Trainer
