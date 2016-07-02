----------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN
--  https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
--
-- 
--  This is a testing code for implementing the RNN model with LSTM 
--  written by Chih-Yao Ma. 
-- 
--  The code will take feature vectors (from CNN model) from contiguous 
--  frames and train against the ground truth, i.e. the labeling of video classes. 
-- 
--  Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------
require 'torch'
require 'sys'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'

print(sys.COLORS.red .. '==> defining some tools')

-- model:
local m = require 'model'
local model = m.model
local criterion = m.criterion

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes) 

-- Logger:
local trainLogger = optim.Logger(paths.concat(opt.save,'train.log'))

-- Batch test:
local inputs = torch.Tensor(opt.batchSize, TrainData:size(2), TrainData:size(3))
local targets = torch.Tensor(opt.batchSize)

if opt.cuda == true then
   inputs = inputs:cuda()
   targets = targets:cuda()
end

print(sys.COLORS.red ..  '==> configuring optimizer')
-- Pass learning rate from command line
local optimState = {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   learningRateDecay = opt.learningRateDecay
}

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()

function train(TrainData, TrainTarget)

   -- epoch tracker
   epoch = epoch or 1

   local time = sys.clock()

   model:training()

   -- shuffle at each epoch
   local shuffle = torch.randperm(TrainData:size(1))

   print(sys.COLORS.green .. '==> doing epoch on training data:') 
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   if opt.optimizer == 'adam' or 'adamax' or 'rmsprop' then
         -- Maybe decay learning rate
      if epoch % opt.lrDecayEvery == 0 then
         print(sys.COLORS.yellow ..  '==> Updating learning rate .. ')
         local old_learningRate = optimState.learningRate
         optimState = {learningRate = old_learningRate * opt.lrDecayFactor}
      end
   end

   print(sys.COLORS.yellow ..  '==> Learning rate is: ' .. optimState.learningRate .. '')

   for t = 1,TrainData:size(1),opt.batchSize do

      if opt.progress == true then
         -- disp progress
         xlua.progress(t, TrainData:size(1))
      end
      collectgarbage()

      -- batch fits?
      if (t + opt.batchSize - 1) > TrainData:size(1) then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         inputs[idx] = TrainData[shuffle[i]]
         targets[idx] = TrainTarget[shuffle[i]]
         idx = idx + 1
      end

      --------------------------------------------------------
      -- Using optim package for training
      --------------------------------------------------------
      local eval_E = function(w)
         -- reset gradients
         dE_dw:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local E = criterion:forward(outputs,targets)

         -- estimate df/dW
         local dE_dy = criterion:backward(outputs,targets)   
         model:backward(inputs,dE_dy)

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i],targets[i])
         end

         -- return f and df/dX
         return E,dE_dw
      end

      -- optimize on current mini-batch
      if opt.optimizer == 'sgd' then
         -- use SGD
         optim.sgd(eval_E, w, optimState)
      elseif opt.optimizer == 'adam' then
         -- use adam
         optim.adam(eval_E, w, optimState)
      elseif opt.optimizer == 'adamax' then
         -- use adamax
         optim.adamax(eval_E, w, optimState)
      elseif opt.optimizer == 'rmsprop' then
         -- use RMSProp
         optim.rmsprop(eval_E, w, optimState)
      end

   end

   -- time taken
   time = sys.clock() - time
   time = time / TrainData:size(1)
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end
   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end

-- Export:
return train