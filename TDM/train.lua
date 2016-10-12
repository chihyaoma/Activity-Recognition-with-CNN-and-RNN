----------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN
--  https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
--
--  Temporal Dynamic Model: Multidimensional LSTM and Temporal ConvNet
--  
-- 
--  Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------

local sys = require 'sys'
local xlua = require 'xlua'    -- xlua provides useful tools, like progress bars
local optim = require 'optim'

print(sys.COLORS.red .. '==> defining some tools')

-- model:
local m = require 'model'
local model = m.model
local criterion = m.criterion

-- Batch test:
local inputs = torch.Tensor(opt.batchSize, opt.inputSize, opt.rho)

local targets = torch.Tensor(opt.batchSize)

if opt.cuda == true then
   inputs = inputs:cuda()
   targets = targets:cuda()
end

print(sys.COLORS.red ..  '==> configuring optimizer')
-- Pass learning rate from command line
optimState = optimState or {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   lrMethod = opt.lrMethod,
   epochUpdateLR = opt.epochUpdateLR,
   learningRateDecay = opt.learningRateDecay,
   lrDecayFactor = opt.lrDecayFactor,
   nesterov = true, 
   dampening = 0.0
}

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()

function train(trainData, trainTarget)

   -- epoch tracker
   epoch = epoch or 1

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   model:training()

   -- shuffle at each epoch
   local shuffle = torch.randperm(trainData:size(1))

   print(sys.COLORS.green .. '==> doing epoch on training data:') 
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   -- adjust learning rate
   optimState.learningRate = adjustLR(opt.learningRate, epoch)

   print(sys.COLORS.yellow ..  '==> Learning rate is: ' .. optimState.learningRate .. '')

   local top1Sum, top3Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   for t = 1,trainData:size(1),opt.batchSize do
      local dataTime = dataTimer:time().real

      if opt.progress == true then
         -- disp progress
         xlua.progress(t, trainData:size(1))
      end
      collectgarbage()

      -- batch fits?
      if (t + opt.batchSize - 1) > trainData:size(1) then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         inputs[idx] = trainData[shuffle[i]]:float()
         targets[idx] = trainTarget[shuffle[i]]
         idx = idx + 1
      end

      -- replicate the training data and feed into LSTM and T-CNN
      local repeatInputs = torch.repeatTensor(inputs,2,1,1)

      --------------------------------------------------------
      -- Using optim package for training
      --------------------------------------------------------
      local loss, top1, top3
      local eval_E = function(w)
         -- reset gradients
         dE_dw:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(repeatInputs)

         -- print(outputs)
         -- error('test')

         loss = criterion:forward(outputs,targets)

         -- estimate df/dW
         local dE_dy = criterion:backward(outputs,targets)
         model:backward(repeatInputs,dE_dy)

         top1, top3 = computeScore(outputs, targets, 1)
         top1Sum = top1Sum + top1*opt.batchSize
         top3Sum = top3Sum + top3*opt.batchSize
         lossSum = lossSum + loss*opt.batchSize
         N = N + opt.batchSize

         -- return f and df/dX
         return loss,dE_dw
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

      print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top3 %7.3f'):format(
         epoch, t, trainData:size(1), timer:time().real, dataTime, loss, top1, top3))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(w:storage() == model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()

   end

   print(sys.COLORS.red .. '==> Best testing accuracy = ' .. bestAcc .. '%')

   epoch = epoch + 1
end

-- TODO: Learning Rate function 
function adjustLR(learningRate, epoch)
   local decayPower = 0
   if optimState.lrMethod == 'manual' then
      decayPower = decayPower
   elseif optimState.lrMethod == 'fixed' then
      decayPower = math.floor((epoch - 1) / optimState.epochUpdateLR)
   end
   
   return learningRate * math.pow(optimState.lrDecayFactor, decayPower)
end

function computeScore(output, target)

   -- Coputes the top1 and top3 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-3 score, if there are at least 3 classes
   local len = math.min(3, correct:size(2))
   local top3 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top3 * 100
end

-- Export:
return train
