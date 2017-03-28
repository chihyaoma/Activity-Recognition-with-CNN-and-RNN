-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- training w/ data augmentation

-- modified by Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 10/11/2016

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'
require 'image'

----------------------------------------------------------------------
-- Model + Loss:
local t
if opt.model == 'model-Conv' then
  t = require 'model-Conv'
elseif opt.model == 'model-Conv-VGG' then
  t = require 'model-Conv-VGG'
elseif opt.model == 'model-Conv-Inception' then
  t = require 'model-Conv-Inception'
elseif opt.model == 'model-Conv-Inception-v1' then
  t = require 'model-Conv-Inception-v1'
elseif opt.model == 'model-Conv-Inception-v1-1L' then
  t = require 'model-Conv-Inception-v1-1L'
elseif opt.model == 'model-Conv-Inception-v1-2L' then
  t = require 'model-Conv-Inception-v1-2L'
elseif opt.model == 'model-Conv_2stride-Inception-v1' then
  t = require 'model-Conv_2stride-Inception-v1'
elseif opt.model == 'model-1L-MultiFlow' then
  t = require 'model-1L-MultiFlow'
elseif opt.model == 'model-1L-SplitST' then
  t = require 'model-1L-SplitST'
elseif opt.model == 'model-1L' then
  t = require 'model-1L'
elseif opt.model == 'model-2L' then
  t = require 'model-2L'
end
local model = t.model
local loss = t.loss
local model_name = t.model_name
local nframeAll = t.nframeAll
local nframeUse = t.nframeUse
local nfeature = t.nfeature
----------------------------------------------------------------------
-- Save light network tools:
function nilling(module)
   module.gradBias   = nil
   if module.finput then module.finput = torch.Tensor() end
   module.gradWeight = nil
   module.output     = torch.Tensor()
   if module.fgradInput then module.fgradInput = torch.Tensor() end
   module.gradInput  = nil
end

function netLighter(network)
   nilling(network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         netLighter(a)
      end
   end
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining some tools')

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-- Log results to files
local trainLogger = optim.Logger(paths.concat(opt.save,'train.log'))

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> flattening model parameters')

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()
print('parameter #: '..tostring(w:size()))
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> configuring optimizer')

local optimState = {
   learningRate = tonumber(opt.learningRate),
   momentum = tonumber(opt.momentum),
   weightDecay = tonumber(opt.weightDecay),
   learningRateDecay = tonumber(opt.learningRateDecay),
   decayRate = tonumber(opt.decayRate)
}
local function deepCopy(object)
    local lookup_table = {}
    local function _copy(object)
        if type(object) ~= "table" then
            return object
        elseif lookup_table[object] then
            return lookup_table[object]
        end
        local new_table = {}
        lookup_table[object] = new_table
        for index, value in pairs(object) do
            new_table[_copy(index)] = _copy(value)
        end
        return setmetatable(new_table, getmetatable(object))
    end
    return _copy(object)
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> allocating minibatch memory')

local nCrops
if opt.methodCrop == 'centerCropFlip' then
  nCrops = 2
-- elseif opt.methodCrop == 'tenCrop' then
--   nCrops = 10
else 
  nCrops = 1
end

local numTrainVideo = trainData:size(1)/nCrops

local batchSize = tonumber(opt.batchSize)
-- print('training batch size: '..batchSize)
local x = torch.Tensor(batchSize*nCrops,1, nfeature, nframeUse) -- data (32x1x4096x25)
local yt = torch.Tensor(batchSize*nCrops)
if opt.type == 'cuda' then 
   x = x:cuda()
   yt = yt:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining training procedure')

local epoch
local function data_augmentation(inputs)
		-- TODO: appropriate augmentation methods
        -- DATA augmentation (only random 1-D cropping here)
    local frameSkip = nframeAll - nframeUse
		-- -- 1. crop a clip of video
  --   local i = torch.random(1,frameSkip+1)
		-- local outputs = inputs[{{},{},{i,i+nframeUse-1}}]
		-- return outputs

    -- 2. randomly drop frames
    local nframe = nframeAll
    local outputs = inputs:clone()
    for t = 1,frameSkip do      
      local id = torch.random(nframe)
      nframe = nframe-1

      if id == 1 then
        outputs = outputs[{{},{},{2,nframe+1}}]
      elseif id == nframe+1 then
        outputs = outputs[{{},{},{1,nframe}}]
      else
        local clip_1 = outputs[{{},{},{1,id-1}}]
        local clip_2 = outputs[{{},{},{id+1,nframe+1}}]
        outputs = torch.cat(clip_1,clip_2,3)
      end      
    end
		return outputs
end

local LR_ini = optimState.learningRate

local function train(trainData)
   model:training()

   -- epoch tracker
   epoch = epoch or 1

   -- update the learning rate
   local decayPower = math.floor((epoch-1)/optimState.decayRate)
   local LR_new = LR_ini * math.pow(0.1,decayPower)
   optimState.learningRate = LR_new
   print(optimState)

   -- local vars
   local time = sys.clock()
   local run_passed = 0
   local mean_dfdx = torch.Tensor():typeAs(w):resizeAs(w):zero()

   -- shuffle at each epoch
   local shuffle = torch.randperm(numTrainVideo)

   -- do one epoch
   print(sys.COLORS.green .. '==> doing epoch on training data:') 
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,numTrainVideo,batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())
      collectgarbage()

      -- batch fits?
      if (t + batchSize - 1) > numTrainVideo then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+batchSize-1 do
      	 local i_shuf = shuffle[i]
         
         x[{{(idx-1)*nCrops+1,idx*nCrops}}] = data_augmentation(trainData.data[{{(i_shuf-1)*nCrops+1,i_shuf*nCrops}}])

         -- print(x:size())
         -- x[{idx,1}] = trainData.data[shuffle[i]]
         yt[{{(idx-1)*nCrops+1,idx*nCrops}}] = trainData.labels[{{(i_shuf-1)*nCrops+1,i_shuf*nCrops}}]

         idx = idx + 1
      end

         -- create closure to evaluate f(X) and df/dX
         local eval_E = function(w)
            -- reset gradients
            dE_dw:zero()

            -- evaluate function for complete mini batch
            local y
            local x_final
            if opt.model == 'model-1L-SplitST' then
              x_final = {x[{{},{},{1,nfeature/2},{}}],x[{{},{},{nfeature/2+1,nfeature},{}}]}
            elseif opt.model == 'model-Conv-Inception-v1-TemSeg3' then
              x_final = {x[{{},{},{},{1,8}}],x[{{},{},{},{9,17}}],x[{{},{},{},{18,25}}]}
            elseif opt.model == 'model-Conv-Inception-TemSeg5' then
              x_final = {x[{{},{},{},{1,5}}],x[{{},{},{},{6,10}}],x[{{},{},{},{11,15}}],x[{{},{},{},{16,20}}],x[{{},{},{},{21,25}}]}
              -- print(x_final)
              -- error(test)
            else
              x_final = x
            end
            -- x_final = x_final:transpose(2,3)-- transpose (BN x 1 x 4096 x 25 --> BN x 4096 x 1 x 25)
            y = model:forward(x_final)
            --print(y:size())
            --error(test)
            local E = loss:forward(y,yt)

            -- estimate df/dW
            local dE_dy = loss:backward(y,yt)   
            model:backward(x_final,dE_dy)

            -- update confusion
            for i = 1,batchSize do
               confusion:add(y[i],yt[i])
            end

            local yProb, yLabels = y:topk(1, true, true)

            -- print(y)
            -- print(yProb)
            -- print(yLabels, yt)
            
            -- return f and df/dX
            return E,dE_dw
         end

         -- optimize on current mini-batch
         if opt.optMethod == 'sgd' then
            optim.sgd(eval_E, w, optimState)
         elseif opt.optMethod == 'adam' then
            optim.adam(eval_E, w, optimState)
         elseif opt.optMethod == 'rmsprop' then
            optim.rmsprop(eval_E, w, optimState)
         elseif opt.optMethod == 'asgd' then
            run_passed = run_passed + 1
            mean_dfdx  = asgd(eval_E, w, run_passed, mean_dfdx, optimState)
         end
         -- error(test)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot == 'Yes' then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- -- save/log current net
   -- local filename = paths.concat(opt.save, model_name)
   -- os.execute('mkdir -p ' .. sys.dirname(filename))
   -- print('==> saving model to '..filename)
   -- model1 = model:clone()
   -- netLighter(model1)
   -- torch.save(filename, model1)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end

-- Export:
return train

