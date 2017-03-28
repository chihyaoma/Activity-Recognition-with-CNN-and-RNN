-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- This script implements a test procedure, to report accuracy on the test data

-- TODO:
-- 1. 
-- 2. 

-- modified by Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 10/22/2016


require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'image'


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
print(sys.COLORS.red .. '==> defining some tools')

-- model:
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
-- local loss = t.loss
local model_name = t.model_name
local nframeAll = t.nframeAll
local nframeUse = t.nframeUse
local nfeature = t.nfeature

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes) 

-- Logger:
local testLogger = optim.Logger(paths.concat(opt.save,'test.log'))

local nCrops
if opt.methodCrop == 'centerCropFlip' then
  nCrops = 2
-- elseif opt.methodCrop == 'tenCrop' then
--   nCrops = 10
else 
  nCrops = 1
end

local numTestVideo = testData:size(1)/nCrops

-- Batch test:
local batchSize = tonumber(opt.batchSize)
-- print('testing batch size: '..batchSize)
local inputs = torch.Tensor(batchSize*nCrops,1, nfeature, nframeUse) -- get size from data
local targets = torch.Tensor(batchSize*nCrops)
if opt.type == 'cuda' then 
   inputs = inputs:cuda()
   targets = targets:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')

accMax = 0 -- record the highest testing accuracy
epoBest = 0 -- record the epoch # of the best testing accuracy

-- test function
function test(testData, classes, epo)
   model:evaluate()

   local bSize = batchSize
   
   -- local vars
   local time = sys.clock()
   
   -- test over test data
   print(sys.COLORS.red .. '==> testing on test set:')
   local predlabeltxt = {}
   local prob = {}
   for t = 1,numTestVideo,batchSize do
      -- disp progress
	  collectgarbage()
      xlua.progress(t, numTestVideo)

      -- batch fits?
      if (t + bSize - 1) > numTestVideo then
         bSize = numTestVideo - (t-1)
         -- break
      end

      -- create mini batch
      local frameSkip = nframeAll-nframeUse
      local idx = 1 -- index only for this mini-batch
      for i = t,t+bSize-1 do
         --inputs[{idx,1}] = testData.data[{i,{},{1,nframeUse}}]
         inputs[{{(idx-1)*nCrops+1,idx*nCrops},1}] = testData.data[{{(i-1)*nCrops+1,i*nCrops},{},{1+frameSkip/2,nframeAll-frameSkip/2}}] -- take the middle part
	      targets[{{(idx-1)*nCrops+1,idx*nCrops}}] = testData.labels[{{(i-1)*nCrops+1,i*nCrops}}]
         idx = idx + 1
      end

      -- test sample
      local preds
      local input_final
      if opt.model == 'model-1L-SplitST' then
         input_final = {inputs[{{},{},{1,nfeature/2},{}}],inputs[{{},{},{nfeature/2+1,nfeature},{}}]}
      elseif opt.model == 'model-Conv-Inception-v1-TemSeg3' then
         input_final = {inputs[{{},{},{},{1,8}}],inputs[{{},{},{},{9,17}}],inputs[{{},{},{},{18,25}}]}
      elseif opt.model == 'model-Conv-Inception-TemSeg5' then
         input_final = {inputs[{{},{},{},{1,5}}],inputs[{{},{},{},{6,10}}],inputs[{{},{},{},{11,15}}],inputs[{{},{},{},{16,20}}],inputs[{{},{},{},{21,25}}]}
      else
         input_final = inputs
      end
      -- input_final = input_final:transpose(2,3)-- transpose (BN x 1 x 4096 x 25 --> BN x 4096 x 1 x 25)
      preds = model:forward(input_final)

      preds = preds[{{1,bSize*nCrops}}] -- for the last few test data
      targets_batch = targets[{{1,bSize*nCrops}}] -- for the last few test data

      -- print(inputs[{{},1,1,{}}])
      -- print(preds)
      -- print(targets_batch)

      ---- Compute n-Crop score ----
      preds = preds:view(preds:size(1) / nCrops, nCrops, preds:size(2)):exp():mean(2):squeeze(2)
      targets_batch = targets_batch:view(targets_batch:size(1) / nCrops, nCrops):mean(2):squeeze(2)
	--print(preds:sum(2))
      -- Get the top N class indexes and probabilities
      local N = 3
      local probLog, predLabels = preds:topk(N, true, true)
	--print(probLog)
      -- print(predLabels)
       --error(test)
      
      -- Convert log probabilities back to [0, 1]
      -- probLog:exp()

      -- --print(preds:size())
      -- _,indices = torch.sort(preds,2,true)
      -- predlabels = indices[{{},1}]
      -- --print(predlabels:size())

      
      for i = 1,bSize do
         --predlabeltxt[i-1+t] = classes[predlabels[i]]
         predlabeltxt[i-1+t] = {}
         prob[i-1+t] = {}
         for j = 1, N do
            predlabeltxt[i-1+t][j] = classes[predLabels[i][j]]
            prob[i-1+t][j] = probLog[i][j]
         end

      end

      -- idx = 1
      -- for i = t,t+bSize-1 do
      --    -- inputs[{idx,1}] = testData.data[{i,{},{1,nframeUse}}]
      --    -- targets[idx] = testData.labels[i]
      --    predlabeltxt[i] = classes[predlabels[idx]]
      --    idx = idx + 1
      -- end


      -- confusion
      for i = 1,bSize do
         confusion:add(preds[i], targets_batch[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if confusion.totalValid > accMax then
      print(sys.COLORS.yellow ..  'Updating the predicted labels!!!')
      accMax = confusion.totalValid
      epoBest = epo
      torch.save('labels.txt',predlabeltxt,'ascii')
      torch.save('prob.txt',prob,'ascii')

      -- save/log current net
      if opt.saveModel == 'Yes' then
         local filename = paths.concat(opt.save, model_name..'.t7')
         -- os.execute('mkdir -p ' .. sys.dirname(filename))
         print('==> saving model to '..filename)
         --model1 = model:clone()
         --netLighter(model1)
         torch.save(filename, model)
      end
      
   end
   print("\n the max accuracy is " .. accMax ..' in the epoch '.. epoBest)

   if opt.plot == 'Yes' then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end
   confusion:zero()
end

-- Export:
return test

