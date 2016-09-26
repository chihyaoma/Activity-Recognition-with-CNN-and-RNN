----------------------------------------------------------------
-- Georgia Tech 2016 Spring
-- Deep Learning for Perception
-- Final Project: LRCN model for Video Classification
--
-- 
-- This is a testing code for implementing the RNN model with LSTM 
-- written by Chih-Yao Ma. 
-- 
-- The code will take feature vectors (from CNN model) from contiguous 
-- frames and train against the ground truth, i.e. the labeling of video classes. 
-- 
-- Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------

local nn = require 'nn'
local rnn = require 'rnn'
local sys = require 'sys'

print(sys.COLORS.red ..  '==> construct RNN')

if checkpoint then
   local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
   assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
   print('=> Resuming model from ' .. modelPath)
   model = torch.load(modelPath)
else
   -- Video Classification model
   model = nn.Sequential()

   local inputSize = opt.inputSize

   -- input layer 
   assert((opt.batchSize%(#opt.hiddenSize)) == 0, 'batchSize need to be the multiple of # of hidden layer.')
   model:add(nn.View(#opt.hiddenSize, opt.batchSize/#opt.hiddenSize, inputSize, -1))
   model:add(nn.SplitTable(1,2)) -- tensor to table of tensors
   
   local p = nn.ParallelTable()
   for i=1,#opt.hiddenSize do 
      p:add(nn.SplitTable(3,1))
   end
   model:add(p)

   local pFC = nn.ParallelTable()
   
   if opt.fcSize ~= nil then
      -- for i,fcSize in ipairs(opt.fcSize) do 
      for i=1,#opt.hiddenSize do 
         -- add fully connected layers to fuse spatial and temporal features
         pFC:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
      end
   end
   model:add(pFC)

   -- setup two LSTM cells with different dimensions
   local lstmTable = nn.ParallelTable()
   -- recurrent layer
   for i,hiddenSize in ipairs(opt.hiddenSize) do 
      lstmTable:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
   end
   model:add(lstmTable)

   -- output layer
   local p1 = nn.ParallelTable()
   for i in ipairs(opt.hiddenSize) do 
      -- select the last output from each of the LSTM cells
      p1:add(nn.SelectTable(-1))
   end
   
   local p2 = nn.ParallelTable()
   for i,hiddenSize in ipairs(opt.hiddenSize) do
      -- full-connected layers for each LSTM output
      p2:add(nn.Linear(hiddenSize, nClass))
   end
   -- concat the prediction from all FC layers
   model:add(p1):add(p2)
   model:add(nn.JoinTable(1))

   if opt.uniform > 0 then
      for k,param in ipairs(model:parameters()) do
         param:uniform(-opt.uniform, opt.uniform)
      end
   end

   -- will recurse a single continuous sequence
   model:remember((opt.lstm or opt.gru) and 'both' or 'eval')
end

-- build criterion
criterion = nn.CrossEntropyCriterion()

print(sys.COLORS.red ..  '==> here is the network:')
print(model)

if opt.cuda == true then
   model:cuda()
   criterion:cuda()
end

return
{
   model = model,
   criterion = criterion
}
