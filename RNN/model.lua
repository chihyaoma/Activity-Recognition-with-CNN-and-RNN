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
   if opt.fcSize ~= nil then
      for i,fcSize in ipairs(opt.fcSize) do 
         -- add fully connected layers to fuse spatial and temporal features
         model:add(nn.Sequencer(nn.Linear(inputSize, fcSize)))
         inputSize = fcSize
      end
   end

   -- setup two LSTM cells with different dimensions
   local conTable = nn.ConcatTable()
   -- recurrent layer
   for i,hiddenSize in ipairs(opt.hiddenSize) do 
      conTable:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
   end
   model:add(conTable)

   -- input layer 
   model:insert(nn.SplitTable(3,1), 1) -- tensor to table of tensors

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

   model:add(p1):add(p2)
   -- average the prediction from all FC layers
   model:add(nn.CAddTable())
   -- model:add(nn.CMaxTable())

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
