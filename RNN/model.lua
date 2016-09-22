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

   if opt.dropout > 0 then
      -- model:add(nn.Sequencer(nn.Dropout(opt.dropoutProb), 1))
   end

   for i,hiddenSize in ipairs(opt.hiddenSize) do 

      if i~= 1 and (not opt.lstm) and (not opt.gru) then
         model:add(nn.Sequencer(nn.Linear(inputSize, hiddenSize)))
      end
      
      -- recurrent layer
      local rnn
      if opt.gru then
         -- Gated Recurrent Units
         rnn = nn.Sequencer(nn.GRU(inputSize, hiddenSize))
      elseif opt.lstm then
         -- Long Short Term Memory
         require 'nngraph'
         nn.FastLSTM.usenngraph = true -- faster
         nn.FastLSTM.bn = opt.bn
         -- rnn = nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize))
         rnn = nn.Sequencer(nn.LSTM(inputSize, hiddenSize))
      else
         -- simple recurrent neural network
         rnn = nn.Recurrent(
            hiddenSize, -- first step will use nn.Add
            nn.Identity(), -- for efficiency (see above input layer) 
            nn.Linear(hiddenSize, hiddenSize), -- feedback layer (recurrence)
            nn.Sigmoid(), -- transfer function 
            --99999 -- maximum number of time-steps per sequence
            opt.rho
         )
         if opt.zeroFirst then
            -- this is equivalent to forwarding a zero vector through the feedback layer
            rnn.startModule:share(rnn.feedbackModule, 'bias')
         end
         rnn = nn.Sequencer(rnn)
      end
      
      model:add(rnn)

      if opt.dropout > 0 then -- dropout it applied between recurrent layers
         -- model:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
      end
      
      inputSize = hiddenSize
   end

   -- input layer 
   model:insert(nn.SplitTable(3,1), 1) -- tensor to table of tensors

   if opt.dropout > 0 then
      model:insert(nn.Dropout(opt.dropoutProb), 1)
   end

   -- output layer
   model:add(nn.SelectTable(-1)) -- this selects the last time-step of the rnn output sequence
   model:add(nn.Linear(inputSize, nClass))
   model:add(nn.LogSoftMax())

   if opt.uniform > 0 then
      for k,param in ipairs(model:parameters()) do
         param:uniform(-opt.uniform, opt.uniform)
      end
   end

   -- will recurse a single continuous sequence
   model:remember((opt.lstm or opt.gru) and 'both' or 'eval')
end

-- build criterion
criterion = nn.ClassNLLCriterion()

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
