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

require 'torch'   -- torch
require 'nn'
require 'rnn'
require 'sys'

print(sys.COLORS.red ..  '==> construct RNN')


--[[
----------------------------------------------------------
--if only using simple Sequencer
----------------------------------------------------------
rnn = nn.Recurrent(
opt.hiddenSize[1], -- size of output
nn.Linear(ds.FeatureDims, opt.hiddenSize[1]), -- input layer
nn.Linear(opt.hiddenSize[1], opt.hiddenSize[1]), -- recurrent layer
nn.Sigmoid(), -- transfer function
opt.rho
)

model = nn.Sequential()
-- model:insert(nn.SplitTable(3,1), 1) -- tensor to table of tensors, which can't not be used in 'nn.Sequencer'
-- model:add(rnn)
:add(nn.FastLSTM(ds.FeatureDims, opt.hiddenSize[1]))
:add(nn.FastLSTM(opt.hiddenSize[1], opt.hiddenSize[2]))
:add(nn.Linear(opt.hiddenSize[2], nClass))
:add(nn.LogSoftMax())

model = nn.Sequencer(model)
----------------------------------------------------------
]]


-- Video Classification model
model = nn.Sequential()

-- local inputSize = opt.hiddenSize[1]
local inputSize = opt.inputSize

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
      rnn = nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize))
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

   if opt.dropout then -- dropout it applied between recurrent layers
      model:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
   end
   
   inputSize = hiddenSize
end

-- input layer 
model:insert(nn.SplitTable(3,1), 1) -- tensor to table of tensors

if opt.dropout then
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
