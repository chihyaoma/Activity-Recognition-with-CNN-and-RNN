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

require 'nn'

local rnn = require 'rnn'
local sys = require 'sys'

print(sys.COLORS.red ..  '==> construct RNN')

function model_temporal_BN_max_LSTM_BN_FC()
    -- Video Classification model
    local model = nn.Sequential()
    local inputSize = opt.inputSize

    local split = nn.ParallelTable()
    for i = 1, opt.numSegment do 
      split:add(nn.SplitTable(3,1))
    end
    model:add(split)

    local pBN = nn.ParallelTable()
    for i = 1, opt.numSegment do 
        pBN:add(nn.Sequencer(nn.BatchNormalization(inputSize)))
    end
    model:add(pBN)

    local mergeTable1 = nn.ParallelTable()
    for i = 1, opt.numSegment do 
        mergeTable1:add(nn.MapTable(nn.Unsqueeze(1)))
    end
    model:add(mergeTable1)

    local mergeTable2 = nn.ParallelTable()
    for i = 1, opt.numSegment do 
        mergeTable2:add(nn.JoinTable(1))
    end
    model:add(mergeTable2)

    local poolingTable = nn.ParallelTable()
    for i = 1, opt.numSegment do 
       poolingTable:add(nn.Max(1, -1))
    --   poolingTable:add(nn.Mean(1, -1))
    end
    model:add(poolingTable)


    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        model:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
        inputSize = hiddenSize
    end

    model:add(nn.SelectTable(-1))

    model:add(nn.BatchNormalization(inputSize))
       
    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(inputSize, nClass))

    return model
end

-------------------------------------------------------
-------------------------------------------------------
-- 
--                        Main
-- 
-------------------------------------------------------
-------------------------------------------------------

if checkpoint then
   local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
   assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
   print('=> Resuming model from ' .. modelPath)
   model = torch.load(modelPath)
else

    -- construct model
    model = model_temporal_BN_max_LSTM_BN_FC()
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
local criterion = {}
if opt.seqCriterion == true then
    criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
else
    criterion = nn.ClassNLLCriterion()
    -- criterion = nn.CrossEntropyCriterion()
end


print(sys.COLORS.red ..  '==> here is the network:')
print(model)

if opt.cuda == true then
    model:cuda()
    criterion:cuda()

    -- Wrap the model with DataParallelTable, if using more than one GPU
    if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            local rnn = require 'rnn'
            -- require 'TemporalDropout'
            
            -- Set the CUDNN flags
            cudnn.fastest = true
            cudnn.benchmark = true
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
    end
end

-- Export:
return
{
    model = model,
    criterion = criterion
}