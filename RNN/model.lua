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


function model_temporal_BN_FC_max_LSTM_BN_FC()
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

    -- local pFC = nn.ParallelTable()
    -- if opt.fcSize ~= nil then
    --   -- for i,fcSize in ipairs(opt.fcSize) do 
    --   for i = 1, opt.numSegment do 
    --      -- add fully connected layers to fuse spatial and temporal features
    --      pFC:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
    --   end
    --   inputSize = opt.fcSize[1]
    -- end
    -- model:add(pFC)

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

    local maxTable = nn.ParallelTable()
    -- recurrent layer
    for i = 1, opt.numSegment do 
      maxTable:add(nn.Max(1, -1))
    --   maxTable:add(nn.Mean(1, -1))
    end
    model:add(maxTable)

    -- model:add(nn.Sequencer(nn.BatchNormalization(inputSize)))

    model:add(nn.Sequencer(nn.LSTM(inputSize, opt.hiddenSize[1])))
    -- model:add(nn.Sequencer(nn.LSTM(opt.hiddenSize[1], opt.hiddenSize[2])))
    model:add(nn.SelectTable(-1))
    inputSize = opt.hiddenSize[1]

    model:add(nn.BatchNormalization(inputSize))
       
    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(inputSize, nClass))

    return model
end


function model_temporal_BN_LSTM_BN_LSTM_BN_FC()
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

    -- local pFC = nn.ParallelTable()
    -- if opt.fcSize ~= nil then
    --   -- for i,fcSize in ipairs(opt.fcSize) do 
    --   for i = 1, opt.numSegment do 
    --      -- add fully connected layers to fuse spatial and temporal features
    --      pFC:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
    --   end
    --   inputSize = opt.fcSize[1]
    -- end
    -- model:add(pFC)

    -- setup two LSTM cells with different dimensions
    local lstmTable = nn.ParallelTable()
    -- recurrent layer
    for i = 1, opt.numSegment do 
      lstmTable:add(nn.Sequencer(nn.LSTM(inputSize, opt.hiddenSize[1])))
    end
    model:add(lstmTable)
    inputSize = opt.hiddenSize[1]

    -- output layer
    local s1 = nn.ParallelTable()
    for i = 1, opt.numSegment do 
      -- select the last output from each of the LSTM cells
      s1:add(nn.SelectTable(-1))
    end
    model:add(s1)

    local pBN1 = nn.ParallelTable()
    for i = 1, opt.numSegment do 
        pBN1:add(nn.BatchNormalization(inputSize))
    end
    model:add(pBN1)


    model:add(nn.Sequencer(nn.LSTM(inputSize, opt.hiddenSize[2])))
    model:add(nn.SelectTable(-1))
    inputSize = opt.hiddenSize[2]

    model:add(nn.BatchNormalization(inputSize))
       
    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(inputSize, nClass))

    return model
end

function model_temporal_pooling_BN_FC_LSTM_BN_FC()
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

    local pFC = nn.ParallelTable()
    if opt.fcSize ~= nil then
      -- for i,fcSize in ipairs(opt.fcSize) do 
      for i = 1, opt.numSegment do 
         -- add fully connected layers to fuse spatial and temporal features
         pFC:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
      end
      inputSize = opt.fcSize[1]
    end
    model:add(pFC)

    -- setup two LSTM cells with different dimensions
    local lstmTable = nn.ParallelTable()
    -- recurrent layer
    for i = 1, opt.numSegment do 
      lstmTable:add(nn.Sequencer(nn.LSTM(inputSize, opt.hiddenSize[1])))
    end
    model:add(lstmTable)

    -- output layer
    local s1 = nn.ParallelTable()
    for i = 1, opt.numSegment do 
      -- select the last output from each of the LSTM cells
      s1:add(nn.SelectTable(-1))
    end

    model:add(s1)

    model:add(nn.JoinTable(2))
    model:add(nn.BatchNormalization(opt.numSegment*opt.hiddenSize[1]))
       
    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(opt.numSegment*opt.hiddenSize[1], nClass))

    return model
end


function model_temporal_BN_FC_maxpooling_BN_FC()
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

    local pFC = nn.ParallelTable()
    if opt.fcSize ~= nil then
      -- for i,fcSize in ipairs(opt.fcSize) do 
      for i = 1, opt.numSegment do 
         -- add fully connected layers to fuse spatial and temporal features
         pFC:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
      end
      inputSize = opt.fcSize[1]
    end
    model:add(pFC)

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

    local maxTable = nn.ParallelTable()
    -- recurrent layer
    for i = 1, opt.numSegment do 
      maxTable:add(nn.Max(1, -1))
    --   maxTable:add(nn.Mean(1, -1))
    end
    model:add(maxTable)

    model:add(nn.JoinTable(2))
    model:add(nn.BatchNormalization(opt.numSegment*inputSize))
       
    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(opt.numSegment*inputSize, nClass))

    return model
end

function model_temporal_pooling_BN_FC_LSTMmax_max_BN_FC()
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

    local pFC = nn.ParallelTable()
    if opt.fcSize ~= nil then
      -- for i,fcSize in ipairs(opt.fcSize) do 
      for i = 1, opt.numSegment do 
         -- add fully connected layers to fuse spatial and temporal features
         pFC:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
      end
      inputSize = opt.fcSize[1]
    end
    model:add(pFC)

    -- setup two LSTM cells with different dimensions
    local lstmTable = nn.ParallelTable()
    -- recurrent layer
    for i = 1, opt.numSegment do 
      lstmTable:add(nn.Sequencer(nn.LSTM(inputSize, opt.hiddenSize[1])))
    end
    model:add(lstmTable)

    -- output layer
    local s1 = nn.ParallelTable()
    for i = 1, opt.numSegment do 
      s1:add(nn.MapTable(nn.Unsqueeze(1)))
    end
    model:add(s1)

    local s2 = nn.ParallelTable()
    for i = 1, opt.numSegment do 
      s2:add(nn.JoinTable(1))
    end
    model:add(s2)

    model:add(nn.JoinTable(1))

    model:add(nn.Max(1, -1))
    -- --   model:add(nn.Mean(1, -1))

    model:add(nn.BatchNormalization(opt.hiddenSize[1]))
       
    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(opt.hiddenSize[1], nClass))

    return model
end

function model_temporal_pooling_BN_FC_LSTM_max_BN_FC()
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

    local pFC = nn.ParallelTable()
    if opt.fcSize ~= nil then
      -- for i,fcSize in ipairs(opt.fcSize) do 
      for i = 1, opt.numSegment do 
         -- add fully connected layers to fuse spatial and temporal features
         pFC:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
      end
      inputSize = opt.fcSize[1]
    end
    model:add(pFC)

    -- setup two LSTM cells with different dimensions
    local lstmTable = nn.ParallelTable()
    -- recurrent layer
    for i = 1, opt.numSegment do 
      lstmTable:add(nn.Sequencer(nn.LSTM(inputSize, opt.hiddenSize[1])))
    end
    model:add(lstmTable)

    -- output layer
    local s1 = nn.ParallelTable()
    for i = 1, opt.numSegment do 
      -- select the last output from each of the LSTM cells
      s1:add(nn.SelectTable(-1))
    end
    model:add(s1)

    model:add(nn.MapTable(nn.Unsqueeze(1)))
    model:add(nn.JoinTable(1))

    model:add(nn.Max(1, -1))
    -- --   model:add(nn.Mean(1, -1))

    model:add(nn.BatchNormalization(opt.hiddenSize[1]))
       
    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(opt.hiddenSize[1], nClass))

    return model
end


function model_triData_BN_3FC_3LSTM_Avg_CAdd()
    -- Video Classification model
    local model = nn.Sequential()

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


    local pBN = nn.ParallelTable()
    for i=1,#opt.hiddenSize do 
        pBN:add(nn.Sequencer(nn.BatchNormalization(inputSize)))
    end
    model:add(pBN)

    local pFC = nn.ParallelTable()

    if opt.fcSize ~= nil then
      -- for i,fcSize in ipairs(opt.fcSize) do 
      for i=1,#opt.hiddenSize do 
         -- add fully connected layers to fuse spatial and temporal features
         pFC:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
      end
      inputSize = opt.fcSize[1]
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
        -- select the last output from each of the LSTM cells
        p2:add(nn.BatchNormalization(hiddenSize))
    end

    local p3 = nn.Identity()
    if opt.dropout > 0 then 
        p3 = nn.ParallelTable()
        for i in ipairs(opt.hiddenSize) do 
            -- Dropout layer
            p3:add(nn.Dropout(opt.dropout))
        end
    end

    local p4 = nn.ParallelTable()
    for i,hiddenSize in ipairs(opt.hiddenSize) do
      -- full-connected layers for each LSTM output
      p4:add(nn.Linear(hiddenSize, nClass))
    end

    -- concat the prediction from all FC layers
    model:add(p1):add(p2):add(p3):add(p4)
    model:add(nn.JoinTable(1))

    return model
end

function model_FC_LSTM_CAdd()
    -- Video Classification model
    local model = nn.Sequential()
    local inputSize = opt.inputSize

    -- input layer 
    model:add(nn.SplitTable(3,1)) -- tensor to table of tensors
    model:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
    inputSize = opt.fcSize[1]

    -- setup two LSTM cells with different dimensions
    local lstmTable = nn.ConcatTable()
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
        p2:add(nn.BatchNormalization(hiddenSize))
    end

    local p3 = nn.Identity()
    if opt.dropout > 0 then 
        p3 = nn.ParallelTable()
        for i in ipairs(opt.hiddenSize) do 
            -- Dropout layer
            p3:add(nn.Dropout(opt.dropout))
        end
    end

    local p4 = nn.ParallelTable()
    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        -- full-connected layers for each LSTM output
        p4:add(nn.Linear(hiddenSize, nClass))
    end

    model:add(p1):add(p2):add(p3):add(p4)
    model:add(nn.CAddTable())

    return model
end


function model_twostream_seqLSTM_FC()
    local model = nn.Sequential()
    local inputSize = opt.inputSize

    model:add(nn.View(48,2048,25,2))
    inputSize = inputSize/2

    -- model:add(nn.SplitTable(1,1)) -- tensor to table of tensors

    
    for i,hiddenSize in ipairs(opt.hiddenSize) do 

        local splitTable = nn.ParallelTable()
        splitTable:add(nn.SplitTable(3,1)) -- tensor to table of tensors
        splitTable:add(nn.SplitTable(3,1)) -- tensor to table of tensors
        -- model:add(splitTable)
        -- recurrent layer
        local lstmTable = nn.ParallelTable()
        lstmTable:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
        lstmTable:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
        
        -- model:add(lstmTable)
        inputSize = hiddenSize
        fcInputSize = hiddenSize*2
    end

    error('spatial and temporal feature issue need to be fixed ')

    local s1 = nn.ParallelTable()
    s1:add(nn.SelectTable(-1))
    s1:add(nn.SelectTable(-1))

    -- model:add(s1)
    -- model:add(nn.JoinTable(2))
    
    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    -- model:add(nn.Linear(fcInputSize, nClass))

    return model
end

function model_stackedSeqLSTMP_FC()
    local model = nn.Sequential()
    local inputSize = opt.inputSize
    -- recurrent layer
    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        -- seqlstm = nn.SeqLSTM(inputSize, hiddenSize)
        seqlstm = nn.SeqLSTMP(inputSize, opt.projSize, hiddenSize)
        model:add(seqlstm)
        inputSize = hiddenSize
    end

    model:add(nn.Select(1, -1))

    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(inputSize, nClass))
    
    return model
end

function model_seqLSTMP_FC()
    -- Video Classification model
    local model = nn.Sequential()
    local inputSize = opt.inputSize
    local fcInputSize = 0

    -- input layer 
    -- model:add(nn.SplitTable(3,1)) -- tensor to table of tensors

    -- model:add(nn.Reshape(opt.batchSize, opt.rho, opt.fcSize, true))
    -- model:add(nn.Transpose(2,3))

    local fcTable = nn.ConcatTable()
    -- recurrent layer
    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        fcTable:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))   
    end
    -- model:add(fcTable)

    local joinTable = nn.ParallelTable()
    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        joinTable:add(nn.JoinTable(1))   
    end
    -- model:add(joinTable)

    local viewTable = nn.ParallelTable()
    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        viewTable:add(nn.View(-1, opt.batchSize, opt.fcSize[1]))   
    end
    -- model:add(viewTable)


    -- setup LSTM cells with different dimensions
    -- local lstmTable = nn.ParallelTable()
    local lstmTable = nn.ConcatTable()
    -- recurrent layer
    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        -- seqlstm = nn.SeqLSTM(inputSize, hiddenSize)
        seqlstm = nn.SeqLSTMP(inputSize, opt.projSize, hiddenSize)
        -- seqlstm.batchfirst = true
        lstmTable:add(seqlstm)
        -- lstmTable:add(nn.SeqLSTMP(inputSize, opt.projSize, hiddenSize))
        fcInputSize = fcInputSize + hiddenSize
    end
   
    model:add(lstmTable)

    -- output layer
    local s1 = nn.ParallelTable()
    for i in ipairs(opt.hiddenSize) do 
        -- select the last output from each of the LSTM cells
        -- s1:add(nn.Mean(1))
        s1:add(nn.Select(1, -1))
    end

    model:add(s1)
    model:add(nn.JoinTable(2))
    
    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(fcInputSize, nClass))
    
    return model
end

function model_BN_FC_LSTM_FC()
    -- Video Classification model
    local model = nn.Sequential()
    local inputSize = opt.inputSize
    local fcInputSize = 0

    -- input layer 
    
    model:add(nn.SplitTable(3,1)) -- tensor to table of tensors

    model:add(nn.Sequencer(nn.BatchNormalization(inputSize)))
    model:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
    inputSize = opt.fcSize[1]
    
    -- setup two LSTM cells with different dimensions
    local lstmTable = nn.ConcatTable()
    -- recurrent layer
    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        lstmTable:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
        fcInputSize = fcInputSize + hiddenSize
    end
   
    model:add(lstmTable)
    inputSize = fcInputSize

    -- output layer
    local p1 = nn.ParallelTable()
    for i in ipairs(opt.hiddenSize) do 
        -- select the last output from each of the LSTM cells
        p1:add(nn.SelectTable(-1))
    end

    local p2 = nn.ParallelTable()
    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        -- select the last output from each of the LSTM cells
        p2:add(nn.BatchNormalization(hiddenSize))
    end

    model:add(p1):add(p2)
    model:add(nn.JoinTable(2))

    -- model:add(nn.ReLU())
    
    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(inputSize, nClass))

    return model
end

function model_BN_3FC_3LSTM_BN_FC()
    -- Video Classification model
    local model = nn.Sequential()
    local inputSize = opt.inputSize
    local fcInputSize = 0

    -- input layer 
    model:add(nn.SplitTable(3,1)) -- tensor to table of tensors

    model:add(nn.Sequencer(nn.BatchNormalization(inputSize)))

    local fcTable = nn.ConcatTable()
    -- recurrent layer
    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        fcTable:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))   
    end
    model:add(fcTable)
    inputSize = opt.fcSize[1]

    -- setup two LSTM cells with different dimensions
    local lstmTable = nn.ParallelTable()
    -- recurrent layer
    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        lstmTable:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
        fcInputSize = fcInputSize + hiddenSize
    end
    model:add(lstmTable)
    inputSize = fcInputSize
    
    -- output layer
    local p1 = nn.ParallelTable()
    for i in ipairs(opt.hiddenSize) do 
        -- select the last output from each of the LSTM cells
        p1:add(nn.SelectTable(-1))
    end

    -- local p2 = nn.ParallelTable()
    -- for i,hiddenSize in ipairs(opt.hiddenSize) do 
    --     -- select the last output from each of the LSTM cells
    --     p2:add(nn.BatchNormalization(hiddenSize))
    -- end

    model:add(p1)--:add(p2)

    model:add(nn.JoinTable(2))
    model:add(nn.BatchNormalization(inputSize))
       
    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(inputSize, nClass))
    
    return model
end

function model_BN_3LSTM_FC()
    -- Video Classification model
    local model = nn.Sequential()
    local inputSize = opt.inputSize
    local fcInputSize = 0

    -- input layer 
    model:add(nn.SplitTable(3,1)) -- tensor to table of tensors

    model:add(nn.Sequencer(nn.BatchNormalization(inputSize)))

    -- local fcTable = nn.ConcatTable()
    -- -- recurrent layer
    -- for i,hiddenSize in ipairs(opt.hiddenSize) do 
    --     fcTable:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))   
    -- end
    -- model:add(fcTable)
    -- inputSize = opt.fcSize[1]

    -- setup two LSTM cells with different dimensions
    local lstmTable = nn.ConcatTable()
    -- recurrent layer
    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        lstmTable:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
        fcInputSize = fcInputSize + hiddenSize
    end
    model:add(lstmTable)
    inputSize = fcInputSize
    
    -- output layer
    local p1 = nn.ParallelTable()
    for i in ipairs(opt.hiddenSize) do 
        -- select the last output from each of the LSTM cells
        p1:add(nn.SelectTable(-1))
    end

    -- local p2 = nn.ParallelTable()
    -- for i,hiddenSize in ipairs(opt.hiddenSize) do 
    --     -- select the last output from each of the LSTM cells
    --     p2:add(nn.BatchNormalization(hiddenSize))
    -- end

    model:add(p1)--:add(p2)

    model:add(nn.JoinTable(2))
    model:add(nn.BatchNormalization(inputSize))
       
    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(inputSize, nClass))
    
    return model
end

function model_FC_stackedRNN_FC()
    -- Video Classification model
    local model = nn.Sequential()
    local inputSize = opt.inputSize

    -- input layer 
    -- model:add(nn.TemporalDropout(0.5))
    model:add(nn.SplitTable(3,1)) -- tensor to table of tensors

    model:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
    inputSize = opt.fcSize[1]

    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        
        if opt.bn then
            nn.FastLSTM.bn = true
        end
        if opt.lstm then
            model:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
            -- model:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize)))
        elseif opt.gru then
            model:add(nn.Sequencer(nn.GRU(inputSize, hiddenSize)))
        else
            error('invalid RNN cells')
        end
        -- model:add(nn.Sequencer(nn.NormStabilizer()))
        inputSize = hiddenSize
    end
    
    model:add(nn.SelectTable(-1))

    -- model:add(nn.BatchNormalization(inputSize))
    
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end

    model:add(nn.Linear(inputSize, nClass))

    return model
end

function model_BN_FC_stackedRNN_BN_FC()
    -- Video Classification model
    local model = nn.Sequential()
    local inputSize = opt.inputSize

    -- input layer 
    -- model:add(nn.TemporalDropout(0.5))
    model:add(nn.SplitTable(3,1)) -- tensor to table of tensors

    model:add(nn.Sequencer(nn.BatchNormalization(inputSize)))
    model:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
    inputSize = opt.fcSize[1]

    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        
        if opt.bn then
            nn.FastLSTM.bn = true
        end
        if opt.lstm then
            model:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
            -- model:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize)))
        elseif opt.gru then
            model:add(nn.Sequencer(nn.GRU(inputSize, hiddenSize)))
        else
            error('invalid RNN cells')
        end
        -- model:add(nn.Sequencer(nn.NormStabilizer()))
        inputSize = hiddenSize
    end
    
    model:add(nn.SelectTable(-1))

    model:add(nn.BatchNormalization(inputSize))
    
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end

    model:add(nn.Linear(inputSize, nClass))

    return model
end

function model_FC_stackedLSTM_3FC()
    
    error('unfinished...')
    -- Video Classification model
    local model = nn.Sequential()
    local inputSize = opt.inputSize

    -- input layer 
    model:add(nn.SplitTable(3,1)) -- tensor to table of tensors

    local firstLSTM = nn.Sequencer(nn.LSTM(inputSize, hiddenSize))
    
    
    -- for i,hiddenSize in ipairs(opt.hiddenSize) do 
    local concatTable = nn.ConcatTable()
    concatTable:add(nn.Identity())
    concatTable:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
    firstLSTM:add(concatTable)
    -- end

    model:add(firstLSTM)

    return model
end

function model_FCs()
    -- Video Classification model
    local model = nn.Sequential()
    local inputSize = opt.inputSize

    -- input layer 
    model:add(nn.SplitTable(3,1)) -- tensor to table of tensors

    model:add(nn.Sequencer(nn.Linear(inputSize, opt.fcSize[1])))
    -- model:add(nn.Sequencer(nn.Linear(opt.fcSize[1], 2048)))
    -- model:add(nn.Sequencer(nn.Linear(2048, nClass)))

    -- Dropout layer
    if opt.dropout > 0 then 
       model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Sequencer(nn.Linear(opt.fcSize[1], nClass)))

    -- average the prediciton from spatialRNN and temporalRNN
    model:add(nn.CAddTable())

    return model
end


function model_stackedRNN_FC()
    -- Video Classification model
    local model = nn.Sequential()
    local inputSize = opt.inputSize

    -- input layer 
    -- model:add(nn.TemporalDropout(0.5))
    model:add(nn.SplitTable(3,1)) -- tensor to table of tensors

    for i,hiddenSize in ipairs(opt.hiddenSize) do 
        
        if opt.bn then
            nn.FastLSTM.bn = true
        end
        if opt.lstm then
            model:add(nn.Sequencer(nn.LSTM(inputSize, hiddenSize)))
            -- model:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize)))
        elseif opt.gru then
            model:add(nn.Sequencer(nn.GRU(inputSize, hiddenSize)))
        else
            error('invalid RNN cells')
        end
        -- model:add(nn.Sequencer(nn.NormStabilizer()))
        inputSize = hiddenSize
    end
    
    model:add(nn.SelectTable(-1))

    -- model:add(nn.BatchNormalization(inputSize))
    
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
    model = model_temporal_BN_FC_max_LSTM_BN_FC()
    -- model:add(nn.LogSoftMax())

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
    -- criterion = nn.ClassNLLCriterion()
    criterion = nn.CrossEntropyCriterion()
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