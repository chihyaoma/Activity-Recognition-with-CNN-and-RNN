-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- Create CNN and loss to optimize.
-- parameters for the Res-101 network

-- TODO:
-- 1. change nstate
-- 2. change convsize, convstep, poolsize, poolstep

-- modified by Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 10/11/2016

require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique
require 'sys'
local data = require 'data-2Stream'


if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end

----------------------------------------------------------------------
print '==> processing options'
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> define parameters')

-- 101 classes problem
local noutputs = 101

-- input: 
local frameSkip = 0
local nframeAll = data.trainData.data:size(3)
local nframeUse = nframeAll - frameSkip
local nfeature = data.trainData.data:size(2)
local bSize = tonumber(opt.batchSize)
local dropout = tonumber(opt.dropout)
local dimMap = 1

-- hidden units, filter sizes (for ConvNet only): 		
-- experiments for 25fps (25 frames)
local nstates = {4,2048}
local convsize = 5
local convstep = 1
local convpad  = (convsize-1)/2
local poolsize = 2
local poolstep = 2

----------------------------------------------------------------------
local model = nn.Sequential()
local model_name = "model"

------------------------------------
-- Full-connected layer for input --
------------------------------------
local batch_FC = nn.Sequential()
-- 1. dispart the mini-patch feature maps to lots of feature vectors
batch_FC:add(nn.SplitTable(1)) -- split the mini-batches into single feature maps

local vectorTable = nn.ParallelTable()
for b=1,bSize do 
   vectorTable:add(nn.SplitTable(-1)) -- split one feature map into feature vectors
end
batch_FC:add(vectorTable)

-- 2. duplicate thr fully-connected layer to fit the input
local mapFC = nn.MapTable():add(nn.MapTable():add(nn.Linear(nfeature,nfeature)))
batch_FC:add(mapFC)

-- 3. convert the whole table back to the original mini-batch
local combineTable = nn.ParallelTable()
for b=1,bSize do 
   combineTable:add(nn.JoinTable(-1)) -- merge all the vectors back to map
end
batch_FC:add(combineTable)

local viewTable = nn.ParallelTable()
for b=1,bSize do 
   viewTable:add(nn.View(dimMap,nfeature,nframeUse)) -- (1,4096*25) --> (1,4096,25)
end
batch_FC:add(viewTable)

batch_FC:add(nn.JoinTable(1)) -- merge all the maps back to mini-batch
batch_FC:add(nn.View(bSize,dimMap,nfeature,nframeUse)) -- 32x1x4096x25

-----------------------
-- Main Architecture --
-----------------------
if opt.model == 'model-1L' then
   print(sys.COLORS.red ..  '==> construct 1-layer T-CNN')

   model_name = 'model_best'

   -- -- stage 0: mini-batch FC
   -- model:add(batch_FC)

   -- stage 1: conv -> ReLU -> Pooling
   model:add(nn.SpatialConvolutionMM(dimMap,nstates[1],convsize,1,convstep,1,convpad,0))
   if opt.batchNormalize == 'Yes' then model:add(nn.SpatialBatchNormalization(nstates[1])) end
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,1,poolstep,1))

   model:add(nn.SpatialDropout(dropout)) -- dropout

   -- stage 2: linear -> ReLU -> linear
   local ninputFC = nstates[1]*nfeature*torch.floor(nframeUse/poolsize) -- temporal kernel

   model:add(nn.Reshape(ninputFC))
   model:add(nn.Linear(ninputFC,nstates[2]))
   model:add(nn.ReLU())

   --model:add(nn.Dropout(opt.dropout)) -- dropout

   model:add(nn.Linear(nstates[2],noutputs))

   -- stage 4 : log probabilities
   model:add(nn.LogSoftMax())

end
   
-- Loss: NLL
loss = nn.ClassNLLCriterion()

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the network:')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
   nfeature = nfeature,
   nframeAll = nframeAll,
   nframeUse = nframeUse,
   model_name = model_name,
}
