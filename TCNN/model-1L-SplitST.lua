-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- split spatial and temporal network
-- results are not good

-- modified by Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 10/13/2016

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
local nfeature_h = nfeature/2
local bSize = opt.batchSize
local dimMap = 1

-- hidden units, filter sizes: 		
local nstate_CNN = {4, 4} -- neuron # after pooling
local nstate_FC = 2048 -- neuron # after 1st FC

local convsize = {5,5}
local convstep = {1,1}
local convpad  = {(convsize[1]-1)/2,(convsize[2]-1)/2}
local poolsize = {2,2}
local poolstep = {2,2}

local numStream = #convsize
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
local mapFC = nn.MapTable():add(nn.MapTable():add(nn.Linear(nfeature_h,nfeature_h)))
batch_FC:add(mapFC)

-- 3. convert the whole table back to the original mini-batch
local combineTable = nn.ParallelTable()
for b=1,bSize do 
   combineTable:add(nn.JoinTable(-1)) -- merge all the vectors back to map
end
batch_FC:add(combineTable)

local viewTable = nn.ParallelTable()
for b=1,bSize do 
   viewTable:add(nn.View(dimMap,nfeature_h,nframeUse)) -- (1,2048*25) --> (1,2048,25)
end
batch_FC:add(viewTable)

batch_FC:add(nn.JoinTable(1)) -- merge all the maps back to mini-batch
batch_FC:add(nn.View(bSize,dimMap,nfeature_h,nframeUse)) -- 32x1x2048x25

-----------------------
-- Main Architecture --
-----------------------
if opt.model == 'model-1L-SplitST' then
	print(sys.COLORS.red ..  '==> construct 1-layer T-CNN (split spatial & temporal network)')

   model_name = 'model_best'

   ------------------------------
   --  Two-stream CNN block    --
   ------------------------------
   local CNN_branches = nn.ParallelTable()

  	local noutput_1 = torch.zeros(numStream)
	for n=1,numStream do
		noutput_1[n] = nstate_CNN[n]*nfeature_h*torch.floor(nframeUse/poolsize[n]) -- temporal kernel

		local CNN_1L = nn.Sequential()
		-- stage 0: mini-batch FC (not used here)
		-- CNN_1L:add(batch_FC)

		-- stage 1: Conv -> ReLU -> Pooling
		CNN_1L:add(nn.SpatialConvolutionMM(dimMap,nstate_CNN[n],convsize[n],1,convstep[n],1,convpad[n],0))
		if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN[n])) end
      CNN_1L:add(nn.ReLU())
		CNN_1L:add(nn.SpatialMaxPooling(poolsize[n],1,poolstep[n],1))
				
		CNN_branches:add(CNN_1L)
   end
   	
   model:add(CNN_branches)

   ------------------------------
   --   Combine two streams    --
   ------------------------------
	model:add(nn.JoinTable(3)) -- merge two streams: bSizex1x(2048+2048)x?
	model:add(nn.Dropout(opt.dropout)) -- dropout

   	-- stage 2: linear -> ReLU -> linear

	local ninputFC = noutput_1:sum()
	model:add(nn.Reshape(ninputFC))
   	model:add(nn.Linear(ninputFC,nstate_FC)) -- add one more FC layer
   	model:add(nn.ReLU())
	model:add(nn.Linear(nstate_FC,noutputs)) -- output layer (output: 101 prediction probability)

   	-- stage 3 : log probabilities
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
