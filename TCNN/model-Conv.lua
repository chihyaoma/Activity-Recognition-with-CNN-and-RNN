-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- single-flow
-- reduce dim. by convolution (25 --> 23 --> 21 --> ... --> 1)

-- TODO:
-- 1. change nstate
-- 2. change convsize, convstep, poolsize, poolstep

-- modified by Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 01/14/2017

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
local noutputs = #data.classes

-- input: 
local frameSkip = 0
local nframeAll = data.trainData.data:size(3)
local nframeUse = nframeAll - frameSkip
local nfeature = data.trainData.data:size(2)
local bSize = tonumber(opt.batchSize)
local dropout1 = tonumber(opt.dropout1)
local dropout2 = tonumber(opt.dropout2)
local dimMap = 1

-- hidden units, filter sizes  	
---- 2-flow	
local nstate_CNN = dimMap -- neuron # after convolution
local nstate_FC = 1024 -- neuron # after 1st FC
-- local nstate_FC_all = 1024 -- neuron # after the last FC

local convsize = 3
-- local convpad  = (convsize-1)/2 -- size don't change after convolution
local convpad  = 0
local convstep_1 = 1

-- local poolsize_1 = 2
-- local poolstep_1 = poolsize_1
-- local poolsize_2 = 3
-- local poolstep_2 = poolsize_2

-- local numConvLayer = torch.log(nframeUse)/torch.log(poolsize)
local numConvLayer = torch.floor(nframeUse/(convsize-1))

-- local numFlow = #convsize
local numFlow = 1

----------------------------------------------------------------------
local model = nn.Sequential()
local model_name = "model"

-----------------------
-- Main Architecture --
-----------------------
if opt.model == 'model-Conv' then
   	print(sys.COLORS.red ..  '==> construct T-CNN w/o dropout and pooling')

   	model_name = 'model_best'

   	-------------------------------------
   	--  Single-Flow Multi-Conv CNN  --
   	-------------------------------------
   	local CNN_flows = nn.ConcatTable()

   	local noutput_1L = torch.zeros(numFlow)
   	for n=1,numFlow do
      	-- local ninputFC = nstate_CNN[n]*nfeature*torch.floor(nframeUse/torch.pow(poolsize,numConvLayer)) -- temporal kernel
      	-- local ninputFC = nstate_CNN[n]*nfeature*(nframeUse-(convsize[1]-1)*numConvLayer) -- temporal kernel
      	local ninputFC = nstate_CNN*nfeature

      	local CNN_1L = nn.Sequential()

      	-- stage 0: input --> first layer
      	CNN_1L:add(nn.SpatialConvolutionMM(dimMap,nstate_CNN,convsize,1,convstep,1,convpad,0)) -- 25 --> 23
		if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN)) end
		CNN_1L:add(nn.ReLU())
	    -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN[n])) end
	    -- CNN_1L:add(nn.SpatialDropout(dropout1)) -- dropout

      	-- stage 1: (conv --> BN --> ReLU)*(numConvLayer-1)
      	for m=1,numConvLayer-1 do
      		CNN_1L:add(nn.SpatialConvolutionMM(nstate_CNN,nstate_CNN,convsize,1,convstep,1,convpad,0)) -- size - 2 
		    if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN)) end
		    CNN_1L:add(nn.ReLU())
	    	-- -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN[n])) end
	    	-- CNN_1L:add(nn.SpatialDropout(dropout1)) -- dropout
      	end
	    CNN_1L:add(nn.SpatialDropout(dropout1)) -- dropout
	    
      	-- stage 2: linear -> ReLU
      	CNN_1L:add(nn.Reshape(ninputFC))
      	CNN_1L:add(nn.Linear(ninputFC,nstate_FC))
      	if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.BatchNormalization(nstate_FC)) end
      	CNN_1L:add(nn.ReLU())
		-- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.BatchNormalization(nstate_FC)) end
      	-- CNN_1L:add(nn.Dropout(dropout2)) -- dropout
      	
    	CNN_flows:add(CNN_1L)
   		noutput_1L[n] = nstate_FC
   		-- noutput_1L[n] = ninputFC
   	end
      
   	model:add(CNN_flows)

   	------------------------------

   	
   	--  Combine multiple flows  --
   	------------------------------
   	model:add(nn.JoinTable(2)) -- merge two streams: bSizex(dim1+dim2+dim3)
   	-- model:add(nn.Dropout(opt.dropout)) -- dropout
   	local ninputFC_all = noutput_1L:sum()
   	model:add(nn.Linear(ninputFC_all,noutputs)) -- output layer (output: 101 prediction probability)	
   	-- stage 4 : log probabilities
	model:add(nn.LogSoftMax())

end
   
-- Loss: NLL
loss = nn.ClassNLLCriterion()

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the network:')
print(model)
print('Multi-Flow method: '..opt.typeMF)

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
