-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- 2-flow
-- by pooling with size 2 ==> dim. for frame: 25 --> 12 --> 6 --> 3 --> 1

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
-- local nstate_CNN = {dimMap, dimMap} -- neuron # after convolution
local nstate_CNN_1 = {1, 1} -- neuron # after convolution
local nstate_CNN_2 = {nstate_CNN_1[1]*2, nstate_CNN_1[1]*2} -- neuron # after convolution
local nstate_CNN_3 = {nstate_CNN_2[1]*2, nstate_CNN_2[1]*2} -- neuron # after convolution
local nstate_CNN_4 = {nstate_CNN_3[1]*2, nstate_CNN_3[1]*2} -- neuron # after convolution
local nstate_FC = 1024 -- neuron # after 1st FC & 2nd FC
-- local nstate_FC_all = 1024 -- neuron # after the last FC

local convsize = {5,7} 
local convpad  = {(convsize[1]-1)/2,(convsize[2]-1)/2}
local convstep_1 = {1,1}
local convstep_2 = {2,2}
-- local convpad  = {0,(convsize[2]-convsize[1])/2}

--local poolsize = {3,4}
local poolsize_1 = 2
local poolstep_1 = poolsize_1
local poolsize_2 = 3
local poolstep_2 = poolsize_2

-- local numConvLayer = torch.log(nframeUse)/torch.log(poolsize)
-- local numConvLayer = torch.floor(nframeUse/(convsize[1]-1))

local numFlow = #convsize
-- local numFlow = 1

----------------------------------------------------------------------
local model = nn.Sequential()
local model_name = "model"

------------------
-- Architecture --
------------------
if opt.model == 'model-Conv-Inception' then
   	print(sys.COLORS.red ..  '==> construct T-CNN w/ Inception-style modules')

   	model_name = 'model_best'

   	-----------------------------
   	--  Inception-style module --
   	-----------------------------
   	local function multiFlow(inDim, outDim, convSize, convStep, convPad, poolSize, poolStep)
   		local CNN_flows = nn.ConcatTable()
	   	for n=1,numFlow do
	      	local CNN_1L = nn.Sequential()

	     --  	CNN_1L:add(nn.SpatialConvolutionMM(inDim,outDim[n],1,1,1,1,0,0)) -- 25 --> 25 
		    -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(outDim[n])) end
		    -- CNN_1L:add(nn.ReLU())
		    -- -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(outDim[n])) end

		    CNN_1L:add(nn.SpatialConvolutionMM(inDim,outDim[n],convSize[n],1,convStep[n],1,convPad[n],0)) -- 25 --> 25 
		    if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(outDim[n])) end
		    CNN_1L:add(nn.ReLU())
		    -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(outDim[n])) end
        
		    CNN_1L:add(nn.SpatialMaxPooling(poolSize,1,poolStep,1)) -- floor(dim/2)
		    -- CNN_1L:add(nn.SpatialDropout(dropout1)) -- dropout    
		    CNN_flows:add(CNN_1L)  	
	   	end
   		return CNN_flows
   	end
	
	-----------------------
	-- Main Architecture --
	-----------------------   	
    -- model:add(nn.SpatialBatchNormalization(dimMap))
   	-- stage 1: inception module: (Conv --> BN --> ReLU)*N      
   	model:add(multiFlow(dimMap,nstate_CNN_1,convsize,convstep_1,convpad,poolsize_1,poolstep_1)) -- bSize*1*4096*25 --> bSize*1*4096*12
   	model:add(nn.JoinTable(2)) 	-- merge two streams: bSize*2*4096*12
   	-- model:add(nn.SpatialDropout(dropout1)) -- dropout
   	
   	model:add(multiFlow(nstate_CNN_1[1]+nstate_CNN_1[2],nstate_CNN_2,convsize,convstep_1,convpad,poolsize_1,poolstep_1)) -- bSize*2*4096*12 --> bSize*2*4096*6
   	model:add(nn.JoinTable(2)) 	-- merge two streams: bSize*4*4096*6
	-- model:add(nn.SpatialDropout(dropout1)) -- dropout
	
	model:add(multiFlow(nstate_CNN_2[1]+nstate_CNN_2[2],nstate_CNN_3,convsize,convstep_1,convpad,poolsize_1,poolstep_1)) -- bSize*4*4096*6 --> bSize*4*4096*3
   	model:add(nn.JoinTable(2)) 	-- merge two streams: bSize*8*4096*3
   	-- model:add(nn.SpatialDropout(dropout1)) -- dropout
   	
   	-- model:add(nn.SpatialAveragePooling(poolsize_2,1,poolstep_2,1)) -- bSize*8*4096*3 --> bSize*8*4096*1

	model:add(multiFlow(nstate_CNN_3[1]+nstate_CNN_3[2],nstate_CNN_4,convsize,convstep_1,convpad,poolsize_2,poolstep_2)) -- bSize*8*4096*3 --> bSize*8*4096*1
   	model:add(nn.JoinTable(2)) 	-- merge two streams: bSize*16*4096*1
   	
   	model:add(nn.SpatialDropout(dropout1)) -- dropout

   	-- stage 2: linear -> ReLU
	local ninputFC = (nstate_CNN_4[1]*numFlow)*nfeature
    model:add(nn.Reshape(ninputFC))
    model:add(nn.Linear(ninputFC,nstate_FC))
    -- if opt.batchNormalize == 'Yes' then model:add(nn.BatchNormalization(nstate_FC)) end
    model:add(nn.BatchNormalization(nstate_FC))
    model:add(nn.ReLU())
    -- if opt.batchNormalize == 'Yes' then model:add(nn.BatchNormalization(nstate_FC)) end

   	model:add(nn.Linear(nstate_FC,noutputs)) -- output layer (output: 101 prediction probability)	
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
