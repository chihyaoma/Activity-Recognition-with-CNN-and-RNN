-- Georgia Institute of Technology 

-- 2-flow
-- by pooling with size 2 ==> dim. for frame: 25 --> 12 --> 6 --> 3 --> 1

-- TODO:
-- use conv ==> 16*4096*1 --> 1*4096*1

-- modified by Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 02/26/2017

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
local nstate_FC1 = 1024
--local nstate_FC2 = 1024 

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

local numFlow = #convsize
-- local numFlow = 1

----------------------------------------------------------------------
local model = nn.Sequential()
local model_name = "model"

------------------
-- Architecture --
------------------
if opt.model == 'model-Conv-Inception-v1' then
   	print(sys.COLORS.red ..  '==> construct T-CNN w/ Inception-style modules')

   	model_name = 'model_best'

   	-----------------------------
   	--  Inception-style module --
   	-----------------------------
   	local function multiFlow(inDim, outDim, convSize, convStep, convPad, poolSize, poolStep)
   		local CNN_multiflow = nn.Sequential()
		local CNN_flows = nn.ConcatTable()
	   	for n=1,numFlow do
	      	local CNN_1L = nn.Sequential()

		    CNN_1L:add(nn.SpatialConvolutionMM(inDim,outDim[n],convSize[n],1,convStep[n],1,convPad[n],0)) -- 25 --> 25 
		    if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(outDim[n])) end
		    CNN_1L:add(nn.ReLU())
		    -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(outDim[n])) end
        
		    CNN_1L:add(nn.SpatialMaxPooling(poolSize,1,poolStep,1)) -- floor(dim/2)
		    -- CNN_1L:add(nn.SpatialDropout(dropout1)) -- dropout    
		    CNN_flows:add(CNN_1L)  	
	   	end
		CNN_multiflow:add(CNN_flows)
		CNN_multiflow:add(nn.JoinTable(2))
--   		CNN_multiflow:add(nn.SpatialMaxPooling(poolSize,1,poolStep,1)) -- floor(dim/2)
		return CNN_multiflow
   	end
	
	-----------------------
	-- Main Architecture --
	-----------------------   	
   	---- stage 0: change the dimension (worse performance)
	--model:add(nn.Transpose({2,3})) -- bSize*1*4096*25 --> bSize*4096*1*25
	--model:add(nn.SpatialConvolutionMM(nfeature,nstate_FC1,1,1,1,1,0,0)) -- 4096 --> 1024 
	--model:add(nn.Transpose({2,3})) -- bSize*nstate_FC1*1*25 --> bSize*1*nstate_FC1*25
	
	-- stage 1: inception module: (Conv --> BN --> ReLU)*N      
   	model:add(multiFlow(dimMap,nstate_CNN_1,convsize,convstep_1,convpad,poolsize_1,poolstep_1)) -- bSize*1*4096*25 --> bSize*2*4096*12
	-- model:add(nn.SpatialDropout(dropout1)) -- dropout
   	
   	model:add(multiFlow(nstate_CNN_1[1]*2,nstate_CNN_2,convsize,convstep_1,convpad,poolsize_1,poolstep_1)) -- bSize*2*4096*12 --> bSize*4*4096*6
	-- model:add(nn.SpatialDropout(dropout1)) -- dropout
	
	model:add(multiFlow(nstate_CNN_2[1]*2,nstate_CNN_3,convsize,convstep_1,convpad,poolsize_1,poolstep_1)) -- bSize*4*4096*6 --> bSize*8*4096*3
	-- model:add(nn.SpatialDropout(dropout1)) -- dropout
   	
	model:add(multiFlow(nstate_CNN_3[1]*2,nstate_CNN_4,convsize,convstep_1,convpad,poolsize_2,poolstep_2)) -- bSize*8*4096*3 --> bSize*16*4096*1
   	
	--model:add(nn.SpatialMaxPooling(poolsize_1,1,poolstep_1,1))	

	-- stage 2: reduce the dimension
	outDimConv1 = 4
	outDimConv2 = 2
	outDimConv3 = 1
--	outDimConv4 = 1	

	model:add(nn.SpatialConvolutionMM(16,outDimConv1,1,1,1,1,0,0)) -- 16 --> outDimConv 
        model:add(nn.SpatialBatchNormalization(outDimConv1))
        model:add(nn.ReLU())
	
	model:add(nn.SpatialConvolutionMM(outDimConv1,outDimConv2,1,1,1,1,0,0)) -- 16 --> outDimConv
        model:add(nn.SpatialBatchNormalization(outDimConv2))
        model:add(nn.ReLU())

	model:add(nn.SpatialConvolutionMM(outDimConv2,outDimConv3,1,1,1,1,0,0)) -- 16 --> outDimConv
        model:add(nn.SpatialBatchNormalization(outDimConv3))
        model:add(nn.ReLU())

--	model:add(nn.SpatialConvolutionMM(outDimConv3,outDimConv4,1,1,1,1,0,0)) -- 16 --> outDimConv
--        model:add(nn.SpatialBatchNormalization(outDimConv4))
--        model:add(nn.ReLU())

--	model:add(nn.Transpose({2,4}))	
--	model:add(nn.SpatialMaxPooling(16,1,16,1))
--	model:add(nn.Transpose({2,4}))

   	model:add(nn.SpatialDropout(dropout1)) -- dropout

   	-- stage 3: linear -> ReLU
	local ninputFC = nfeature*outDimConv3
--	local ninputFC = nfeature
	model:add(nn.Reshape(ninputFC))
	model:add(nn.Linear(ninputFC,nstate_FC1))
	model:add(nn.BatchNormalization(nstate_FC1))
   	model:add(nn.ReLU())
	
	model:add(nn.Dropout(dropout2))	

   	model:add(nn.Linear(nstate_FC1,noutputs)) -- output layer (output: 101 prediction probability)	
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
