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
local nstate_CNN_2 = {nstate_CNN_1[1], nstate_CNN_1[1]} -- neuron # after convolution
local nstate_CNN_3 = {nstate_CNN_2[1], nstate_CNN_2[1]} -- neuron # after convolution
local nstate_CNN_4 = {nstate_CNN_3[1], nstate_CNN_3[1]} -- neuron # after convolution
local nstate_FC = {1024, 1024} -- neuron # after 1st FC & 2nd FC
-- local nstate_FC_alil = 1024 -- neuron # after the last FC

local convsize_1 = {1,9} 
local convpad_1  = {(convsize_1[1]-1)/2,(convsize_1[2]-1)/2}
local convsize_2 = {5,7} 
local convpad_2  = {(convsize_2[1]-1)/2,(convsize_2[2]-1)/2}
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

--local numFlow = #convsize_2
local numFlow = 1

----------------------------------------------------------------------
local model = nn.Sequential()
local model_name = "model"

-----------------------
-- Main Architecture --
-----------------------
if opt.model == 'model-Conv-VGG' then
   	print(sys.COLORS.red ..  '==> construct T-CNN w/o dropout and pooling')

   	model_name = 'model_best'

   	-------------------------------------
   	--  Multi-Flow 1-layer CNN  --
   	-------------------------------------
   	local CNN_flows = nn.ConcatTable()

   	local noutput_1L = torch.zeros(numFlow)
   	for n=1,numFlow do
      	-- local ninputFC = nstate_CNN[n]*nfeature*torch.floor(nframeUse/torch.pow(poolsize,numConvLayer)) -- temporal kernel
      	-- local ninputFC = nstate_CNN[n]*nfeature*(nframeUse-(convsize[1]-1)*numConvLayer) -- temporal kernel
      	local ninputFC = nstate_CNN_4[n]*nfeature

      	local CNN_1L = nn.Sequential()

	    CNN_1L:add(nn.SpatialConvolutionMM(dimMap,nstate_CNN_1[n],convsize_2[n],1,convstep_1[n],1,convpad_2[n],0)) -- 25 --> 25 
	    -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN_1[n])) end
       CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN_1[n])) 
	    CNN_1L:add(nn.ReLU())
	    -- -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN[n])) end
	    CNN_1L:add(nn.SpatialMaxPooling(poolsize_1,1,poolstep_1,1)) -- 25 --> 12
	    -- CNN_1L:add(nn.SpatialDropout(dropout1)) -- dropout

	    CNN_1L:add(nn.SpatialConvolutionMM(nstate_CNN_1[n],nstate_CNN_2[n],convsize_2[n],1,convstep_1[n],1,convpad_2[n],0)) -- 12 --> 12 
	    if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN_2[n])) end
       
	    CNN_1L:add(nn.ReLU())
	    -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN[n])) end
	    CNN_1L:add(nn.SpatialMaxPooling(poolsize_1,1,poolstep_1,1)) -- 12 --> 6
	    -- CNN_1L:add(nn.SpatialDropout(dropout1)) -- dropout

	    CNN_1L:add(nn.SpatialConvolutionMM(nstate_CNN_2[n],nstate_CNN_3[n],convsize_2[n],1,convstep_1[n],1,convpad_2[n],0)) -- 6 --> 6
	    -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN_3[n])) end
       CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN_1[n])) 
	    CNN_1L:add(nn.ReLU())
	    -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN[n])) end
	    -- CNN_1L:add(nn.SpatialConvolutionMM(nstate_CNN_3[n],nstate_CNN_3[n],convsize_2[n],1,convstep_1[n],1,convpad_2[n],0)) -- 6 --> 6
	    -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN_3[n])) end
	    -- CNN_1L:add(nn.ReLU())
	    -- -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN[n])) end
	    CNN_1L:add(nn.SpatialMaxPooling(poolsize_1,1,poolstep_1,1)) -- 6 --> 3
	    -- CNN_1L:add(nn.SpatialDropout(dropout1)) -- dropout

 		 CNN_1L:add(nn.SpatialConvolutionMM(nstate_CNN_3[n],nstate_CNN_4[n],convsize_2[n],1,convstep_1[n],1,convpad_2[n],0)) -- 3 --> 3
	    if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN_4[n])) end
       -- CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN_1[n])) 
	    CNN_1L:add(nn.ReLU())
	    -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN[n])) end
	    -- CNN_1L:add(nn.SpatialConvolutionMM(nstate_CNN_4[n],nstate_CNN_4[n],convsize_2[n],1,convstep_1[n],1,convpad_2[n],0)) -- 3 --> 3
	    -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN_4[n])) end
	    -- CNN_1L:add(nn.ReLU())
	    -- -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN[n])) end
	    CNN_1L:add(nn.SpatialMaxPooling(poolsize_2,1,poolstep_2,1)) -- 3 --> 1
	    CNN_1L:add(nn.SpatialDropout(dropout1)) -- dropout

	    -- CNN_1L:add(nn.SpatialAveragePooling(poolsize_2,1,poolstep_2,1)) -- 3 --> 1
	    -- CNN_1L:add(nn.SpatialDropout(dropout1)) -- dropout
      	

      	-- stage 2: linear -> ReLU
      	CNN_1L:add(nn.Reshape(ninputFC))
      	CNN_1L:add(nn.Linear(ninputFC,nstate_FC[n]))
      	-- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.BatchNormalization(nstate_FC[n])) end
         CNN_1L:add(nn.BatchNormalization(nstate_FC[n])) 
      	CNN_1L:add(nn.ReLU())
		-- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.BatchNormalization(nstate_FC[n])) end
      --    CNN_1L:add(nn.Linear(nstate_FC[n],nstate_FC[n]))
      --    if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.BatchNormalization(nstate_FC[n])) end
      --    CNN_1L:add(nn.ReLU())
      -- -- if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.BatchNormalization(nstate_FC[n])) end
      	-- CNN_1L:add(nn.Dropout(dropout2)) -- dropout
      	
    	CNN_flows:add(CNN_1L)
   		noutput_1L[n] = nstate_FC[n]
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
