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
local nstate_CNN = {4, 4} -- neuron # after pooling
local nstate_FC = {1024, 1024} -- neuron # after 1st FC
local nstate_FC_all = 1024 -- neuron # after the last FC

local convsize = {5,9}
local convstep = {1,1}
local convpad  = {(convsize[1]-1)/2,(convsize[2]-1)/2}
--local poolsize = {3,4}
local poolsize = {2,2}
local poolstep = poolsize

-- ---- 3-flow	
-- local nstate_CNN = {4, 4, 4} -- neuron # after pooling
-- local nstate_FC = {1024, 1024, 1024} -- neuron # after 1st FC
-- local nstate_FC_all = 1024 -- neuron # after the last FC

-- -- local convsize = {5,5,5}
-- local convsize = {5,7,13}
-- local convstep = {1,1,1}
-- local convpad  = {(convsize[1]-1)/2,(convsize[2]-1)/2,(convsize[3]-1)/2}
-- local poolsize = {2,2,2}
-- local poolstep = {2,2,2}

local numFlow = #convsize
----------------------------------------------------------------------
local model = nn.Sequential()
local model_name = "model"

-- ------------------------------------
-- -- Full-connected layer for input --
-- ------------------------------------
-- local batch_FC = nn.Sequential()
-- -- 1. dispart the mini-patch feature maps to lots of feature vectors
-- batch_FC:add(nn.SplitTable(1)) -- split the mini-batches into single feature maps

-- local vectorTable = nn.ParallelTable()
-- for b=1,bSize do 
--    vectorTable:add(nn.SplitTable(-1)) -- split one feature map into feature vectors
-- end
-- batch_FC:add(vectorTable)

-- -- 2. duplicate thr fully-connected layer to fit the input
-- -- FC layer
-- local vectorFC = nn.Sequential()
-- vectorFC:add(nn.Linear(nfeature,nfeature))
-- -- if opt.batchNormalize == 'Yes' then vectorFC:add(nn.BatchNormalization(nfeature)) end
-- -- vectorFC:add(nn.ReLU())
-- -- vectorFC:add(nn.Dropout(0.5)) -- dropout

-- local mapFC = nn.MapTable():add(nn.MapTable():add(vectorFC))
-- batch_FC:add(mapFC)

-- -- 3. convert the whole table back to the original mini-batch
-- local combineTable = nn.ParallelTable()
-- for b=1,bSize do 
--    combineTable:add(nn.JoinTable(-1)) -- merge all the vectors back to map
-- end
-- batch_FC:add(combineTable)

-- local viewTable = nn.ParallelTable()
-- for b=1,bSize do 
--    viewTable:add(nn.View(dimMap,nfeature,nframeUse)) -- (1,4096*25) --> (1,4096,25)
-- end
-- batch_FC:add(viewTable)

-- batch_FC:add(nn.JoinTable(1)) -- merge all the maps back to mini-batch
-- batch_FC:add(nn.View(bSize,dimMap,nfeature,nframeUse)) -- 32x1x4096x25

-----------------------
-- Main Architecture --
-----------------------
if opt.model == 'model-1L-MultiFlow' then
   	print(sys.COLORS.red ..  '==> construct multi-flow 1-layer T-CNN')

   	model_name = 'model_best'

   	-------------------------------------
   	--  Multi-Flow 1-layer CNN  --
   	-------------------------------------
   	local CNN_flows = nn.ConcatTable()

   	local noutput_1L = torch.zeros(numFlow)
   	for n=1,numFlow do
      	local ninputFC = nstate_CNN[n]*nfeature*torch.floor(nframeUse/poolsize[n]) -- temporal kernel

      	local CNN_1L = nn.Sequential()

         -- -- stage 0: mini-batch FC
         -- CNN_1L:add(batch_FC)

      	-- stage 1: Conv -> ReLU -> Pooling
      	CNN_1L:add(nn.SpatialConvolutionMM(dimMap,nstate_CNN[n],convsize[n],1,convstep[n],1,convpad[n],0))
      	if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN[n])) end
      	CNN_1L:add(nn.ReLU())
        --if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.SpatialBatchNormalization(nstate_CNN[n])) end
      	CNN_1L:add(nn.SpatialMaxPooling(poolsize[n],1,poolstep[n],1))
      	CNN_1L:add(nn.SpatialDropout(dropout1)) -- dropout
      
      	-- stage 2: linear -> ReLU -> linear
      	CNN_1L:add(nn.Reshape(ninputFC))
      	CNN_1L:add(nn.Linear(ninputFC,nstate_FC[n]))
      	if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.BatchNormalization(nstate_FC[n])) end
      	CNN_1L:add(nn.ReLU())
	--if opt.batchNormalize == 'Yes' then CNN_1L:add(nn.BatchNormalization(nstate_FC[n])) end
      	CNN_1L:add(nn.Dropout(dropout2)) -- dropout

      	if opt.typeMF == 'LS-Joint-LS' or opt.typeMF == 'LS-Add' or opt.typeMF == 'S-Add-L' or opt.typeMF == 'Add-LS' or opt.typeMF == 'Add-S' then
      		CNN_1L:add(nn.Linear(nstate_FC[n],noutputs)) -- output layer (output: 101 prediction probability)

		    -- stage 3 : probabilities
      		if opt.typeMF == 'LS-Joint-LS' or opt.typeMF == 'LS-Add' then
	   			CNN_1L:add(nn.LogSoftMax())
	   		elseif opt.typeMF == 'S-Add-L' then
	   			CNN_1L:add(nn.SoftMax())
	   		end
      	end
      	
    	CNN_flows:add(CNN_1L)

    	if opt.typeMF == 'LS-Joint-LS' or opt.typeMF == 'LS-Add' or opt.typeMF == 'S-Add-L' or opt.typeMF == 'Add-LS' or opt.typeMF == 'Add-S' then
      		noutput_1L[n] = noutputs
      	else
      		noutput_1L[n] = nstate_FC[n]
      	end
   	end
      
   	model:add(CNN_flows)

   	------------------------------
   	--  Combine multiple flows  --
   	------------------------------
   	if opt.typeMF == 'LS-Add' or opt.typeMF == 'S-Add-L' or opt.typeMF == 'Add-LS' or opt.typeMF == 'Add-S' then
   		model:add(nn.CAddTable())
   		-- model:add(nn.MulConstant(1/numFlow)) -- take the average

   		-- stage 4 : probabilities
      	if opt.typeMF == 'Add-LS' then
	   		model:add(nn.LogSoftMax())
	   	elseif opt.typeMF == 'Add-S' then
	   		model:add(nn.SoftMax())
  		elseif opt.typeMF == 'S-Add-L' then
  			model:add(nn.Log())
	   	end
   	elseif opt.typeMF == 'LS-Joint-LS' or opt.typeMF == 'Joint-LS' or opt.typeMF == 'Joint-S' or opt.typeMF == 'Joint-FC-LS' or opt.typeMF == 'Joint-FC-S' then
   		model:add(nn.JoinTable(2)) -- merge two streams: bSizex(dim1+dim2+dim3)
   		-- model:add(nn.Dropout(opt.dropout)) -- dropout
   		local ninputFC_all = noutput_1L:sum()
   		
   		if opt.typeMF == 'LS-Joint-LS' or opt.typeMF == 'Joint-LS' or opt.typeMF == 'Joint-S' then
	   		model:add(nn.Linear(ninputFC_all,noutputs)) -- output layer (output: 101 prediction probability)	
   		   	-- stage 4 : log probabilities
		   	if opt.typeMF == 'LS-Joint-LS' or opt.typeMF == 'Joint-LS' then
		   		model:add(nn.LogSoftMax())
	   		elseif opt.typeMF == 'Joint-S' then
	   			model:add(nn.SoftMax())
	   		end
		   	
   		elseif opt.typeMF == 'Joint-FC-LS' or opt.typeMF == 'Joint-FC-S' then
   		   	model:add(nn.Reshape(ninputFC_all))
		   	model:add(nn.Linear(ninputFC_all,nstate_FC_all)) -- add one more FC layer
		   	model:add(nn.ReLU())

		   	model:add(nn.Dropout(dropout)) -- dropout

		   	model:add(nn.Linear(nstate_FC_all,noutputs)) -- output layer (output: 101 prediction probability)

		   	-- stage 4 : log probabilities
		   	if opt.typeMF == 'Joint-FC-LS' then
		   		model:add(nn.LogSoftMax())
	   		elseif opt.typeMF == 'Joint-FC-S' then
	   			model:add(nn.SoftMax())
	   		end
		   	
   		end


   	end
   	
   	


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
