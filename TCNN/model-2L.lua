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
local nframeAll = data.nframeAll
local nframeUse = nframeAll - frameSkip
local nfeature = 4096

-- hidden units, filter sizes (for ConvNet only): 		
local nstates = {4,32,768}
local convsize = {5,13} 
local convstep = {1,1}
local convpad  = {(convsize[1]-1)/2, (convsize[2]-1)/2}
local poolsize = {2, 2}
local poolstep = {2, 2}

----------------------------------------------------------------------
local model = nn.Sequential()
local model_name = "model"

if opt.model == 'TCNN-2' then
   print(sys.COLORS.red ..  '==> construct 2-layer T-CNN')

   model_name = 'model_best'

   ---- stage 0: FC layer for inputs (not help)
   --model:add(nn.Reshape(opt.batchSize,nfeature,1,nframeUse))
   --model:add(nn.SpatialConvolutionMM(nfeature,nfeature,nframeUse,1,1,1,(nframeUse-1)/2,0))
   --model:add(nn.ReLU())
   --model:add(nn.Reshape(opt.batchSize,1,nfeature,nframeUse))

   -- stage 1: conv -> ReLU -> Pooling
   model:add(nn.SpatialConvolutionMM(1,nstates[1],convsize[1],1,convstep[1],1,convpad[1],0))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize[1],1,poolstep[1],1))

   --model:add(nn.Dropout(opt.dropout)) -- dropout

   -- stage 2: conv -> ReLU -> Pooling   
   model:add(nn.SpatialConvolutionMM(nstates[1],nstates[2],convsize[2],1,convstep[2],1,convpad[2],0))
   model:add(nn.ReLU()) 
   model:add(nn.SpatialMaxPooling(poolsize[2],1,poolstep[2],1))

   model:add(nn.Dropout(opt.dropout)) -- dropout

   -- stage 3: linear -> ReLU -> linear
   local ninputFC = nstates[2]*nfeature*torch.floor(torch.floor(nframeUse/poolsize[1])/poolsize[2]) -- temporal kernel

   model:add(nn.Reshape(ninputFC))
   model:add(nn.Linear(ninputFC,nstates[3]))
   model:add(nn.ReLU())

   --model:add(nn.Dropout(opt.dropout)) -- dropout

   model:add(nn.Linear(nstates[3],noutputs))

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
