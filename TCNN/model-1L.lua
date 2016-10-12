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

if opt.model == 'TCNN-1' then
   print(sys.COLORS.red ..  '==> construct 1-layer T-CNN')

   model_name = 'model_best'

   -- stage 1: conv -> ReLU -> Pooling
   model:add(nn.SpatialConvolutionMM(1,nstates[1],convsize,1,convstep,1,convpad,0))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(poolsize,1,poolstep,1))

   model:add(nn.Dropout(opt.dropout)) -- dropout

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
