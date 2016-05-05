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
-- Last updated: 04/26/2016


require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique
require 'sys'


if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end

----------------------------------------------------------------------
print '==> processing options'
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> define parameters')

-- 101 classes problem
local noutputs = 101

-- input dimensions: 
local nframe = 48
local nfeature = 2048

-- hidden units, filter sizes (for ConvNet only):
--local nstates = {20,50,250} 		-- exp. 1
local nstates = {16,32,256}      -- exp. 2, 3
local convsize = {3, 11}        	-- exp. 1, 2, 3
local convstep = {1, 1}
local convpad  = {(convsize[1]-1)/2, (convsize[2]-1)/2}
local poolsize = {4, 2} 			-- exp. 1, 2
local poolstep = {4, 2} 			-- exp. 1, 2
--local poolsize = {2, 2}          -- exp. 3
--local poolstep = {2, 2}          -- exp. 3

----------------------------------------------------------------------
local classifier = nn.Sequential()
local model = nn.Sequential()
local model_name = "model.net"

if opt.model == 'CNN' then
   print(sys.COLORS.red ..  '==> construct CNN')
   ---- TODO --------
   --Create a CNN network as mentioned in the write-up
   --followed by a 2 layer fully connected layers
   --Use ReLU as yoru activations
   model_name = 'CNN.net'

   local CNN = nn.Sequential()
   
   -- stage 1: conv -> ReLU -> Pooling
   CNN:add(nn.SpatialConvolutionMM(1,nstates[1],1,convsize[1],1,convstep[1],0,convpad[1])) 
   local sizeConv1_w = nfeature
   local sizeConv1_h = (nframe-convsize[1]+2*convpad[1])/convstep[1]+1

   CNN:add(nn.ReLU())
   CNN:add(nn.SpatialMaxPooling(1,poolsize[1],1,poolstep[1])) 
   local sizePool1_w = sizeConv1_w
   local sizePool1_h = (sizeConv1_h-poolsize[1])/poolstep[1]+1

   --CNN:add(nn.Dropout(opt.dropout)) -- dropout

   -- stage 2: conv -> ReLU -> Pooling   
   CNN:add(nn.SpatialConvolutionMM(nstates[1],nstates[2],1,convsize[2],1,convstep[2],0,convpad[2]))
   local sizeConv2_w = sizePool1_w
   local sizeConv2_h = (sizePool1_h-convsize[2]+2*convpad[2])/convstep[2]+1

   CNN:add(nn.ReLU()) 
   CNN:add(nn.SpatialMaxPooling(1,poolsize[2],1,poolstep[2])) 
   local sizePool2_w = sizeConv2_w
   local sizePool2_h = (sizeConv2_h-poolsize[2])/poolstep[2]+1

   CNN:add(nn.Dropout(opt.dropout)) -- dropout

   -- stage 3: linear -> ReLU -> linear
   local ninputFC = sizePool2_w*sizePool2_h*nstates[2] -- exp. 1, 2, 3

   CNN:add(nn.Reshape(ninputFC))
   CNN:add(nn.Linear(ninputFC,nstates[3]))
   CNN:add(nn.ReLU())

   --CNN:add(nn.Dropout(opt.dropout)) -- dropout

   CNN:add(nn.Linear(nstates[3],noutputs))

   -- stage 4 : log probabilities
   CNN:add(nn.LogSoftMax())

   model:add(CNN)

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
   nfeature = 2048,
   nframe = 48,
   model_name = model_name,
}
