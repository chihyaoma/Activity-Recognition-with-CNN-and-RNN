----------------------------------------------------------------
-- Georgia Tech 2016 Spring
-- Deep Learning for Perception
-- Final Project: LRCN model for Video Classification
--
-- 
-- This is a testing code for implementing the RNN model with LSTM 
-- written by Chih-Yao Ma. 
-- 
-- The code will take feature vectors (from CNN model) from contiguous 
-- frames and train against the ground truth, i.e. the labeling of video classes. 
-- 
-- Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------

require 'dp'
require 'sys'

version = 1

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Model on video classification dataset using RNN with LSTM or GRU')
cmd:text('Example:')
cmd:text("RNN_LSTM.lua --cuda --useDevice 1 --progress --opt.rho 36")
cmd:text('Options:')
cmd:option('--learningRate', 5e-4, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--learningRateDecay', 1e-7, 'learningRateDecay')
cmd:option('--saturateEpoch', 400, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--weightDecay', 1e-5, 'weightDecay')
cmd:option('--optimizer', 'adam', 'Use different optimizer, e.g. sgd, adam, adamax, rmsprop for now')
cmd:option('--lrDecayEvery', 5, 'learning rate decay per epochs for optimizer adam')
cmd:option('--lrDecayFactor', 0.8, 'learning rate decay factor for optimizer adam')
cmd:option('--maxOutNorm', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--batchSize', 64, 'number of examples per batch') -- how many examples per training 
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--progress', true, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')

-- recurrent layer 
cmd:option('--lstm', true, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--gru', false, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
cmd:option('--rho', 36, 'number of frames for each video')
cmd:option('--inputSize', 2048, 'dimension of the feature vector from CNN')
cmd:option('--hiddenSize', '{1024}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
-- cmd:option('--hiddenSize', 1024, 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
cmd:option('--dropout', true, 'apply dropout after each recurrent layer')
cmd:option('--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')

-- testing process
cmd:option('--AveragePred', true, 'average the predictions from each time step per video')

-- data
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation') 

cmd:option('--featFile', '', 'file of feature vectors')
cmd:option('--targFile', '', 'file of target vector')
dname,fname = sys.fpath()
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('--plot', true, 'Plot the training and testing accuracy')

cmd:text()
opt = cmd:parse(arg or {})

opt.rundir = cmd:string(fname:gsub('.lua',''), opt, {dir=true})
paths.mkdir(opt.rundir)

-- create log file
cmd:log(opt.rundir .. '/log.txt', opt)

opt.hiddenSize = dp.returnString(opt.hiddenSize)
if not opt.silent then
   table.print(opt)
end

-- type:
if opt.cuda == true then
   print(sys.COLORS.red ..  '==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')

local data  = require 'data'
local train = require 'train'
local test  = require 'test'

------------------------------------------------------------
-- Run
------------------------------------------------------------
-- initialize bestAcc
bestAcc = 0

print(sys.COLORS.red .. '==> training!')


for iteration = 1, opt.maxEpoch do

   -- Begin training process
   train(data.TrainData, data.TrainTarget)

   -- Begin testing with trained model
	test(data.TestData, data.TestTarget)

end

--[[
local answer
repeat
   io.write("continue with this operation (y/n)? ")
   io.flush()
   answer=io.read()
until answer=="y" or answer=="n"
--]]
