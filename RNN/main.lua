----------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN
--  https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
--
-- 
--  This is a testing code for implementing the RNN model with LSTM 
--  written by Chih-Yao Ma. 
-- 
--  The code will take feature vectors (from CNN model) from contiguous 
--  frames and train against the ground truth, i.e. the labeling of video classes. 
-- 
--  Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------

local sys = require 'sys'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Model on video classification dataset using RNN with LSTM or GRU')
cmd:text('Example:')
cmd:text("main.lua --cuda --useDevice 1 --progress --opt.rho 48")
cmd:text('Options:')
cmd:option('--learningRate', 5e-3, 'learning rate at t=0')
cmd:option('--minLR', 1e-5, 'minimum learning rate')
cmd:option('--learningRateDecay', 0, 'learningRateDecay')
cmd:option('--saturateEpoch', 400, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--weightDecay', 1e-4, 'weightDecay')
cmd:option('--optimizer', 'sgd', 'Use different optimizer, e.g. sgd, adam, adamax, rmsprop for now')
cmd:option('--lrMethod',  'fixed',   'methods for tuning the learning rate: manual | fixed ')
cmd:option('--epochUpdateLR', 2, 'learning rate decay per epochs for optimizer adam')
cmd:option('--lrDecayFactor', 0.1, 'learning rate decay factor for optimizer adam')
cmd:option('--maxOutNorm', -1, 'max l2-norm of each layer`s output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--batchSize', 128, 'number of examples per batch') -- how many examples per training 
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--progress', true, 'print progress bar')
cmd:option('--silent', false, 'don`t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
-- recurrent layer 
cmd:option('--lstm', true, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--bn', true, 'use batch normalization. Only supported with --lstm')
cmd:option('--gru', false, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
cmd:option('--rho', 50, 'number of frames for each video')
-- cmd:option('--inputSize', 2048, 'dimension of the feature vector from CNN')
cmd:option('--hiddenSize', '{512, 256}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
cmd:option('--dropout', true, 'apply dropout after each recurrent layer')
cmd:option('--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')
-- testing process
cmd:option('--averagePred', true, 'average the predictions from each time step per video')
-- checkpoint
cmd:option('-resume', 'false',  'Path to directory containing checkpoint')
-- data
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation') 
cmd:option('--spatFeatDir', '/home/chih-yao/Downloads/', 'directory of spatial feature vectors')
cmd:option('--tempFeatDir', 'none', 'directory of temporal feature vectors (from optical flow)')

dname,fname = sys.fpath()
cmd:option('--plot', true, 'Plot the training and testing accuracy')

cmd:text()
opt = cmd:parse(arg or {})
opt.save = 'log' .. '_' .. opt.hiddenSize .. '_' .. opt.learningRate
paths.mkdir(opt.save)

-- create log file
cmd:log(opt.save .. '/log.txt', opt)

opt.hiddenSize = loadstring(" return "..opt.hiddenSize)()

-- type:
if opt.cuda == true then
	print(sys.COLORS.red ..  '==> switching to CUDA')
	require 'cunn'
	cutorch.setDevice(opt.useDevice)
	print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

-- check if rgb or flow features wanted to be used
opt.spatial = paths.dirp(opt.spatFeatDir) and true or false
opt.temporal = paths.dirp(opt.tempFeatDir) and true or false
assert(opt.spatial or opt.temporal, 'no spatial or temporal features found!')

if opt.spatial and opt.temporal then
	opt.inputSize = 4096
else
	opt.inputSize = 2048
end

------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')

-- checkpoints
checkpoints = require 'checkpoints'
-- Load previous checkpoint, if it exists
checkpoint, optimState = checkpoints.latest(opt)

local data  = require 'data-ucf101'
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
	train(data.trainData, data.trainTarget)

	-- Begin testing with trained model
	test(data.testData, data.testTarget)
end
