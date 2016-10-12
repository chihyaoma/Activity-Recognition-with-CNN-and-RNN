-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- Train a ConvNet on UCF101

-- TODO:
-- 1. cross-validation
-- 2. 

-- modified by Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 10/11/2016


require 'pl'
require 'trepl'
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
   -z,--sourcePath         (default workstation) source path (local | workstation)
   -r,--learningRate       (default 1e-3)        learning rate
   -l,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples)
   -w,--weightDecay        (default 1e-5)        L2 penalty on the weights
   -m,--momentum           (default 0.1)         momentum
   -d,--dropout            (default 0.5)         dropout amount
   -b,--batchSize          (default 32)          batch size
   -t,--threads            (default 2)           number of threads
   -p,--type               (default cuda)        float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
   -o,--save               (default results)     save directory
   -s,--splitId            (default 1)           split number
      --patches            (default all)         percentage of samples to use for testing'
      --model              (default TCNN-2)      network model (TCNN-1 | TCNN-2)
      --optMethod          (default sgd)         optimization method
      --plot               (default no)       	 plot the training and test accuracies
]]
-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
   print(sys.COLORS.red ..  '==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

----------------------------------------------------------------------
print(sys.COLORS.cyan ..  '==> load data')
local data  = require 'data-2Stream'
print(sys.COLORS.cyan ..  '==> prepare for training')
local train = require 'train'
print(sys.COLORS.cyan ..  '==> prepare for testing')
local test  = require 'test'
--
------------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')
--

for epo=1,100 do 
--while true do
	train(data.trainData)
   test(data.testData, data.classes, epo)
end
