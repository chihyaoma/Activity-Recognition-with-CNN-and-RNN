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
   -e,--epoch	 	         (default 30)			   epoch number
   -z,--sourcePath         (default workstation)         source path (local | workstation)
   -r,--learningRate       (default 5e-5)          learning rate
   -l,--learningRateDecay  (default 1e-7)          learning rate decay (in # samples)
   -w,--weightDecay        (default 1e-1)          L2 penalty on the weights
   -m,--momentum           (default 0.9)           momentum
   -b,--batchSize          (default 16)            batch size
   -t,--threads            (default 2)             number of threads
   -p,--type               (default cuda)          float or cuda
   -i,--devid              (default 1)             device ID (if using CUDA)
   -o,--save               (default results)       save directory
   -s,--splitId            (default 1)             split number
   -d,--dropout1           (default 0.2)           dropout amount (spatial dropout)
      --decayRate	   (default 10)		   divide the learning rate with this number
      --dataset            (default UCF-101)       datset (UCF-101 | HMDB-51)
      --idPart             (default 0)             part of data (0 means full-data)
      --dropout2           (default 0.2)           dropout amount (dropout)
      --model              (default model-Conv-Inception-v1)      network model (model-Conv | model-Conv-VGG | model-Conv-Inception | model-Conv-Inception-TemSeg | model-1L | model-2L)
      --typeMF             (default Joint-LS)        multi-flow type (LS-Add | S-Add-L | Add-LS | Add-S | Joint-LS | Joint-S | Joint-FC-LS | Joint-FC-S | LS-Joint-LS)
      --batchNormalize	   (default Yes)		   do batch-normalization or not
      --methodCrop         (default centerCrop)    cropping method (tenCrop | centerCrop | centerCropMirror | centerCropFlip)
      --optMethod          (default adam)           optimization method
      --plot               (default No)       	   plot the training and test accuracies (Yes | No)
      --saveModel          (default No)            save the model or not (Yes | No)
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

numEpoch = tonumber(opt.epoch)

for epo=1,numEpoch do 
--while true do
   train(data.trainData)
   test(data.testData, data.classes, epo)
end
