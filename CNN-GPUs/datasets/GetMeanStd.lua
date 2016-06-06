---------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN
--  https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
-- 
-- 
--  Calculate mean and std for different datasets, e.g. imagenet or UCF-101
-- 
-- 
--  Contact: Chih-Yao Ma at <cyma@gatech.edu>
---------------------------------------------------------------
require 'torch'
require 'nn'
local image = require 'image'
local paths = require 'paths'
local ffi = require 'ffi'

local tm = torch.Timer()
local nSamples = 10000 -- number of samples to calculate the mean and std
local dataset = 'imagenet' -- specify which dataset to use
local dataDir = '/home/chih-yao/Downloads/dataset/' .. dataset
local meanstdCache = dataset .. '-meanstdCache.t7'

-- load the cachefile saved before 
local dataCache = torch.load('../gen/' .. dataset .. '.t7')

-- generate #nSamples random number
local randMean = torch.Tensor(nSamples) 
randMean:random(1,dataCache.train.imagePath:size(1))

-- the calculation for mean and std are on training split
local split = 'train'

print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
local meanEstimate = {0,0,0}
for i=1,nSamples do
	local path = ffi.string(dataCache.train.imagePath[randMean[1]]:data())
	local img = image.load(paths.concat(dataDir, split, path), 3, 'float')
	for j=1,3 do
		meanEstimate[j] = meanEstimate[j] + img[j]:mean()
	end
end
for j=1,3 do
	meanEstimate[j] = meanEstimate[j] / nSamples
end
mean = meanEstimate


-- generate #nSamples random number
local randstd = torch.Tensor(nSamples) 
randstd:random(1,dataCache.train.imagePath:size(1))


print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
local stdEstimate = {0,0,0}
for i=1,nSamples do
	local path = ffi.string(dataCache.train.imagePath[randstd[1]]:data())
	local img = image.load(paths.concat(dataDir, split, path), 3, 'float')
	for j=1,3 do
		stdEstimate[j] = stdEstimate[j] + img[j]:std()
	end
end
for j=1,3 do
	stdEstimate[j] = stdEstimate[j] / nSamples
end
std = stdEstimate

print(mean, std)

local cache = {}
cache.mean = mean
cache.std = std
torch.save(meanstdCache, cache)
print('Time to estimate:', tm:time().real)