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
local t = require 'transforms'

local tm = torch.Timer()
local nSamples = 500000 -- number of samples to calculate the mean and std
local dataset = 'ucf101-flow' -- specify which dataset to use
local dataDir = '/home/chih-yao/Downloads/dataset/UCF-101/FlowMap-M-frame'
local meanstdCache = dataset .. '-meanstdCache.t7'

-- load the cachefile saved before 
local dataCache = torch.load('../gen/' .. dataset .. '.t7')

-- generate #nSamples random number
local randNum = torch.Tensor(nSamples) 
randNum:random(1,dataCache.train.imagePath:size(1))

-- the calculation for mean and std are on training split
local split = 'train'

print('Estimating the mean and std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
local meanEstimate = {0,0,0}
local stdEstimate = {0,0,0}

for i=1,nSamples do
	if i % 100 == 1 then
		print(('Calculating mean and std [ %d / %d]'):format(i, nSamples))
	end

	local path = ffi.string(dataCache.train.imagePath[randNum[i]]:data())
	local img = image.load(paths.concat(dataDir, split, path), 3, 'float')
	local cropImg = image.scale(img, 224, 224)
	
	for j=1,3 do
		meanEstimate[j] = meanEstimate[j] + cropImg[j]:mean()
		stdEstimate[j] = stdEstimate[j] + cropImg[j]:std()
	end

end

for j=1,3 do
	meanEstimate[j] = meanEstimate[j] / nSamples
	stdEstimate[j] = stdEstimate[j] / nSamples
end

mean = meanEstimate
std = stdEstimate

print(mean, std)

local cache = {}
cache.mean = mean
cache.std = std
torch.save(meanstdCache, cache)
print('Time to estimate:', tm:time().real)