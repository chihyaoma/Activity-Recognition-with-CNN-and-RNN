---------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN
--  https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
-- 
-- 
--  Train a CNN on flow map of UCF-101 dataset 
-- 
-- 
--  Contact: Chih-Yao Ma at <cyma@gatech.edu>
---------------------------------------------------------------
--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  UCF-101 flow maps dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local UCF101Dataset = torch.class('resnet.UCF101Dataset', M)

function UCF101Dataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split] -- imagePath & imageClass
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.data, split)
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function UCF101Dataset:get(i, nChannel, nStacking)
   local path = ffi.string(self.imageInfo.imagePath[i]:data()) -- e.g. Nunchucks/v_Nunchucks_g16_c03_flow_50.png
   local image

   -- extract frame number
   local afterPath = path:match("^.+flow(.+)$") -- e.g. _50.png
   local prePath = path:match("^.-flow")        -- e.g. Nunchucks/v_Nunchucks_g16_c03_flow
   local frameStr = afterPath:match("%d+")      -- e.g. 50
   local fileExten = afterPath:match("%a+")     -- e.g. png

   -- which frames will be stacked
   local frameStack = torch.range(frameStr-(nStacking-1), frameStr) -- e.g. 41~50

   -- read and stack images
   for i=1,frameStack:size(1) do
      path = prePath .. '_' .. frameStack[i] .. '.' .. fileExten
      local imageTmp = self:_loadImage(paths.concat(self.dir, path))
      imageTmp2 = imageTmp[{{3-(nChannel-1),3}}] -- the first channel is R
      if not image then
         image = imageTmp2
      else
         image = torch.cat(image, imageTmp2, 1) -- final dimension = 2*nStacking
      end
   end

   -- get image class as target
   local class = self.imageInfo.imageClass[i]

   return {
      input = image,
      target = class,
   }
end

function UCF101Dataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float') -- RGB (R is 0)
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function UCF101Dataset:size()
   return self.imageInfo.imageClass:size(1)
end

-- Computed from random subset of UCF-101 training Brox flow maps
local meanstd = {
   -- -- Middlebury
   -- mean = { 0.951, 0.918, 0.955 },
   -- std = { 0.043, 0.052, 0.044 },
   
   -- x, y displacement (2-channel)
   mean = { 0.009, 0.492, 0.498 },
   std = { 0.006, 0.071, 0.081 },
}

function UCF101Dataset:preprocess(opt)
   if self.split == 'train' then
      return t.Compose{
         t.RandomSizedCrop(224),
         -- t.ColorJitter({
         --    brightness = 0.4,
         --    contrast = 0.4,
         --    saturation = 0.4,
         -- }),
         t.ColorNormalize(meanstd,opt.nChannel),
         t.HorizontalFlip(1),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(256),
         t.ColorNormalize(meanstd,opt.nChannel),
         Crop(224),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.UCF101Dataset
