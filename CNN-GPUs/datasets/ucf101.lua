--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet dataset loader
--

-- modified by: 
-- Chih-Yao Ma at cyma@gatech.edu
-- Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>

-- Last updated: 06/04/2016

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local UCF101Dataset = torch.class('resnet.UCF101Dataset', M)

function UCF101Dataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.data, split)
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function UCF101Dataset:get(i)
   local path = ffi.string(self.imageInfo.imagePath[i]:data())

   local image = self:_loadImage(paths.concat(self.dir, path))
   local class = self.imageInfo.imageClass[i]

   return {
      input = image,
      target = class,
   }
end

function UCF101Dataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
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

function UCF101Dataset:preprocess(opt)

   -- Computed from random subset of ImageNet training images
   -- local meanstd = {
   --    mean = { 0.485, 0.456, 0.406 },
   --    std = { 0.229, 0.224, 0.225 },
   -- }

   -- RGB
   -- 10 fps
   -- meanstd = {ean = { 0.392, 0.376, 0.348 },
   -- std = { 0.241, 0.234, 0.231 }}

   -- 25 fps
   meanstd = {mean = { 0.39234371606738, 0.37576219443075, 0.34801909196893 },
               std = { 0.24149100687454, 0.23453123289779, 0.23117322727131 }}

   local scaleSize, imageSize = 256, 224
   if opt.downsample ~='false' then -- downsample to half
      scaleSize = scaleSize / 2
      imageSize = imageSize / 2
   end

   
   if self.split == 'train' then
      return t.Compose{
         t.RandomSizedCrop(imageSize),
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.ColorNormalize(meanstd,opt.nChannel),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(scaleSize),
         t.ColorNormalize(meanstd,opt.nChannel),
         Crop(imageSize),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.UCF101Dataset