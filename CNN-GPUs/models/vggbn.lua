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
local nn = require 'nn'
require 'cunn'

local function createModel(opt)
   local modelType = 'A' -- on a titan black, B/D/E run out of memory even for batch-size 32

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   if modelType == 'A' then
      cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'B' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'D' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'}
   elseif modelType == 'E' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end

   local model = nn.Sequential()
   do
      local iChannels = opt.nChannel
      if opt.nStacking ~= 'false' then
         iChannels = opt.nChannel * opt.nStacking
      end

      for k,v in ipairs(cfg) do
         if v == 'M' then
            model:add(nn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            local conv3 = nn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
            model:add(conv3)
            model:add(nn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end

   model:add(nn.View(512*7*7))
   model:add(nn.Linear(512*7*7, 4096))
   model:add(nn.Threshold(0, 1e-6))
   model:add(nn.BatchNormalization(4096, 1e-3))
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(4096, 4096))
   model:add(nn.Threshold(0, 1e-6))
   model:add(nn.BatchNormalization(4096, 1e-3))
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(4096, 1000))

   model:cuda()

   model:get(1).gradInput = nil

   return model
end

return createModel