--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local checkpoint = {}

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestFile = opt.resumeFile .. '.t7'
   local latestPath = paths.concat(opt.resume, latestFile)

   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   
   if opt.optimState == 'none' then
        return latest, nil
   else
        local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))
        return latest, optimState
   end
end

function checkpoint.save(epoch, model, optimState, bestModel, bestTop1)
   -- Don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   local modelFile = 'model_current.t7'
   local optimFile = 'optimState_current.t7'

   torch.save(opt.save .. '/' .. modelFile, model)
   torch.save(opt.save .. '/' .. optimFile, optimState)

   torch.save(opt.save .. '/' .. 'latest_current.t7', {
      epoch = epoch,
      bestTop1 = bestTop1,
      modelFile = modelFile,
      optimFile = optimFile,
   })

   if bestModel then
      torch.save(opt.save .. '/' .. 'model_best.t7', model)
      torch.save(opt.save .. '/' .. 'optimState_best.t7', optimState)
      torch.save(opt.save .. '/' .. 'latest_best.t7', {
      epoch = epoch,
      bestTop1 = bestTop1,
      modelFile = 'model_best.t7',
      optimFile = 'optimState_best.t7',
   })
   end

end

return checkpoint
