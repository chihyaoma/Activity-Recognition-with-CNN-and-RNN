----------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN
--  https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
--
--  Temporal Dynamic Model: Multidimensional LSTM and Temporal ConvNet
--  
-- 
--  Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------

local checkpoint = {}

function checkpoint.latest(opt)
   if opt.lstmResume and opt.tcnnResume == 'none' then
      return nil
   end

   -- load LSTM model 
   local lstmLatestFile = opt.resumeFile .. '.t7'
   local lstmLatestPath = paths.concat(opt.lstmResume, lstmLatestFile)

   if not paths.filep(lstmLatestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. lstmLatestPath)
   local lstmLatest = torch.load(lstmLatestPath)
   
   -- load T-CNN model
   -- local tcnnLatestFile = 'TCNN.t7'
   -- local tcnnLatestPath = paths.concat(opt.tcnnResume, tcnnLatestFile)

   -- if not paths.filep(tcnnLatestPath) then
   --    return nil
   -- end

   -- print('=> Loading checkpoint ' .. tcnnLatestPath)
   -- local tcnnLatest = torch.load(tcnnLatestPath)   

   local latest = lstmLatest
   -- local latest[2] = tcnnLatest

   return latest

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
