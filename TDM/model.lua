----------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN
--  https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
--
--  Temporal Dynamic Model: Multidimensional LSTM and Temporal ConvNet
--  
-- 
--  Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------

local nn = require 'nn'
local rnn = require 'rnn'
local sys = require 'sys'

print(sys.COLORS.red ..  '==> construct LSTM + T-CNN')

if checkpoint then

   -- load T-CNN model
   local tcnnModelPath = paths.concat(opt.tcnnResume, 'TCNN.net')
   assert(paths.filep(tcnnModelPath), 'Saved model not found: ' .. tcnnModelPath)
   print('=> Resuming model from ' .. tcnnModelPath)
   
   tcnn = torch.load(tcnnModelPath)
   tcnn:insert(nn.View(opt.batchSize, 1, opt.inputSize, -1),1)
   tcnn:get(2):remove(9)
   
   -- load LSTM model
   local lstmModelPath = paths.concat(opt.lstmResume, checkpoint.modelFile)
   assert(paths.filep(lstmModelPath), 'Saved model not found: ' .. lstmModelPath)
   print('=> Resuming model from ' .. lstmModelPath)
   
   lstm = torch.load(lstmModelPath)

   lstm:remove(1)
   lstm:insert(nn.View(3, opt.batchSize/3, opt.inputSize, -1),1)
   lstm:remove(#lstm.modules)
   lstm:remove(#lstm.modules)
   lstm:add(nn.JoinTable(1))

   -- Video Classification model
   model = nn.Sequential()

   local inputSize = opt.inputSize

   model:add(nn.View(2, opt.batchSize, inputSize, -1))
   model:add(nn.SplitTable(1,2)) -- tensor to table of tensors

   local p = nn.ParallelTable()
   p:add(lstm)
   p:add(tcnn)
   model:add(p)

   model:add(nn.CAddTable())
   model:add(nn.LogSoftMax())

else
   error('cannot load LSTM or T-CNN model')
end

-- build criterion
criterion = nn.ClassNLLCriterion()

print(sys.COLORS.red ..  '==> here is the network:')
print(model)

error('test')

if opt.cuda == true then
   model:cuda()
   criterion:cuda()
end

return
{
   model = model,
   criterion = criterion
}
