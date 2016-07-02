----------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN
--  https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
--
-- 
--  This is a testing code for implementing the RNN model with LSTM 
--  written by Chih-Yao Ma. 
-- 
--  The code will take feature vectors (from CNN model) from contiguous 
--  frames and train against the ground truth, i.e. the labeling of video classes. 
-- 
--  Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------

require 'torch'
require 'sys'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'

print(sys.COLORS.red .. '==> defining some tools')

-- model:
local m = require 'model'
local model = m.model
local criterion = m.criterion

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes) 

-- Logger:
local testLogger = optim.Logger(paths.concat(opt.save,'test.log'))

-- Batch test:
local inputs = torch.Tensor(opt.batchSize, TestData:size(2), TestData:size(3))
local targets = torch.Tensor(opt.batchSize)
local labels = {}
local prob = {}

if opt.averagePred == true then 
	predsFrames = torch.Tensor(opt.batchSize, nClass, opt.rho-1)
end

if opt.cuda == true then
	inputs = inputs:cuda()
	targets = targets:cuda()
	if opt.averagePred == true then 
	   predsFrames = predsFrames:cuda()
	end
end

-- test function
function test(TestData, TestTarget)

	-- local vars
	local time = sys.clock() 

	-- Sets Dropout layer to have a different behaviour during evaluation.
	-- TODO: is it okay if we don't un-evaluate the model?
	-- TODO: out of memory...
	model:evaluate() 

	-- test over test data
	print(sys.COLORS.red .. '==> testing on test set:')

	for t = 1,TestData:size(1),opt.batchSize do
		-- disp progress
		xlua.progress(t, TestData:size(1))

		-- batch fits?
		if (t + opt.batchSize - 1) > TestData:size(1) then
			break
		end

		-- create mini batch
		local idx = 1
		for i = t,t+opt.batchSize-1 do
			inputs[idx] = TestData[i]
			targets[idx] = TestTarget[i]
			idx = idx + 1
		end

		if opt.averagePred == true then 
			-- make prediction for each of the images frames, start from frame #2
			idx = 1
			for i = 2, opt.rho do
				-- extract various length of frames 
				local Index = torch.range(1, i)
				local indLong = torch.LongTensor():resize(Index:size()):copy(Index)
				local inputsPreFrames = inputs:index(3, indLong)

				-- feedforward pass the trained model
				predsFrames[{{},{},idx}] = model:forward(inputsPreFrames)

				idx = idx + 1
			end
			-- average all the prediction across all frames
			preds = torch.mean(predsFrames, 3):squeeze()

			-- Get the top N class indexes and probabilities
			local N = 1
			local probLog, predLabels = preds:topk(N, true, true)

			-- Convert log probabilities back to [0, 1]
			probLog:exp()

			idx = 1
			for i = t,t+opt.batchSize-1 do
				labels[i] = {}
				prob[i] = {}
				for j = 1, N do
					-- local indClass = predLabels[idx][j]
					labels[i][j] = classes[predLabels[idx][j]]
					prob[i][j] = probLog[idx][j]
				end
				idx = idx + 1
			end

		else
			-- test sample
			preds = model:forward(inputs)
		end

		-- confusion
		for i = 1,opt.batchSize do
			confusion:add(preds[i], targets[i])
		end
	end

	-- timing
	time = sys.clock() - time
	time = time / TestData:size(1)
	print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

  	-- print confusion matrix
  	print(confusion)

  	-- if the performance is so far the best..
	if confusion.totalValid * 100 >= bestAcc then
		bestAcc = confusion.totalValid * 100
		-- save the labels and probabilities into file
		torch.save('labels.txt', labels,'ascii')
		torch.save('prob.txt', prob,'ascii')
	end
	print(sys.COLORS.red .. '==> Best testing accuracy = ' .. bestAcc .. '%')

	-- update log/plot
	testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
	if opt.plot then
		testLogger:style{['% mean class accuracy (test set)'] = '-'}
		testLogger:plot()
	end
	confusion:zero()
end

-- Export:
return test

