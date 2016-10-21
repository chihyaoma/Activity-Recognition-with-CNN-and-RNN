-- Activity-Recognition-with-CNN-and-RNN
-- https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN

-- load the RGB & TVL1 features generated using ResNet-101, and then concatenate them
-- load the TCNN & RNN models
-- feed the concatenated features into TCNN and RNN separately, and then fuse the score

-- Reference:
-- Khurram Soomro, Amir Roshan Zamir and Mubarak Shah, 
-- "UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild.", 
-- CRCV-TR-12-01, November, 2012. 

-- TODO: top3 accuracy

-- contact:
-- Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
-- Chih-Yao Ma at <cyma@gatech.edu>
-- Last updated: 10/05/2016

require 'xlua'
require 'optim'   -- an optimization package, for online and batch methods
require 'torch'
require 'ffmpeg'
require 'image'
require 'nn'
require 'cudnn' 
require 'cunn'
require 'cutorch'
require 'rnn'
t = require './transforms'

----------------
-- parse args --
----------------
op = xlua.OptionParser('%prog [options]')

op:option{'-bS1', '--batchSize1', action='store', dest='batchSize1',
          help='batch size of inputs for 1st model', default=192} 
op:option{'-bS2', '--batchSize2', action='store', dest='batchSize2',
          help='batch size of inputs for 2nd model', default=32} -- max: 224

op:option{'-nH', '--numHidden', action='store', dest='numHidden',
          help='number of the hidden LSTM layers', default=3}  

op:option{'-tn', '--numTopN', action='store', dest='numTopN',
          help='Top N accuracy', default=1}      
op:option{'-idS', '--idSplit', action='store', dest='idSplit',
          help='which split used for testing', default=1}      
op:option{'-nM', '--numModel', action='store', dest='numModel',
          help='how many models to fuse', default=2}      
op:option{'-nS', '--numStream', action='store', dest='numStream',
          help='how many streams of inputs', default=2}      
op:option{'-fp', '--fpsTe', action='store', dest='fpsTe',
          help='fps for testing videos', default=25}
op:option{'-f', '--nframeAll', action='store', dest='nframeAll',
          help='frame # used for each video', default=25}      
op:option{'-mO', '--methodOF', action='store', dest='methodOF',
          help='optical flow method (TVL1|Brox)', default='TVL1'}
op:option{'-mC', '--methodCrop', action='store', dest='methodCrop',
          help='cropping method (tenCrop|centerCrop)', default='tenCrop'}
op:option{'-sM', '--softMax', action='store', dest='softMax',
          help='use the softmax layer or not', default=false}
op:option{'-m', '--mode', action='store', dest='mode',
          help='option for generating features (pred|feat)', default='pred'}
op:option{'-p', '--type', action='store', dest='type',
          help='option for CPU/GPU', default='cuda'}
op:option{'-i1', '--devid1', action='store', dest='devid1',
          help='1st GPU', default=1}      
-- op:option{'-i2', '--devid2', action='store', dest='devid2',
--           help='2nd GPU', default=2}      
op:option{'-s', '--save', action='store', dest='save',
          help='save the intermediate data or not', default=false}

opt,args = op:parse()

----------------------------------------------
-- 			User-defined parameters			--
----------------------------------------------
print('')
weightRNN = 0.25
weight = torch.Tensor({weightRNN,1-weightRNN})


if opt.softMax then
	print('Using SoftMax layer')
end
print('Using '..opt.methodCrop)
print('frame length per video: '..opt.nframeAll)

nameOutFile = 'acc_RNN-TCNN_'..opt.nframeAll..'Frames'..'-'..opt.methodCrop..'-'..opt.fpsTe..'fps.txt' -- output the video accuracy

----------------------------------------------
-- 				  Data paths			    --
----------------------------------------------
source = 'local' -- local | workstation
if source == 'local' then
	dirSource = '/home/cmhung/Code/'
elseif source == 'workstation' then	
	dirSource = '/home/chih-yao/Downloads/'
end

---- Input Features ----
dirFeature = dirSource..'Features/'

---- Input Models ----
dirModel = {} 
table.insert(dirModel, dirSource..'Models-RNN/')-- RNN
table.insert(dirModel, dirSource..'Models-TCNN/1L_bS_32-sp'..opt.idSplit..'/')-- TCNN

modelName = {}
table.insert(modelName, 'model_best.t7')-- RNN
table.insert(modelName, 'model_best.t7')-- TCNN


modelPath = {}
for nM=1,opt.numModel do
	table.insert(modelPath, dirModel[nM]..modelName[nM])
end
--print(modelPath)

outTest = {} -- intermediate data
table.insert(outTest, {name = 'data_'..opt.mode..'_RNN-TCNN_sp'..opt.idSplit..'.t7'})

----------------------------------------------
-- 				  Load Data 			    --
----------------------------------------------
nameType = {}
table.insert(nameType, 'RGB')
table.insert(nameType, 'FlowMap-TVL1-crop20')

---- Load testing data ----
dataTestAll = {}

for nS=1,opt.numStream do
  print('==> load test data: '..'data_feat_test_'..nameType[nS]..'_'..opt.methodCrop..'_'..opt.nframeAll..'f_sp'..opt.idSplit..'.t7')
  table.insert(dataTestAll, torch.load(dirFeature..'data_feat_test_'..nameType[nS]..'_'..opt.methodCrop..'_'..opt.nframeAll..'f_sp'..opt.idSplit..'.t7'))
end

-- concatenation
dataTest = {}
dataTest.featMats = torch.cat(dataTestAll[1].featMats,dataTestAll[2].featMats,2)
dataTest.labels = dataTestAll[1].labels
dataTestAll = nil
collectgarbage()

-- information for the data
dimFeat = dataTest.featMats:size(2)
numFrame = dataTest.featMats:size(3)
teSize = (#dataTest.labels)[1]
numClass = dataTest.labels[teSize]
-- local shuffleTest = torch.randperm(teSize)

----------------------------------------------
-- 					Functions 				--
----------------------------------------------

----------------------------------------------
-- 					Class		        	--
----------------------------------------------
-- load the label
ucf101Label = require './ucf-101'
table.sort(ucf101Label)

-- count the video # for each class
numVideoClass = torch.Tensor(numClass) -- 101x1
for i=1,numClass do
	numVideoClass[i] = torch.sum(torch.eq(dataTest.labels,i))
	-- print(numVideoClass[i])
end
-- input()
cutorch.setDevice(opt.devid1)
print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
print(sys.COLORS.white ..  ' ')

--====================================================================--
--                     Run all the videos in UCF-101                  --
--====================================================================--
BatchSize = torch.Tensor(opt.numModel)
BatchSize[1] = opt.batchSize1 -- for RNN
BatchSize[2] = opt.batchSize2 -- for TCNN

if opt.mode == 'pred' then
	fd = io.open(nameOutFile,'w')
	fd:write('RNN TCNN RNN+TCNN \n')
end
print '==> Processing all the testing data...'

-- Load the intermediate feature data or generate a new one --
if not (opt.save and paths.filep(outTest[idSplit].name)) then
	Te = {} -- output
	Te.finished = false
	--==== Accuracy ====--
	--== RNN+TCNN
	Te.accVideoClass = torch.zeros(numClass)
	Te.accVideoAll = 0
	--== RNN & TCNN
	Te.accVideoClassModels = torch.zeros(numClass,opt.numModel) -- 101x2
	Te.accVideoAllModels = torch.zeros(opt.numModel)

else
	Te = torch.load(outTest[idSplit].name) -- output
end
collectgarbage()

timerAll = torch.Timer() -- count the whole processing time

----------------------------------
-- 			Main Program		--
----------------------------------
if Te.finished then
	print('The prediction of split '..idSplit..' is finished!!!!!!')
else
	if not paths.filep('score-twoModels.t7') then
	----------------------------------------------
    --    	Obtain the prediction scores 	    --
    ----------------------------------------------  	
    ----==== RNN & TCNN separately ====----
    scoreVideoModels = torch.zeros(teSize,numClass,opt.numModel) -- 3783x101x2
	
	for nM=1,opt.numModel do
		-----====== Load Models	======-----
		print('==> Loading '..modelName[nM]..'......')
		local net = torch.load(modelPath[nM]):cuda() -- Torch model
		print(net)

		--- model modification ---
		if opt.mode == 'pred' then
			if opt.softMax then
				softMaxLayer = cudnn.SoftMax():cuda()
			    net:add(softMaxLayer)
			end
		end

		net:evaluate() -- Evaluate mode

		-----====== Processing ======-----
		local countVideo = 0

        local bSize = BatchSize[nM]
		for t=countVideo+1, teSize, BatchSize[nM] do
			---- 0. check the batch size ----
	  		if (t-1 + bSize) > teSize then
         		bSize = teSize - (t-1)
			end

	  		-- countVideo = countVideo + bSize -- count the video
	  		print('Processing: ['..t..'/'..teSize..']')
		    
	      	---- 1. Extract the features and lables ----
	      	-- fix the input dimension for the RNN model
	      	local featVideo = torch.zeros(BatchSize[nM],dimFeat,numFrame) -- 192x4096x25
	      	featVideo[{{1,bSize}}] = dataTest.featMats[{{t,t+bSize-1}}]
	      	
	      	local preds

	      	if nM == 1 then --== (1) RNN ==--
		      	---- 2. Calculate the scores ----
		      	---- modify the input dimension of the model
			    local inputFeat = featVideo:cuda() -- transform the input into batch mode (192x4096x25)
			    local predsFrames = torch.Tensor(BatchSize[nM], numClass, numFrame-1):cuda() -- 192x4096xframe#
		        -- make prediction for each of the images frames, start from frame #2
				idx = 1
				for i = 2, numFrame do
					-- extract various length of frames 
					local Index = torch.range(1, i)
					local indLong = torch.LongTensor():resize(Index:size()):copy(Index)
					local inputsPreFrames = inputFeat:index(3, indLong)

					-- replicate the testing data and feed into LSTM cells seperately
					inputsPreFrames = torch.repeatTensor(inputsPreFrames,opt.numHidden,1,1)
		
					-- feedforward pass the trained model
					predsFrames[{{},{},idx}] = net:forward(inputsPreFrames)
					
					idx = idx + 1
				end
				-- average all the prediction across all frames
				-- preds = torch.mean(predsFrames, 3):squeeze():float()
				preds = torch.mean(predsFrames:exp(), 3):squeeze():float()
				-- print(preds:sum(2))


			elseif nM == 2 then --== (2) TCNN ==--
				local inputFeat = nn.Reshape(1, dimFeat, numFrame):forward(featVideo):cuda() -- transform the input into batch mode (bSizex1x4096x25)
				-- preds = net:forward(inputFeat):float() -- bSizex101
				preds = net:forward(inputFeat):float():exp() -- bSizex101

			end

			scoreVideoModels[{{t,t-1+bSize},{},{nM}}] = preds[{{1,bSize}}]

		end

		net = nil
		collectgarbage()
	end

	torch.save('score-twoModels.t7', scoreVideoModels)

	-- -- This matrix records the current confusion across classes
	-- confusion = optim.ConfusionMatrix(ucf101Label) 
	-- for i=1,teSize do
	-- confusion:add(scoreVideoModels[{{i},{},{2}}]:squeeze(), dataTest.labels[i])
	-- end
	-- print(confusion) 
	-- print(confusion.totalValid)
	-- input()


	else
		scoreVideoModels = torch.load('score-twoModels.t7')
	end

	----==== Fusion of RNN & TCNN ====----
	-- convert the score to probability (in order to fuse two stream)
	probVideoModels = scoreVideoModels
	-- probVideoModels = torch.exp(scoreVideoModels) -- change from power to nuber
	-- probVideoModels_sum = torch.sum(probVideoModels,2):expandAs(probVideoModels) -- 3783x101x2
	-- probVideoModels = torch.cdiv(probVideoModels,probVideoModels_sum)

	-- torch.save('prob-twoModels.t7', probVideoModels)

	-- scoreVideoModelsFusion = torch.mean(scoreVideoModels,3):squeeze(3) -- 3783x101
	probVideoModelsWeighted = torch.Tensor():resizeAs(probVideoModels):copy(probVideoModels)
	for nM=1,opt.numModel do
		probVideoModelsWeighted[{{},{},{nM}}] = torch.mul(probVideoModels[{{},{},{nM}}],weight[nM])
	end
	probVideoModelsFusion = torch.sum(probVideoModelsWeighted,3):squeeze(3) -- 3783x101

	-- torch.save('data-Fusion.t7', scoreVideoModelsFusion)
	----------------------------------------------
    --    	 Obtain the prediction labels 	    --
    ---------------------------------------------- 
    ----==== RNN & TCNN separately ====----
    probLogModels, predLabelsModels = probVideoModels:sort(2,true) 
	predVideoModels = predLabelsModels[{{},{1,opt.numTopN}}] -- 3783xtop#x2
	-- torch.save('pred.t7',predVideoModels)

 --    ----==== Fusion of RNN & TCNN ====----
	probLog, predLabels = probVideoModelsFusion:sort(2,true)         
	predVideoFusion = predLabels[{{},{1,opt.numTopN}}] -- 3783xtop#

	-- -- select labels by the confidence
	-- -- confVideoModels = probLogModels[{{},{1,2}}]:sum(2):squeeze(2)
	-- confVideoModels = torch.abs(probLogModels[{{},{1}}]:csub(probLogModels[{{},{2}}])):squeeze(2)
		

	-- -- torch.save('conf_1.t7',probLogModels[{{},{1}}]:squeeze(2))
	-- -- torch.save('conf_2.t7',probLogModels[{{},{2}}]:squeeze(2))
	-- -- torch.save('conf_3.t7',probLogModels[{{},{3}}]:squeeze(2))
	-- -- print(confVideoModels)
	-- -- input()

	-- predVideoFusion = torch.zeros(teSize,opt.numTopN):long()
	-- for t=1,teSize do
	-- 	if confVideoModels[t][1]*weight[1] > confVideoModels[t][2]*weight[2] then
	-- 		predVideoFusion[t] = predLabelsModels[{{t},{1,opt.numTopN},{1}}]:squeeze()
	-- 	else
	-- 		predVideoFusion[t] = predLabelsModels[{{t},{1,opt.numTopN},{2}}]:squeeze()
	-- 	end
	-- end
	-- -- print(predVideoFusion)
	-- -- input()
	----------------------------------------------
    --    	 	Calculate the accuracy 		    --
    ---------------------------------------------- 
    --== 1. Class accuracy ==--
    countVideo = 0
    for c=1,numClass do
    	idStart = countVideo + 1
    	idEnd = countVideo + numVideoClass[c]
    	
    	local labelClass = dataTest.labels[{{idStart,idEnd}}]
    	----==== RNN & TCNN separately ====----
    	local predClassModels = predVideoModels[{{idStart,idEnd}}] -- video#xtop#x2
    	    	
    	-- replicate the tensor
    	local labelClassModels = labelClass:repeatTensor(opt.numTopN,opt.numModel,1):transpose(2,3):transpose(1,2)

    	local hitClassModelsOri = torch.cmul(predClassModels:eq(c), labelClassModels:eq(c)) -- video#xtop#x2
    	local hitClassModels = hitClassModelsOri:sum(2):gt(0):float():sum(1)
    	Te.accVideoClassModels[{{c}}] = hitClassModels/numVideoClass[c]

    	----==== Fusion of RNN & TCNN ====----
    	local predClassFusion = predVideoFusion[{{idStart,idEnd}}] -- video#xtop#

		-- replicate the tensor
    	local labelClassFusion = labelClass:repeatTensor(opt.numTopN,1):transpose(1,2)

    	local hitClassFusionOri = torch.cmul(predClassFusion:eq(c), labelClassFusion:eq(c)) -- video#xtop#
    	local hitClassFusion = hitClassFusionOri:sum(2):gt(0):float():sum(1)
    	Te.accVideoClass[c] = hitClassFusion/numVideoClass[c]

    	-- print --
    	-- print('Class '..ucf101Label[c]..' accuracy: '..Te.accVideoClassModels[{{c},{1}}]:squeeze()..' | '..Te.accVideoClassModels[{{c},{2}}]:squeeze()..' | '..Te.accVideoClass[c])
    	fd:write(Te.accVideoClassModels[{{c},{1}}]:squeeze(), ' ', Te.accVideoClassModels[{{c},{2}}]:squeeze(), ' ', Te.accVideoClass[c], '\n')

    	countVideo = countVideo + numVideoClass[c]
    end

    --== 2. Total accuracy ==--
    ----==== RNN & TCNN separately ====----
    -- replicate the tensor
   	labelAllModels = dataTest.labels:repeatTensor(opt.numTopN,opt.numModel,1):transpose(2,3):transpose(1,2):long()

	hitAllModelsOri = predVideoModels:eq(labelAllModels) -- teSizextop#x2
    hitAllModels = hitAllModelsOri:sum(2):gt(0):float():sum(1):squeeze()
    
    Te.accVideoAllModels = hitAllModels/teSize
    ----==== Fusion of RNN & TCNN ====----
    -- replicate the tensor
    labelAllFusion = dataTest.labels:repeatTensor(opt.numTopN,1):transpose(1,2):long()

    hitAllFusionOri = predVideoFusion:eq(labelAllFusion) -- video#xtop#
    hitAllFusion = hitAllFusionOri:sum(2):gt(0):float():sum(1):squeeze()

    Te.accVideoAll = hitAllFusion/teSize

    -- print --
    print('Total accuracy: '..Te.accVideoAllModels[1]..' | '..Te.accVideoAllModels[2]..' | '..Te.accVideoAll)
    fd:write(Te.accVideoAllModels[1], ' ', Te.accVideoAllModels[2], ' ', Te.accVideoAll, '\n')

  	if opt.save then
  		torch.save(outTest[opt.idSplit].name, Te)
	end
	collectgarbage()
  	print(' ')
end

Te.finished = true

print('The total elapsed time in the split '..opt.idSplit..': ' .. timerAll:time().real .. ' seconds')

print('Total video numbers: '..countVideo)
print('Total video accuracy for the whole dataset: '..Te.accVideoAll)
print ' '

Te = nil
collectgarbage()

if opt.mode == 'pred' then
	fd:close()
end
