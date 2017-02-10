-- Activity-Recognition-with-CNN-and-RNN
-- https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN

-- Load all the videos & predict the video labels
-- Follow the split sets provided in the UCF-101 website
-- load the trained Spatial-stream ConvNet and Temporal-stream ConvNet

-- contact:
-- Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
-- Chih-Yao Ma at <cyma@gatech.edu>
-- Last updated: 12/16/2016

-- TODO
-- 1. HMDB-51

require 'xlua'
require 'torch'
require 'image'
require 'nn'
require 'cudnn' 
require 'cunn'
require 'cutorch'
t = require './transforms'

local videoDecoder = assert(require("libvideo_decoder")) -- package 3

----------------
-- parse args --
----------------
op = xlua.OptionParser('%prog [options]')

op:option{'-bS', '--batchSize', action='store', dest='batchSize',
          help='batch size of inputs for the model', default=32} 

op:option{'-sP', '--sourcePath', action='store', dest='sourcePath',
          help='source path (local | workstation)', default='local'}
op:option{'-dB', '--nameDatabase', action='store', dest='nameDatabase',
          help='used database (UCF-101 | HMDB-51)', default='UCF-101'}
op:option{'-sT', '--stream', action='store', dest='stream',
          help='type of optical stream (FlowMap-TVL1-crop20 | FlowMap-Brox)', default='FlowMap-TVL1-crop20'}
op:option{'-iSp', '--idSplit', action='store', dest='idSplit',
          help='index of the split set', default=1}

op:option{'-tn', '--numTopN', action='store', dest='numTopN',
          help='Top N accuracy', default=1} 
op:option{'-mC', '--methodCrop', action='store', dest='methodCrop',
          help='cropping method (tenCrop|centerCrop)', default='centerCrop'}
op:option{'-mP', '--methodPred', action='store', dest='methodPred',
          help='prediction method (scoreMean | classVoting)', default='scoreMean'}

op:option{'-f', '--frame', action='store', dest='frame',
          help='frame length for each video', default=25}
op:option{'-sA', '--sampleAll', action='store', dest='sampleAll',
          help='use all the frames or not', default=false}

op:option{'-tH', '--threads', action='store', dest='threads',
          help='number of threads', default=2}
op:option{'-i', '--devid1', action='store', dest='devid',
          help='GPU', default=1}           
op:option{'-s', '--save', action='store', dest='save',
          help='save the intermediate data or not', default=true}

numStream = 2 -- stream number of the input data
opt,args = op:parse()
-- convert strings to numbers --
idSplit = tonumber(opt.idSplit)
numTopN = tonumber(opt.numTopN)
frame = tonumber(opt.frame)

devid = tonumber(opt.devid)
threads = tonumber(opt.threads)

nameDatabase = opt.nameDatabase
methodOF = opt.stream 
BatchSize = opt.batchSize 

print('split #: '..idSplit)
print('source path: '..opt.sourcePath)
print('Database: '..opt.nameDatabase)
print('Temporal Stream: '..opt.stream)

print('frame length per video: '..frame)

----------------------------------------------
-- 				Data paths				    --
----------------------------------------------
source = opt.sourcePath -- local | workstation
if source == 'local' then
	dirSource = '/home/cmhung/Code/'
elseif source == 'workstation' then	
	dirSource = '/home/chih-yao/Downloads/'
end

if nameDatabase == 'UCF-101' then
	dirDatabase = 'Models-UCF101/'
elseif nameDatabase == 'HMDB-51' then	
	 dirDatabase = 'Models-HMDB51/'
end

dirModel = dirSource..dirDatabase..'Models-Temporal-ConvNet/sp'..idSplit..'/'

---- Input Features ----
dirFeature = dirSource..'Features/'..nameDatabase..'/'

----------------------------------------------
--  		       CPU option	 	        --
----------------------------------------------
-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------
--         Input/Output information         --
----------------------------------------------
nameType = {}
table.insert(nameType, 'RGB')
table.insert(nameType, 'FlowMap-TVL1-crop20')

---- Load testing data ----
dataTestAll = {}

for nS=1,numStream do
  print('==> load test data: '..'data_feat_test_'..nameType[nS]..'_'..opt.methodCrop..'_'..opt.frame..'f_sp'..opt.idSplit..'.t7')
  table.insert(dataTestAll, torch.load(dirFeature..'data_feat_test_'..nameType[nS]..'_'..opt.methodCrop..'_'..opt.frame..'f_sp'..opt.idSplit..'.t7'))
end

-- concatenation
dataTest = {}
dataTest.featMats = torch.cat(dataTestAll[1].featMats,dataTestAll[2].featMats,2)
dataTest.labels = dataTestAll[1].labels
nameVideo = dataTestAll[1].name
labels = dataTestAll[1].labels
dataTestAll = nil
collectgarbage()

-- information for the data
dimFeatIn = dataTest.featMats:size(2)
numFrame = dataTest.featMats:size(3)
teSize = (#dataTest.labels)[1]
numClass = dataTest.labels[teSize]
-- local shuffleTest = torch.randperm(teSize)

----------------------------------------------
-- 			User-defined parameters			--
----------------------------------------------
-- will combine to 'parse args' later
numFrameSample = frame
numSplit = 3
dimFeatOut = 1024

softMax = false
print('')
print('method for video prediction: ' .. opt.methodPred)
if softMax then
	print('Using SoftMax layer')
end
print('Using '..opt.methodCrop)

nameOutFile = 'acc_TemConv_'..nameDatabase..'_'..opt.frame..'f'..'-'..opt.methodCrop..'-sp'..idSplit..'.txt' -- output the video accuracy

-- Output information --
nameScoreTe = 'data_score_test_TemConv_'..nameDatabase..'_sp'..idSplit..'.t7'
namePredTe = 'data_pred_test_TemConv_'..nameDatabase..'_sp'..idSplit..'.t7'
nameFeatTe = 'data_feat_test_TemConv_'..nameDatabase..'_sp'..idSplit..'.t7'

------ model selection ------
-- ResNet model (from Torch) ==> need cudnn
modelName = 'model_best.t7'
modelPath = dirModel..modelName

----------------------------------------------
-- 					Classes 				--
----------------------------------------------
-- count the video # for each class
numVideoClass = torch.Tensor(numClass) -- 101x1
for i=1,numClass do
	numVideoClass[i] = torch.sum(torch.eq(dataTest.labels,i))
	-- print(numVideoClass[i])
end

----------------------------------------------
-- 					Models		        	--
----------------------------------------------
devID = devid 

--- choose GPU ---
cutorch.setDevice(devID)
print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
print(sys.COLORS.white ..  ' ')

print ' '
local net = torch.load(modelPath):cuda() -- Torch model

------ model modification ------	
if softMax then
	softMaxLayer = cudnn.SoftMax():cuda()
	net:add(softMaxLayer)
end

net:evaluate() -- Evaluate mode

print(net)
print ' '

----------------------------------------------
-- 			Loading UCF-101 labels	  		--
----------------------------------------------
-- imagenetLabel = require './imagenet'
ucf101Label = require './ucf-101'
table.sort(ucf101Label)

print ' '

--====================================================================--
--                     Run all the videos in UCF-101                  --
--====================================================================--
fd = io.open(nameOutFile,'w')
fd:write('Acc (video) \n')

print '==> Processing all the testing data...'

-- Load the intermediate feature data or generate a new one --
-- Training data --
existTe = opt.save and paths.filep(namePredTe) and paths.filep(nameFeatTe) and paths.filep(nameScoreTe)
-- Testing data --
Te = {} -- output prediction & info
featTe = {}
if not existTe then
--	Te = {} -- output prediction & info
	Te.finished = false

	--==== Prediction ====--
	Te.accVideoClass = torch.zeros(numClass)
	Te.accVideoAll = 0

	--==== Feature (Spatial & Temporal) ====--
	--featTe = {}
	featTe.name = {}
	featTe.featMats = torch.DoubleTensor()
	featTe.labels = torch.DoubleTensor()

	else
		Te = torch.load(namePredTe) -- output prediction
		featTe = torch.load(nameFeatTe) -- output features
	end
	collectgarbage()

	timerAll = torch.Timer() -- count the whole processing time

----------------------------------
-- 			Main Program		--
----------------------------------
if Te.finished then
	print('The prediction and feature extrction of split '..idSplit..' is finished!!!!!!')
else
	finish_pass = paths.filep(nameFeatTe) and paths.filep(nameScoreTe)
	if not finish_pass then
		----------------------------------------------
	    --    	Obtain the prediction scores 	    --
	    ----------------------------------------------
	    scoreVideos = torch.zeros(teSize,numClass) -- 3783x101
	    featVideos = torch.zeros(teSize,dimFeatOut) -- 3783x1024

	    -----====== Processing ======-----
		local countVideo = 0

	    local bSize = BatchSize
		for t=countVideo+1, teSize, BatchSize do
			---- 0. check the batch size ----
			if (t-1 + bSize) > teSize then
	       		bSize = teSize - (t-1)
			end
			-- countVideo = countVideo + bSize -- count the video
			print('Processing: ['..t..'/'..teSize..']')
			    
		   	---- 1. Extract the features and labels ----
		   	local featVideo = torch.zeros(BatchSize,dimFeatIn,numFrame) -- e.g. 32x4096x25
		   	featVideo[{{1,bSize}}] = dataTest.featMats[{{t,t+bSize-1}}]
		   	local inputFeat = nn.Reshape(1, dimFeatIn, numFrame):forward(featVideo):cuda() -- transform the input into batch mode (bSizex1x4096x25)
			
		   	--====== prediction ======--
			-- preds = net:forward(inputFeat):float() -- bSizex101
			local preds = net:forward(inputFeat):float():exp() -- bSizex101

			scoreVideos[{{t,t-1+bSize},{}}] = preds[{{1,bSize}}]

			--====== feature ======--
	        local feat_now = net.modules[13].output:float() -- bSizex1024
	        featVideos[{{t,t-1+bSize},{}}] = feat_now[{{1,bSize}}]
		end
		
		featTe.name = nameVideo
		featTe.labels = labels
		featTe.featMats = featVideos

		net = nil
		collectgarbage()

		if opt.save then
			torch.save(nameScoreTe, scoreVideos)
			torch.save(nameFeatTe, featTe)
		end
	else
		scoreVideos = torch.load(paths.filep(nameScoreTe))
	end

	----------------------------------------------
    --    	 Obtain the prediction labels 	    --
    ---------------------------------------------- 
    probLog, predLabels = scoreVideos:sort(2,true)         
	predVideos = predLabels[{{},{1,numTopN}}] -- 3783xtop#
	
	----------------------------------------------
    --    	 	Calculate the accuracy 		    --
    ---------------------------------------------- 
    --== 1. Class accuracy ==--
    countVideo = 0
    for c=1,numClass do
    	idStart = countVideo + 1
    	idEnd = countVideo + numVideoClass[c]
    	
    	local labelClass = dataTest.labels[{{idStart,idEnd}}]
    	local predClass = predVideos[{{idStart,idEnd}}] -- video#xtop#

		-- replicate the tensor (for Top-N accuracy)
    	local labelClassTopN = labelClass:repeatTensor(opt.numTopN,1):transpose(1,2)

    	local hitClassOri = torch.cmul(predClass:eq(c), labelClassTopN:eq(c)) -- video#xtop#
    	local hitClass = hitClassOri:sum(2):gt(0):float():sum(1)
    	Te.accVideoClass[c] = hitClass/numVideoClass[c]

    	-- print --
    	fd:write(Te.accVideoClass[c], '\n')

    	countVideo = countVideo + numVideoClass[c]
    end

    --== 2. Total accuracy ==--
    -- replicate the tensor (for Top-N accuracy)
    labelAll = dataTest.labels:repeatTensor(opt.numTopN,1):transpose(1,2):long()

    hitAllOri = predVideos:eq(labelAll) -- video#xtop#
    hitAll = hitAllOri:sum(2):gt(0):float():sum(1):squeeze()

    Te.accVideoAll = hitAll/teSize

    -- print --
    print('Total accuracy: '..Te.accVideoAll)
    fd:write(Te.accVideoAll, '\n')

    Te.finished = true

  	if opt.save then
  		torch.save(namePredTe, Te)
	end
	collectgarbage()
  	print(' ')
end

print('The total elapsed time in the split '..opt.idSplit..': ' .. timerAll:time().real .. ' seconds')

print('Total video numbers: '..countVideo)
print('Total video accuracy for the whole dataset: '..Te.accVideoAll)
print ' '

Te = nil
featTe = nil

collectgarbage()

fd:close()

