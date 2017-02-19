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

op:option{'-sP', '--sourcePath', action='store', dest='sourcePath',
          help='source path (local | workstation)', default='workstation'}
op:option{'-dB', '--nameDatabase', action='store', dest='nameDatabase',
          help='used database (UCF-101 | HMDB-51)', default='HMDB-51'}
op:option{'-sT', '--stream', action='store', dest='stream',
          help='type of optical stream (FlowMap-TVL1-crop20 | FlowMap-Brox)', default='FlowMap-TVL1-crop20'}
op:option{'-iSp', '--idSplit', action='store', dest='idSplit',
          help='index of the split set', default=1}

op:option{'-iP', '--idPart', action='store', dest='idPart',
          help='index of the divided part', default=1}
op:option{'-nP', '--numPart', action='store', dest='numPart',
          help='number of parts to divide', default=2}
op:option{'-mD', '--manualDivide', action='store', dest='manualDivide',
          help='manually set the range', default=false}
op:option{'-iS', '--idStart', action='store', dest='idStart',
          help='manually set the starting class', default=1}
op:option{'-iE', '--idEnd', action='store', dest='idEnd',
          help='manually set the ending class', default=51}

op:option{'-mC', '--methodCrop', action='store', dest='methodCrop',
          help='cropping method (tenCrop | centerCrop)', default='centerCrop'}
op:option{'-mP', '--methodPred', action='store', dest='methodPred',
          help='prediction method (scoreMean | classVoting)', default='scoreMean'}

op:option{'-f', '--frame', action='store', dest='frame',
          help='frame length for each video', default=25}
-- op:option{'-fpsTr', '--fpsTr', action='store', dest='fpsTr',
--           help='fps of the trained model', default=25}
-- op:option{'-fpsTe', '--fpsTe', action='store', dest='fpsTe',
--           help='fps for testing', default=25}
op:option{'-sA', '--sampleAll', action='store', dest='sampleAll',
          help='use all the frames or not', default=false}

op:option{'-p', '--type', action='store', dest='type',
          help='option for CPU/GPU', default='cuda'}
op:option{'-tH', '--threads', action='store', dest='threads',
          help='number of threads', default=2}
op:option{'-i1', '--devid1', action='store', dest='devid1',
          help='1st GPU', default=1}      
op:option{'-i2', '--devid2', action='store', dest='devid2',
          help='2nd GPU', default=2}      
op:option{'-s', '--save', action='store', dest='save',
          help='save the intermediate data or not', default=true}

op:option{'-t', '--time', action='store', dest='seconds',
          help='length to process (in seconds)', default=2}
op:option{'-w', '--width', action='store', dest='width',
          help='resize video, width', default=320}
op:option{'-h', '--height', action='store', dest='height',
          help='resize video, height', default=240}

numStream = 2 -- stream number of the input data
opt,args = op:parse()
-- convert strings to numbers --
idPart = tonumber(opt.idPart)
numPart = tonumber(opt.numPart)
idSplit = tonumber(opt.idSplit)
idStart = tonumber(opt.idStart)
idEnd = tonumber(opt.idEnd)
frame = tonumber(opt.frame)
-- fpsTr = tonumber(opt.fpsTr)
-- fpsTe = tonumber(opt.fpsTe)
devid1 = tonumber(opt.devid1)
devid2 = tonumber(opt.devid2)
threads = tonumber(opt.threads)

nameDatabase = opt.nameDatabase
methodOF = opt.stream 

print('split #: '..idSplit)
print('source path: '..opt.sourcePath)
print('Database: '..opt.nameDatabase)
print('Stream: '..opt.stream)

-- print('fps for training: '..fpsTr)
-- print('fps for testing: '..fpsTe)

print('frame length per video: '..frame)
print('Data part '..idPart)

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

DIR = {}
dataFolder = {}
pathDatabase = dirSource..'dataset/'..nameDatabase..'/'
---- Temporal ----
table.insert(DIR, {dirModel = dirSource..dirDatabase..'Models-TwoStreamConvNets/ResNet-'..methodOF..'-sgd-sp'..idSplit..'/', 
		pathVideoIn = pathDatabase..methodOF..'/'})

---- Spatial ----
table.insert(DIR, {dirModel = dirSource..dirDatabase..'Models-TwoStreamConvNets/ResNet-RGB-sgd-sp'..idSplit..'/', 
	pathVideoIn = pathDatabase..'/RGB/'})

pathTxtSplit = pathDatabase..'testTrainMulti_7030_splits/' -- for HMDB-51

for nS=1,numStream do
	table.insert(dataFolder, paths.basename(DIR[nS].pathVideoIn))
end

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
-- select the number of classes, groups & videos you want to use
-- numClass = 101
dimFeat = 2048
numTopN = 3

numStack = torch.Tensor(numStream)
nChannel = torch.Tensor(numStream)

-- Temporal
numStack[1] = 10
nChannel[1] = 2
-- Spatial
numStack[2] = 1
nChannel[2] = 3

----------------------------------------------
-- 			User-defined parameters			--
----------------------------------------------
-- will combine to 'parse args' later
numFrameSample = frame
numSplit = 3
softMax = false
nCrops = (opt.methodCrop == 'tenCrop') and 10 or 1
print('')
print('method for video prediction: ' .. opt.methodPred)
if softMax then
	print('Using SoftMax layer')
end
print('Using '..opt.methodCrop)

nameOutFile = 'acc_'..nameDatabase..'_'..numFrameSample..'Frames'..'-'..opt.methodCrop..'-sp'..idSplit..'_part'..idPart..'.txt' -- output the video accuracy

------------------------------------------
--  	Train/Test split (UCF-101)		--
------------------------------------------
groupSplit = {}
for sp=1,numSplit do
	if sp==1 then
		table.insert(groupSplit, {setTr = torch.Tensor({{8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}}), 
			setTe = torch.Tensor({{1,2,3,4,5,6,7}})})
	elseif sp==2 then
		table.insert(groupSplit, {setTr = torch.Tensor({{1,2,3,4,5,6,7,15,16,17,18,19,20,21,22,23,24,25}}), 
			setTe = torch.Tensor({{8,9,10,11,12,13,14}})})
	elseif sp==3 then
		table.insert(groupSplit, {setTr = torch.Tensor({{1,2,3,4,5,6,7,8,9,10,11,12,13,14,22,23,24,25}}), 
			setTe = torch.Tensor({{15,16,17,18,19,20,21}})})
	end
end

-- Output information --
namePredTr = 'data_pred_train_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..idSplit..'_part'..idPart..'.t7'
namePredTe = 'data_pred_test_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..idSplit..'_part'..idPart..'.t7'

nameFeatTr = {}
nameFeatTe = {}
---- Temporal ----
table.insert(nameFeatTr, 'data_feat_train_'..methodOF..'_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..idSplit..'_part'..idPart..'.t7')
table.insert(nameFeatTe, 'data_feat_test_'..methodOF..'_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..idSplit..'_part'..idPart..'.t7')
---- Spatial ----
table.insert(nameFeatTr, 'data_feat_train_RGB_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..idSplit..'_part'..idPart..'.t7')
table.insert(nameFeatTe, 'data_feat_test_RGB_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..idSplit..'_part'..idPart..'.t7')


-- outTrain = {}
-- for sp=1,numSplit do
-- 	--table.insert(outTrain, {name = 'data_'..opt.mode..'_train_'..dataFolder..'_'..opt.methodCrop..'_sp'..sp..'.t7'})
-- 	table.insert(outTrain, {name = 'data_pred_train_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..sp..'.t7'})
-- end

-- outTest = {}
-- for sp=1,numSplit do
-- 	--table.insert(outTest, {name = 'data_'..opt.mode..'_test_'..dataFolder..'_'..opt.methodCrop..'_sp'..sp..'.t7'})
-- 	table.insert(outTest, {name = 'data_pred_test_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..sp..'.t7'})
-- end


------ model selection ------
-- ResNet model (from Torch) ==> need cudnn
modelName = 'model_best.t7'

modelPath = {}
for nS=1,numStream do
	table.insert(modelPath, DIR[nS].dirModel..modelName)
end
----------------------------------------------
-- 					Functions 				--
----------------------------------------------
meanstd = {}
-- Temporal
if dataFolder[1] == 'FlowMap-Brox' then
	if fpsTr == 10 then
		table.insert(meanstd, {mean = { 0.0091950063390791, 0.4922446721625, 0.49853131534726}, 
					std = { 0.0056229398806939, 0.070845543666524, 0.081589332546496}})
	elseif fpsTr == 25 then
		table.insert(meanstd, {mean = { 0.0091796917475333, 0.49176131835977, 0.49831646616289 },
               		std = { 0.0056094466799444, 0.070888495268898, 0.081680047609585 }})
	end
elseif dataFolder[1] == 'FlowMap-Brox-crop40' then
	table.insert(meanstd, {mean = { 0.0091936888040752, 0.49204453841557, 0.49857498097595},
      			std = { 0.0056320802048129, 0.070939325098903, 0.081698516724234}})
elseif dataFolder[1] == 'FlowMap-Brox-crop20' then
   	table.insert(meanstd, {mean = { 0.0092002901164412, 0.49243926742539, 0.49851170257907},
                std = { 0.0056614266189997, 0.070921186231261, 0.081781848181796}})
elseif dataFolder[1] == 'FlowMap-Brox-M' then
	table.insert(meanstd, {mean = { 0.951, 0.918, 0.955 },
                std = { 0.043, 0.052, 0.044 }})
elseif dataFolder[1] == 'FlowMap-FlowNet' then
	table.insert(meanstd, {mean = { 0.009, 0.510, 0.515 },
                std = { 0.007, 0.122, 0.124 }})
elseif dataFolder[1] == 'FlowMap-FlowNet-M' then
	table.insert(meanstd, {mean = { 0.951, 0.918, 0.955 },
                std = { 0.043, 0.052, 0.044 }})
elseif dataFolder[1] == 'FlowMap-TVL1-crop20' then
	-- if fpsTr == 10 then
	-- 	table.insert(meanstd, {mean = { 0.0078286737613148, 0.49277467447062, 0.42283539438139 },
	--                  std = { 0.0049402251681559, 0.060421647049655, 0.058913364961995 }})
	-- elseif fpsTr == 25 then
	-- 	table.insert(meanstd, {mean = { 0.0078368888567733, 0.49304171615406, 0.42294166284263 },
	--                   std = { 0.0049412518723573, 0.060508027119622, 0.058952390342379 }})
	-- else
	if nameDatabase == 'UCF-101' then
		-- if idSplit == 2
		-- 	table.insert(meanstd, {mean = {0.0077500744698257, 0.49250124870187, 0.41921449413675},
  --                  			std = {0.0049062763442844, 0.059868330437194, 0.057967578047986}})
		-- else
			table.insert(meanstd, {mean = { 0.0077904963214443, 0.49308556329956, 0.42114283484146 },
							std = { 0.0049190714163826, 0.060068045559535, 0.058203296730741 }})
		-- end
	elseif nameDatabase == 'HMDB-51' then
		table.insert(meanstd, {mean = { 0.0018654796792839, 0.34654865925634, 0.49215155492952 },
                  		std = {0.0037040848522728, 0.050646484297722, 0.073084135799887 }})
	end
	-- end
else
    error('no mean and std defined for temporal network... ')
end

-- Spatial
if dataFolder[2] == 'RGB' then
	-- if fpsTr == 10 then
	-- 	table.insert(meanstd, {mean = { 0.392, 0.376, 0.348 },
	--    				std = { 0.241, 0.234, 0.231 }})
	-- elseif fpsTr == 25 then
	-- 	table.insert(meanstd, {mean = { 0.39234371606738, 0.37576219443075, 0.34801909196893 },
	--                std = { 0.24149100687454, 0.23453123289779, 0.23117322727131 }})
	-- else
	if nameDatabase == 'UCF-101' then
		table.insert(meanstd, {mean = {0.39743499656438, 0.38846055375943, 0.35173909269078},
								std = {0.24145608138375, 0.23480329347676, 0.2306657093885}})
	elseif nameDatabase == 'HMDB-51' then
		table.insert(meanstd, {mean = {0.36410178082273, 0.36032826208483, 0.31140866484224},
  								std = {0.20658244577568, 0.20174469333003, 0.19790770088352}})
	end
	-- end
else
    error('no mean and std defined for spatial network... ')
end

Crop = (opt.methodCrop == 'tenCrop') and t.TenCrop or t.CenterCrop

----------------------------------------------
-- 					Class		        	--
----------------------------------------------
nameClass = paths.dir(DIR[1].pathVideoIn) 
table.sort(nameClass)
table.remove(nameClass,1) -- remove "."
table.remove(nameClass,1) -- remove ".."
numClass = #nameClass -- 101 classes 

---- divide the whole dataset into several parts ----
if not opt.manualDivide then
	numClassSub = torch.floor(numClass/numPart)
	rangeClassPart = {}
	numClassAcm = torch.zeros(numPart)
	Acm = 0
	for i=1,numPart do
		if i==numPart then
			table.insert(rangeClassPart, torch.range((i-1)*numClassSub+1,numClass))
		else
			table.insert(rangeClassPart, torch.range((i-1)*numClassSub+1,i*numClassSub))
		end
		
		Acm = Acm + rangeClassPart[i]:nElement()
		numClassAcm[i] = Acm
	end
end

----------------------------------------------
-- 					Models		        	--
----------------------------------------------
devID = torch.Tensor(numStream)
devID[1] = devid1 -- for temporal
devID[2] = devid2 -- for spatial
net = {}

for nS=1,numStream do
	--- choose GPU ---
	cutorch.setDevice(devID[nS])
	print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
	print(sys.COLORS.white ..  ' ')

	print ' '
	if nS == 1 then
		print '==> Loading the temporal model...'
	elseif nS == 2 then
		print '==> Loading the spatial model...'
	end
	local netTemp = torch.load(modelPath[nS]):cuda() -- Torch model
	
	------ model modification ------	
	if softMax then
		softMaxLayer = cudnn.SoftMax():cuda()
		netTemp:add(softMaxLayer)
	end

	netTemp:evaluate() -- Evaluate mode

	table.insert(net, netTemp)

	-- print(netTemp)
	print ' '
end

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
fd:write('S(frame) S(video) T(frame) T(video) S+T(frame) S+T(video) \n')

print '==> Processing all the videos...'

-- Load the intermediate feature data or generate a new one --
-- for sp=1,numSplit do
sp = idSplit
	-- Training data --
	existTr = opt.save and paths.filep(namePredTr) and paths.filep(nameFeatTr[1]) and paths.filep(nameFeatTr[2])
	Tr = {} -- output prediction & info
	featTr = {}
	if not existData then
	--	Tr = {} -- output prediction & info
		Tr.countVideo = 0
		if opt.manualDivide then
			Tr.countClass = idStart - 1
		else
			Tr.countClass = rangeClassPart[idPart][1]-1
		end
		Tr.c_finished = 0 

		--==== Feature (Spatial & Temporal) ====--
	--	featTr = {}
		--== Temporal
		table.insert(featTr,{name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
		--== Spatial
		table.insert(featTr,{name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})

	else
		Tr = torch.load(namePredTr) -- output prediction
		featTr[1] = torch.load(nameFeatTr[1]) -- output temporal features
		featTr[2] = torch.load(nameFeatTr[2]) -- output spatial features
	end

	existTe = opt.save and paths.filep(namePredTe) and paths.filep(nameFeatTe[1]) and paths.filep(nameFeatTe[2])
	-- Testing data --
	Te = {} -- output prediction & info
        featTe = {}
	if not existTe then
	--	Te = {} -- output prediction & info
		Te.countFrame = 0
		Te.countVideo = 0
		if opt.manualDivide then
			Te.countClass = idStart - 1
		else
			Te.countClass = rangeClassPart[idPart][1]-1
		end

		Te.accFrameClass = {}
		Te.accFrameAll = 0
		Te.accVideoClass = {}
		Te.accVideoAll = 0
		Te.c_finished = 0 

		Te.hitTestFrameAll = 0
		Te.hitTestVideoAll = 0

		--==== Prediction (Spatial & Temporal) ====--
		--== Temporal
		Te.accFrameClassT = {}
		Te.accFrameAllT = 0
		Te.accVideoClassT = {}
		Te.accVideoAllT = 0
		Te.hitTestFrameAllT = 0
		Te.hitTestVideoAllT = 0
		--== Spatial
		Te.accFrameClassS = {}
		Te.accFrameAllS = 0
		Te.accVideoClassS = {}
		Te.accVideoAllS = 0
		Te.hitTestFrameAllS = 0
		Te.hitTestVideoAllS = 0

		--==== Feature (Spatial & Temporal) ====--
		--featTe = {}
		--== Temporal
		table.insert(featTe, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})
		--== Spatial
		table.insert(featTe, {name = {}, path = {}, featMats = torch.DoubleTensor(), labels = torch.DoubleTensor()})


	else
		Te = torch.load(namePredTe) -- output prediction
		featTe[1] = torch.load(nameFeatTe[1]) -- output temporal features
		featTe[2] = torch.load(nameFeatTe[2]) -- output spatial features
	end
	collectgarbage()

	timerAll = torch.Timer() -- count the whole processing time

	if Tr.countClass == numClass and Te.countClass == numClass then
		print('The feature data of split '..sp..' is already in your folder!!!!!!')
	else
		local classStart, classEnd
		if opt.manualDivide then
			classStart = idStart
			classEnd = idEnd
		else
			classStart = Te.countClass + 1
			classEnd = numClassAcm[idPart]
		end

		for c=classStart, classEnd do

				local numTestFrameClass = 0
				local numTestVideoClass = 0

				local hitTestFrameClassB = 0
				local hitTestVideoClassB = 0

				--==== Separate Spatial & Temporal ====--
				local hitTestFrameClassT = 0
				local hitTestVideoClassT = 0
				local hitTestFrameClassS = 0
				local hitTestVideoClassS = 0
						
				print('Current Class: '..c..'. '..nameClass[c])

				Tr.countClass = Tr.countClass + 1
				Te.countClass = Te.countClass + 1
			  	
			  	------ Data paths ------
			  	local pathClassIn = {}
			  	local nameSubVideo = {}

			  	for nS=1,numStream do
			  		pathClassInTemp = DIR[nS].pathVideoIn..nameClass[c]..'/'
			  		table.insert(pathClassIn, pathClassInTemp)

			  		local nameSubVideoTemp = {}
				  	if nameDatabase == 'UCF-101' then
					  	nameSubVideoTemp = paths.dir(pathClassInTemp)
						table.sort(nameSubVideo) -- ascending order
						table.remove(nameSubVideoTemp,1) -- remove "."
						table.remove(nameSubVideoTemp,1) -- remove ".."
					elseif nameDatabase == 'HMDB-51' then
						local nameTxt = pathTxtSplit..nameClass[c]..'_test_split'..idSplit..'.txt'
						for l in io.lines(nameTxt) do
							table.insert(nameSubVideoTemp,l)
						end
					end
			  		table.insert(nameSubVideo, nameSubVideoTemp)
			  	end

			  	local numSubVideoTotal = #nameSubVideo[1] -- videos 


			  	local timerClass = torch.Timer() -- count the processing time for one class
			  	
			  	for sv=1, numSubVideoTotal do
			      	--------------------
			      	-- Load the video --
			      	--------------------  
			   		local videoName = {}
			   		local videoPath = {}
			   		local videoPathLocal = {}
			   		local videoIndex -- for HMDB-51

			      	for nS=1,numStream do
			      		local videoNameTemp
			      		local videoPathTemp
			       		if nameDatabase == 'UCF-101' then
							videoNameTemp = paths.basename(nameSubVideo[nS][sv],'avi')
							videoPathTemp = pathClassIn[nS]..videoNameTemp..'.avi'
						elseif nameDatabase == 'HMDB-51' then
							local i,j = string.find(nameSubVideo[nS][sv],' ') -- find the location of the group info in the string
						   	videoNameTemp = string.sub(nameSubVideo[nS][sv],1,j-5) -- get the video name
							videoIndex = tonumber(string.sub(nameSubVideo[nS][sv],j+1)) -- get the index for training/testing
							if nS == 2 then
						       	videoPathTemp = pathClassIn[nS]..videoNameTemp..'.avi'
						    else
						    	videoPathTemp = pathClassIn[nS]..videoNameTemp..'_flow.avi'
						    end
						end

		        		table.insert(videoName, videoNameTemp)
		        		table.insert(videoPath, videoPathTemp)
		        		table.insert(videoPathLocal, nameClass[c] .. '/' .. videoNameTemp .. '.avi')
					end

					----------------------------------------------
			       	--          Train/Test feature split        --
			       	----------------------------------------------
			        -- find out whether the video is in training set or testing set
					
					local flagTrain, flagTest = false
					if nameDatabase == 'UCF-101' then
						local i,j = string.find(videoName[1],'_g') -- find the location of the group info in the string
					    local videoGroup = tonumber(string.sub(videoName[1],j+1,j+2)) -- get the group#
					    
					    if groupSplit[idSplit].setTe:eq(videoGroup):sum() == 0 then -- training data
					    	flagTrain = true
					    	flagTest = false
					    else -- testing data
					    	flagTrain = false
					    	flagTest = true
					    end
					elseif nameDatabase == 'HMDB-51' then
						if videoIndex == 1 then -- training data
							flagTrain = true
							flagTest = false
						elseif videoIndex == 2 then -- testing data
							flagTrain = false
							flagTest = true
						else
							flagTrain = false
							flagTest = false
						end
					end
	
					--=========================================--
			        --            Process the video            --
					--=========================================--
			        --if not flagTrain and flagTest then -- testing data 
				    --== Read the video ==--
					local vidTensor = {}

					for nS=1,numStream do
						--print(videoName[nS])
						---- read the video to a tensor ----
						local status, height, width, length, fps = videoDecoder.init(videoPath[nS])
						local vidTensorTemp
		    			local inFrame = torch.ByteTensor(3, height, width)
		    			local countFrame = 0
						while true do
							status = videoDecoder.frame_rgb(inFrame)
							if not status then
					   			break
							end
							countFrame = countFrame + 1
					   		local frameTensor = torch.reshape(inFrame, 1, 3, height, width):double():div(255)
										
					   		if countFrame == 1 then -- the first frame
						        vidTensorTemp = frameTensor
						    else -- from the second or the following videos
						        vidTensorTemp = torch.cat(vidTensorTemp,frameTensor,1)	            	
						    end			      
									
						end
						videoDecoder.exit()
									
						table.insert(vidTensor, vidTensorTemp) -- read the whole video & turn it into a 4D tensor (e.g. 150x3x240x320)
					end
					        	
					local numFrame = vidTensor[1]:size(1) -- same frame # for two streams
					local height =  vidTensor[1]:size(3)
					local width = vidTensor[1]:size(4)
					        	
					------ Video prarmeters (same for two streams) ------				        	
				    local numFrameAvailable = numFrame - numStack[1] + 1 -- for 10-stacking
				    local numFrameInterval = opt.sampleAll and 1 or torch.floor(numFrameAvailable/numFrameSample)
				    numFrameInterval = torch.Tensor({{1,numFrameInterval}}):max() -- make sure larger than 0
				    local numFrameUsed = opt.sampleAll and numFrameAvailable or numFrameSample -- choose frame # for one video
					          	
				    ------ Initialization of the prediction ------
				    local predFramesB = torch.Tensor(numFrameUsed):zero() -- e.g. 25
				    local scoreFramesB = torch.Tensor(numFrameUsed,numClass):zero() -- e.g. 25x101

				    --==== Separate Spatial & Temporal ====--
				    local predFramesT = torch.Tensor(numFrameUsed):zero() -- e.g. 25
				    local predFramesS = torch.Tensor(numFrameUsed):zero() -- e.g. 25
				    local scoreFramesT = torch.Tensor(numFrameUsed,numClass):zero() -- e.g. 25x101
				    local scoreFramesS = torch.Tensor(numFrameUsed,numClass):zero() -- e.g. 25x101

				    --==== Outputs ====--
				    local scoreFrameTS = torch.Tensor(numStream,numFrameUsed,numClass):zero() -- 2x25x101
				    local featFrameTS = torch.Tensor(numStream,dimFeat,numFrameUsed):zero() -- 2x2048x25

				    ----------------------------------------------------------
				    -- Forward pass the model (get features and prediction) --
				    ----------------------------------------------------------
				    -- print '==> Begin predicting......'
				    for nS=1,numStream do
				      	cutorch.setDevice(devID[nS])
					   	--- transform ---
					   	transform = t.Compose{t.Scale(256), t.ColorNormalize(meanstd[nS], nChannel[nS]), Crop(224)}

					   	local I_all -- concatenate all sampled frames for a video
						    
					    for i=1, numFrameUsed do
				       		local f = (i-1)*numFrameInterval+5 -- current frame sample (middle in 10-stacking)
				       		f = torch.Tensor({{f,numFrame-5}}):min() -- make sure we can extract the corresponding 10-stack optcial flow
				        	
					   		-- extract the input
					   		-- Temporal:	2-channel, 10-stacking
					   		-- Spatial:		3-channel, none-stacking
					   		local inFrames = vidTensor[nS][{{torch.floor(f-numStack[nS]/2)+1,torch.floor(f+numStack[nS]/2)},
					              		{3-(nChannel[nS]-1),3},{},{}}]

					   		-- change the dimension for the input to "transform" 
					   		-- Temporal:	20x240x320
					   		-- Spatial:		3x240x320					              		
					   		local netInput = torch.Tensor(inFrames:size(1)*nChannel[nS],height,width):zero()
					   		for x=0,numStack[nS]-1 do
					   			netInput[{{x*nChannel[nS]+1,(x+1)*nChannel[nS]}}] = inFrames[{{x+1},{},{},{}}]
					   		end

					  		local I = transform(netInput) -- e.g. 20x224x224 or 10x20x224x224 (tenCrop)
							I = I:view(1, table.unpack(I:size():totable())) -- 20x224x224 --> 1x20x224x224

							-- concatenation
							if i==1 then
								I_all = I
							else
								I_all = torch.cat(I_all,I,1)
							end

					    end

					    --====== prediction ======--
					    local scoreFrame_now = torch.Tensor(numFrameUsed,numClass):zero() -- 25x101
				        if (opt.methodCrop == 'tenCrop') then
					    	local outputTen = net[nS]:forward(I_all:cuda()) -- 25x10x101
				           	scoreFrame_now = torch.mean(outputTen,1) -- 25x101
				        else
				          	--I = I:view(1, table.unpack(I:size():totable())) -- 25x20x224x224
				           	local output = net[nS]:forward(I_all:cuda()) -- 25x101
				           	scoreFrame_now = output
				        end

				        -- scoreFrame_now = cudnn.SoftMax():cuda():forward(scoreFrame_now)	-- add a softmax layer to convert the value to probability				              		
				        scoreFrameTS[nS] = scoreFrame_now:float()

				        --====== feature ======--
				        local feat_now = net[nS].modules[10].output:float() -- 25x2048
					feat_now = feat_now:transpose(1,2)
				        local featFrame_now = feat_now:reshape(1, table.unpack(feat_now:size():totable())):double() -- 1x2048x25
				        featFrameTS[nS] = featFrame_now
					
				    end
					
				    ------------------------------------------------------------------
				    -- Training:	features										--
				    -- Testing:		features + frame prediction + video prediction 	--
				    ------------------------------------------------------------------
				    if flagTrain and not flagTest then -- training data
				    	Tr.countVideo = Tr.countVideo + 1

				    	--==== Feature (Spatial & Temporal) ====--
						for nS=1,numStream do
							featTr[nS].name[Tr.countVideo] = videoName[nS]
							featTr[nS].path[Tr.countVideo] = videoPathLocal[nS]
							if Tr.countVideo == 1 then -- the first video
				        		featTr[nS].featMats = featFrameTS[{{nS},{},{}}]
				        		featTr[nS].labels = torch.DoubleTensor(nCrops):fill(Tr.countClass)
					        else 					-- from the second or the following videos
					        	featTr[nS].featMats = torch.cat(featTr[nS].featMats,featFrameTS[{{nS},{},{}}],1)
					        	featTr[nS].labels = torch.cat(featTr[nS].labels,torch.DoubleTensor(nCrops):fill(Tr.countClass),1)
					        end
						end

					elseif not flagTrain and flagTest then -- testing data	
						Te.countVideo = Te.countVideo + 1

						--==== Feature (Spatial & Temporal) ====--
						for nS=1,numStream do
							featTe[nS].name[Te.countVideo] = videoName[nS]
							featTe[nS].path[Te.countVideo] = videoPathLocal[nS]
							if Te.countVideo == 1 then -- the first video
				        		featTe[nS].featMats = featFrameTS[{{nS},{},{}}]
				        		featTe[nS].labels = torch.DoubleTensor(nCrops):fill(Te.countClass)
					        else 					-- from the second or the following videos
					        	featTe[nS].featMats = torch.cat(featTe[nS].featMats,featFrameTS[{{nS},{},{}}],1)
					        	featTe[nS].labels = torch.cat(featTe[nS].labels,torch.DoubleTensor(nCrops):fill(Te.countClass),1)
					        end
						end

						-- scores --> prediction --
						Te.countFrame = Te.countFrame + numFrameUsed
						numTestFrameClass = numTestFrameClass + numFrameUsed
						
						--==== Baseline ====--
						scoreFramesB = torch.mean(scoreFrameTS,1):squeeze(1) -- 101 probabilities of the frame (baseline)
						local probLogB, predLabelsB = scoreFramesB:topk(numTopN, true, true) -- 5 (probabilities + labels)        
						local predFramesB = predLabelsB[{{},1}] -- predicted label of the frame
				        	local hitTestFrameB = predFramesB:eq(c):sum() 
				        	hitTestFrameClassB = hitTestFrameClassB + hitTestFrameB -- accumulate the score for frame prediction
				        
						--==== Separate Spatial & Temporal ====--
						--== Temporal
						scoreFramesT = scoreFrameTS[1] 
						local probLogT, predLabelsT = scoreFramesT:topk(numTopN, true, true) -- 5 (probabilities + labels)        
						local predFramesT = predLabelsT[{{},1}] -- predicted label of the frame
					        local hitTestFrameT = predFramesT:eq(c):sum()
						hitTestFrameClassT = hitTestFrameClassT + hitTestFrameT -- accumulate the score for frame prediction
					    
						--== Spatial
						scoreFramesS = scoreFrameTS[2] 
						local probLogS, predLabelsS = scoreFramesS:topk(numTopN, true, true) -- 5 (probabilities + labels)        
						local predFramesS = predLabelsS[{{},1}] -- predicted label of the frame
						local hitTestFrameS = predFramesS:eq(c):sum()
						hitTestFrameClassS = hitTestFrameClassS + hitTestFrameS -- accumulate the score for frame prediction
					

						----------------------
				        -- Video Prediction --
				        ----------------------
					    local predVideoB

					    --==== Separate Spatial & Temporal ====--
					    local predVideoT
						local predVideoS

					    if opt.methodPred == 'classVoting' then 
					       	local predVideoTensor = torch.mode(predFramesB)
					    	predVideoB = predVideoTensor[1]
						elseif opt.methodPred == 'scoreMean' then
							local scoreMean = torch.mean(scoreFramesB,1) -- 1x101
							local probLogB, predLabelsB = scoreMean:topk(numTopN, true, true) -- 5 (probabilities + labels)
						   	predVideoB = predLabelsB[{{},1}]

							--==== Separate Spatial & Temporal ====--
							--== Temporal
							local scoreMeanT = torch.mean(scoreFramesT,1)
							local probLogT, predLabelsT = scoreMeanT:topk(numTopN, true, true) -- 5 (probabilities + labels)

							predVideoT = predLabelsT[{{},1}]
							
							--== Spatial
							local scoreMeanS = torch.mean(scoreFramesS,1)
							local probLogS, predLabelsS = scoreMeanS:topk(numTopN, true, true) -- 5 (probabilities + labels)
							predVideoS = predLabelsS[{{},1}]
						end
					    
				        -- accumulate the score for video prediction
				        numTestVideoClass = numTestVideoClass + 1
				        local hitTestVideoB = predVideoB:eq(c):sum()
					hitTestVideoClassB = hitTestVideoClassB + hitTestVideoB -- baseline
						--print(predVideoT)
						--print(predVideoT:eq(c))
						--print(predVideoT:eq(c):sum())
						--error(test)
					local hitTestVideoT = predVideoT:eq(c):sum() 
					hitTestVideoClassT = hitTestVideoClassT + hitTestVideoT -- temporal
				        local hitTestVideoS = predVideoS:eq(c):sum()
					hitTestVideoClassS = hitTestVideoClassS + hitTestVideoS -- spatial
        				print(videoName[1],hitTestFrameB,hitTestFrameT,hitTestFrameS,hitTestVideoB,hitTestVideoT,hitTestVideoS)
				    end				        
			      	collectgarbage()
			    end
				Te.c_finished = c -- save the index
 				
	          	----------------------------------------------
	          	--       Print the prediction results       --
	          	----------------------------------------------
					Te.hitTestFrameAll = Te.hitTestFrameAll + hitTestFrameClassB
					local acc_frame_class_ST = hitTestFrameClassB/numTestFrameClass
					acc_frame_all_ST = Te.hitTestFrameAll/Te.countFrame
					print('Class frame accuracy: '..acc_frame_class_ST)
					print('Accumulated frame accuracy: '..acc_frame_all_ST)
					Te.accFrameClass[Te.countClass] = acc_frame_class_ST
					Te.accFrameAll = acc_frame_all_ST

					-- video prediction
					Te.hitTestVideoAll = Te.hitTestVideoAll + hitTestVideoClassB
					local acc_video_class_ST = hitTestVideoClassB/numTestVideoClass
					acc_video_all_ST = Te.hitTestVideoAll/Te.countVideo
					print('Class video accuracy: '..acc_video_class_ST)
					print('Accumulated video accuracy: '..acc_video_all_ST)
					Te.accVideoClass[Te.countClass] = acc_video_class_ST
					Te.accVideoAll = acc_video_all_ST

					--==== Separate Spatial & Temporal ====--
					--== Temporal
					Te.hitTestFrameAllT = Te.hitTestFrameAllT + hitTestFrameClassT
					local acc_frame_class_T = hitTestFrameClassT/numTestFrameClass
					acc_frame_all_T = Te.hitTestFrameAllT/Te.countFrame
					print('Class frame accuracy (Temporal): '..acc_frame_class_T)
					print('Accumulated frame accuracy (Temporal): '..acc_frame_all_T)
					Te.accFrameClassT[Te.countClass] = acc_frame_class_T
					Te.accFrameAllT = acc_frame_all_T

					-- video prediction
					Te.hitTestVideoAllT = Te.hitTestVideoAllT + hitTestVideoClassT
					local acc_video_class_T = hitTestVideoClassT/numTestVideoClass
					acc_video_all_T = Te.hitTestVideoAllT/Te.countVideo
					print('Class video accuracy (Temporal): '..acc_video_class_T)
					print('Accumulated video accuracy (Temporal): '..acc_video_all_T)
					Te.accVideoClassT[Te.countClass] = acc_video_class_T
					Te.accVideoAllT = acc_video_all_T

					--== Spatial
					Te.hitTestFrameAllS = Te.hitTestFrameAllS + hitTestFrameClassS
					local acc_frame_class_S = hitTestFrameClassS/numTestFrameClass
					acc_frame_all_S = Te.hitTestFrameAllS/Te.countFrame
					print('Class frame accuracy (Spatial): '..acc_frame_class_S)
					print('Accumulated frame accuracy (Spatial): '..acc_frame_all_S)
					Te.accFrameClassS[Te.countClass] = acc_frame_class_S
					Te.accFrameAllS = acc_frame_all_S

					-- video prediction
					Te.hitTestVideoAllS = Te.hitTestVideoAllS + hitTestVideoClassS
					local acc_video_class_S = hitTestVideoClassS/numTestVideoClass
					acc_video_all_S = Te.hitTestVideoAllS/Te.countVideo
					print('Class video accuracy (Spatial): '..acc_video_class_S)
					print('Accumulated video accuracy (Spatial): '..acc_video_all_S)
					Te.accVideoClassS[Te.countClass] = acc_video_class_S
					Te.accVideoAllS = acc_video_all_S


					fd:write(acc_frame_class_S, ' ', acc_video_class_S, ' ', acc_frame_class_T, ' ', acc_video_class_T, ' ', acc_frame_class_ST, ' ', acc_video_class_ST, '\n')



			  	print('The elapsed time for the class '..nameClass[c]..': ' .. timerClass:time().real .. ' seconds')
			  	
			  	if opt.save then
					torch.save(namePredTr, Tr)
			  		torch.save(namePredTe, Te)

			  		for nS=1,numStream do
			  			print(featTr[nS].featMats:size())
			  			print(featTr[nS].labels:size())
			  			print(featTe[nS].featMats:size())
			  			print(featTe[nS].labels:size())
			  			
				  		torch.save(nameFeatTr[nS], featTr[nS])
				  		torch.save(nameFeatTe[nS], featTe[nS])
			  		end
				end

			  	collectgarbage()
			  	print(' ')
		end
	end

	print('The total elapsed time in the split '..sp..': ' .. timerAll:time().real .. ' seconds')

	-- Final Outputs --
	print('Total frame numbers: '..Te.countFrame)
	print('Total frame accuracy for the whole dataset: '..Te.accFrameAll)
	print('Total frame accuracy for the whole dataset (Temporal): '..Te.accFrameAllT)
	print('Total frame accuracy for the whole dataset (Spatial): '..Te.accFrameAllS)
	print('Total video numbers: '..Te.countVideo)
	print('Total video accuracy for the whole dataset: '..Te.accVideoAll)
	print('Total video accuracy for the whole dataset (Temporal): '..Te.accVideoAllT)
	print('Total video accuracy for the whole dataset (Spatial): '..Te.accVideoAllS)
	
	fd:write(acc_frame_all_S, ' ', acc_video_all_S, ' ', acc_frame_all_T, ' ', acc_video_all_T, ' ', acc_frame_all_ST, ' ', acc_video_all_ST, '\n')
		


	
	print ' '

	Tr = nil
	Te = nil
	featTr = nil
	featTe = nil
	collectgarbage()
-- end


	fd:close()

