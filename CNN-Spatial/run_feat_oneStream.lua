-- Activity-Recognition-with-CNN-and-RNN
-- https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN

-- Load all the videos & Generate a feature matrix for each video
-- Follow the split sets provided in the UCF-101 website
-- load the fine-tuned ResNet model (We use Res-101 now)

-- contact:
-- Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
-- Chih-Yao Ma at <cyma@gatech.edu>
-- Last updated: 12/16/2016

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
          help='source path (local | workstation)', default='local'}
op:option{'-dB', '--nameDatabase', action='store', dest='nameDatabase',
          help='used database (UCF-101 | HMDB-51)', default='UCF-101'}
op:option{'-sT', '--stream', action='store', dest='stream',
          help='type of stream (RGB | FlowMap-TVL1-crop20 | FlowMap-Brox)', default='FlowMap-TVL1-crop20'}
op:option{'-iSp', '--idSplit', action='store', dest='idSplit',
          help='index of the split set', default=1}
          
op:option{'-iP', '--idPart', action='store', dest='idPart',
          help='index of the divided part', default=1}
op:option{'-nP', '--numPart', action='store', dest='numPart',
          help='number of parts to divide', default=1}

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
          help='number of threads', default=1}
op:option{'-i', '--devid', action='store', dest='devid',
          help='device ID (if using CUDA)', default=1}
op:option{'-s', '--save', action='store', dest='save',
          help='save the intermediate data or not', default=true}

op:option{'-w', '--width', action='store', dest='width',
          help='resize video, width', default=320}
op:option{'-h', '--height', action='store', dest='height',
          help='resize video, height', default=240}
op:option{'-t', '--time', action='store', dest='seconds',
          help='length to process (in seconds)', default=1000}

opt,args = op:parse()
-- convert strings to numbers --
idPart = tonumber(opt.idPart)
numPart = tonumber(opt.numPart)
idSplit = tonumber(opt.idSplit)
frame = tonumber(opt.frame)
-- fpsTr = tonumber(opt.fpsTr)
-- fpsTe = tonumber(opt.fpsTe)
devid = tonumber(opt.devid)
threads = tonumber(opt.threads)

nameDatabase = opt.nameDatabase
dirVideoIn = opt.stream 

print('Split #: '..idSplit)
print('threads #: '..threads)
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
	-- dirSource = '/home/chih-yao/'
end

dirModel = dirSource..'Models-UCF101/Models-TwoStreamConvNets/ResNet-'..dirVideoIn..'-sgd-sp'..idSplit..'/'
pathVideoIn = dirSource..'dataset/'..nameDatabase..'/'..dirVideoIn..'/'

dataFolder = paths.basename(pathVideoIn)
print('Video type: '..dataFolder)

----------------------------------------------
--  		       CPU option	 	        --
----------------------------------------------
-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------
--  		       GPU option	 	        --
----------------------------------------------
cutorch.setDevice(devid)
print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
print(sys.COLORS.white ..  ' ')

----------------------------------------------
--         Input/Output information         --
----------------------------------------------
-- select the number of classes, groups & videos you want to use
-- numClass = 101
dimFeat = 2048
numStack = 10
nChannel = 2
numTopN = 1

if dataFolder == 'RGB' then
	numStack = 1
	nChannel = 3
end

----------------------------------------------
-- 			User-defined parameters			--
----------------------------------------------
-- will combine to 'parse args' later
numFrameSample = frame
numSplit = 3
nCrops = (opt.methodCrop == 'tenCrop') and 10 or 1

print('')

print('Using '..opt.methodCrop)

-- Train/Test split
groupSplit = {}
for sp=1,numSplit do
	if sp==1 then
		table.insert(groupSplit, {setTr = torch.Tensor({{8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}}), setTe = torch.Tensor({{1,2,3,4,5,6,7}})})
	elseif sp==2 then
		table.insert(groupSplit, {setTr = torch.Tensor({{1,2,3,4,5,6,7,15,16,17,18,19,20,21,22,23,24,25}}),setTe = torch.Tensor({{8,9,10,11,12,13,14}})})
	elseif sp==3 then
		table.insert(groupSplit, {setTr = torch.Tensor({{1,2,3,4,5,6,7,8,9,10,11,12,13,14,22,23,24,25}}),setTe = torch.Tensor({{15,16,17,18,19,20,21}})})
	end
end

-- Output information --
outTrain = {}
for sp=1,numSplit do
	table.insert(outTrain, {name = 'data_feat_train_'..dataFolder..'_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..sp..'_part'..idPart..'.t7'})
end

outTest = {}
for sp=1,numSplit do
	table.insert(outTest, {name = 'data_feat_test_'..dataFolder..'_'..opt.methodCrop..'_'..numFrameSample..'f_sp'..sp..'_part'..idPart..'.t7'})
end

------ model selection ------
-- ResNet model (from Torch) ==> need cudnn
modelName = 'model_best.t7'
modelPath = dirModel..modelName

----------------------------------------------
-- 					Functions 				--
----------------------------------------------
if dataFolder == 'RGB' then
	if fpsTr == 10 then
		meanstd =  {mean = { 0.392, 0.376, 0.348 },
	   				std = { 0.241, 0.234, 0.231 }}
	elseif fpsTr == 25 then
		meanstd = {mean = { 0.39234371606738, 0.37576219443075, 0.34801909196893 },
		std = { 0.24149100687454, 0.23453123289779, 0.23117322727131 }}
	else
		meanstd = {mean = {0.39743499656438, 0.38846055375943, 0.35173909269078},
		std = {0.24145608138375, 0.23480329347676, 0.2306657093885}}
	end
elseif dataFolder == 'FlowMap-Brox' then
    if fpsTr == 10 then
		meanstd = {mean = { 0.0091950063390791, 0.4922446721625, 0.49853131534726}, 
					std = { 0.0056229398806939, 0.070845543666524, 0.081589332546496}}
	elseif fpsTr == 25 then
		meanstd = {mean = { 0.0091796917475333, 0.49176131835977, 0.49831646616289 },
               		std = { 0.0056094466799444, 0.070888495268898, 0.081680047609585 }}
    else
    	meanstd = {mean = { 0.0091796917475333, 0.49176131835977, 0.49831646616289 },
               		std = { 0.0056094466799444, 0.070888495268898, 0.081680047609585 }}
	end
elseif dataFolder == 'FlowMap-Brox-crop40' then
    meanstd = {mean = { 0.0091936888040752, 0.49204453841557, 0.49857498097595},
      			std = { 0.0056320802048129, 0.070939325098903, 0.081698516724234}}
   elseif dataFolder == 'FlowMap-Brox-crop20' then
    meanstd = {mean = { 0.0092002901164412, 0.49243926742539, 0.49851170257907},
                std = { 0.0056614266189997, 0.070921186231261, 0.081781848181796}}
elseif dataFolder == 'FlowMap-Brox-M' then
    meanstd = {mean = { 0.951, 0.918, 0.955 },
                std = { 0.043, 0.052, 0.044 }}
elseif dataFolder == 'FlowMap-FlowNet' then
    meanstd = {mean = { 0.009, 0.510, 0.515 },
                std = { 0.007, 0.122, 0.124 }}
elseif dataFolder == 'FlowMap-FlowNet-M' then
    meanstd = {mean = { 0.951, 0.918, 0.955 },
                std = { 0.043, 0.052, 0.044 }}
elseif dataFolder == 'FlowMap-TVL1-crop20' then
    if fpsTr == 10 then
		meanstd = {mean = { 0.0078286737613148, 0.49277467447062, 0.42283539438139 },
	                std = { 0.0049402251681559, 0.060421647049655, 0.058913364961995 }}
	elseif fpsTr == 25 then
		meanstd = {mean = { 0.0078368888567733, 0.49304171615406, 0.42294166284263 },
	                  std = { 0.0049412518723573, 0.060508027119622, 0.058952390342379 }}
	else
		meanstd = {mean = { 0.0077904963214443, 0.49308556329956, 0.42114283484146 },
		std = { 0.0049190714163826, 0.060068045559535, 0.058203296730741 }}
	end
else
    error('no mean and std defined ... ')
end

Crop = (opt.methodCrop == 'tenCrop') and t.TenCrop or t.CenterCrop
transform = t.Compose{
     t.Scale(256),
     t.ColorNormalize(meanstd, nChannel),
     Crop(224),
  }

----------------------------------------------
-- 					Class		        	--
----------------------------------------------
nameClass = paths.dir(pathVideoIn) 
table.sort(nameClass)
table.remove(nameClass,1) -- remove "."
table.remove(nameClass,1) -- remove ".."
numClass = #nameClass -- 101 classes


---- divide the whole dataset into several parts ----
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

----------------------------------------------
-- 					Models		        	--
----------------------------------------------
------ Loading the model ------
print ' '
print '==> Loading the model...'
-- Torch model
print(modelPath)
net = torch.load(modelPath):cuda()

------ model modification ------
-- Remove the fully connected layer
assert(torch.type(net:get(#net.modules)) == 'nn.Linear')
net:remove(#net.modules)

-- -- Evaluate mode
net:evaluate()

print ' '

--====================================================================--
--                     Run all the videos in UCF-101                  --
--====================================================================--
print '==> Processing all the videos...'

-- Load the intermediate feature data or generate a new one --
-- for sp=2,numSplit do
sp = idSplit
	
-- Training data --
if not (opt.save and paths.filep(outTrain[sp].name)) then
	Tr = {} -- output
	Tr.name = {}
	Tr.path = {}
	Tr.featMats = torch.DoubleTensor()
	Tr.labels = torch.DoubleTensor()
	Tr.countVideo = 0
	Tr.countClass = rangeClassPart[idPart][1]-1
else
	Tr = torch.load(outTrain[sp].name) -- output
end

-- Testing data --
if not (opt.save and paths.filep(outTest[sp].name)) then
	Te = {} -- output
	Te.name = {}
	Te.path = {}
	Te.featMats = torch.DoubleTensor()
	Te.labels = torch.DoubleTensor()
	Te.countVideo = 0
	Te.countClass = rangeClassPart[idPart][1]-1
else
	Te = torch.load(outTest[sp].name) -- output
end
collectgarbage()

timerAll = torch.Timer() -- count the whole processing time

if Tr.countClass == numClass and Te.countClass == numClass then
	print('The feature data of split '..sp..' is already in your folder!!!!!!')
else
	for c=Te.countClass+1, numClassAcm[idPart] do
	-- for c=69, 69 do		
		print('Current Class: '..c..'. '..nameClass[c])

		Tr.countClass = Tr.countClass + 1
		Te.countClass = Te.countClass + 1

	  	------ Data paths ------
	  	local pathClassIn = pathVideoIn..nameClass[c]..'/' 
	  	if nameDatabase == 'UCF-101' then
		  	nameSubVideo = paths.dir(pathClassIn)
			table.sort(nameSubVideo) -- ascending order
			table.remove(nameSubVideo,1) -- remove "."
			table.remove(nameSubVideo,1) -- remove ".."
		elseif nameDatabase == 'HMDB-51' then
			local nameTxt = pathTxtSplit..nameClass[c]..'_test_split'..idSplit..'.txt'
			for l in io.lines(nameTxt) do
				table.insert(nameSubVideo,l)
			end
		end

	  	local numSubVideoTotal = #nameSubVideo 

	  	local timerClass = torch.Timer() -- count the processing time for one class
			  	
	  	for sv=1, numSubVideoTotal do
	  	-- for sv=21, 21 do
	      	--------------------
	      	-- Load the video --
	      	--------------------  
	      	---- obtain the video path ----
	      	local videoName
			local videoIndex -- for HMDB-51
		   	local videoPath
			    	
			if nameDatabase == 'UCF-101' then
				videoName = paths.basename(nameSubVideo[sv],'avi')
				videoPath = pathClassIn..videoName..'.avi'
			elseif nameDatabase == 'HMDB-51' then
				local i,j = string.find(nameSubVideo[sv],' ') -- find the location of the group info in the string
			   	videoName = string.sub(nameSubVideo[sv],1,j-5) -- get the video name
				videoIndex = tonumber(string.sub(nameSubVideo[sv],j+1)) -- get the index for training/testing
				if dirVideoIn == 'RGB' then
			       	videoPath = pathClassIn..videoName..'.avi'
			    else
			    	videoPath = pathClassIn..videoName..'_flow.avi'
			    end
			end

			---- read the video to a tensor ----
			local status, height, width, length, fps = videoDecoder.init(videoPath)

			----------------------------------------------
			--           Process with the video         --
			----------------------------------------------
			--== Read the Video ==--
		    -- print('==> Loading the video: '..videoPath)
		    local vidTensor
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
	        		vidTensor = frameTensor
	        	else -- from the second or the following videos
	            	vidTensor = torch.cat(vidTensor,frameTensor,1)	            	
	            end			      
				        
				
			end
			videoDecoder.exit()


			------ Video prarmeters ------
			local numFrame = vidTensor:size(1)

			local numFrameAvailable = numFrame - (numStack-1) -- for 10-stacking
			local numFrameInterval = opt.sampleAll and 1 or torch.floor(numFrameAvailable/numFrameSample)
			numFrameInterval = torch.Tensor({{1,numFrameInterval}}):max() -- make sure larger than 0
			local numFrameUsed = opt.sampleAll and numFrameAvailable or numFrameSample -- choose frame # for one video
			--== Extract Features ==--
			local featMatsVideo = torch.DoubleTensor(nCrops,dimFeat,numFrameUsed):zero() -- 1x2048x25 or 10x2048x25
			--print '==> Generating the feature matrix......'
			for i=1, numFrameUsed do
				local f = (i-1)*numFrameInterval+5 -- current frame sample
				f = torch.Tensor({{f,numFrame-5}}):min() -- make sure we can extract the corresponding 10-stack optcial flow

				local inFrames = vidTensor[{{torch.floor(f-numStack/2)+1,torch.floor(f+numStack/2)},{3-(nChannel-1),3},{},{}}]					              	
				local netInput = torch.Tensor(inFrames:size(1)*nChannel,height,width):zero() -- 20x240x320
				for x=0,numStack-1 do
					netInput[{{x*nChannel+1,(x+1)*nChannel}}] = inFrames[{{x+1},{},{},{}}]
				end
				local I = transform(netInput) -- 20x224x224 or 10x20x224x224 (tenCrop)
				I = I:view(nCrops, table.unpack(I:size():totable())) -- 20x224x224 --> 1x20x224x224
				local feat_now = net:forward(I:cuda()):float() -- 1x2048 or 10x2048

				-- store the feature matrix for this video
				feat_now:resize(nCrops,dimFeat,1)
				featMatsVideo[{{},{},{i}}] = feat_now:double()

			end
			-- print(featMatsVideo:size())

	        ----------------------------------------------
	       	--          Train/Test feature split        --
	       	----------------------------------------------
	        -- find out whether the video is in training set or testing set
			local flagTrain, flagTest = false
			if nameDatabase == 'UCF-101' then
				local i,j = string.find(videoName,'_g') -- find the location of the group info in the string
			    local videoGroup = tonumber(string.sub(videoName,j+1,j+2)) -- get the group#
			    local videoPathLocal = nameClass[c] .. '/' .. videoName .. '.avi'

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


	        if flagTrain and not flagTest then -- training data
	        	Tr.countVideo = Tr.countVideo + 1
	        	Tr.name[Tr.countVideo] = videoName
	        	Tr.path[Tr.countVideo] = videoPathLocal
	        	if Tr.countVideo == 1 then -- the first video
	        		Tr.featMats = featMatsVideo
	        		Tr.labels = torch.DoubleTensor(nCrops):fill(Tr.countClass)
		        else 					-- from the second or the following videos
		        	Tr.featMats = torch.cat(Tr.featMats,featMatsVideo,1)
		        	Tr.labels = torch.cat(Tr.labels,torch.DoubleTensor(nCrops):fill(Tr.countClass),1)
		        end			            	
	        elseif not flagTrain and flagTest then -- testing data
	        	Te.countVideo = Te.countVideo + 1
	        	Te.name[Te.countVideo] = videoName
		       	Te.path[Te.countVideo] = videoPathLocal
	        	if Te.countVideo == 1 then -- the first video
	        		Te.featMats = featMatsVideo
	        		Te.labels = torch.DoubleTensor(nCrops):fill(Te.countClass)
	        	else -- from the second or the following videos
	            	Te.featMats = torch.cat(Te.featMats,featMatsVideo,1)
	            	Te.labels = torch.cat(Te.labels,torch.DoubleTensor(nCrops):fill(Te.countClass),1)
	            end			            	
	        end
	          	
	     	collectgarbage()
	    end
		Te.countClass = c -- save the index

		print('Generated training data#: '..Tr.countVideo)
		print('Generated testing data#: '..Te.countVideo)
		print('The elapsed time for the class '..nameClass[c]..': ' .. timerClass:time().real .. ' seconds')
			  	
		if opt.save then
			print('saving data......')
			torch.save(outTrain[sp].name, Tr)
			torch.save(outTest[sp].name, Te)
		end

		collectgarbage()
		print(' ')
		
	end

	print('The total elapsed time in the split '..sp..': ' .. timerAll:time().real .. ' seconds')

	print('The total training class numbers in the split'..sp..': ' .. Tr.countClass)
	print('The total training video numbers in the split'..sp..': ' .. Tr.countVideo)
	print('The total testing class numbers in the split'..sp..': ' .. Te.countClass)
	print('The total testing video numbers in the split'..sp..': ' .. Te.countVideo)
	
	print ' '

	Tr = nil
	Te = nil
	collectgarbage()
end
