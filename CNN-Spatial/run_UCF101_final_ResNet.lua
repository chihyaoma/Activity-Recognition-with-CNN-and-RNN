-- Activity-Recognition-with-CNN-and-RNN
-- https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN

-- Load all the videos & Generate a feature matrix for each video
-- Select all the videos which have the frame numbers at least "numFrameMin"
-- No need to specify the video number
-- Follow the split sets provided in the UCF-101 website
-- Generate the name list corresponding to each video as well
-- load ResNet model (We use Res-101 now)
-- Process images based on the ResNet sample codes

-- Reference:
-- Khurram Soomro, Amir Roshan Zamir and Mubarak Shah, 
-- "UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild.", 
-- CRCV-TR-12-01, November, 2012. 

-- ffmpeg usage:
-- Video{
--     [path = string]          -- path to video
--     [width = number]         -- width  [default = 224]
--     [height = number]        -- height  [default = 224]
--     [zoom = number]          -- zoom factor  [default = 1]
--     [fps = number]           -- frames per second  [default = 25]
--     [length = number]        -- length, in seconds  [default = 2]
--     [seek = number]          -- seek to pos. in seconds  [default = 0]
--     [channel = number]       -- video channel  [default = 0]
--     [load = boolean]         -- loads frames after conversion  [default = true]
--     [delete = boolean]       -- clears (rm) frames after load  [default = true]
--     [encoding = string]      -- format of dumped frames  [default = png]
--     [tensor = torch.Tensor]  -- provide a packed tensor (NxCxHxW or NxHxW), that bypasses path
--     [destFolder = string]    -- destination folder  [default = out_frames]
--     [silent = boolean]       -- suppress output  [default = false]
-- }

-- contact:
-- Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
-- Chih-Yao Ma at <cyma@gatech.edu>
-- Last updated: 7/2/2016

require 'xlua'
require 'torch'
require 'ffmpeg'
require 'image'
require 'nn'
require 'cudnn' 
require 'cunn'
t = require './transforms'

----------------------------------------------
-- 					Functions 				--
----------------------------------------------
meanstd = {
   mean = { 0.392, 0.376, 0.348 },
   std = { 0.241, 0.234, 0.231 },
	}

transform = t.Compose{
     t.Scale(256),
     t.ColorNormalize(meanstd),
     t.CenterCrop(224),
  }

----------------------------------------------
--         Input/Output information         --
----------------------------------------------
-- select the number of classes, groups & videos you want to use
numClass = 101
dimFeat = 2048

----------------------------------------------
-- 				Data paths				    --
----------------------------------------------
dirModel = '/home/chih-yao/Downloads/'
dirDatabase = '/home/chih-yao/Downloads/dataset/UCF-101/RGB/'

----------------------------------------------
-- 			User-defined parameters			--
----------------------------------------------
numFrameMin = 50
numSplit = 1

-- Train/Test split
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
outTrain = {}
for sp=1,numSplit do
	table.insert(outTrain, {name = 'data_UCF101_train_'..sp..'.t7'})
end

outTest = {}
for sp=1,numSplit do
	table.insert(outTest, {name = 'data_UCF101_test_'..sp..'.t7'})
end

------ model selection ------
-- ResNet model (from Torch) ==> need cudnn
modelName = 'model_best_Steve.t7'
modelPath = dirModel..modelName

----------------
-- parse args --
----------------
op = xlua.OptionParser('%prog [options]')
-- op:option{'-c', '--camera', action='store', dest='camidx',
--           help='camera index: /dev/videoIDX (if no video given)', 
--           default=0}
-- op:option{'-v', '--video', action='store', dest='video',
--           help='video file to process', default=videoPath}
op:option{'-f', '--fps', action='store', dest='fps',
          help='number of frames per second', default=25}
op:option{'-t', '--time', action='store', dest='seconds',
          help='length to process (in seconds)', default=2}
op:option{'-w', '--width', action='store', dest='width',
          help='resize video, width', default=320}
op:option{'-h', '--height', action='store', dest='height',
          help='resize video, height', default=240}
op:option{'-z', '--zoom', action='store', dest='zoom',
          help='display zoom', default=1}
op:option{'-fe', '--feat', action='store', dest='feat',
          help='option for generating features', default=true}
op:option{'-pr', '--pred', action='store', dest='pred',
          help='option for prediction', default=flase}
op:option{'-p', '--type', action='store', dest='type',
          help='option for CPU/GPU', default='cuda'}
op:option{'-i', '--devid', action='store', dest='devid',
          help='device ID (if using CUDA)', default=1}      
opt,args = op:parse()

----------------------------------------------
-- 					Class		        	--
----------------------------------------------
nameClass = paths.dir(dirDatabase) 
table.sort(nameClass)
numClassTotal = #nameClass -- 101 classes + "." + ".."

----------------------------------------------
-- 					Models		        	--
----------------------------------------------
------ Loading the model ------
print ' '
print '==> Loading the model...'
-- Torch model
net = torch.load(modelPath):cuda()

-- -- Evaluate mode
net:evaluate()

------ model modification ------
if opt.feat then
	-- Remove the fully connected layer
    assert(torch.type(net:get(#net.modules)) == 'nn.Linear')
    net:remove(#net.modules)
end

if opt.pred then
	softMaxLayer = cudnn.SoftMax():cuda()
    net:add(softMaxLayer)
end

print(net)
print ' '

----------------------------------------------
--  		       GPU option	 	        --
----------------------------------------------
cutorch.setDevice(opt.devid)
print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
print(sys.COLORS.white ..  ' ')

----------------------------------------------
-- 			Loading ImageNet labels	  		--
----------------------------------------------
if opt.pred then
	-- imagenetLabel = require './imagenet'
	ucf101Label = require './ucf-101'
end
print ' '

--====================================================================--
--                     Run all the videos in UCF-101                  --
--====================================================================--
print '==> Processing all the videos...'

-- Load the intermediate feature data or generate a new one --
for sp=1,numSplit do
	-- Training data --
	if not paths.filep(outTrain[sp].name) then
		Tr = {} -- output
		Tr.name = {}
		Tr.path = {}
		Tr.featMats = torch.DoubleTensor()
		Tr.labels = torch.DoubleTensor()
		Tr.countVideo = 0
		Tr.countClass = 0
		Tr.c_finished = 0 -- different from countClass since there are also "." and ".."
	else
		Tr = torch.load(outTrain[sp].name) -- output
	end

	-- Testing data --
	if not paths.filep(outTest[sp].name) then
		Te = {} -- output
		Te.name = {}
		Te.path = {}
		Te.featMats = torch.DoubleTensor()
		Te.labels = torch.DoubleTensor()
		Te.countVideo = 0
		Te.countClass = 0
		Te.c_finished = 0 -- different from countClass since there are also "." and ".."
	else
		Te = torch.load(outTest[sp].name) -- output
	end
	collectgarbage()

	timerAll = torch.Timer() -- count the whole processing time

	if Tr.countClass == numClass and Te.countClass == numClass then
		print('The feature data of split '..sp..' is already in your folder!!!!!!')
	else
		for c=Tr.c_finished+1, numClassTotal do
			if nameClass[c] ~= '.' and nameClass[c] ~= '..' then
				print('Current Class: '..c..'. '..nameClass[c])
				Tr.countClass = Tr.countClass + 1
				Te.countClass = Te.countClass + 1
			  	------ Data paths ------
			  	local dirClass = dirDatabase..nameClass[c]..'/' 
			  	local nameSubVideo = paths.dir(dirClass)
			  	local numSubVideoTotal = #nameSubVideo -- videos + '.' + '..'
			  	
			  	local timerClass = torch.Timer() -- count the processing time for one class
			  	
			    for sv=1, numSubVideoTotal do
			      	--------------------
			      	-- Load the video --
			      	--------------------  
			      	if nameSubVideo[sv] ~= '.' and nameSubVideo[sv] ~= '..' then
			        	local videoName = nameSubVideo[sv]
			        	local videoPath = dirClass..videoName
			        	
			        	print('==> Loading the video: '..videoName)
			        	-- TODO --
			        	-- now:     fixed frame rate
			        	-- future:  fixed frame #
			        	
			        	local video = ffmpeg.Video{path=videoPath, width=opt.width, height=opt.height, fps=opt.fps, length=opt.seconds, delete=true, destFolder='out_frames',silent=true}
			        	
			        	-- --video:play{} -- play the video
			        	local vidTensor = video:totensor{} -- read the whole video & turn it into a 4D tensor
				        
				        ------ Video prarmeters ------
				        local numFrame = vidTensor:size(1)
				        
				        if numFrame >= numFrameMin then
				          	--countVideo = countVideo + 1 -- save this video only when frame# >= min. frame#
				          	local featMatsVideo = torch.DoubleTensor(1,dimFeat,numFrameMin):zero()
				          	----------------------------------------------
				          	--           Process with the video         --
				          	----------------------------------------------
				          	if opt.pred then -- prediction (useless now)
				            	print '==> Begin predicting......'
				            	for f=1, numFrameMin do
				              		local inFrame = vidTensor[f]
				              		print('frame '..tostring(f)..': ')
				              		local I = transform(inFrame)
				              		I = I:view(1, table.unpack(I:size():totable()))
				              		local output = net:forward(I:cuda())

				              		local N = 5
									local probLog, predLabels = output:float():topk(N, true, true)
				              		print(probLog)
				              		print(probLog, ucf101Label[predLabels[1][1]])
				              		print('=================================')
				            	end
				          	end
				          	--
				          	if opt.feat then -- feature extraction
				            	--print '==> Generating the feature matrix......'
				            	for f=1, numFrameMin do
				              		local inFrame = vidTensor[f]

				              		--print('frame '..tostring(f)..'...')
				              		--local feat = gen_feature(inFrame, net, opt)
				              		
				          			------ Image pre-processing ------
				              		local I = transform(inFrame)
				              		-- View as mini-batch of size 1
    								I = I:view(1, table.unpack(I:size():totable()))
								    -- Get the output of the layer before the (removed) fully connected layer
								    local feat = net:forward(I:cuda()):squeeze(1)
				              		--
				              		-- store the feature matrix for this video
							  		feat:resize(1,torch.numel(feat),1)

							  		featMatsVideo[{{},{},{f}}] = feat:double()
				            	end

				            	----------------------------------------------
				          		--          Train/Test feature split        --
				          		----------------------------------------------
				            	-- store the feature and label for the whole dataset
				            	local i,j = string.find(videoName,'_g') -- find the location of the group info in the string
				            	local videoGroup = tonumber(string.sub(videoName,j+1,j+2)) -- get the group#
				            	local videoPathLocal = nameClass[c]..'/'..videoName
				            	
				            	if groupSplit[sp].setTe:eq(videoGroup):sum() == 0 then -- training data
				            		Tr.countVideo = Tr.countVideo + 1
				            		Tr.name[Tr.countVideo] = videoName
				            		Tr.path[Tr.countVideo] = videoPathLocal
				            		if Tr.countVideo == 1 then -- the first video
				            			Tr.featMats = featMatsVideo
				            			Tr.labels = torch.DoubleTensor(1):fill(Tr.countClass)
				            		else 					-- from the second or the following videos
				            			Tr.featMats = torch.cat(Tr.featMats,featMatsVideo,1)
				            			Tr.labels = torch.cat(Tr.labels,torch.DoubleTensor(1):fill(Tr.countClass),1)
				            		end			            	
				            	else -- testing data
				            		Te.countVideo = Te.countVideo + 1
				            		Te.name[Te.countVideo] = videoName
					            	Te.path[Te.countVideo] = videoPathLocal
				            		if Te.countVideo == 1 then -- the first video
				            			Te.featMats = featMatsVideo
				            			Te.labels = torch.DoubleTensor(1):fill(Te.countClass)
				            		else 					-- from the second or the following videos
				            			Te.featMats = torch.cat(Te.featMats,featMatsVideo,1)
				            			Te.labels = torch.cat(Te.labels,torch.DoubleTensor(1):fill(Te.countClass),1)
				            		end			            	
				            	end
				            	
				          	end
				        end
			      	end
			      	collectgarbage()
			    end
			    
				Tr.c_finished = c -- save the index
				Te.c_finished = c -- save the index
				print('Split: '..sp)
				print('Finished class#: '..Tr.countClass)
				print('Generated training data#: '..Tr.countVideo)
				print('Generated testing data#: '..Te.countVideo)
			  	print('The elapsed time for the class '..nameClass[c]..': ' .. timerClass:time().real .. ' seconds')
			  	torch.save(outTrain[sp].name, Tr)
			  	torch.save(outTest[sp].name, Te)

			  	collectgarbage()
			  	print(' ')
			end
		end
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
