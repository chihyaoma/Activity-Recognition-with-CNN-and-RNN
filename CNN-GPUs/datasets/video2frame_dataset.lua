-- Load all the videos & Extract all the frames for these videos
-- Target dataset: UCF-101 (flowmaps)

-- ffmpeg usage:
-- Video{
--     [path = string]          -- path to video
--     [load = boolean]         -- loads frames after conversion  [default = true]
--     [delete = boolean]       -- clears (rm) frames after load  [default = true]
--     [destFolder = string]    -- destination folder  [default = out_frames]
--     [silent = boolean]       -- suppress output  [default = false]
-- }

-- author: Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 09/08/2016

-- require 'xlua'
require 'torch'
-- require 'imgraph'
-- require 'nnx'
require 'ffmpeg'
require 'torch-ffmpeg'
require 'image'
require 'cutorch'

----------------
-- parse args --
----------------
op = xlua.OptionParser('%prog [options]')

op:option{'-sP', '--sourcePath', action='store', dest='sourcePath',
          help='source path (local | workstation)', default='local'}
op:option{'-dB', '--nameDatabase', action='store', dest='nameDatabase',
          help='used database (UCF-101 | HMDB-51)', default='HMDB-51'}
op:option{'-sT', '--stream', action='store', dest='stream',
          help='type of stream (FlowMap-Brox | FlowMap-TVL1-crop20 | RGB)', default='RGB'}
op:option{'-iSp', '--idSplit', action='store', dest='idSplit',
          help='index of the split set', default=1}
op:option{'-f', '--fps', action='store', dest='fps',
          help='number of frames per second', default=30}
op:option{'-i', '--devid', action='store', dest='devid',
          help='device ID (if using CUDA)', default=1}
op:option{'-s', '--save', action='store', dest='save',
          help='save the intermediate data or not', default=false}

op:option{'-t', '--time', action='store', dest='seconds',
          help='length to process (in seconds)', default=100}
op:option{'-w', '--width', action='store', dest='width',
          help='resize video, width', default=320}
op:option{'-h', '--height', action='store', dest='height',
          help='resize video, height', default=240}

opt,args = op:parse()
-- convert strings to numbers --
idSplit = tonumber(opt.idSplit)
fps = tonumber(opt.fps)
devid = tonumber(opt.devid)

print('Split #: '..idSplit)
print('source path: '..opt.sourcePath)
print('Database: '..opt.nameDatabase)
print('Stream: '..opt.stream)
-- print('fps: '..fps)

----------------------------------------------
-- 			User-defined parameters			--
----------------------------------------------
numSplit = 3

----------------------------------------------
-- 				Data paths				    --
----------------------------------------------
source = opt.sourcePath -- local | workstation
if source == 'local' then
	dirSource = '/home/cmhung/Code/'
elseif source == 'workstation' then	
	dirSource = '/home/chih-yao/Downloads/'
end

nameDatabase = opt.nameDatabase
pathDatabase = dirSource..'dataset/'..nameDatabase..'/'

-- input
dirVideoIn = opt.stream 
pathVideoIn = pathDatabase .. dirVideoIn .. '/'
pathTxtSplit = pathDatabase .. 'testTrainMulti_7030_splits/' -- for HMDB-51

-- outputs
dirVideoOut = dirVideoIn .. '-frame-sp' .. idSplit
dirTrain = 'train' 
dirTest = 'val' 
pathTrain = pathDatabase .. dirVideoOut .. '/' .. dirTrain .. '/'
pathTest = pathDatabase .. dirVideoOut .. '/' .. dirTest .. '/'

-- make output folders
if not paths.dirp(pathTrain) then
	paths.mkdir(pathTrain)
end

if not paths.dirp(pathTest) then
	paths.mkdir(pathTest)
end

-- Output information --
outTrain = {}
table.insert(outTrain, {name = 'data_'..nameDatabase..'_train_'..dirVideoIn..'_sp'..idSplit..'.t7'})

outTest = {}
table.insert(outTest, {name = 'data_'..nameDatabase..'_val_'..dirVideoIn..'_sp'..idSplit..'.t7'})

----------------------------------------------
--  		       GPU option	 	        --
----------------------------------------------
cutorch.setDevice(opt.devid)
print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
print(sys.COLORS.white ..  ' ')

-- ----------------------------------------------
-- --         Input/Output information         --
-- ----------------------------------------------
-- -- select the number of classes, groups & videos you want to use
-- numClass = 101
-- dimFeat = 2048

-- Train/Test split
if nameDatabase == 'UCF-101' then
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
end
----------------------------------------------
-- 					Class		        	--
----------------------------------------------
nameClass = paths.dir(pathVideoIn) 
table.sort(nameClass) -- ascending order
table.remove(nameClass,1) -- remove "."
table.remove(nameClass,1) -- remove ".."
numClassTotal = #nameClass -- 101 classes

--====================================================================--
--                     Run all the videos in UCF-101                  --
--====================================================================--
fd = io.open('frameNum_'..opt.nameDatabase..'_'..opt.stream..'_sp'..opt.idSplit..'.txt','w')

-- Load the intermediate data or generate a new one --
-- Training data --
if not (opt.save and paths.filep(outTrain[1].name)) then
	Tr = {} -- output
	Tr.name = {}
	Tr.path = {}
	Tr.countVideo = 0
	Tr.countClass = 0
	Tr.c_finished = 0 -- different from countClass since there are also "." and ".."
else
	Tr = torch.load(outTrain[1].name) -- output
end

-- Testing data --
if not (opt.save and paths.filep(outTest[1].name)) then
	Te = {} -- output
	Te.name = {}
	Te.path = {}
	Te.countVideo = 0
	Te.countClass = 0
	Te.c_finished = 0 -- different from countClass since there are also "." and ".."
else
	Te = torch.load(outTest[1].name) -- output
end
collectgarbage()

timerAll = torch.Timer() -- count the whole processing time

if Tr.c_finished == numClassTotal and Te.c_finished == numClassTotal then
	print('The feature data of split '..idSplit..' is already in your folder!!!!!!')
else
	for c=Tr.c_finished+1, numClassTotal do
		
			print('Current Class: '..c..'. '..nameClass[c])
			fd:write('Class: '..c..'. '..nameClass[c], '\n')
			Tr.countClass = Tr.countClass + 1
			Te.countClass = Te.countClass + 1
			countVideoClassTr = 0
			countVideoClassTe = 0
		  	------ Data paths ------
		  	-- input
		  	local pathClassIn = pathVideoIn .. nameClass[c] .. '/' 

		  	local nameSubVideo = {}
		  	if nameDatabase == 'UCF-101' then
		  		nameSubVideo = paths.dir(pathClassIn)
			  	table.sort(nameSubVideo) -- ascending order
		  	elseif nameDatabase == 'HMDB-51' then
		  		local nameTxt = pathTxtSplit..nameClass[c]..'_test_split'..idSplit..'.txt'
		  		for l in io.lines(nameTxt) do
					table.insert(nameSubVideo,l)
				end
		  	end
		  	
		  	local numSubVideoTotal = #nameSubVideo -- videos + '.' + '..'
		  	
		  	-- outputs
		  	local pathClassTrain = pathTrain .. nameClass[c] .. '/'  
		  	local pathClassTest = pathTest .. nameClass[c] .. '/'  

		  	-- make output folders
			if not paths.dirp(pathClassTrain) then
				paths.mkdir(pathClassTrain)
			end

			if not paths.dirp(pathClassTest) then
				paths.mkdir(pathClassTest)
			end

		  	local timerClass = torch.Timer() -- count the processing time for one class

		  	for sv=1, numSubVideoTotal do
			    if nameSubVideo[sv] ~= '.' and nameSubVideo[sv] ~= '..' then
			       	--------------------
			    	-- Load the video --
			    	--------------------  
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
			    
			       	--
			       	-- print('==> Current video: '..videoName)
			    	
			    	local vid = FFmpeg(videoPath)
			    	local vidStats = vid:stats() -- avg_frame_rate, duration, width, height
			    	local video = ffmpeg.Video{path=videoPath, fps=vidStats.avg_frame_rate, length = vidStats.duration, width = vidStats.width, height = vidStats.height, delete=true, destFolder='out_frames',silent=true}
			       	--
			       	local vidTensor = video:totensor{} -- read the whole video & turn it into a 4D tensor

			       	------ Video prarmeters ------
				    local numFrame = vidTensor:size(1)
				    fd:write(numFrame, ' ')
				   	-- local numChannel = vidTensor:size(2)
				    -- local height = vidTensor:size(3)
				    -- local width = vidTensor:size(4)

				    ----------------------
				    -- Train/Test split --
				    ----------------------
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
					  	countVideoClassTr = countVideoClassTr + 1
					    Tr.name[Tr.countVideo] = videoName
					    Tr.path[Tr.countVideo] = videoPathLocal
					    pathClassOut = pathClassTrain
					elseif not flagTrain and flagTest then -- testing data
					  	Te.countVideo = Te.countVideo + 1
					  	countVideoClassTe = countVideoClassTe + 1
		            	Te.name[Te.countVideo] = videoName
			           	Te.path[Te.countVideo] = videoPathLocal
			           	pathClassOut = pathClassTest
			        end

				    -- save all the frames
					for f=1, numFrame do
				        local inFrame = vidTensor[f]
				        nameImage = videoName .. '_' .. tostring(f) 
				        pathImageOut = pathClassOut .. nameImage .. '.png'
				        
				        image.save(pathImageOut, inFrame)
				    end
				    
				    
				end
				collectgarbage()
			end
			Tr.c_finished = c -- save the index
			Te.c_finished = c -- save the index
			-- print('Split: '..idSplit)
			-- print('Finished class#: '..Tr.countClass)
			print('Generated training data# in this class: '..countVideoClassTr)
			print('Generated testing data# in this class: '..countVideoClassTe)
			print('Generated total training data#: '..Tr.countVideo)
			print('Generated total testing data#: '..Te.countVideo)
			print('The elapsed time for the class '..nameClass[c]..': ' .. timerClass:time().real .. ' seconds')
			
			if opt.save then
				torch.save(outTrain[idSplit].name, Tr)
				torch.save(outTest[idSplit].name, Te)
			end

			collectgarbage()
			print(' ')
			fd:write('\n')
		
	end
end       	

print('The total elapsed time in the split '..idSplit..': ' .. timerAll:time().real .. ' seconds')
print('The total training class numbers in the split'..idSplit..': ' .. Tr.countClass)
print('The total training video numbers in the split'..idSplit..': ' .. Tr.countVideo)
print('The total testing class numbers in the split'..idSplit..': ' .. Te.countClass)
print('The total testing video numbers in the split'..idSplit..': ' .. Te.countVideo)
print ' '

Tr = nil
Te = nil
collectgarbage()

fd:close()