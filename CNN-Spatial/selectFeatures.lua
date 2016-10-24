-- Select n-Crop features from the Ten-Crop features

-- author: Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 10/23/2016

require 'xlua'

----------------
-- parse args --
----------------
op = xlua.OptionParser('%prog [options]')

op:option{'-fT', '--featType', action='store', dest='featType',
          help='type of feature (train | test)', default='test'}
op:option{'-sT', '--stream', action='store', dest='stream',
          help='type of stream (RGB | FlowMap-TVL1-crop20 | FlowMap-Brox)', default='RGB'}
op:option{'-iC', '--inputCrop', action='store', dest='inputCrop',
          help='input cropping method (tenCrop | centerCropFlip)', default='tenCrop'}
op:option{'-oC', '--outputCrop', action='store', dest='outputCrop',
          help='output cropping method (centerCrop | centerCropFlip | centerCropMirror)', default='centerCropFlip'}

opt,args = op:parse()

featType = opt.featType
stream = opt.stream
inputCrop = opt.inputCrop
outputCrop  = opt.outputCrop

------------------
-- main program --
------------------

inputName = 'data_feat_'..featType..'_'..stream..'_'..inputCrop..'_25f_sp1'
outputName = 'data_feat_'..featType..'_'..stream..'_'..outputCrop..'_25f_sp1'

if inputCrop == 'tenCrop' then
	nCropsIn = 10
elseif inputCrop == 'centerCropFlip' then
	nCropsIn = 2
else
	nCropsIn = 1
end

if outputCrop == 'tenCrop' then
	nCropsOut = 10
elseif outputCrop == 'centerCropFlip' then
	nCropsOut = 2
else
	nCropsOut = 1
end

print('Loading features: '..inputName)
dataIn = torch.load(inputName..'.t7')

numVideo = dataIn.labels:size(1)/nCropsIn

if outputCrop == 'centerCropFlip' then
	idx_Selected = {1,6} -- 1: centerCrop; 6: centerCrop + horizontal flipping
elseif inputCrop == 'centerCropFlip' and outputCrop == 'centerCropMirror' then
	idx_Selected = {2} -- 1: centerCrop; 2: centerCrop + horizontal flipping
end

dataOut = {}
dataOut.featMats = torch.zeros(numVideo*nCropsOut,dataIn.featMats:size(2),dataIn.featMats:size(3)):double()
dataOut.labels = torch.zeros(numVideo*nCropsOut):double()

print('Selecting features......')
for i=1,numVideo do
	if i%500 == 0 then
		print(i..'/'..numVideo)
	end

	local featsIn = dataIn.featMats[{{(i-1)*nCropsIn+1,i*nCropsIn}}]
	local labelsIn = dataIn.labels[{{(i-1)*nCropsIn+1,i*nCropsIn}}]
	for j=1,nCropsOut do
		dataOut.featMats[(i-1)*nCropsOut+j] = featsIn[idx_Selected[j]]
		dataOut.labels[(i-1)*nCropsOut+j] = labelsIn[idx_Selected[j]]
	end
end

print('Saving features: '..outputName)
torch.save(outputName..'.t7',dataOut)