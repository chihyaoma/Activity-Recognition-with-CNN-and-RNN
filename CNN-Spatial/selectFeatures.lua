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
          help='input cropping method', default='tenCrop'}
op:option{'-oC', '--outputCrop', action='store', dest='outputCrop',
          help='output cropping method (centerCrop | centerCropFlip)', default='centerCropFlip'}

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

nCropsIn = (inputCrop == 'tenCrop') and 10 or 1
nCropsOut = (outputCrop == 'centerCropFlip') and 2 or 1

print('Loading features: '..inputName)
dataIn = torch.load(inputName..'.t7')

numVideo = dataIn.labels:size(1)/nCropsIn

idx_Selected = {1,6} -- 1: centerCrop; 6: centerCrop + horizontal flipping

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