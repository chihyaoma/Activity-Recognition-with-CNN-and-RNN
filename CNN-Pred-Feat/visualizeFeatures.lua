-- Visualize the feature vectors

-- contact:
-- Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
-- Last updated: 02/01/2018

require 'xlua'
require 'torch'
require 'image'

----------------
-- parse args --
----------------
op = xlua.OptionParser('%prog [options]')

op:option{'-fP', '--featurePath', action='store', dest='featurePath',
          help='full source path', default='local'}
op:option{'-dB', '--nameDatabase', action='store', dest='nameDatabase',
          help='used database (UCF-101 | HMDB-51)', default='UCF-101'}
op:option{'-dM', '--dataModality', action='store', dest='dataModality',
          help='modality of the feature (RGB | FlowMap-TVL1-crop20)', default='RGB'}
op:option{'-idS', '--idSplit', action='store', dest='idSplit',
          help='which split used for testing', default=1}     
op:option{'-idV', '--idVideo', action='store', dest='idVideo',
          help='video id of the extracted video', default=0}  
op:option{'-s', '--save', action='store', dest='save',
          help='full path to the saved images', default='feature_images/'}

opt,args = op:parse()

-- 1500: HighJump
-- 2700: PushUps

-- convert strings
idSplit = tonumber(opt.idSplit)
idVideo = tonumber(opt.idVideo)

print('featurePath: '..opt.featurePath)
print('Database: '..opt.nameDatabase)
print('Modality: '..opt.dataModality)
print('split #: '..idSplit)

----------------------------------------------
-- 				  Load Data 			    --
----------------------------------------------
feat = torch.load(opt.featurePath..opt.nameDatabase..'/data_feat_test_'..opt.dataModality..'_centerCrop_25f_sp'..idSplit..'.t7')

if idVideo > 0 then
	print(feat.name[idVideo])
	test = feat.featMats[idVideo]
	image.save(opt.save..'test_'..feat.name[idVideo]..'.png', test)
else 
	for i=1,feat.featMats:size(1) do
		print(feat.name[i])
		test = feat.featMats[i]
		image.save(opt.save..'test_'..feat.name[i]..'.png', test)
	end
end