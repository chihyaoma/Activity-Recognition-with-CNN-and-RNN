-- Visualize the feature vectors

-- contact:
-- Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>
-- Last updated: 02/03/2018

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
op:option{'-sP', '--savePath', action='store', dest='savePath',
          help='full path to the saved images', default='feature_images/'}
op:option{'-s', '--save', action='store', dest='save',
          help='save the images or not', default='No'}
op:option{'-n', '--normalize', action='store', dest='normalize',
          help='normalize the features or not', default='Yes'}

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
if opt.save == 'Yes' then
	print('will save images')
end
if opt.normalize == 'Yes' then
	print('will normalize the features')
end

-- load the label
ucf101Label = require './ucf-101'
table.sort(ucf101Label)

----------------------------------------------
-- 				  Load Data 			    --
----------------------------------------------
feat = torch.load(opt.featurePath..opt.nameDatabase..'/data_feat_test_'..opt.dataModality..'_centerCrop_25f_sp'..idSplit..'.t7')
-- normalize the features 
if opt.normalize == 'Yes' then
	featMats = (feat.featMats-torch.mean(feat.featMats))/torch.std(feat.featMats)
else
	featMats = feat.featMats
end

if idVideo > 0 then
	print(feat.name[idVideo])
	test = featMats[idVideo]

	var_test = torch.mean(torch.var(test,2))
	std_test = torch.mean(torch.std(test,2))
	print(var_test, std_test)
	if opt.save == 'Yes' then
		image.save(opt.savePath..'test_'..feat.name[idVideo]..'.png', test)
	end
else
	var_std_allClass = torch.zeros(#ucf101Label,2)
	for j=1, #ucf101Label do
		print(ucf101Label[j])
		var_allVideo = torch.Tensor()
		std_allVideo = torch.Tensor()
		for i=1,featMats:size(1) do
			if string.match(feat.name[i], ucf101Label[j]) then
				-- print(feat.name[i])
				test = featMats[i]
				var_test = torch.mean(torch.var(test,2))
				std_test = torch.mean(torch.std(test,2))
				var_allVideo = torch.cat(var_allVideo, torch.Tensor({var_test}))
				std_allVideo = torch.cat(std_allVideo, torch.Tensor({std_test}))
			end
		end
		var_std_allClass[j][1] = torch.mean(var_allVideo)
		var_std_allClass[j][2] = torch.mean(std_allVideo)
	end
	print('variance & standard deviation of each class for '..opt.dataModality)
	print(var_std_allClass)
	torch.save('var_std_'..opt.dataModality..'.txt', var_std_allClass, 'ascii')
end