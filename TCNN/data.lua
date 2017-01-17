-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- Load Data and separate training and testing samples

-- TODO:
-- 1. subtract by mean (?)
-- 2. cross-validation 

-- modified by Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 09/17/2016


require 'torch'   -- torch

----------------------------------------------
-- 				  Data paths                --
----------------------------------------------
idSplit = tonumber(opt.splitId)
print('split #: '..idSplit)

idPart = tonumber(opt.idPart)

source = opt.sourcePath
if source == 'local' then
	dirSource = '/home/cmhung/Code/'
elseif source == 'workstation' then	
	dirSource = '/home/chih-yao/Downloads/'
end

if idPart == 0 then
  dirFeature = dirSource..'Features/'
else
  print('Data part '..idPart)
  dirFeature = dirSource..'Features/parts/sp'..idSplit..'/'
end

----------------------------------------------
-- 			User-defined parameters		    --
----------------------------------------------
--ratioTrain = 0.8
nframeAll = 25

----------------------------------------------
-- 			      Load Data	    	        --
----------------------------------------------
-- Load all the feature matrices & labels

nameType = 'FlowMap-TVL1-crop20' -- FlowMap-TVL1-crop20 | RGB
methodCrop = opt.methodCrop

if idPart == 0 then
  print('==> load training data')
  dataTrain = torch.load(dirFeature..'data_feat_train_'..nameType..'_'..methodCrop..'_'..nframeAll..'f_sp'..idSplit..'.t7')
  print('==> load test data')
  dataTest = torch.load(dirFeature..'data_feat_test_'..nameType..'_'..methodCrop..'_'..nframeAll..'f_sp'..idSplit..'.t7')
else
  print('==> load training data')
  dataTrain = torch.load(dirFeature..'data_feat_train_'..nameType..'_'..methodCrop..'_'..nframeAll..'f_sp'..idSplit..'_part'..idPart..'.t7')
  print('==> load test data')
  dataTest = torch.load(dirFeature..'data_feat_test_'..nameType..'_'..methodCrop..'_'..nframeAll..'f_sp'..idSplit..'_part'..idPart..'.t7')
end


-- information for the data
local dimFeat = dataTrain.featMats:size(2)
local numFrame = dataTrain.featMats:size(3)
local trsize = (#dataTrain.labels)[1]
local tesize = (#dataTest.labels)[1]
local shuffleTrain = torch.randperm(trsize)
local shuffleTest = torch.randperm(tesize)

-- create the train set:
trainData = {
   data = torch.Tensor(trsize, dimFeat, numFrame),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}

-- create testing set:
testData = {
      data = torch.Tensor(tesize, dimFeat, numFrame),
      labels = torch.Tensor(tesize),
      size = function() return tesize end
   }

-- classes in UCF-11
-- classes = {'basketball','biking','diving','golf_swing','horse_riding','soccer_juggling',
-- 			'swing','tennis_swing','trampoline_jumping','volleyball_spiking','walking'}

-- classes in UCF-101
classes = {
"BoxingSpeedBag", "Surfing", "FloorGymnastics", "IceDancing", "Lunges", "Swing", "SkyDiving", "MilitaryParade", "PlayingPiano", "Punch",
"HulaHoop", "VolleyballSpiking", "Skijet", "JavelinThrow", "LongJump", "Mixing", "Shotput", "BandMarching", "Kayaking", "StillRings",
"PushUps", "Archery", "FieldHockeyPenalty", "BoxingPunchingBag", "PlayingCello", "FrontCrawl", "Billiards", "Rowing", "ApplyLipstick", "TrampolineJumping",
"CuttingInKitchen", "BodyWeightSquats", "JugglingBalls", "Nunchucks", "JumpRope", "PlayingViolin", "PlayingGuitar", "YoYo", "SumoWrestling", "SoccerJuggling",
"CliffDiving", "CricketBowling", "PlayingDhol", "HorseRiding", "BabyCrawling", "PlayingSitar", "TaiChi", "BenchPress", "PommelHorse", "BrushingTeeth",
"Hammering", "PlayingTabla", "HandstandWalking", "Typing", "CleanAndJerk", "TennisSwing", "CricketShot", "BlowDryHair", "HeadMassage", "BalanceBeam",
"TableTennisShot", "MoppingFloor", "Drumming", "PlayingFlute", "FrisbeeCatch", "ApplyEyeMakeup", "SkateBoarding", "BaseballPitch", "SoccerPenalty", "ThrowDiscus",
"RopeClimbing", "HorseRace", "HighJump", "PullUps", "Diving", "BreastStroke", "ParallelBars", "WalkingWithDog", "PizzaTossing", "BlowingCandles",
"GolfSwing", "PoleVault", "UnevenBars", "HandstandPushups", "JumpingJack", "WallPushups", "WritingOnBoard", "Skiing", "Bowling", "BasketballDunk",
"SalsaSpin", "ShavingBeard", "Basketball", "Knitting", "RockClimbingIndoor", "Haircut", "Biking", "Fencing", "Rafting", "PlayingDaf",
"HammerThrow"
}
table.sort(classes)
-- local numClass = #classes

-- -- class ID range for the part
-- local subClasses = {}
-- local numPart = 4
-- local numClassSub = torch.floor(numClass/numPart)
-- local idStart, idEnd -- the range of this part
-- if idPart==numPart then
--   idStart = (idPart-1)*numClassSub+1
--   idEnd = numClass
-- else
--   idStart = (idPart-1)*numClassSub+1
--   idEnd = idPart*numClassSub
-- end

-- for i = idStart,idEnd do
--   table.insert(subClasses,classes[i])
-- end

-- print(idStart)
-- print(idEnd)
-- print(subClasses)
-- -- classes = subClasses

-- save data
for i = 1,trsize do
    trainData.data[i] = dataTrain.featMats[shuffleTrain[i]]:clone()
    trainData.labels[i] = dataTrain.labels[shuffleTrain[i]]
end

for i= 1,tesize do
   -- testData.data[i] = dataTest.featMats[shuffleTest[i]]:clone()
   -- testData.labels[i] = dataTest.labels[shuffleTest[i]]
   testData.data[i] = dataTest.featMats[i]:clone()
   testData.labels[i] = dataTest.labels[i]
end

print(trainData)
print(testData)

dataTrain = nil
destTrain = nil
collectgarbage()


return {
   trainData = trainData,
   testData = testData,
   classes = classes
}

