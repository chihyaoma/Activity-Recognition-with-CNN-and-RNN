-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- Load Data and separate training and testing samples

-- TODO:
-- 1. subtract by mean (?)
-- 2. cross-validation 

-- modified by Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 09/20/2016


require 'torch'   -- torch

----------------------------------------------
--      User-defined parameters     --
----------------------------------------------
numStream = 2
local nframe = 50

----------------------------------------------
-- 				  Data paths                --
----------------------------------------------
source = 'local' -- local | workstation
if source == 'local' then
	dirSource = '/home/cmhung/Code/'
elseif source == 'workstation' then	
	dirSource = '/home/chih-yao/Downloads/'
end

dirFeature = dirSource..'Features/'
-- dirFeature = dirSource..'Features/feat-10fps/'

----------------------------------------------
-- 			User-defined parameters		    --
----------------------------------------------
--ratioTrain = 0.8

----------------------------------------------
-- 			      Load Data	    	        --
----------------------------------------------
-- Load all the feature matrices & labels
idSplit = 1

nameType = {}
table.insert(nameType, 'FlowMap-TVL1-crop20')
table.insert(nameType, 'RGB')

methodCrop = 'tenCrop' -- tenCrop | centerCrop

---- Load testing data ----
dataTestAll = {}

for nS=1,numStream do
  print('==> load test data: '..'data_feat_test_'..nameType[nS]..'_'..methodCrop..'_'..nframe..'f_sp'..idSplit..'.t7')
  table.insert(dataTestAll, torch.load(dirFeature..'data_feat_test_'..nameType[nS]..'_'..methodCrop..'_'..nframe..'f_sp'..idSplit..'.t7'))
end

-- concatenation
dataTest = {}
dataTest.featMats = torch.cat(dataTestAll[1].featMats,dataTestAll[2].featMats,2)
dataTest.labels = dataTestAll[1].labels
dataTestAll = nil
collectgarbage()

-- information for the data
local dimFeat = dataTest.featMats:size(2)
local numFrame = dataTest.featMats:size(3)
local tesize = (#dataTest.labels)[1]
local shuffleTest = torch.randperm(tesize)

-- create testing set:
testData = {
      data = torch.Tensor(tesize, dimFeat, numFrame),
      labels = torch.Tensor(tesize),
      size = function() return tesize end
   }

for i= 1,tesize do
   -- testData.data[i] = dataTest.featMats[shuffleTest[i]]:clone()
   -- testData.labels[i] = dataTest.labels[shuffleTest[i]]
   testData.data[i] = dataTest.featMats[i]:clone()
   testData.labels[i] = dataTest.labels[i]
end

dataTest = nil
collectgarbage()


---- Load training data ----
dataTrainAll = {}

for nS=1,numStream do
  print('==> load training data: '..'data_feat_train_'..nameType[nS]..'_'..methodCrop..'_'..nframe..'f_sp'..idSplit..'.t7')
  table.insert(dataTrainAll, torch.load(dirFeature..'data_feat_train_'..nameType[nS]..'_'..methodCrop..'_'..nframe..'f_sp'..idSplit..'.t7'))
end

dataTrain = {}
dataTrain.featMats = torch.cat(dataTrainAll[1].featMats,dataTrainAll[2].featMats,2)
dataTrain.labels = dataTrainAll[1].labels
dataTrainAll = nil
collectgarbage()

-- information for the data
-- local dimFeat = dataTrain.featMats:size(2)
-- local numFrame = dataTrain.featMats:size(3)
local trsize = (#dataTrain.labels)[1]
local shuffleTrain = torch.randperm(trsize)

-- create the train set:
trainData = {
   data = torch.Tensor(trsize, dimFeat, numFrame),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}

for i = 1,trsize do
    trainData.data[i] = dataTrain.featMats[shuffleTrain[i]]:clone()
    trainData.labels[i] = dataTrain.labels[shuffleTrain[i]]
end

dataTrain = nil
collectgarbage()

print(trainData)
print(testData)

------ classes in UCF-11 ----
-- classes = {'basketball','biking','diving','golf_swing','horse_riding','soccer_juggling',
-- 			'swing','tennis_swing','trampoline_jumping','volleyball_spiking','walking'}

---- classes in UCF-101 ----
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

return {
   trainData = trainData,
   testData = testData,
   classes = classes
}

