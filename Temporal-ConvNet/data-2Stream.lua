-- Georgia Institute of Technology 
-- CS8803DL Spring 2016 (Instructor: Zsolt Kira)
-- Final Project: Video Classification

-- Load Data and separate training and testing samples

-- TODO:
-- 1. subtract by mean (?)
-- 2. cross-validation 

-- modified by Min-Hung Chen
-- contact: cmhungsteve@gatech.edu
-- Last updated: 10/11/2016


require 'torch'   -- torch

----------------------------------------------
--      User-defined parameters     --
----------------------------------------------
numStream = 2
nframeAll = 25

----------------------------------------------
-- 				  Data paths                --
----------------------------------------------
source = opt.sourcePath
if source == 'local' then
	dirSource = '/home/cmhung/Code/'
elseif source == 'workstation' then	
	dirSource = '/home/chih-yao/Downloads/'
end

dirFeature = dirSource..'Features/'..opt.dataset..'/'
--dirFeature = dirSource..'Features/'

----------------------------------------------
-- 			User-defined parameters		    --
----------------------------------------------
idSplit = tonumber(opt.splitId)
print('split #: '..idSplit)

----------------------------------------------
-- 			      Load Data	    	        --
----------------------------------------------
-- Load all the feature matrices & labels

nameType = {}
table.insert(nameType, 'RGB')
table.insert(nameType, 'FlowMap-TVL1-crop20')

methodCrop = opt.methodCrop

---- Load testing data ----
dataTestAll = {}

for nS=1,numStream do
  print('==> load test data: '..'data_feat_test_'..nameType[nS]..'_'..methodCrop..'_'..nframeAll..'f_sp'..idSplit..'.t7')
  table.insert(dataTestAll, torch.load(dirFeature..'data_feat_test_'..nameType[nS]..'_'..methodCrop..'_'..nframeAll..'f_sp'..idSplit..'.t7'))
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
  print('==> load training data: '..'data_feat_train_'..nameType[nS]..'_'..methodCrop..'_'..nframeAll..'f_sp'..idSplit..'.t7')
  table.insert(dataTrainAll, torch.load(dirFeature..'data_feat_train_'..nameType[nS]..'_'..methodCrop..'_'..nframeAll..'f_sp'..idSplit..'.t7'))
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

-- --------------------
-- -- pre-processing --
-- --------------------
-- -- calculate residual
-- local dataTr_rshift = torch.zeros(trainData.data:size())
-- dataTr_rshift[{{},{},{2,numFrame}}] = trainData.data[{{},{},{1,numFrame-1}}]
-- dataTr_rshift[{{},{},{1}}] = trainData.data[{{},{},{numFrame}}]
-- trainData.data = trainData.data - dataTr_rshift

-- local dataTe_rshift = torch.zeros(testData.data:size())
-- dataTe_rshift[{{},{},{2,numFrame}}] = testData.data[{{},{},{1,numFrame-1}}]
-- dataTe_rshift[{{},{},{1}}] = testData.data[{{},{},{numFrame}}]
-- testData.data = testData.data - dataTe_rshift


-- -- calculate mean, std
-- local dataMean = trainData.data:mean(1):mean(3)
-- local dataStd = trainData.data:std(1):std(3)
-- -- replicate to match the dimension
-- local dataMean_repTr = dataMean:expand(trsize, dimFeat, numFrame)
-- local dataStd_repTr = dataStd:expand(trsize, dimFeat, numFrame)
-- local dataMean_repTe = dataMean:expand(tesize, dimFeat, numFrame)
-- local dataStd_repTe = dataStd:expand(tesize, dimFeat, numFrame)

-- trainData.data = torch.cdiv((trainData.data - dataMean_repTr),dataStd_repTr)
-- testData.data = torch.cdiv((testData.data - dataMean_repTe),dataStd_repTe)

print(trainData)
print(testData)
print(trainData.data:mean())
print(testData.data:mean())
----------------------------
--         Classes        --
----------------------------
------ UCF-11 ----
-- classes = {'basketball','biking','diving','golf_swing','horse_riding','soccer_juggling',
-- 			'swing','tennis_swing','trampoline_jumping','volleyball_spiking','walking'}

------ UCF-101 & HMDB-51 ----
if opt.dataset == 'UCF-101' then
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
elseif opt.dataset == 'HMDB-51' then
  classes = {
  "brush_hair", "kick_ball", "ride_horse", "pour", "jump", "smile", "stand", "shake_hands", "flic_flac", 
  "golf", "wave", "cartwheel", "clap", "dive", "ride_bike", "turn", "chew", "draw_sword", "push", "hug", 
  "shoot_gun", "pullup", "sit", "smoke", "somersault", "shoot_bow", "kick", "kiss", "shoot_ball", "run", 
  "walk", "situp", "sword", "drink", "pushup", "fall_floor", "climb", "hit", "laugh", "eat", "pick", 
  "swing_baseball", "dribble", "talk", "climb_stairs", "catch", "fencing", "punch", "throw", 
  "sword_exercise", "handstand"
}
end
table.sort(classes)

return {
   trainData = trainData,
   testData = testData,
   classes = classes,
}

