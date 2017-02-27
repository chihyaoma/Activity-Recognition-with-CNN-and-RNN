----------------------------------------------------------------
--  Activity-Recognition-with-CNN-and-RNN
--  https://github.com/chihyaoma/Activity-Recognition-with-CNN-and-RNN
--
-- 
--  This is a testing code for implementing the RNN model with LSTM 
--  written by Chih-Yao Ma. 
-- 
--  The code will take feature vectors (from CNN model) from contiguous 
--  frames and train against the ground truth, i.e. the labeling of video classes. 
-- 
--  Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------
require 'torch'
require 'sys'

if opt.dataset == 'ucf101' then
    -- classes in UCF-101
    classes = {
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam", "BandMarching",
    "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress", "Biking", "Billiards",
    "BlowDryHair", "BlowingCandles", "BodyWeightSquats", "Bowling", "BoxingPunchingBag",
    "BoxingSpeedBag", "BreastStroke", "BrushingTeeth", "CleanAndJerk", "CliffDiving",
    "CricketBowling", "CricketShot", "CuttingInKitchen", "Diving", "Drumming", "Fencing",
    "FieldHockeyPenalty", "FloorGymnastics", "FrisbeeCatch", "FrontCrawl", "GolfSwing",
    "Haircut", "HammerThrow", "Hammering", "HandstandPushups", "HandstandWalking",
    "HeadMassage", "HighJump", "HorseRace", "HorseRiding", "HulaHoop", "IceDancing",
    "JavelinThrow", "JugglingBalls", "JumpRope", "JumpingJack", "Kayaking", "Knitting",
    "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor", "Nunchucks", "ParallelBars",
    "PizzaTossing", "PlayingCello", "PlayingDaf", "PlayingDhol", "PlayingFlute", "PlayingGuitar",
    "PlayingPiano", "PlayingSitar", "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse",
    "PullUps", "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing", "Rowing",
    "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding", "Skiing", "Skijet", "SkyDiving",
    "SoccerJuggling", "SoccerPenalty", "StillRings", "SumoWrestling", "Surfing", "Swing",
    "TableTennisShot", "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
    "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard", "YoYo"
    }
elseif opt.dataset == 'hmdb51' then
    -- classes in HMDB51
    classes = {
    "brush_hair", "kick_ball", "ride_horse", "pour", "jump", "smile", "stand", "shake_hands", "flic_flac", "golf", "wave", "cartwheel", "clap", "dive", "ride_bike", "turn", "chew", "draw_sword", "push", "hug", "shoot_gun", "pullup", "sit", "smoke", "somersault", "shoot_bow", "kick", "kiss", "shoot_ball", "run", "walk", "situp", "sword", "drink", "pushup", "fall_floor", "climb", "hit", "laugh", "eat", "pick", "swing_baseball", "dribble", "talk", "climb_stairs", "catch", "fencing", "punch", "throw", "sword_exercise", "handstand"
    }
else
    error('Unknown dataset: ' .. opt.dataset)
end
nClass = #classes -- UCF101 has 101 categories

------------------------------------------------------------
-- Only use a certain number of frames from each video
------------------------------------------------------------
function extractFrames(inputData, rho)
   print(sys.COLORS.green ..  '==> Extracting only ' .. rho .. ' frames per video')

   if inputData:size(3) == rho then 
      return inputData
   end

   local timeStep = inputData:size(3) / rho
   local dataOutput = torch.Tensor(inputData:size(1), inputData:size(2), rho)

   local idx = 1
   for j = 1,inputData:size(3),timeStep do
      dataOutput[{{},{},idx}] = inputData[{{},{},j}]
      idx = idx + 1
   end
   return dataOutput
end

------------------------------------------------------------
-- Only use a certain number of consecutive frames from each video
------------------------------------------------------------
function extractConsecutiveFrames(inputData, rho)
   print(sys.COLORS.green ..  '==> Extracting random ' .. rho .. ' consecutive frames per video')

   local dataOutput = torch.Tensor(inputData:size(1), inputData:size(2), rho)
   local nProb = inputData:size(3) - rho
   local ind_start = torch.Tensor(1):random(1,nProb)
   
   local index = torch.range(ind_start[1], ind_start[1]+rho-1)
   local indLong = torch.LongTensor():resize(index:size()):copy(index)

   -- extracting data according to the Index
   local dataOutput = inputData:index(3,indLong)

   return dataOutput
end

------------------------------------------------------------
-- n-fold cross-validation function
-- this is only use a certain amount of data for training, and the rest of data for testing
------------------------------------------------------------
function crossValidation(dataset, target, nFolds)
   print(sys.COLORS.green ..  '==> Train on ' .. (1-1/nFolds)*100 .. '% of data ..')
   print(sys.COLORS.green ..  '==> Test on ' .. 100/nFolds .. '% of data ..')
   -- shuffle the dataset
   local shuffle = torch.randperm(dataset:size(1))
   local index = torch.ceil(dataset:size(1)/nFolds)
   -- extract test data
   local testIndices = shuffle:sub(1,index)
   local test_ind = torch.LongTensor():resize(testIndices:size()):copy(testIndices)
   local testData = dataset:index(1,test_ind)
   local testTarget = target:index(1,test_ind)
   -- extract train data
   local trainIndices = shuffle:sub(index+1,dataset:size(1))
   local train_ind = torch.LongTensor():resize(trainIndices:size()):copy(trainIndices)
   local trainData = dataset:index(1,train_ind)
   local trainTarget = target:index(1,train_ind)

   return trainData, trainTarget, testData, testTarget
end

print(sys.COLORS.green .. '==> Reading UCF101 external feature vector and target file ...')

----------------------------------------------
-- spatial feature matrix from CNN model
----------------------------------------------
local spaTrainData, spaTestData, spaTrainTarget, spaTestTarget

if opt.spatial == true then
   -- training and testing data from UCF101 website
   assert(paths.filep(paths.concat(opt.spatFeatDir, 'data_feat_train_RGB_centerCrop_25f_sp' .. opt.split .. '.t7')), 'no spatial training feature file found.')
   local spaTrainFeatureLabels = torch.load(paths.concat(opt.spatFeatDir, 'data_feat_train_RGB_centerCrop_25f_sp' .. opt.split .. '.t7'))
   spaTrainData = spaTrainFeatureLabels.featMats

   -- check if there are enough frames to extract and extract
   if spaTrainData:size(3) >= opt.rho then
      -- extract #rho of frames
      spaTrainData = extractFrames(spaTrainData, opt.rho)  
   else
      error('total number of frames lower than the extracting frames')
   end
   spaTrainTarget = spaTrainFeatureLabels.labels

   assert(paths.filep(paths.concat(opt.spatFeatDir, 'data_feat_test_RGB_centerCrop_25f_sp' .. opt.split .. '.t7')), 'no spatial testing feature file found.')
   local spaTestFeatureLabels = torch.load(paths.concat(opt.spatFeatDir, 'data_feat_test_RGB_centerCrop_25f_sp' .. opt.split .. '.t7'))
   spaTestData = spaTestFeatureLabels.featMats

   if opt.averagePred == false then 
      if spaTestData:size(3) >= opt.rho then
         -- extract #rho of frames
         spaTestData = extractFrames(spaTestData, opt.rho)  
      else
         error('total number of frames lower than the extracting frames')
      end
   end
   spaTestTarget = spaTestFeatureLabels.labels
end

----------------------------------------------
-- temporal feature matrix from CNN model
----------------------------------------------
local tempTrainData, tempTestData, tempTrainTarget, tempTestTarget

if opt.temporal == true then
   -- training and testing data from UCF101 website
   assert(paths.filep(paths.concat(opt.tempFeatDir, 'data_feat_train_FlowMap-TVL1-crop20_centerCrop_25f_sp' .. opt.split .. '.t7')), 'no temporal training feature file found.')
   local tempTrainFeatureLabels = torch.load(paths.concat(opt.tempFeatDir, 'data_feat_train_FlowMap-TVL1-crop20_centerCrop_25f_sp' .. opt.split .. '.t7'))
   tempTrainData = tempTrainFeatureLabels.featMats

   -- check if there are enough frames to extract and extract
   if tempTrainData:size(3) >= opt.rho then
      -- extract #rho of frames
      tempTrainData = extractFrames(tempTrainData, opt.rho)  
   else
      error('total number of frames lower than the extracting frames')
   end

   tempTrainTarget = tempTrainFeatureLabels.labels

   assert(paths.filep(paths.concat(opt.tempFeatDir, 'data_feat_test_FlowMap-TVL1-crop20_centerCrop_25f_sp' .. opt.split .. '.t7')), 'no temporal testing feature file found.')
   local tempTestFeatureLabels = torch.load(paths.concat(opt.tempFeatDir, 'data_feat_test_FlowMap-TVL1-crop20_centerCrop_25f_sp' .. opt.split .. '.t7'))
   tempTestData = tempTestFeatureLabels.featMats

   if opt.averagePred == false then 
      if tempTestData:size(3) >= opt.rho then
         -- extract #rho of frames
         tempTestData = extractFrames(tempTestData, opt.rho)  
      else
         error('total number of frames lower than the extracting frames')
      end
   end
   tempTestTarget = tempTestFeatureLabels.labels
end

local trainData, testData
if opt.spatial and opt.temporal then 
   -- spatial and temporal feature concatenation 
   trainData = torch.cat(spaTrainData, tempTrainData, 2)
   testData = torch.cat(spaTestData, tempTestData, 2)

   -- check if training & testing target are the same for spatial and temporal network
   assert(spaTrainTarget:equal(tempTrainTarget), 'training target of spatial and temporal features don`t match')
   assert(spaTestTarget:equal(tempTestTarget), 'testing target of spatial and temporal features don`t match')
else
   trainData = spaTrainData or tempTrainData
   testData = spaTestData or tempTestData
end

local trainTarget = spaTrainTarget or tempTrainTarget
local testTarget = spaTestTarget or tempTestTarget

collectgarbage()

return 
{
   trainData = trainData,
   trainTarget = trainTarget,
   testData = testData, 
   testTarget = testTarget
}
