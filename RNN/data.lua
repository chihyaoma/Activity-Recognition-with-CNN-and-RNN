----------------------------------------------------------------
-- Georgia Tech 2016 Spring
-- Deep Learning for Perception
-- Final Project: LRCN model for Video Classification
--
-- 
-- This is a testing code for implementing the RNN model with LSTM 
-- written by Chih-Yao Ma. 
-- 
-- The code will take feature vectors (from CNN model) from contiguous 
-- frames and train against the ground truth, i.e. the labeling of video classes. 
-- 
-- Contact: Chih-Yao Ma at <cyma@gatech.edu>
----------------------------------------------------------------
require 'torch'
require 'sys'
-- nClass = 11 -- UCF11 has 11 categories
nClass = 101 -- UCF101 has 101 categories

-- generate strings for classes
-- classes = {}
-- for c = 1, nClass do
--    classes[c] = tostring(c)
-- end

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

------------------------------------------------------------
-- Only use a certain number of frames from each video
------------------------------------------------------------
function ExtractFrames(InputData, rho)
   print(sys.COLORS.green ..  '==> Extracting only ' .. rho .. ' frames per video')
   local TimeStep = InputData:size(3) / rho
   local DataOutput = torch.Tensor(InputData:size(1), InputData:size(2), rho)

   local idx = 1
   for j = 1,InputData:size(3),TimeStep do
      DataOutput[{{},{},idx}] = InputData[{{},{},j}]
      idx = idx + 1
   end
   return DataOutput
end

------------------------------------------------------------
-- Only use a certain number of consecutive frames from each video
------------------------------------------------------------
function ExtractConsecutiveFrames(InputData, rho)
   print(sys.COLORS.green ..  '==> Extracting random ' .. rho .. ' consecutive frames per video')

   local DataOutput = torch.Tensor(InputData:size(1), InputData:size(2), rho)
   local nProb = InputData:size(3) - rho
   local ind_start = torch.Tensor(1):random(1,nProb)
   
   local Index = torch.range(ind_start[1], ind_start[1]+rho-1)
   local IndLong = torch.LongTensor():resize(Index:size()):copy(Index)

   -- extracting data according to the Index
   local DataOutput = InputData:index(3,IndLong)

   return DataOutput
end

------------------------------------------------------------
-- n-fold cross-validation function
-- this is only use a certain amount of data for training, and the rest of data for testing
------------------------------------------------------------
function CrossValidation(Dataset, Target, nFolds)
   print(sys.COLORS.green ..  '==> Train on ' .. (1-1/nFolds)*100 .. '% of data ..')
   print(sys.COLORS.green ..  '==> Test on ' .. 100/nFolds .. '% of data ..')
   -- shuffle the dataset
   local shuffle = torch.randperm(Dataset:size(1))
   local Index = torch.ceil(Dataset:size(1)/nFolds)
   -- extract test data
   local TestIndices = shuffle:sub(1,Index)
   local Test_ind = torch.LongTensor():resize(TestIndices:size()):copy(TestIndices)
   local TestData = Dataset:index(1,Test_ind)
   local TestTarget = Target:index(1,Test_ind)
   -- extract train data
   local TrainIndices = shuffle:sub(Index+1,Dataset:size(1))
   local Train_ind = torch.LongTensor():resize(TrainIndices:size()):copy(TrainIndices)
   local TrainData = Dataset:index(1,Train_ind)
   local TrainTarget = Target:index(1,Train_ind)

   return TrainData, TrainTarget, TestData, TestTarget
end


print(sys.COLORS.green .. '==> Reading UCF101 external feature vector and target file ...')
-- load saved feature matrix from CNN model
local UCF101_list = true

if UCF101_list == true then 

   -- training and testing data from UCF101 website
   -- local TrainFeatureLabels = torch.load('/home/chih-yao/Downloads/feat_label_UCF101_train_1.t7')
   local TrainFeatureLabels = torch.load('/home/chih-yao/Downloads/data_UCF101_train_1.t7')
   TrainData = TrainFeatureLabels.featMats
   TrainData = ExtractFrames(TrainData, opt.rho)
   TrainTarget = TrainFeatureLabels.labels

   -- local TestFeatureLabels = torch.load('/home/chih-yao/Downloads/feat_label_UCF101_test_1.t7')
   local TestFeatureLabels = torch.load('/home/chih-yao/Downloads/data_UCF101_test_1.t7')
   TestData = TestFeatureLabels.featMats
   if opt.AveragePred == false then 
      TestData = ExtractFrames(TestData, opt.rho)
   end
   TestTarget = TestFeatureLabels.labels

else
   print(sys.COLORS.green .. '==> Reading self-defined external feature vector and target file . . .')
   
   ds = {}

   -- local InputFeatureLabels = torch.load('feat_label_UCF11.t7')
   -- local InputFeatureLabels = torch.load('/home/chih-yao/Downloads/feat_label_UCF101.t7')
   local InputFeatureLabels = torch.load('/home/chih-yao/Downloads/feat_label_UCF101_2.t7')

   ds.input = InputFeatureLabels.featMats
   ds.target = InputFeatureLabels.labels
   ds.size = ds.input:size(1)
   ds.FeatureDims = ds.input:size(2)

   if not InputFeatureLabels then error("Cannot read feature vector file") end

   -- Only use a certain number of (consecutive) frames from each video
   ds.input = ExtractFrames(ds.input, opt.rho)
   -- ds.input = ExtractConsecutiveFrames(ds.input, opt.rho)

   -- n-fold cross-validation
   TrainData, TrainTarget, TestData, TestTarget = CrossValidation(ds.input, ds.target, 5)

end





if (false) then
-- input dimension = ds.size x ds.FeatureDims x opt.rho = 1100 x 1024 x time
   if not (opt.featFile == '') then
      -- read feature file from command line
      print(' - - Reading external feature file . . .')
      file = torch.DiskFile(opt.featFile, 'r')
      ds.input = file:readObject()
   else
      -- generate random feature file
      print(sys.COLORS.red .. ' - - No --featFile specified. Generating random feature matrix . . . ' .. c.white)
      ds.input = torch.randn(ds.input:size(1), ds.FeatureDims, opt.rho)
   end

   -- target dimension = ds.size x 1 = 1100 x 1
   if not (opt.targFile == '') then
      -- read feature file from command line
      print(' - - Reading external target file . . .')
      file = torch.DiskFile(opt.targFile, 'r')
      ds.target = file:readObject()
   else
      print(sys.COLORS.red .. ' - - No --targFile specified. Generating random target vector . . . ' .. c.white)
      ds.target = torch.DoubleTensor(ds.input:size(1)):random(nClass)
   end
end


return 
{
   TrainData = TrainData,
   TrainTarget = TrainTarget,
   TestData = TestData, 
   TestTarget = TestTarget
}
