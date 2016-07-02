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

nClass = 101 -- UCF101 has 101 categories

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

-- training and testing data from UCF101 website
assert(paths.filep(paths.concat(opt.featFile, 'data_UCF101_train_1.t7')), 'no training feature file found.')
local TrainFeatureLabels = torch.load(paths.concat(opt.featFile, 'data_UCF101_train_1.t7'))
TrainData = TrainFeatureLabels.featMats

-- check if there are enough frames to extract
assert(TrainData:size(3) >= opt.rho, '# of frames lower than rho')

-- extract #rho of frames
TrainData = ExtractFrames(TrainData, opt.rho)
TrainTarget = TrainFeatureLabels.labels

assert(paths.filep(paths.concat(opt.featFile, 'data_UCF101_test_1.t7')), 'no testing feature file found.')
local TestFeatureLabels = torch.load(paths.concat(opt.featFile, 'data_UCF101_test_1.t7'))
TestData = TestFeatureLabels.featMats

if opt.averagePred == false then 
   -- check if there are enough frames to extract
   assert(TrainData:size(3) >= opt.rho, '# of frames lower than rho')
   -- extract #rho of frames
   TestData = ExtractFrames(TestData, opt.rho)
end
TestTarget = TestFeatureLabels.labels

return 
{
   TrainData = TrainData,
   TrainTarget = TrainTarget,
   TestData = TestData, 
   TestTarget = TestTarget
}
