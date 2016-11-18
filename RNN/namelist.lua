--usage: th namelist.lua > labels_rnn.txt

predlabeltxt = torch.load('labels.txt','ascii')
probtxt = torch.load('prob.txt','ascii')

-- read the testing feature file to get the video path and its name
nameucf = torch.load('/home/chih-yao/Downloads/Features/data_feat_test_RGB_tenCrop_25f_sp1.t7')

for i = 1, #nameucf.path do
	for j = 1,3 do
	    -- print(nameucf.path[i]..' '..probtxt[i][j])
    print(nameucf.path[i]..' '..predlabeltxt[i][j]..' '..probtxt[i][j])
	end
end

