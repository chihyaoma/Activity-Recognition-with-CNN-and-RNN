--usage: th namelist.lua > labels_tcnn.txt

predlabeltxt = torch.load('labels.txt','ascii')
probtxt = torch.load('prob.txt','ascii')
nameucf = torch.load('/media/cmhung/MyDisk/CMHung_FS/Big_and_Data/PhDResearch/Code/Dataset/Features/Res/name_UCF101_test_1.t7')
for i = 1, #nameucf.path do
	for j = 1,3 do
	    -- print(nameucf.path[i]..' '..probtxt[i][j])
    print(nameucf.path[i]..' '..predlabeltxt[i][j]..' '..probtxt[i][j])
	end
end

