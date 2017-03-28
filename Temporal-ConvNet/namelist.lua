--usage: th namelist.lua > labels_tcnn.txt

predlabeltxt = torch.load('labels.txt','ascii')
probtxt = torch.load('prob.txt','ascii')
nameucf = torch.load('/home/cmhung/Code/Features/UCF-101/data_feat_test_RGB_centerCrop_25f_sp1.t7')

for i = 1, #nameucf.name do
	for j = 1,3 do
	    -- print(nameucf.name[i]..' '..probtxt[i][j])
    print(nameucf.name[i]..' '..predlabeltxt[i][j]..' '..probtxt[i][j])
	end
end
