predlabeltxt=torch.load('labels.txt','ascii')
nameucf = torch.load('/home/chih-yao/Downloads/name_UCF101_test_1.t7')
for i = 1,3754 do
    print(nameucf.path[i]..' '..predlabeltxt[i])
end

