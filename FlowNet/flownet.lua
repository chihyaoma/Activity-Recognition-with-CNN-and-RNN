-- This is a sample code of using FlowNet to generate the optical flow map from a pair of input images

-- Contact: Chih-Yao Ma at cyma@gatech.edu
-- 05/08/2016

require 'image'
require 'loadcaffe'
require 'xlua'
require 'optim'

-- load the network and print it's structure
prototxt = 'deploy.tpl.prototxt'
binary = 'flownet_official.caffemodel'
net = loadcaffe.load(prototxt, binary)

print('Here is the FlowNet..')
print(net)



-- Read pair of images
filePath = '../data/'
fileName_0 = '0000000-img0.ppm'
fileName_1 = '0000000-img1.ppm'
img_0 = image.load(filePath .. fileName_0)
img_1 = image.load(filePath .. fileName_1)

-- get the dimension of the images
imgSize_0 = img_0:size()
imgSize_1 = img_1:size()

-- define the dimension for the input images
inputSize = torch.LongStorage({512,384})

-- check the pair of images if they are in the same size
if img_0:isSameSizeAs(img_1) then 
	
	if not img_0:isSize(inputSize) and not img_1:isSize(inputSize) then
	-- resize the image into 512 x 384 if they are not ..
	end
	-- pass forward into the network 
else
	error('dimension of the images does not match..')
end

