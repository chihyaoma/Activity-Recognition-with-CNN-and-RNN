# This is a Python code using OpenCV to process the two consecutive frames to compute optical flow
#
# 
#
# Contact: Chih-Yao Ma at cyma@gatech.edu
# 05/13/2016

import os, sys
import numpy as np
import cv2
import subprocess
from math import ceil

class FlowNet:

    caffe_bin = './bin/caffe'
    img_size_bin = './bin/get_image_size'
    template = 'deploy.tpl.prototxt'

    # model_folder = './model_simple' # use FlowNetSimple
    model_folder = './model_corr' # use FlowNetCorr


    @staticmethod
    def run(img0):

        model_folder = './model_corr'

        # check if the compiled FLowNet existed
        # TODO: is img_size_bin necessary? 
        if not (os.path.isfile(FlowNet.caffe_bin) and os.path.isfile(FlowNet.img_size_bin)):
            print('Caffe tool binaries not found. Did you compile caffe with tools (make all tools)?')
            sys.exit(1)

        # Get image sizes
        height, width, channels = img0.shape

        # Prepare prototxt
        subprocess.call('mkdir -p tmp', shell=True)

        divisor = 64.
        adapted_width = ceil(width/divisor) * divisor
        adapted_height = ceil(height/divisor) * divisor
        rescale_coeff_x = width / adapted_width
        rescale_coeff_y = height / adapted_height

        replacement_list = {
            '$ADAPTED_WIDTH': ('%d' % adapted_width),
            '$ADAPTED_HEIGHT': ('%d' % adapted_height),
            '$TARGET_WIDTH': ('%d' % width),
            '$TARGET_HEIGHT': ('%d' % height),
            '$SCALE_WIDTH': ('%.8f' % rescale_coeff_x),
            '$SCALE_HEIGHT': ('%.8f' % rescale_coeff_y)
        }

        proto = ''
        with open(os.path.join(model_folder, FlowNet.template), "r") as tfile:
            proto = tfile.read()

        for r in replacement_list:
            proto = proto.replace(r, replacement_list[r])

        with open('tmp/deploy.prototxt', "w") as tfile:
            tfile.write(proto)

        list_length = 1

        # Run caffe
        args = [FlowNet.caffe_bin, 'test', '-model', 'tmp/deploy.prototxt',
                '-weights', model_folder + '/flownet_official.caffemodel',
                '-iterations', str(list_length),
                '-gpu', '0']

        cmd = str.join(' ', args)
        print('Executing %s' % cmd)

        subprocess.call(args)


