#!/usr/bin/env python
# coding: utf-8
from __future__ import division
import os
os.environ['GLOG_minloglevel'] = '2' ## ignore the caffe log
import warnings
warnings.filterwarnings('ignore') ## ignore Warning log
import numpy as np
import cv2  ## 3.4.5+ or 4.0 +
import math
import argparse
from tqdm import tqdm
from nms import nms
import cfg
############ Add argument parser for command line arguments ############
parser = argparse.ArgumentParser(description='Use this script to run EAST-caffe')
parser.add_argument('--input', default='TB1698ILXXXXXXaXFXXunYpLFXX.jpg',
                    help='Path to input image for single demo')
parser.add_argument('--input_dir', default='imgs/ic15_test', 
                    help='Path to input image for batch demo')
parser.add_argument('--output_dir', default='results', 
                    help='Path to input image for batch demo')                         
parser.add_argument('--model_def', default='mbv3/deploy.prototxt',
                    help='prototxt file')
parser.add_argument('--model_weights', default='saved_model/mbv3_iter_117800.caffemodel',
                    help='caffemodel file')   
parser.add_argument('--thr',type=float, default=0.9,
                    help='Confidence threshold.')
parser.add_argument('--nms',type=float, default=0.1,
                    help='Non-maximum suppression threshold.')
parser.add_argument('--infer', default='dnn',
                    help='Inference API, dnn or caffe, recommand dnn inference')                     
parser.add_argument('--gpu',type=int, default=0,
                    help='GPU id (only set when inference API is caffe)')                    
args = parser.parse_args()

############ Utility functions ############
def resize_image(im, max_img_size=cfg.max_train_img_size):
    im_width = np.minimum(im.shape[1], max_img_size)#####min(746,512)=512
    if im_width == max_img_size < im.shape[1]:
        im_height = int((im_width / im.shape[1]) * im.shape[0])####512/746 * 564==387
    else:
        im_height = im.shape[0]#### 564
    o_height = np.minimum(im_height, max_img_size) ####min(387,512)===387
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width###512
    d_wight = o_width - (o_width % 32)####512-0=512
    d_height = o_height - (o_height % 32)#####387-3=384

    return d_wight, d_height

def decode(east_detect, confThreshold):

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(east_detect.shape) == 4, "Incorrect dimensions of east_detect"
    assert east_detect.shape[0] == 1, "Invalid dimensions of east_detect"
    assert east_detect.shape[1] == 7, "Invalid dimensions of east_detect"
    y = np.squeeze(east_detect, axis=0)
    cond = np.greater_equal(y[0, :, :], confThreshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    return quad_scores, quad_after_nms

    
def single_demo(input, output_dir):

    img = cv2.imread(input)

    inpWidth,inpHeight=resize_image(img)
    # print inpWidth,inpHeight 512 384
    im=cv2.resize(img,(inpWidth,inpHeight))
    # im_name = input[:-4]
    # txt_name = 'res_' + im_name + '.txt'
    
    if Inference_API == 'caffe':
        import caffe
        import time
        
        gpu = args.gpu
        caffe.set_device(gpu)  # GPU_id pick
        caffe.set_mode_gpu() # gpu mode

        net = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)
        # new_shape = [im.shape[2], im.shape[0], im.shape[1]]
        # net.blobs['image'].reshape(1, *im.shape)

        mu = np.array([103.94, 116.78, 123.68]) # the mean (BGR) pixel values

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
        transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

        image = caffe.io.load_image(input)
        transformed_image = transformer.preprocess('data', image)
        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image
        ### perform classification
        start = time.time()
        output = net.forward() # forward
        elapsed = (time.time() - start) * 1000
        print("CAFFE Inference time: %.2f ms" % elapsed)

        east_detect  = output['east_concat']
        print east_detect.shape


    if Inference_API == 'dnn':

        net = cv2.dnn.readNet(model_weights, model_def, 'caffe')
        blob = cv2.dnn.blobFromImage(im, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
        net.setInput(blob)
        # outs = net.forward(['ScoreMap/score', 'GeoMap'])
        outs = net.forward('east_concat')
        t, _ = net.getPerfProfile()
        print('OPENCV-DNN Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))

        east_detect = outs
        print east_detect.shape

    # Decode
    quad_scores, quad_after_nms = decode(east_detect, confThreshold)
    for score, geo, s in zip(quad_scores, quad_after_nms,
                             range(len(quad_scores))):
        # print geo.shape
        if np.amin(score) > 0:

            for i in range(1, len(geo)):
                cv2.line(im, (int(round(geo[i - 1][0])), int(round(geo[i - 1][1]))),
                         (int(round(geo[i][0])), int(round(geo[i][1]))), (0, 255, 0), 2,
                         lineType=cv2.LINE_AA)
            cv2.line(im, (int(round(geo[3][0])), int(round(geo[3][1]))),
                     (int(round(geo[0][0])), int(round(geo[0][1]))), (0, 255, 0), 2,
                     lineType=cv2.LINE_AA)
    # cv2.imwrite("012_pre.jpg",im)
    cv2.imshow("1",im)
    cv2.waitKey(0)

    


############ Parse Args ############
model_def = args.model_def
model_weights = args.model_weights
confThreshold = args.thr
nmsThreshold = args.nms
input = args.input  ## single demo
input_dir = args.input_dir  ## batch demo
output_dir = args.output_dir
Inference_API = args.infer

if __name__ == "__main__":
    single_demo(input, output_dir)

