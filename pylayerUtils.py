#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import caffe
import numpy as np
import cv2
import random
import os
import math
import csv


from generator import gen

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # data layer config
        params = eval(self.param_str)
        self.patch_size = 512
        self.batch_size = int(params['batch_size'])

        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        if len(top) != 2:
            raise Exception("data top need three output")
            
        ## set directory for each dataset here
        
        self.dataset = 'tianchi'
        datasetDict = {
            'ic13': '/home1/surfzjy/data/ic13',
            'ic15': '/home/csy/AttentionOCR/DB/datasets/icdar2015',
            'tianchi':'/home/csy/ICDAR/icpr/train_1000/images_3T512',
        }
        self.basedir = datasetDict[self.dataset]
        self.fnLst = os.listdir(self.basedir)
        self.nb_img = len(self.fnLst)
        self.index = np.arange(0, self.nb_img)
        np.random.shuffle(self.index)
        self.idx = 0         

    def reshape(self, bottom, top):
        # load image, label and weight
        self.data, self.ture_map = self.load()
        # print(self.data.shape)
        # reshape tops to fit
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.ture_map.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.ture_map
    def backward(self, top, propagate_down, bottom):
        pass

    def load(self):
        load_index = self.index[self.idx: min((self.idx + self.batch_size), self.nb_img)]
        self.idx += self.batch_size
        if self.idx >= self.nb_img:
            np.random.shuffle(self.index)
            self.idx = 0

        #TODO
        input_image ,batch_data= gen(batch_size=self.batch_size, is_val=False)
        input_images = np.array(input_image).transpose(0, 3, 1, 2)

        ture_map = batch_data.transpose(0, 3, 1, 2)
        return (input_images, ture_map)
#TODO
class DiceCoefLossLayer(caffe.Layer):
    """
    self designed loss layer for segmentation. Class weighted, per pixel loss
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self.epsilon = 1e-4
        self.pixel_num=bottom[1].data[:,:1,:,:].shape[0]*bottom[1].data[:,:1,:,:].shape[1]*bottom[1].data[:,:1,:,:].shape[2]*bottom[1].data[:,:1,:,:].shape[3]

    def reshape(self, bottom, top):
        # check input dimensions match
        # N 7 128 128
        if bottom[0].count!=bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        self.score_gt=np.zeros_like(bottom[1].data[:,:1,:,:],dtype=np.float32)
        self.inside_score=np.zeros_like(bottom[0].data[:,:1,:,:],dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.score_gt[...]=bottom[1].data[:,:1,:,:]
        self.inside_score[...]=bottom[0].data[:,:1,:,:]
        self.beta = 1. - np.mean(self.score_gt)
        self.L_S = np.mean(-1.*(self.beta * self.score_gt * np.log(self.inside_score+self.epsilon) + (1. - self.beta)*(1. - self.score_gt)*np.log(1. - self.inside_score +self.epsilon )))

        # top[0].data[...] = self.L_S
        top[0].data[...] = self.L_S

    def backward(self, top, propagate_down, bottom):

        A=(-1.*self.beta * self.score_gt)/(self.inside_score +self.epsilon)
        B=((1.-self.beta)*(1.-self.score_gt))/(1. -self.inside_score +self.epsilon)
        self.score_grad = (A+B)/self.pixel_num

        # bottom[0].diff[...] = self.score_grad
        bottom[0].diff[:,:1,:,:] = self.score_grad


class VertexcodeLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need three inputs to compute loss.")
        self.epsilon = 1e-4

    def reshape(self, bottom, top):
        #N 7 128 128
        if bottom[0].count != bottom[1].count:
            raise Exception("First Two Inputs must have the same dimension.")

        self.score_gt = np.zeros_like(bottom[1].data[:, :1, :, :], dtype=np.float32)
        self.geo_gt = np.zeros_like(bottom[1].data[:, 1:3, :, :], dtype=np.float32)
        # self.side_v_code = np.zeros_like(bottom[0].data[:, 1:3, :, :], dtype=np.float32)
        # self.pos = np.zeros_like(bottom[0].data, dtype=np.float32)
        # self.neg = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.v_p = np.zeros_like(bottom[0].data[:, 1:3, :, :], dtype=np.float32)
        top[0].reshape(1)  # 1,

    def forward(self, bottom, top):
        self.score_gt[...] = bottom[1].data[:, :1, :, :]
        self.geo_gt[...] = bottom[1].data[:, 1:3, :, :]
        self.v_p[...] = bottom[0].data[:, 1:3, :, :]
        self.vertex_beta = 1.-(np.mean(bottom[1].data[:, 1:2, :, :])/(np.mean(self.score_gt)+self.epsilon))
        self.pos = -1.*self.vertex_beta * self.geo_gt * np.log(self.v_p + self.epsilon)
        self.neg = -1.*(1. - self.vertex_beta)*(1. - self.geo_gt)*np.log(1.-self.v_p + self.epsilon)
        self.p_w = np.equal(bottom[1].data[:, 0, :, :],1).astype(np.float32)
        loss = np.sum(np.sum(self.pos + self.neg,axis=1)*self.p_w)/(np.sum(self.p_w)+self.epsilon)
        # top[0].data[...] = loss
        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        grad_p = (-1.*self.vertex_beta*self.geo_gt)/(self.v_p + self.epsilon)
        grad_n = ((1.-self.vertex_beta)*(1.-self.geo_gt))/(1.-self.v_p + self.epsilon)


        # bottom[0].diff[...] =(grad_p + grad_n)/(np.sum(self.p_w)+self.epsilon)
        bottom[0].diff[:, 1:3, :, :] = (grad_p + grad_n) / (np.sum(self.p_w) + self.epsilon)
class VertexcoordLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need three inputs to compute loss.")
        self.epsilon = 1e-4
        self.batch = bottom[0].data.shape[0]
        self.pixel_num = bottom[0].data.shape[2]*bottom[0].data.shape[3]
        self.channels=bottom[0].data[:, 3:, :, :].shape[1]

    def reshape(self, bottom, top):
        #n 7 128 128
        if bottom[0].count != bottom[1].count:
            raise Exception("First Two Inputs must have the same dimension.")

        self.coord_gt = np.zeros_like(bottom[1].data[:, 3:, :, :], dtype=np.float32)
        self.coord_pred = np.zeros_like(bottom[0].data[:, 3:, :, :], dtype=np.float32)

        self.v_w_a = np.zeros_like(bottom[1].data[:, 3:, :, :], dtype=np.float32)
        self.n_q_a = np.zeros_like(bottom[1].data[:, 3:, :, :], dtype=np.float32)
        top[0].reshape(1)  # 1,

    def forward(self, bottom, top):
        self.coord_gt[...]=bottom[1].data[:, 3:, :, :]
        self.coord_pred[...]=bottom[0].data[:, 3:, :, :]
        self.v_w=np.equal(bottom[1].data[:, 1, :, :],1).astype(np.float32)

        t_shape = self.coord_gt.transpose(0,2,3,1)# n h w c
        shape = np.shape(t_shape)
        delta_xy_matrix =np.reshape(t_shape,[-1,2,2])
        diff_q = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
        square = np.square(diff_q)
        distance = np.sqrt(np.sum(square, axis=2))
        distance *= 4.0
        distance += self.epsilon
        self.quad_norm =np.reshape(distance, shape[:-1])

        self.n_q = np.reshape(self.quad_norm, np.shape(self.v_w)) #N*128*128
        self.diff = self.coord_pred - self.coord_gt

        self.sum_one = np.sign(self.diff)

        abs_diff = np.abs(self.diff)
        self.abs_diff_lt_1 = np.less(abs_diff, 1) # |x|<1
        self.pixel_wise_smooth_l1norm=(np.sum(np.where(self.abs_diff_lt_1, 0.5 * np.square(abs_diff), abs_diff - 0.5),axis=1) / self.n_q) * self.v_w

        self.side_vertex_coord_loss = np.sum(self.pixel_wise_smooth_l1norm) / (np.sum(self.v_w) + self.epsilon)

        # top[0].data[...] = self.side_vertex_coord_loss
        top[0].data[...] = self.side_vertex_coord_loss
    def backward(self, top, propagate_down, bottom):
        for i in xrange(self.channels):
            for j in xrange(self.batch):
                self.v_w_a[j,i,:,:]=self.v_w[j,:,:]
                self.n_q_a[j,i,:,:]=self.n_q[j,:,:]
        smooth_l1_grad = np.where(self.abs_diff_lt_1, self.diff, self.sum_one) / self.n_q_a * self.v_w_a
        # smooth_l1_grad=np.where(self.abs_diff_lt_1,self.diff , self.sum_one) / self.n_q * self.v_w

        # bottom[0].diff[...] = smooth_l1_grad / ((np.sum(self.v_w) + self.epsilon))
        bottom[0].diff[:, 3:, :, :] = smooth_l1_grad/((np.sum(self.v_w) + self.epsilon))



