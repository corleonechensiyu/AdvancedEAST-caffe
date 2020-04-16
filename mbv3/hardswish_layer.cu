#include <vector>
#include <algorithm>
#include "caffe/layers/hardswish_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void HardSwishForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    //out[index] = 0.5 * tanh(0.5 * in[index]) + 0.5;
    //out[index] = in[index] < -2.5 ? 0:(in[index] > 2.5 ? 1:in[index]*0.2 + 0.5);
    out[index] = in[index]*(in[index] < -2.5 ? 0:(in[index] > 2.5 ? 1:in[index]*0.2 + 0.5));
  }
}

template <typename Dtype>
void HardSwishLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  HardSwishForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void HardSwishBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    //const Dtype sigmoid_x = out_data[index];
    //out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
    //out_diff[index] = in_diff[index]* 0.2 *((in_data[index]>-2.5) && (in_data[index]<2.5));
    out_diff[index] = in_diff[index] * (in_data[index]<-2.5 ? 0:(in_data[index]>2.5 ? 1:0.4*in_data[index]+0.5));
  }
}

template <typename Dtype>
void HardSwishLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    HardSwishBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HardSwishLayer);


}  // namespace caffe
