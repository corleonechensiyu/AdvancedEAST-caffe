#include <vector>
#include <algorithm>

#include "caffe/layers/hardsigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype hardsigmoid(Dtype x) {
//   return 0.5 * tanh(0.5 * x) + 0.5;
    return std::max(Dtype(0),std::min(Dtype(1),Dtype(0.2)*x +Dtype(0.5)));

}

template <typename Dtype>
void HardSigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = hardsigmoid(bottom_data[i]);
  }
}

template <typename Dtype>
void HardSigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    // const Dtype* top_data = top[0]->cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
    //   const Dtype sigmoid_x = top_data[i];
    //   bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
        bottom_diff[i] = top_diff[i]*0.2*((bottom_data[i]>-2.5 && bottom_data[i]<2.5));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(HardSigmoidLayer);
#endif

INSTANTIATE_CLASS(HardSigmoidLayer);
REGISTER_LAYER_CLASS(HardSigmoid);

}  // namespace caffe
