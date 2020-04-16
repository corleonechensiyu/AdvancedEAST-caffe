# add HardSigmoid and HardSwish
put *.h in include/caffe/layers/ ...put *.cpp/*.cu in src/caffe/layers/

    edit caffe.proto
    optional HardSigmoidParameter hardsigmoid_parm =163;
    optional HardSwishParameter hardswish_parm =164;

    message HardSigmoidParameter {
    enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
    }
    optional Engine engine = 1 [default = DEFAULT];
    }
    message HardSwishParameter {
    enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
    }
    optional Engine engine = 1 [default = DEFAULT];
    }
# Demo
    # HardSigmoid SE
    layer {
    name: "conv3-2/dwise-se-fc1"
    type: "InnerProduct"
    bottom: "conv3-2/dwise-se-pool"
    top: "conv3-2/dwise-se-fc1"
    param {
        lr_mult: 1
        decay_mult: 1
    }
    param {
        lr_mult: 2
        decay_mult: 0
    }
    inner_product_param {
        num_output: 18
        weight_filler {
        type: "msra"
    }
    bias_filler {
        type: "constant"
        value: 0
        }
        }
    }
    layer {
    name: "conv3-2/dwise-se-fc1-relu"
    type: "ReLU"
    bottom: "conv3-2/dwise-se-fc1"
    top: "conv3-2/dwise-se-fc1"
    }
    layer {
    name: "conv3-2/dwise-se-fc2"
    type: "InnerProduct"
    bottom: "conv3-2/dwise-se-fc1"
    top: "conv3-2/dwise-se-fc2"
    param {
        lr_mult: 1
        decay_mult: 1
    }
    param {
        lr_mult: 2
        decay_mult: 0
    }
    inner_product_param {
        num_output: 72
        weight_filler {
        type: "msra"
        }
        bias_filler {
        type: "constant"
        value: 0
        }
    }
    }
    layer {
    name: "relu5_2"
    type: "HardSigmoid"
    bottom: "conv3-2/dwise-se-fc2"
    top: "conv3-2/dwise-se-fc2"
    }
    layer {
    name: "conv3-2/dwise/scale"
    type: "Scale"
    bottom: "conv3-2/dwise"
    bottom: "conv3-2/dwise-se-fc2"
    top: "conv3-2/dwise/scale"
    scale_param {
        axis: 0
        }
    }
    layer {
    name: "conv3-2/dwise/scale-relu"
    type: "ReLU6"
    bottom: "conv3-2/dwise/scale"
    top: "conv3-2/dwise/scale"
    }

    # HardSwish
    layer {
    name: "conv1"
    type: "Convolution"
    bottom: "data"
    top: "conv1"
    param {
        lr_mult: 1
        decay_mult: 1
    }
    convolution_param {
        num_output: 16
        bias_term: false
        pad: 1
        kernel_size: 3
        stride: 2
        weight_filler {
        type: "msra"
        }
    }
    }
    layer {
        name: "conv1-bn"
        type: "BatchNorm"
        bottom: "conv1"
        top: "conv1"
    }
    layer {
        name: "conv1-bn-scale"
        type: "Scale"
        bottom: "conv1"
        top: "conv1"
        scale_param {
            bias_term: true
        }
    }
    layer {
        name: "activation1_1"
        type: "HardSwish"
        bottom: "conv1"
        top: "conv1"
    }
