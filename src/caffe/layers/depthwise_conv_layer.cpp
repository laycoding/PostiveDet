#include <vector>
#include "caffe/layers/depthwise_conv_layer.hpp"
#include <algorithm>
#include "caffe/util/math_functions.hpp"
namespace caffe {

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* weight = this->blobs_[0]->cpu_data();
	int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
	int* stride_data = this->stride_.mutable_cpu_data();
	int* pad_data = this->pad_.mutable_cpu_data();


	for (int i = 0; i < bottom.size(); ++i)
  {
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype* top_data = top[i]->mutable_cpu_data();
		vector<int> shape_ = bottom[i]->shape();
		const int channels_ = shape_[1];
		const int height_ = shape_[2];
		const int width_ = shape_[3];

		const int kernel_h_ = kernel_shape_data[0];
		const int kernel_w_ = kernel_shape_data[1];
		const int stride_h_ = stride_data[0];
		const int stride_w_ = stride_data[1];
		const int pad_h_ = pad_data[0];
		const int pad_w_ = pad_data[1];
    const int  num_ = bottom[i]->num();
    //LOG(INFO)<<"size: "<<bottom.size()<<"num:"<<num_<<" channels_: "<<channels_<<" height_: "<<height_<<" width_: "<<width_<<" kernel_h_: "<<kernel_h_<<" kernel_w_: "<<kernel_w_<<" stride_h_: "<<stride_h_<<" stride_w_: "<<stride_w_<<" pad_h_: "<<pad_h_<<" pad_w_: "<<pad_w_;
		const bool bias_term_ = this->bias_term_;
    const int conved_height = this->output_shape_[0];
		const int conved_weight = this->output_shape_[1];
    //int count=0;
    for (int n = 0; n < num_; ++n)
    {
     for (int c = 0; c < channels_; ++c)
     {
       for (int h = 0; h < conved_height; ++h)
       {
         for (int w = 0; w < conved_weight; ++w)
         {
           int hstart = h * stride_h_ - pad_h_;
           int wstart = w * stride_w_ - pad_w_;
           int hend = std::min(hstart + kernel_h_, height_ + pad_h_);
           int wend = std::min(wstart + kernel_w_, width_ + pad_w_);
           hstart = std::max(hstart, 0);
           wstart = std::max(wstart, 0);
           hend = std::min(hend, height_);
           wend = std::min(wend, width_);
           Dtype aveval = 0;
           const Dtype* const bottom_slice =
           bottom_data + (n * channels_ + c) * height_ * width_;
           const Dtype* const weight_slice =
           weight + c * kernel_h_ * kernel_w_;

           int khstart=hend<kernel_h_?kernel_h_-hend:0;
           int kwstart=wend<kernel_w_?kernel_w_-wend:0;
           for (int hh = hstart; hh < hend; ++hh)
           {
            for (int ww = wstart; ww < wend; ++ww)
            {
                aveval += bottom_slice[hh * width_ + ww]*weight_slice[(khstart+hh-hstart) * kernel_w_ + (kwstart+ww-wstart)];
                //if(count++<10)
                //   LOG(INFO)<<bottom_slice[hh * width_ + ww]<<" "<<weight_slice[(khstart+hh-hstart) * kernel_w_ + (kwstart+ww-wstart)];
              }
            }
           if(bias_term_)
           {
             const Dtype* const bias=this->blobs_[1]->cpu_data();
             *(top_data++) = (aveval+bias[c]);
           }else{
             *(top_data++) = aveval;
           }
        }
      }
    }
   }
 }
}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DepthwiseConvolutionLayer);
#endif

INSTANTIATE_CLASS(DepthwiseConvolutionLayer);
REGISTER_LAYER_CLASS(DepthwiseConvolution);
}  // namespace caffe
