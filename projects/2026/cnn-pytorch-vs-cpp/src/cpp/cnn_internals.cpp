#include "include/cnn_internals.h"

#include <limits>

static constexpr float MAXPOOL_NEG_INF = std::numeric_limits<float>::lowest();

// -----------------------------------------------
// Baseline nested loops implementation internals
// -----------------------------------------------
void conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                    int stride, int padding) {
    int out_channels = weight.batches;
    int in_channels  = weight.channels;
    int kernel_h     = weight.height;
    int kernel_w     = weight.width;


    for (int b = 0; b < input.batches; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < output.height; ++oh) {
                for (int ow = 0; ow < output.width; ++ow) {
                    float pixel_value = bias.data[oc];
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && ih < input.height && iw >= 0 && iw < input.width) {
                                    int in_idx = b * (input.channels * input.height * input.width) +
                                                 ic * (input.height * input.width) +
                                                 ih * input.width + iw;
                                    int w_idx  = oc * (weight.channels * weight.height * weight.width) +
                                                 ic * (weight.height * weight.width) +
                                                 kh * weight.width + kw;
                                    pixel_value += input.data[in_idx] * weight.data[w_idx];
                                }
                            }
                        }
                    }
                    int out_idx = b * (output.channels * output.height * output.width) +
                                  oc * (output.height * output.width) +
                                  oh * output.width + ow;
                    output.data[out_idx] = pixel_value;
                }
            }
        }
    }
}

void conv2d_forward_blocked(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                             int stride, int padding) {
    constexpr int OC_TILE = 4;
    constexpr int OW_TILE = 4;

    int out_channels = weight.batches;
    int in_channels  = weight.channels;
    int kernel_h     = weight.height;
    int kernel_w     = weight.width;

    for (int b = 0; b < input.batches; ++b) {
        for (int oh = 0; oh < output.height; ++oh) {
            for (int oc0 = 0; oc0 < out_channels; oc0 += OC_TILE) {
                int oc_lim = std::min(OC_TILE, out_channels - oc0);

                for (int ow0 = 0; ow0 < output.width; ow0 += OW_TILE) {
                    int ow_lim = std::min(OW_TILE, output.width - ow0);

                    float acc[OC_TILE][OW_TILE];
                    for (int t_oc = 0; t_oc < oc_lim; ++t_oc)
                        for (int t_ow = 0; t_ow < ow_lim; ++t_ow)
                            acc[t_oc][t_ow] = bias.data[oc0 + t_oc];

                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            int ih = oh * stride - padding + kh;
                            if (ih < 0 || ih >= input.height) continue;

                            for (int kw = 0; kw < kernel_w; ++kw) {
                                for (int t_ow = 0; t_ow < ow_lim; ++t_ow) {
                                    int ow = ow0 + t_ow;
                                    int iw = ow * stride - padding + kw;
                                    if (iw < 0 || iw >= input.width) continue;

                                    int in_idx = b * (input.channels * input.height * input.width) +
                                                 ic * (input.height * input.width) +
                                                 ih * input.width + iw;
                                    float in_val = input.data[in_idx]; 
                                    for (int t_oc = 0; t_oc < oc_lim; ++t_oc) {
                                        int oc = oc0 + t_oc;
                                        int w_idx = oc * (weight.channels * weight.height * weight.width) +
                                                    ic * (weight.height * weight.width) +
                                                    kh * weight.width + kw;
                                        acc[t_oc][t_ow] += in_val * weight.data[w_idx];
                                    }
                                }
                            }
                        }
                    }

                    for (int t_oc = 0; t_oc < oc_lim; ++t_oc) {
                        int oc = oc0 + t_oc;
                        for (int t_ow = 0; t_ow < ow_lim; ++t_ow) {
                            int ow = ow0 + t_ow;
                            int out_idx = b * (output.channels * output.height * output.width) +
                                          oc * (output.height * output.width) +
                                          oh * output.width + ow;
                            output.data[out_idx] = acc[t_oc][t_ow];
                        }
                    }
                }
            }
        }
    }
}


void conv2d_forward_restructured(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                    int stride, int padding) {
    int out_channels = weight.batches;
    int in_channels  = weight.channels;
    int kernel_h     = weight.height;
    int kernel_w     = weight.width;
    int input_h      = input.height;
    int input_w      = input.width;
    int output_h     = output.height;
    int output_w     = output.width;
    int input_batches = input.batches;

    for (int b = 0; b < input_batches; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            
            for (int oh = 0; oh < output_h; ++oh) {
                int c_h = oh * stride - padding;
                
                int kh_lo = std::max(0, -c_h);
                int kh_hi = std::min(kernel_h, input_h - c_h);
                
                for (int ow = 0; ow < output_w; ++ow) {
                    float pixel_value = bias.data[oc];

                    int c_w = ow * stride - padding;

                    int kw_lo = std::max(0, -c_w);
                    int kw_hi = std::min(kernel_w, input_w - c_w);
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = kh_lo; kh < kh_hi; ++kh) {
                            for (int kw = kw_lo; kw < kw_hi; ++kw) {
                                int ih = c_h + kh;
                                int iw = c_w + kw;
                                int in_idx = b * (in_channels * input_h * input_w) +
                                                ic * (input_h * input_w) +
                                                ih * input_w + iw;
                                int w_idx  = oc * (in_channels * kernel_h * kernel_w) +
                                                ic * (kernel_h * kernel_w) +
                                                kh * kernel_w + kw;
                                pixel_value += input.data[in_idx] * weight.data[w_idx];
                            }
                        }
                    }
                    int out_idx = b * (output.channels * output_h * output_w) +
                                  oc * (output_h * output_w) +
                                  oh * output_w + ow;
                    output.data[out_idx] = pixel_value;
                }
            }
        }
    }
}

void conv2d_forward_hoist_restrict(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                    int stride, int padding) {
    const float* __restrict input_ptr  = input.data.data();
    const float* __restrict weight_ptr = weight.data.data();
    const float* __restrict bias_ptr   = bias.data.data();
    float* __restrict       out_ptr    = output.data.data();
    
    int out_channels = weight.batches;
    int in_channels  = weight.channels;
    int kernel_h     = weight.height;
    int kernel_w     = weight.width;
    int input_h      = input.height;
    int input_w      = input.width;
    int output_h     = output.height;
    int output_w     = output.width;
    int input_batches = input.batches;

    int in_size = input_h * input_w;
    int in_ch_size = in_channels * in_size;
    int out_size = output_h * output_w;
    int out_ch_size = out_channels * out_size;
    int weight_size = kernel_h * kernel_w;
    int weight_ch_size = in_channels * weight_size;

    for (int b = 0; b < input_batches; ++b) {
        int in_b_ch_size = b * in_ch_size;
        int out_b_ch_size = b * out_ch_size;
        for (int oc = 0; oc < out_channels; ++oc) {
            int out_oc_ch_size = oc * weight_ch_size;
            int out_oc_size = oc * out_size;
            
            for (int oh = 0; oh < output_h; ++oh) {
                int c_h = oh * stride - padding;
                
                int kh_lo = std::max(0, -c_h);
                int kh_hi = std::min(kernel_h, input_h - c_h);
                
                for (int ow = 0; ow < output_w; ++ow) {
                    float pixel_value = bias_ptr[oc];

                    int c_w = ow * stride - padding;

                    int kw_lo = std::max(0, -c_w);
                    int kw_hi = std::min(kernel_w, input_w - c_w);
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        int ic_size = ic * in_size;
                        int ic_weight_size = ic * weight_size;
                        
                        for (int kh = kh_lo; kh < kh_hi; ++kh) {
                            for (int kw = kw_lo; kw < kw_hi; ++kw) {
                                int ih = c_h + kh;
                                int iw = c_w + kw;
                                int in_idx = in_b_ch_size + ic_size + ih * input_w + iw;
                                int w_idx  = out_oc_ch_size + ic_weight_size + kh * kernel_w + kw;
                                
                                pixel_value += input_ptr[in_idx] * weight_ptr[w_idx];
                            }
                        }
                    }
                    int out_idx = out_b_ch_size + out_oc_size + oh * output_w + ow;
                    out_ptr[out_idx] = pixel_value;
                }
            }
        }
    }
}

void conv2d_forward_reorder(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                    int stride, int padding) {
    const float* __restrict input_ptr  = input.data.data();
    const float* __restrict weight_ptr = weight.data.data();
    const float* __restrict bias_ptr   = bias.data.data();
    float* __restrict       out_ptr    = output.data.data();

    int out_channels = weight.batches;
    int in_channels  = weight.channels;
    int kernel_h     = weight.height;
    int kernel_w     = weight.width;
    int input_h      = input.height;
    int input_w      = input.width;
    int output_h     = output.height;
    int output_w     = output.width;
    int input_batches = input.batches;

    int in_size = input_h * input_w;
    int in_ch_size = in_channels * in_size;
    int out_size = output_h * output_w;
    int out_ch_size = out_channels * out_size;
    int weight_size = kernel_h * kernel_w;
    int weight_ch_size = in_channels * weight_size;

    std::vector<float> acc(output_w); 
    float* __restrict acc_ptr = acc.data();

    for (int b = 0; b < input_batches; ++b) {
        int in_b_ch_size = b * in_ch_size;
        int out_b_ch_size = b * out_ch_size;

        for (int oc = 0; oc < out_channels; ++oc) {
            int out_oc_ch_size = oc * weight_ch_size;
            int out_oc_size = oc * out_size;
            const float b_val = bias_ptr[oc];
            
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;

                int c_h = oh * stride - padding;
                int kh_lo = std::max(0, -c_h);
                int kh_hi = std::min(kernel_h, input_h - c_h);

                for (int ic = 0; ic < in_channels; ++ic) {
                    int ic_size = ic * in_size;
                    int ic_weight_size = ic * weight_size;
                    
                    for (int kh = kh_lo; kh < kh_hi; ++kh) {
                        int ih = c_h + kh;
                        int in_row_offset = in_b_ch_size + ic_size + ih * input_w;
                        int kh_kernel_w = kh * kernel_w;

                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int num   = padding - kw;
                            int ow_lo = num <= 0 ? 0 : (num + stride - 1) / stride;
                            int ow_hi = std::min(output_w, (input_w + padding - kw - 1) / stride + 1);

                            int w_idx  = out_oc_ch_size + ic_weight_size + kh_kernel_w + kw;
                            float w_val = weight_ptr[w_idx];

                            for (int ow = ow_lo; ow < ow_hi; ++ow) {
                                int iw = ow * stride - padding + kw;
                                
                                acc_ptr[ow] += input_ptr[in_row_offset + iw] * w_val;
                            }
                        }
                    }
                }   
                int out_row = out_b_ch_size + out_oc_size + oh * output_w;
                for (int ow = 0; ow < output_w; ++ow) {
                    out_ptr[out_row + ow] = acc_ptr[ow];
                }
            }
        }
    }
}

void conv2d_forward_specialized(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
    const float* __restrict input_ptr  = input.data.data();
    const float* __restrict weight_ptr = weight.data.data();
    const float* __restrict bias_ptr   = bias.data.data();
    float* __restrict       out_ptr    = output.data.data();

    int out_channels = weight.batches;
    int in_channels  = weight.channels;
    int kernel_h     = weight.height;
    int kernel_w     = weight.width;
    int input_h      = input.height;
    int input_w      = input.width;
    int output_h     = output.height;
    int output_w     = output.width;
    int input_batches = input.batches;

    int in_size = input_h * input_w;
    int in_ch_size = in_channels * in_size;
    int out_size = output_h * output_w;
    int out_ch_size = out_channels * out_size;
    int weight_size = kernel_h * kernel_w;
    int weight_ch_size = in_channels * weight_size;

    constexpr int MAX_OW = 64;
    alignas(16) float acc[MAX_OW]; 

    for (int b = 0; b < input_batches; ++b) {
        int in_b_ch_size = b * in_ch_size;
        int out_b_ch_size = b * out_ch_size;

        for (int oc = 0; oc < out_channels; ++oc) {
            int out_oc_ch_size = oc * weight_ch_size;
            int out_oc_size = oc * out_size;
            const float b_val = bias_ptr[oc];
            
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;

                for (int ic = 0; ic < in_channels; ++ic) {
                    int ic_size = ic * in_size;
                    int ic_weight_size = ic * weight_size;
                    
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        int ih = oh + kh;
                        int in_row_offset = in_b_ch_size + ic_size + ih * input_w;
                        int kh_kernel_w = kh * kernel_w;

                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int w_idx  = out_oc_ch_size + ic_weight_size + kh_kernel_w + kw;
                            float w_val = weight_ptr[w_idx];

                            for (int ow = 0; ow < output_w; ++ow) {
                                acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
                            }
                        }
                    }
                }   
                int out_row = out_b_ch_size + out_oc_size + oh * output_w;
                for (int ow = 0; ow < output_w; ++ow) {
                    out_ptr[out_row + ow] = acc[ow];
                }
            }
        }
    }
}

template <int OC_T>
static void pack_weights(const float* __restrict w, float* __restrict packed, int out_channels, 
                         int in_channels, int kernel_h, int kernel_w) {
    const int ksz  = kernel_h * kernel_w;
    const int wcsz  = in_channels * ksz;  
    const int tiles = out_channels / OC_T;

    for (int tb = 0; tb < tiles; ++tb) {
        for (int ic = 0; ic < in_channels; ++ic)
            for (int k = 0; k < ksz; ++k)
                for (int t = 0; t < OC_T; ++t)
                    *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
    }
}

template <int OC_T, int NV>
static inline void conv_row_tile(const float* __restrict in_batch, const float* __restrict wpack_tile,
                                 const float* __restrict bias_ptr, float* __restrict out_batch, int oc0, int oh,
                                 int in_channels, int in_size, int input_w, int kernel_h, int kernel_w,
                                 int output_w, int out_size) {
    float32x4_t acc[OC_T][NV];
    for (int t = 0; t < OC_T; ++t) {
        const float32x4_t bv = vdupq_n_f32(bias_ptr[oc0 + t]);
        for (int v = 0; v < NV; ++v) acc[t][v] = bv;
    }

    const float* __restrict wp = wpack_tile;

    for (int ic = 0; ic < in_channels; ++ic) {
        const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;

        for (int kh = 0; kh < kernel_h; ++kh) {
            const float* __restrict in_row = in_c + (std::size_t)(oh + kh) * input_w;

            for (int kw = 0; kw < kernel_w; ++kw) {
                float wv[OC_T];

                for (int t = 0; t < OC_T; ++t) wv[t] = wp[t];
                wp += OC_T;
                const float* __restrict x = in_row + kw;

                for (int v = 0; v < NV; ++v) {
                    const float32x4_t xv = vld1q_f32(x + 4 * v);

                    for (int t = 0; t < OC_T; ++t)
                        acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
                }
            }
        }
    }

    
    for (int t = 0; t < OC_T; ++t) {
        float* __restrict o = out_batch
                            + (std::size_t)(oc0 + t) * out_size
                            + (std::size_t)oh * output_w;
        int v = 0, col = 0;
        for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
        if (col < output_w) {                          
            float tmp[4];
            vst1q_f32(tmp, acc[t][v]);
            for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
        }
    }
}


template <int NV, int OC_TILE>
static void conv2d_blocked_impl(const float* __restrict input_ptr, const float* __restrict wpack, const float* __restrict bias_ptr,
                                float* __restrict out_ptr, int batches, int in_channels, int out_channels,
                                int input_h, int input_w, int kernel_h, int kernel_w, int output_h, int output_w) {
    const int in_size     = input_h * input_w;
    const int in_ch_size  = in_channels * in_size;
    const int out_size    = output_h * output_w;
    const int out_ch_size = out_channels * out_size;
    const int ksz         = kernel_h * kernel_w;
    const int tile_wsz    = in_channels * ksz * OC_TILE;

    const int n_full = (out_channels / OC_TILE) * OC_TILE;

    for (int b = 0; b < batches; ++b) {
        const float* __restrict in_b  = input_ptr + (std::size_t)b * in_ch_size;
        float*       __restrict out_b = out_ptr   + (std::size_t)b * out_ch_size;

        for (int oh = 0; oh < output_h; ++oh) {
            int oc0 = 0, tile = 0;

            for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
                conv_row_tile<OC_TILE, NV>(
                    in_b, wpack + (std::size_t)tile * tile_wsz, bias_ptr, out_b,
                    oc0, oh, in_channels, in_size, input_w,
                    kernel_h, kernel_w, output_w, out_size);
            
            for (int r = 0; oc0 < out_channels; ++oc0, ++r)
                conv_row_tile<1, NV>(
                    in_b,
                    wpack + (std::size_t)(n_full / OC_TILE) * tile_wsz
                          + (std::size_t)r * in_channels * ksz,
                    bias_ptr, out_b, oc0, oh, in_channels, in_size, input_w,
                    kernel_h, kernel_w, output_w, out_size);
        }
    }
}


void conv2d_forward_specialized_blocked(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
    const int OC = weight.batches, IC = weight.channels;
    const int KH = weight.height,  KW = weight.width;
    const int IH = input.height,   IW = input.width;
    const int OH = output.height,  OW = output.width;

    assert(input.channels  == IC);
    assert(output.channels == OC);
    assert(output.batches  == input.batches);
    assert((int)bias.data.size() >= OC);
    assert(OH == IH - KH + 1);
    assert(OW == IW - KW + 1);

    const int NV = (OW + 3) / 4;

    const std::size_t numel = (std::size_t)input.batches * IC * IH * IW;
    const std::size_t slack = (std::size_t)(NV * 4 - OW);
    const float* in_ptr = input.data.data();
    std::vector<float> in_pad;
    if (slack && input.data.size() < numel + slack) {
        in_pad.assign(input.data.begin(), input.data.end());
        in_pad.resize(numel + slack, 0.0f);
        in_ptr = in_pad.data();
    }

    const float* bp = bias.data.data();
    float*       op = output.data.data();

    auto run = [&](auto oc_tile_c, auto nv_c) {
        constexpr int OC_TILE = decltype(oc_tile_c)::value;
        constexpr int NVv     = decltype(nv_c)::value;
        std::vector<float> packed((std::size_t)OC * IC * KH * KW);
        pack_weights<OC_TILE>(weight.data.data(), packed.data(), OC, IC, KH, KW);
        
        const int n_full = (OC / OC_TILE) * OC_TILE;
        if (n_full < OC)
            pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
                            packed.data() + (std::size_t)n_full * IC * KH * KW,
                            OC - n_full, IC, KH, KW);
        conv2d_blocked_impl<NVv, OC_TILE>(
            in_ptr, packed.data(), bp, op,
            input.batches, IC, OC, IH, IW, KH, KW, OH, OW);
    };

    #define IC_(x) std::integral_constant<int, x>{}
    switch (NV) {
        case 1: run(IC_(4), IC_(1)); break;   
        case 2: run(IC_(4), IC_(2)); break;   
        case 3: run(IC_(4), IC_(3)); break;  
        case 4: run(IC_(3), IC_(4)); break; 
        case 5: run(IC_(2), IC_(5)); break;
        case 6: run(IC_(2), IC_(6)); break;
        case 7: run(IC_(2), IC_(7)); break; 
        default:
            conv2d_forward_specialized(input, weight, bias, output);
            break;
    }
    #undef IC_
}

void relu_forward(Tensor& tensor) {
    int n = tensor.batches * tensor.channels * tensor.height * tensor.width;
    for (int i = 0; i < n; ++i)
        if (tensor.data[i] < 0.0f) tensor.data[i] = 0.0f;
}

void linear_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
    int in_features  = weight.width;
    int out_features = weight.height;


    for (int b = 0; b < input.batches; ++b) {
        for (int out_f = 0; out_f < out_features; ++out_f) {
            float val = bias.data[out_f];
            for (int in_f = 0; in_f < in_features; ++in_f)
                val += input.data[b * in_features + in_f] * weight.data[out_f * in_features + in_f];
            output.data[b * out_features + out_f] = val;
        }
    }
}

void maxpool2d_forward(const Tensor& input, Tensor& output, int pool_size, int stride) {

    for (int b = 0; b < input.batches; ++b) {
        for (int c = 0; c < input.channels; ++c) {
            for (int oh = 0; oh < output.height; ++oh) {
                for (int ow = 0; ow < output.width; ++ow) {
                    float max_val = MAXPOOL_NEG_INF;
                    for (int ph = 0; ph < pool_size; ++ph) {
                        for (int pw = 0; pw < pool_size; ++pw) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            if (ih < input.height && iw < input.width) {
                                int in_idx = b * (input.channels * input.height * input.width) +
                                             c * (input.height * input.width) +
                                             ih * input.width + iw;
                                max_val = std::max(max_val, input.data[in_idx]);
                            }
                        }
                    }
                    int out_idx = b * (output.channels * output.height * output.width) +
                                  c * (output.height * output.width) +
                                  oh * output.width + ow;
                    output.data[out_idx] = max_val;
                }
            }
        }
    }
}

void maxpool2d_forward_specialized(const Tensor& input, Tensor& output) {

    const float* __restrict input_ptr  = input.data.data();
    float* __restrict       out_ptr    = output.data.data();

    int in_channels   = input.channels;
    int input_h       = input.height;
    int input_w       = input.width;
    int output_h      = output.height;
    int output_w      = output.width;
    int input_batches = input.batches;
    int out_channels  = output.channels;

    int in_size      = input_h * input_w;
    int in_ch_size   = in_channels * in_size;
    int out_size     = output_h * output_w;
    int out_ch_size  = out_channels * out_size;

    constexpr int MAX_OW = 64;
    alignas(16) float acc[MAX_OW]; 

    for (int b = 0; b < input_batches; ++b) {
        int in_b_ch_size  = b * in_ch_size;
        int out_b_ch_size = b * out_ch_size;

        for (int c = 0; c < in_channels; ++c) {
            int ic_size    = c * in_size;
            int out_c_size = c * out_size;

            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) acc[ow] = MAXPOOL_NEG_INF;

                for (int ph = 0; ph < 2; ++ph) {
                    int ih = oh * 2 + ph;
                    int in_row_offset = in_b_ch_size + ic_size + ih * input_w;

                    for (int pw = 0; pw < 2; ++pw) {
                        for (int ow = 0; ow < output_w; ++ow) {
                            int iw = ow * 2 + pw;
                            acc[ow] = std::max(acc[ow], input_ptr[in_row_offset + iw]);
                        }
                    }
                }

                int out_row = out_b_ch_size + out_c_size + oh * output_w;
                for (int ow = 0; ow < output_w; ++ow) {
                    out_ptr[out_row + ow] = acc[ow];
                }
            }
        }
    }
}

void adaptive_avgpool2d_forward(const Tensor& input, Tensor& output) {
    int spatial_size = input.height * input.width;

    for (int b = 0; b < input.batches; ++b) {
        for (int c = 0; c < input.channels; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < spatial_size; ++i)
                sum += input.data[b * (input.channels * spatial_size) + c * spatial_size + i];
            output.data[b * input.channels + c] = sum / spatial_size;
        }
    }
}



void conv2d_forward_im2col(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                             int stride, int padding) {
    int out_channels = weight.batches;   // C_out
    int in_channels  = weight.channels;  // C_in
    int kernel_h     = weight.height;    // K_h
    int kernel_w     = weight.width;     // K_w

    int out_h = output.height;
    int out_w = output.width;

    int K = in_channels * kernel_h * kernel_w; // Matrix inner dimension
    int N = out_h * out_w;                     // Spatial positions per image
    int M = out_channels;                      // Output channels

    // Pre-allocating buffer for the im2col matrix
    std::vector<float> im2col_buf(K * N);

    for (int b = 0; b < input.batches; ++b) {
        // 1. Constructing the im2col matrix for batch 'b'
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int row = (ic * kernel_h + kh) * kernel_w + kw;
                    for (int oh = 0; oh < out_h; ++oh) {
                        int ih = oh * stride - padding + kh;
                        for (int ow = 0; ow < out_w; ++ow) {
                            int iw = ow * stride - padding + kw;
                            int col = oh * out_w + ow;

                            float val = 0.0f;
                            if (ih >= 0 && ih < input.height && iw >= 0 && iw < input.width) {
                                int in_idx = b * (input.channels * input.height * input.width) +
                                             ic * (input.height * input.width) +
                                             ih * input.width + iw;
                                val = input.data[in_idx];
                            }
                            im2col_buf[row * N + col] = val;
                        }
                    }
                }
            }
        }

        // 2. Performing GEMM: Y = W * X + Bias
        int batch_offset = b * (out_channels * N);

        for (int oc = 0; oc < M; ++oc) {
            int out_channel_offset = batch_offset + oc * N;
            float b_val = bias.data[oc];

            // Initializing output row with bias vector
            for (int col = 0; col < N; ++col) {
                output.data[out_channel_offset + col] = b_val;
            }

            // Multiplying row of W with im2col matrix X
            const float* w_row = &weight.data[oc * K];
            for (int k = 0; k < K; ++k) {
                float w_val = w_row[k];
                const float* col_row = &im2col_buf[k * N];
                float* out_row = &output.data[out_channel_offset];

                #pragma omp simd
                for (int col = 0; col < N; ++col) {
                    out_row[col] += w_val * col_row[col];
                }
            }
        }
    }
}

