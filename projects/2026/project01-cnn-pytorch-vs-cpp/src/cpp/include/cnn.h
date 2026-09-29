#ifndef CNN_H
#define CNN_H

#include "cnn_internals.h"

struct CNNContext {
        const Tensor& input_batch;
        const Tensor& conv1_weight; const Tensor& conv1_bias; Tensor& conv1_out; Tensor& pool1_out;
        const Tensor& conv2_weight; const Tensor& conv2_bias; Tensor& conv2_out; Tensor& pool2_out;
        const Tensor& conv3_weight; const Tensor& conv3_bias; Tensor& conv3_out; Tensor& avgpool_out;
        const Tensor& fc_weight;    const Tensor& fc_bias;    Tensor& final_logits;
};

/*
 * General function pointer type for CNN forward-pass implementations, useful
 * for benchmarking different implementations with the same interface (mirrors
 * ahe_func in the AHE project).
 */
typedef void (*cnn_func)(CNNContext&);

/*
 * Every cnn_* implementation below follows the same fixed architecture, just
 * with a different conv2d/maxpool2d kernel plugged in:
 *   1. Conv1 -> ReLU -> MaxPool
 *   2. Conv2 -> ReLU -> MaxPool
 *   3. Conv3 -> ReLU -> AdaptiveAvgPool
 *   4. Linear -> final logits
 */

/*
* Simple nested loops implementation of the forward pass of a CNN.
* This function is intended to be used as a baseline for performance comparison.
*/
void cnn_baseline( CNNContext& ctx );

/*
* Attempted loop restucturing of the forward pass of a CNN.
*/
void cnn_restructured( CNNContext& ctx );

/*
* Hoisted loop implementation of the forward pass of a CNN.
*/
void cnn_hoist_restrict( CNNContext& ctx );

/*
* Reordered loop implementation of the forward pass of a CNN.
*/
void cnn_reorder( CNNContext& ctx );

/*
* Specialized implementation of the forward pass of a CNN.
*/
void cnn_specialized( CNNContext& ctx );

/*
* Specialized blocked implementation of the forward pass of a CNN.
*/
void cnn_specialized_blocked( CNNContext& ctx );

/*
* Specialized maxpool2d implementation of the forward pass of a CNN.
*/
void specialized_maxpool2d( CNNContext& ctx );


/*
 * Reorder loop such that output channel (oc), which is constant in the 
 * input and kernel arguments, varies slower than ow and oh, so these can
 * stay constant 
*/
void cnn_baseline_blocked( CNNContext& ctx );



/*
 * Forward pass using im2col / im2col matrix multiplication.
 */
void cnn_im2col( CNNContext& ctx );


/*
 * Register implementations here to include them in benchmarking. Each entry
 * is (function, display_name), matching the AHE_IMPLEMENTATIONS pattern.
 */
#define CNN_IMPLEMENTATIONS(APPLY) \
    APPLY(cnn_baseline, "Baseline Nested-Loop") \
    APPLY(cnn_baseline_blocked, "Baseline w. blocking") \
    APPLY(cnn_im2col, "Baseline w. im2col matrix multiplication") \
    APPLY(cnn_restructured, "Restructured Nested-Loop") \
    APPLY(cnn_hoist_restrict, "Hoisted Vars + restrict") \
    APPLY(cnn_reorder, "Reordered Nested-Loop") \
    APPLY(cnn_specialized, "Specialized Implementation") \
    APPLY(cnn_specialized_blocked, "NEON Blocked Implementation") \
    APPLY(specialized_maxpool2d, "Specialized MaxPool2D")

#endif
