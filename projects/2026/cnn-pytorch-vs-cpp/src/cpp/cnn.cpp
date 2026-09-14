#include "include/cnn.h"
#include "include/cnn_internals.h"

void cnn_baseline( CNNContext& ctx ) {
    conv2d_forward(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
    relu_forward(ctx.conv1_out);
    maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);

    conv2d_forward(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out, 1, 0);
    relu_forward(ctx.conv2_out);
    maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);

    conv2d_forward(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out, 1, 0);
    relu_forward(ctx.conv3_out);
    adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);

    linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
}

void cnn_baseline_blocked( CNNContext& ctx ) {
    conv2d_forward_blocked(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
    relu_forward(ctx.conv1_out);
    maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);

    conv2d_forward_blocked(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out, 1, 0);
    relu_forward(ctx.conv2_out);
    maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);

    conv2d_forward_blocked(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out, 1, 0);
    relu_forward(ctx.conv3_out);
    adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);

    linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
}

void cnn_restructured( CNNContext& ctx ) {
    conv2d_forward_restructured(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
    relu_forward(ctx.conv1_out);
    maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);

    conv2d_forward_restructured(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out, 1, 0);
    relu_forward(ctx.conv2_out);
    maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);

    conv2d_forward_restructured(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out, 1, 0);
    relu_forward(ctx.conv3_out);
    adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);

    linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
}

void cnn_hoist_restrict( CNNContext& ctx ) {
    conv2d_forward_hoist_restrict(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
    relu_forward(ctx.conv1_out);
    maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);

    conv2d_forward_hoist_restrict(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out, 1, 0);
    relu_forward(ctx.conv2_out);
    maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);

    conv2d_forward_hoist_restrict(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out, 1, 0);
    relu_forward(ctx.conv3_out);
    adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);

    linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
}


void cnn_im2col( CNNContext& ctx ) {
    conv2d_forward_im2col(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
    relu_forward(ctx.conv1_out);
    maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);

    conv2d_forward_im2col(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out, 1, 0);
    relu_forward(ctx.conv2_out);
    maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);

    conv2d_forward_im2col(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out, 1, 0);
    relu_forward(ctx.conv3_out);
    adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);

    linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
}

void cnn_reorder( CNNContext& ctx ) {
    conv2d_forward_reorder(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
    relu_forward(ctx.conv1_out);
    maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);

    conv2d_forward_reorder(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out, 1, 0);
    relu_forward(ctx.conv2_out);
    maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);

    conv2d_forward_reorder(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out, 1, 0);
    relu_forward(ctx.conv3_out);
    adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);

    linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
}

void cnn_specialized( CNNContext& ctx ) {
    conv2d_forward_specialized(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out);
    relu_forward(ctx.conv1_out);
    maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);

    conv2d_forward_specialized(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out);
    relu_forward(ctx.conv2_out);
    maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);

    conv2d_forward_specialized(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out);
    relu_forward(ctx.conv3_out);
    adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);

    linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
}

void cnn_specialized_blocked( CNNContext& ctx ) {
    conv2d_forward_specialized_blocked(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out);
    relu_forward(ctx.conv1_out);
    maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);

    conv2d_forward_specialized_blocked(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out);
    relu_forward(ctx.conv2_out);
    maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);

    conv2d_forward_specialized_blocked(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out);
    relu_forward(ctx.conv3_out);
    adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);

    linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
}

void specialized_maxpool2d( CNNContext& ctx ) {
    conv2d_forward_specialized_blocked(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out);
    relu_forward(ctx.conv1_out);
    maxpool2d_forward_specialized(ctx.conv1_out, ctx.pool1_out);

    conv2d_forward_specialized_blocked(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out);
    relu_forward(ctx.conv2_out);
    maxpool2d_forward_specialized(ctx.conv2_out, ctx.pool2_out);

    conv2d_forward_specialized_blocked(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out);
    relu_forward(ctx.conv3_out);
    adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);

    linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
}
