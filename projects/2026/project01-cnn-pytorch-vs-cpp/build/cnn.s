	.arch armv9-a+crc+i8mm+bf16+sve2-aes+sve2-bitperm+sve2-sha3+sve2-sm4+memtag+profile+nossbs
	.file	"cnn.cpp"
// GNU C++17 (Spack GCC) version 14.2.0 (aarch64-unknown-linux-gnu)
//	compiled by GNU C version 14.2.0, GMP version 6.3.0, MPFR version 4.2.1-p1, MPC version 1.3.1, isl version none
// GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
// options passed: -mlittle-endian -mabi=lp64 -mcpu=neoverse-v2+crc+sve2-aes+sve2-sha3+sve2-sm4+norng+nossbs -O3 -std=c++17 -ffast-math -ftree-vectorize -fopt-info-vec-optimized
	.text
	.align	2
	.p2align 5,,15
	.global	_Z12cnn_baselineR10CNNContext
	.type	_Z12cnn_baselineR10CNNContext, %function
_Z12cnn_baselineR10CNNContext:
.LFB5444:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!	//,,,
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	mov	x29, sp	//,
// src/cpp/cnn.cpp:5:     conv2d_forward(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
	mov	w5, 0	//,
	mov	w4, 1	//,
// src/cpp/cnn.cpp:4: void cnn_baseline( CNNContext& ctx ) {
	str	x19, [sp, 16]	//,
	.cfi_offset 19, -16
// src/cpp/cnn.cpp:4: void cnn_baseline( CNNContext& ctx ) {
	mov	x19, x0	// ctx, tmp151
// src/cpp/cnn.cpp:5:     conv2d_forward(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
	ldp	x2, x3, [x19, 16]	//,,
	ldp	x0, x1, [x0]	//,,* ctx
	bl	_Z14conv2d_forwardRK6TensorS1_S1_RS_ii		//
// src/cpp/cnn.cpp:6:     relu_forward(ctx.conv1_out);
	ldr	x0, [x19, 24]	//, ctx_27(D)->conv1_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:7:     maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);
	mov	w3, 2	//,
	ldp	x0, x1, [x19, 24]	//,,
	mov	w2, w3	//,
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/cnn.cpp:9:     conv2d_forward(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out, 1, 0);
	ldp	x0, x1, [x19, 32]	//,,
	mov	w5, 0	//,
	mov	w4, 1	//,
	ldp	x2, x3, [x19, 48]	//,,
	bl	_Z14conv2d_forwardRK6TensorS1_S1_RS_ii		//
// src/cpp/cnn.cpp:10:     relu_forward(ctx.conv2_out);
	ldr	x0, [x19, 56]	//, ctx_27(D)->conv2_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:11:     maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);
	mov	w3, 2	//,
	ldp	x0, x1, [x19, 56]	//,,
	mov	w2, w3	//,
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/cnn.cpp:13:     conv2d_forward(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out, 1, 0);
	ldp	x0, x1, [x19, 64]	//,,
	mov	w5, 0	//,
	mov	w4, 1	//,
	ldp	x2, x3, [x19, 80]	//,,
	bl	_Z14conv2d_forwardRK6TensorS1_S1_RS_ii		//
// src/cpp/cnn.cpp:14:     relu_forward(ctx.conv3_out);
	ldr	x0, [x19, 88]	//, ctx_27(D)->conv3_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:15:     adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);
	ldp	x0, x1, [x19, 88]	//,,
	bl	_Z26adaptive_avgpool2d_forwardRK6TensorRS_		//
// src/cpp/cnn.cpp:17:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	ldp	x0, x1, [x19, 96]	//,,
	ldp	x2, x3, [x19, 112]	//,,
// src/cpp/cnn.cpp:18: }
	ldr	x19, [sp, 16]	//,
	ldp	x29, x30, [sp], 32	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_def_cfa_offset 0
// src/cpp/cnn.cpp:17:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	b	_Z14linear_forwardRK6TensorS1_S1_RS_		//
	.cfi_endproc
.LFE5444:
	.size	_Z12cnn_baselineR10CNNContext, .-_Z12cnn_baselineR10CNNContext
	.align	2
	.p2align 5,,15
	.global	_Z16cnn_restructuredR10CNNContext
	.type	_Z16cnn_restructuredR10CNNContext, %function
_Z16cnn_restructuredR10CNNContext:
.LFB5445:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!	//,,,
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	mov	x29, sp	//,
// src/cpp/cnn.cpp:21:     conv2d_forward_restructured(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
	mov	w5, 0	//,
	mov	w4, 1	//,
// src/cpp/cnn.cpp:20: void cnn_restructured( CNNContext& ctx ) {
	str	x19, [sp, 16]	//,
	.cfi_offset 19, -16
// src/cpp/cnn.cpp:20: void cnn_restructured( CNNContext& ctx ) {
	mov	x19, x0	// ctx, tmp151
// src/cpp/cnn.cpp:21:     conv2d_forward_restructured(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
	ldp	x2, x3, [x19, 16]	//,,
	ldp	x0, x1, [x0]	//,,* ctx
	bl	_Z27conv2d_forward_restructuredRK6TensorS1_S1_RS_ii		//
// src/cpp/cnn.cpp:22:     relu_forward(ctx.conv1_out);
	ldr	x0, [x19, 24]	//, ctx_27(D)->conv1_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:23:     maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);
	mov	w3, 2	//,
	ldp	x0, x1, [x19, 24]	//,,
	mov	w2, w3	//,
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/cnn.cpp:25:     conv2d_forward_restructured(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out, 1, 0);
	ldp	x0, x1, [x19, 32]	//,,
	mov	w5, 0	//,
	mov	w4, 1	//,
	ldp	x2, x3, [x19, 48]	//,,
	bl	_Z27conv2d_forward_restructuredRK6TensorS1_S1_RS_ii		//
// src/cpp/cnn.cpp:26:     relu_forward(ctx.conv2_out);
	ldr	x0, [x19, 56]	//, ctx_27(D)->conv2_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:27:     maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);
	mov	w3, 2	//,
	ldp	x0, x1, [x19, 56]	//,,
	mov	w2, w3	//,
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/cnn.cpp:29:     conv2d_forward_restructured(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out, 1, 0);
	ldp	x0, x1, [x19, 64]	//,,
	mov	w5, 0	//,
	mov	w4, 1	//,
	ldp	x2, x3, [x19, 80]	//,,
	bl	_Z27conv2d_forward_restructuredRK6TensorS1_S1_RS_ii		//
// src/cpp/cnn.cpp:30:     relu_forward(ctx.conv3_out);
	ldr	x0, [x19, 88]	//, ctx_27(D)->conv3_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:31:     adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);
	ldp	x0, x1, [x19, 88]	//,,
	bl	_Z26adaptive_avgpool2d_forwardRK6TensorRS_		//
// src/cpp/cnn.cpp:33:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	ldp	x0, x1, [x19, 96]	//,,
	ldp	x2, x3, [x19, 112]	//,,
// src/cpp/cnn.cpp:34: }
	ldr	x19, [sp, 16]	//,
	ldp	x29, x30, [sp], 32	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_def_cfa_offset 0
// src/cpp/cnn.cpp:33:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	b	_Z14linear_forwardRK6TensorS1_S1_RS_		//
	.cfi_endproc
.LFE5445:
	.size	_Z16cnn_restructuredR10CNNContext, .-_Z16cnn_restructuredR10CNNContext
	.align	2
	.p2align 5,,15
	.global	_Z18cnn_hoist_restrictR10CNNContext
	.type	_Z18cnn_hoist_restrictR10CNNContext, %function
_Z18cnn_hoist_restrictR10CNNContext:
.LFB5446:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!	//,,,
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	mov	x29, sp	//,
// src/cpp/cnn.cpp:37:     conv2d_forward_hoist_restrict(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
	mov	w5, 0	//,
	mov	w4, 1	//,
// src/cpp/cnn.cpp:36: void cnn_hoist_restrict( CNNContext& ctx ) {
	str	x19, [sp, 16]	//,
	.cfi_offset 19, -16
// src/cpp/cnn.cpp:36: void cnn_hoist_restrict( CNNContext& ctx ) {
	mov	x19, x0	// ctx, tmp151
// src/cpp/cnn.cpp:37:     conv2d_forward_hoist_restrict(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
	ldp	x2, x3, [x19, 16]	//,,
	ldp	x0, x1, [x0]	//,,* ctx
	bl	_Z29conv2d_forward_hoist_restrictRK6TensorS1_S1_RS_ii		//
// src/cpp/cnn.cpp:38:     relu_forward(ctx.conv1_out);
	ldr	x0, [x19, 24]	//, ctx_27(D)->conv1_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:39:     maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);
	mov	w3, 2	//,
	ldp	x0, x1, [x19, 24]	//,,
	mov	w2, w3	//,
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/cnn.cpp:41:     conv2d_forward_hoist_restrict(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out, 1, 0);
	ldp	x0, x1, [x19, 32]	//,,
	mov	w5, 0	//,
	mov	w4, 1	//,
	ldp	x2, x3, [x19, 48]	//,,
	bl	_Z29conv2d_forward_hoist_restrictRK6TensorS1_S1_RS_ii		//
// src/cpp/cnn.cpp:42:     relu_forward(ctx.conv2_out);
	ldr	x0, [x19, 56]	//, ctx_27(D)->conv2_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:43:     maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);
	mov	w3, 2	//,
	ldp	x0, x1, [x19, 56]	//,,
	mov	w2, w3	//,
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/cnn.cpp:45:     conv2d_forward_hoist_restrict(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out, 1, 0);
	ldp	x0, x1, [x19, 64]	//,,
	mov	w5, 0	//,
	mov	w4, 1	//,
	ldp	x2, x3, [x19, 80]	//,,
	bl	_Z29conv2d_forward_hoist_restrictRK6TensorS1_S1_RS_ii		//
// src/cpp/cnn.cpp:46:     relu_forward(ctx.conv3_out);
	ldr	x0, [x19, 88]	//, ctx_27(D)->conv3_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:47:     adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);
	ldp	x0, x1, [x19, 88]	//,,
	bl	_Z26adaptive_avgpool2d_forwardRK6TensorRS_		//
// src/cpp/cnn.cpp:49:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	ldp	x0, x1, [x19, 96]	//,,
	ldp	x2, x3, [x19, 112]	//,,
// src/cpp/cnn.cpp:50: }
	ldr	x19, [sp, 16]	//,
	ldp	x29, x30, [sp], 32	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_def_cfa_offset 0
// src/cpp/cnn.cpp:49:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	b	_Z14linear_forwardRK6TensorS1_S1_RS_		//
	.cfi_endproc
.LFE5446:
	.size	_Z18cnn_hoist_restrictR10CNNContext, .-_Z18cnn_hoist_restrictR10CNNContext
	.align	2
	.p2align 5,,15
	.global	_Z11cnn_reorderR10CNNContext
	.type	_Z11cnn_reorderR10CNNContext, %function
_Z11cnn_reorderR10CNNContext:
.LFB5447:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!	//,,,
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	mov	x29, sp	//,
// src/cpp/cnn.cpp:53:     conv2d_forward_reorder(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
	mov	w5, 0	//,
	mov	w4, 1	//,
// src/cpp/cnn.cpp:52: void cnn_reorder( CNNContext& ctx ) {
	str	x19, [sp, 16]	//,
	.cfi_offset 19, -16
// src/cpp/cnn.cpp:52: void cnn_reorder( CNNContext& ctx ) {
	mov	x19, x0	// ctx, tmp151
// src/cpp/cnn.cpp:53:     conv2d_forward_reorder(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
	ldp	x2, x3, [x19, 16]	//,,
	ldp	x0, x1, [x0]	//,,* ctx
	bl	_Z22conv2d_forward_reorderRK6TensorS1_S1_RS_ii		//
// src/cpp/cnn.cpp:54:     relu_forward(ctx.conv1_out);
	ldr	x0, [x19, 24]	//, ctx_27(D)->conv1_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:55:     maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);
	mov	w3, 2	//,
	ldp	x0, x1, [x19, 24]	//,,
	mov	w2, w3	//,
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/cnn.cpp:57:     conv2d_forward_reorder(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out, 1, 0);
	ldp	x0, x1, [x19, 32]	//,,
	mov	w5, 0	//,
	mov	w4, 1	//,
	ldp	x2, x3, [x19, 48]	//,,
	bl	_Z22conv2d_forward_reorderRK6TensorS1_S1_RS_ii		//
// src/cpp/cnn.cpp:58:     relu_forward(ctx.conv2_out);
	ldr	x0, [x19, 56]	//, ctx_27(D)->conv2_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:59:     maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);
	mov	w3, 2	//,
	ldp	x0, x1, [x19, 56]	//,,
	mov	w2, w3	//,
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/cnn.cpp:61:     conv2d_forward_reorder(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out, 1, 0);
	ldp	x0, x1, [x19, 64]	//,,
	mov	w5, 0	//,
	mov	w4, 1	//,
	ldp	x2, x3, [x19, 80]	//,,
	bl	_Z22conv2d_forward_reorderRK6TensorS1_S1_RS_ii		//
// src/cpp/cnn.cpp:62:     relu_forward(ctx.conv3_out);
	ldr	x0, [x19, 88]	//, ctx_27(D)->conv3_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:63:     adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);
	ldp	x0, x1, [x19, 88]	//,,
	bl	_Z26adaptive_avgpool2d_forwardRK6TensorRS_		//
// src/cpp/cnn.cpp:65:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	ldp	x0, x1, [x19, 96]	//,,
	ldp	x2, x3, [x19, 112]	//,,
// src/cpp/cnn.cpp:66: }
	ldr	x19, [sp, 16]	//,
	ldp	x29, x30, [sp], 32	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_def_cfa_offset 0
// src/cpp/cnn.cpp:65:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	b	_Z14linear_forwardRK6TensorS1_S1_RS_		//
	.cfi_endproc
.LFE5447:
	.size	_Z11cnn_reorderR10CNNContext, .-_Z11cnn_reorderR10CNNContext
	.align	2
	.p2align 5,,15
	.global	_Z15cnn_specializedR10CNNContext
	.type	_Z15cnn_specializedR10CNNContext, %function
_Z15cnn_specializedR10CNNContext:
.LFB5448:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!	//,,,
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	mov	x29, sp	//,
	str	x19, [sp, 16]	//,
	.cfi_offset 19, -16
// src/cpp/cnn.cpp:68: void cnn_specialized( CNNContext& ctx ) {
	mov	x19, x0	// ctx, tmp151
// src/cpp/cnn.cpp:69:     conv2d_forward_specialized(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out);
	ldp	x2, x3, [x19, 16]	//,,
	ldp	x0, x1, [x0]	//,,* ctx
	bl	_Z26conv2d_forward_specializedRK6TensorS1_S1_RS_		//
// src/cpp/cnn.cpp:70:     relu_forward(ctx.conv1_out);
	ldr	x0, [x19, 24]	//, ctx_27(D)->conv1_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:71:     maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);
	mov	w3, 2	//,
	ldp	x0, x1, [x19, 24]	//,,
	mov	w2, w3	//,
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/cnn.cpp:73:     conv2d_forward_specialized(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out);
	ldp	x0, x1, [x19, 32]	//,,
	ldp	x2, x3, [x19, 48]	//,,
	bl	_Z26conv2d_forward_specializedRK6TensorS1_S1_RS_		//
// src/cpp/cnn.cpp:74:     relu_forward(ctx.conv2_out);
	ldr	x0, [x19, 56]	//, ctx_27(D)->conv2_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:75:     maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);
	mov	w3, 2	//,
	ldp	x0, x1, [x19, 56]	//,,
	mov	w2, w3	//,
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/cnn.cpp:77:     conv2d_forward_specialized(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out);
	ldp	x0, x1, [x19, 64]	//,,
	ldp	x2, x3, [x19, 80]	//,,
	bl	_Z26conv2d_forward_specializedRK6TensorS1_S1_RS_		//
// src/cpp/cnn.cpp:78:     relu_forward(ctx.conv3_out);
	ldr	x0, [x19, 88]	//, ctx_27(D)->conv3_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:79:     adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);
	ldp	x0, x1, [x19, 88]	//,,
	bl	_Z26adaptive_avgpool2d_forwardRK6TensorRS_		//
// src/cpp/cnn.cpp:81:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	ldp	x0, x1, [x19, 96]	//,,
	ldp	x2, x3, [x19, 112]	//,,
// src/cpp/cnn.cpp:82: }
	ldr	x19, [sp, 16]	//,
	ldp	x29, x30, [sp], 32	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_def_cfa_offset 0
// src/cpp/cnn.cpp:81:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	b	_Z14linear_forwardRK6TensorS1_S1_RS_		//
	.cfi_endproc
.LFE5448:
	.size	_Z15cnn_specializedR10CNNContext, .-_Z15cnn_specializedR10CNNContext
	.align	2
	.p2align 5,,15
	.global	_Z23cnn_specialized_blockedR10CNNContext
	.type	_Z23cnn_specialized_blockedR10CNNContext, %function
_Z23cnn_specialized_blockedR10CNNContext:
.LFB5449:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!	//,,,
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	mov	x29, sp	//,
	str	x19, [sp, 16]	//,
	.cfi_offset 19, -16
// src/cpp/cnn.cpp:84: void cnn_specialized_blocked( CNNContext& ctx ) {
	mov	x19, x0	// ctx, tmp151
// src/cpp/cnn.cpp:85:     conv2d_forward_specialized_blocked(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out);
	ldp	x2, x3, [x19, 16]	//,,
	ldp	x0, x1, [x0]	//,,* ctx
	bl	_Z34conv2d_forward_specialized_blockedRK6TensorS1_S1_RS_		//
// src/cpp/cnn.cpp:86:     relu_forward(ctx.conv1_out);
	ldr	x0, [x19, 24]	//, ctx_27(D)->conv1_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:87:     maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);
	mov	w3, 2	//,
	ldp	x0, x1, [x19, 24]	//,,
	mov	w2, w3	//,
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/cnn.cpp:89:     conv2d_forward_specialized_blocked(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out);
	ldp	x0, x1, [x19, 32]	//,,
	ldp	x2, x3, [x19, 48]	//,,
	bl	_Z34conv2d_forward_specialized_blockedRK6TensorS1_S1_RS_		//
// src/cpp/cnn.cpp:90:     relu_forward(ctx.conv2_out);
	ldr	x0, [x19, 56]	//, ctx_27(D)->conv2_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:91:     maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);
	mov	w3, 2	//,
	ldp	x0, x1, [x19, 56]	//,,
	mov	w2, w3	//,
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/cnn.cpp:93:     conv2d_forward_specialized_blocked(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out);
	ldp	x0, x1, [x19, 64]	//,,
	ldp	x2, x3, [x19, 80]	//,,
	bl	_Z34conv2d_forward_specialized_blockedRK6TensorS1_S1_RS_		//
// src/cpp/cnn.cpp:94:     relu_forward(ctx.conv3_out);
	ldr	x0, [x19, 88]	//, ctx_27(D)->conv3_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:95:     adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);
	ldp	x0, x1, [x19, 88]	//,,
	bl	_Z26adaptive_avgpool2d_forwardRK6TensorRS_		//
// src/cpp/cnn.cpp:97:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	ldp	x0, x1, [x19, 96]	//,,
	ldp	x2, x3, [x19, 112]	//,,
// src/cpp/cnn.cpp:98: }
	ldr	x19, [sp, 16]	//,
	ldp	x29, x30, [sp], 32	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_def_cfa_offset 0
// src/cpp/cnn.cpp:97:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	b	_Z14linear_forwardRK6TensorS1_S1_RS_		//
	.cfi_endproc
.LFE5449:
	.size	_Z23cnn_specialized_blockedR10CNNContext, .-_Z23cnn_specialized_blockedR10CNNContext
	.align	2
	.p2align 5,,15
	.global	_Z21specialized_maxpool2dR10CNNContext
	.type	_Z21specialized_maxpool2dR10CNNContext, %function
_Z21specialized_maxpool2dR10CNNContext:
.LFB5450:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!	//,,,
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	mov	x29, sp	//,
	str	x19, [sp, 16]	//,
	.cfi_offset 19, -16
// src/cpp/cnn.cpp:100: void specialized_maxpool2d( CNNContext& ctx ) {
	mov	x19, x0	// ctx, tmp151
// src/cpp/cnn.cpp:101:     conv2d_forward_specialized_blocked(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out);
	ldp	x2, x3, [x19, 16]	//,,
	ldp	x0, x1, [x0]	//,,* ctx
	bl	_Z34conv2d_forward_specialized_blockedRK6TensorS1_S1_RS_		//
// src/cpp/cnn.cpp:102:     relu_forward(ctx.conv1_out);
	ldr	x0, [x19, 24]	//, ctx_27(D)->conv1_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:103:     maxpool2d_forward_specialized(ctx.conv1_out, ctx.pool1_out);
	ldp	x0, x1, [x19, 24]	//,,
	bl	_Z29maxpool2d_forward_specializedRK6TensorRS_		//
// src/cpp/cnn.cpp:105:     conv2d_forward_specialized_blocked(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out);
	ldp	x0, x1, [x19, 32]	//,,
	ldp	x2, x3, [x19, 48]	//,,
	bl	_Z34conv2d_forward_specialized_blockedRK6TensorS1_S1_RS_		//
// src/cpp/cnn.cpp:106:     relu_forward(ctx.conv2_out);
	ldr	x0, [x19, 56]	//, ctx_27(D)->conv2_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:107:     maxpool2d_forward_specialized(ctx.conv2_out, ctx.pool2_out);
	ldp	x0, x1, [x19, 56]	//,,
	bl	_Z29maxpool2d_forward_specializedRK6TensorRS_		//
// src/cpp/cnn.cpp:109:     conv2d_forward_specialized_blocked(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out);
	ldp	x0, x1, [x19, 64]	//,,
	ldp	x2, x3, [x19, 80]	//,,
	bl	_Z34conv2d_forward_specialized_blockedRK6TensorS1_S1_RS_		//
// src/cpp/cnn.cpp:110:     relu_forward(ctx.conv3_out);
	ldr	x0, [x19, 88]	//, ctx_27(D)->conv3_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/cnn.cpp:111:     adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);
	ldp	x0, x1, [x19, 88]	//,,
	bl	_Z26adaptive_avgpool2d_forwardRK6TensorRS_		//
// src/cpp/cnn.cpp:113:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	ldp	x0, x1, [x19, 96]	//,,
	ldp	x2, x3, [x19, 112]	//,,
// src/cpp/cnn.cpp:114: }
	ldr	x19, [sp, 16]	//,
	ldp	x29, x30, [sp], 32	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_def_cfa_offset 0
// src/cpp/cnn.cpp:113:     linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	b	_Z14linear_forwardRK6TensorS1_S1_RS_		//
	.cfi_endproc
.LFE5450:
	.size	_Z21specialized_maxpool2dR10CNNContext, .-_Z21specialized_maxpool2dR10CNNContext
	.ident	"GCC: (Spack GCC) 14.2.0"
	.section	.note.GNU-stack,"",@progbits
