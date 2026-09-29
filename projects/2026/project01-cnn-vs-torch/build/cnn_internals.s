	.arch armv9-a+crc+i8mm+bf16+sve2-aes+sve2-bitperm+sve2-sha3+sve2-sm4+memtag+profile+nossbs
	.file	"cnn_internals.cpp"
// GNU C++17 (Spack GCC) version 14.2.0 (aarch64-unknown-linux-gnu)
//	compiled by GNU C version 14.2.0, GMP version 6.3.0, MPFR version 4.2.1-p1, MPC version 1.3.1, isl version none
// GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
// options passed: -mlittle-endian -mabi=lp64 -mcpu=neoverse-v2+crc+sve2-aes+sve2-sha3+sve2-sm4+norng+nossbs -O3 -std=c++17 -ffast-math -ftree-vectorize -fopt-info-vec-optimized
	.text
	.align	2
	.p2align 5,,15
	.global	_Z14conv2d_forwardRK6TensorS1_S1_RS_ii
	.type	_Z14conv2d_forwardRK6TensorS1_S1_RS_ii, %function
_Z14conv2d_forwardRK6TensorS1_S1_RS_ii:
.LFB5444:
	.cfi_startproc
// src/cpp/cnn_internals.cpp:14:     for (int b = 0; b < input.batches; ++b) {
	ldr	w12, [x0, 24]	//, input_63(D)->batches
// src/cpp/cnn_internals.cpp:7:                     int stride, int padding) {
	mov	x8, x0	// input, tmp248
// src/cpp/cnn_internals.cpp:8:     int out_channels = weight.batches;
	ldr	w0, [x1, 24]	//, weight_58(D)->batches
// src/cpp/cnn_internals.cpp:14:     for (int b = 0; b < input.batches; ++b) {
	cmp	w12, 0	// _98,
	ble	.L50		//,
	cmp	w0, 0	// out_channels,
	ble	.L50		//,
// src/cpp/cnn_internals.cpp:7:                     int stride, int padding) {
	stp	x29, x30, [sp, -192]!	//,,,
	.cfi_def_cfa_offset 192
	.cfi_offset 29, -192
	.cfi_offset 30, -184
	mov	x29, sp	//,
	mov	x15, x2	// bias, tmp250
	mov	x14, x1	// weight, tmp249
	mov	x11, x3	// output, tmp251
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -176
	.cfi_offset 20, -168
// src/cpp/cnn_internals.cpp:14:     for (int b = 0; b < input.batches; ++b) {
	mov	w20, 0	// b,
// src/cpp/cnn_internals.cpp:7:                     int stride, int padding) {
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 21, -160
	.cfi_offset 22, -152
	mov	w22, w4	// stride, tmp252
	stp	x23, x24, [sp, 48]	//,,
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 23, -144
	.cfi_offset 24, -136
	.cfi_offset 25, -128
	.cfi_offset 26, -120
	mov	w25, w5	// padding, tmp253
// src/cpp/cnn_internals.cpp:10:     int kernel_h     = weight.height;
	ldp	w5, w2, [x1, 28]	// in_channels, kernel_h,
// src/cpp/cnn_internals.cpp:11:     int kernel_w     = weight.width;
	ldr	w10, [x14, 36]	//, weight_58(D)->width
// src/cpp/cnn_internals.cpp:16:             for (int oh = 0; oh < output.height; ++oh) {
	ldr	w24, [x3, 32]	//, output_65(D)->height
	mul	w1, w5, w2	// _46, in_channels, kernel_h
	mul	w21, w10, w2	// _27, kernel_w, kernel_h
	mul	w13, w1, w10	// _237, _46, kernel_w
.L4:
	cmp	w24, 0	// _106,
	bgt	.L56		//,
// src/cpp/cnn_internals.cpp:14:     for (int b = 0; b < input.batches; ++b) {
	add	w20, w20, 1	// b, b,
// src/cpp/cnn_internals.cpp:14:     for (int b = 0; b < input.batches; ++b) {
	cmp	w20, w12	// b, _98
	bne	.L4		//,
// src/cpp/cnn_internals.cpp:44: }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x25, x26, [sp, 64]	//,,
	ldp	x29, x30, [sp], 192	//,,,
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
.L56:
	.cfi_restore_state
	stp	x27, x28, [sp, 80]	//,,
	.cfi_offset 28, -104
	.cfi_offset 27, -112
	sxtw	x1, w0	// _239, out_channels
	mov	w3, w13	// _237, _237
	sub	w0, w2, w25	// ivtmp.342, kernel_h, padding
	mov	w13, w12	// _98, _98
	sub	w28, w10, w25	// ivtmp.335, kernel_w, padding
	neg	w26, w25	// ivtmp.343, padding
	ldr	w23, [x11, 36]	//, output_65(D)->width
	mov	x6, x11	// output, output
	mov	w30, w25	// padding, padding
	str	w0, [sp, 160]	// ivtmp.342, %sfp
// src/cpp/cnn_internals.cpp:37:                                   oc * (output.height * output.width) +
	mul	w9, w24, w23	// _54, _106, _109
	sxtw	x27, w23	// _162, _109
	mov	w12, w23	// _109, _109
	mov	w23, w22	// stride, stride
	mov	w7, w9	// _54, _54
.L26:
	mov	w25, 0	// ivtmp.349,
	mov	x0, 0	// ivtmp.346,
.L28:
	cmp	w12, 0	// _109,
	bgt	.L57		//,
// src/cpp/cnn_internals.cpp:15:         for (int oc = 0; oc < out_channels; ++oc) {
	add	x0, x0, 1	// ivtmp.346, ivtmp.346,
	add	w25, w25, w3	// ivtmp.349, ivtmp.349, _237
	cmp	x0, x1	// ivtmp.346, _239
	bne	.L28		//,
// src/cpp/cnn_internals.cpp:14:     for (int b = 0; b < input.batches; ++b) {
	add	w20, w20, 1	// b, b,
// src/cpp/cnn_internals.cpp:14:     for (int b = 0; b < input.batches; ++b) {
	cmp	w20, w13	// b, _98
	bne	.L26		//,
.L61:
	ldp	x27, x28, [sp, 80]	//,,
	.cfi_remember_state
	.cfi_restore 28
	.cfi_restore 27
// src/cpp/cnn_internals.cpp:44: }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x25, x26, [sp, 64]	//,,
	ldp	x29, x30, [sp], 192	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
.L57:
	.cfi_restore_state
	ldr	w11, [x6, 28]	//, output_65(D)->channels
// src/cpp/cnn_internals.cpp:37:                                   oc * (output.height * output.width) +
	mul	w4, w7, w0	// _215, _54, ivtmp.346
	lsl	x9, x0, 2	// _164, ivtmp.346,
// src/cpp/cnn_internals.cpp:36:                     int out_idx = b * (output.channels * output.height * output.width) +
	str	w13, [sp, 164]	// _98, %sfp
	mov	x13, x6	// output, output
	ldr	x17, [x6]	// _93, MEM[(struct vector *)output_65(D)].D.54862._M_impl.D.54163._M_start
	mov	x6, x1	// _239, _239
	mov	w1, w12	// _109, _109
	ldr	x16, [x15]	// MEM[(const struct vector *)bias_68(D)].D.54862._M_impl.D.54163._M_start, MEM[(const struct vector *)bias_68(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:36:                     int out_idx = b * (output.channels * output.height * output.width) +
	mul	w11, w24, w11	// _30, _106, output_65(D)->channels
	str	x17, [sp, 152]	// _93, %sfp
// src/cpp/cnn_internals.cpp:36:                     int out_idx = b * (output.channels * output.height * output.width) +
	mov	x17, x15	// bias, bias
// src/cpp/cnn_internals.cpp:36:                     int out_idx = b * (output.channels * output.height * output.width) +
	mul	w11, w11, w12	// _31, _30, _109
// src/cpp/cnn_internals.cpp:36:                     int out_idx = b * (output.channels * output.height * output.width) +
	mul	w22, w11, w20	// _32, _31, b
.L23:
	ldr	w15, [sp, 160]	//, %sfp
// src/cpp/cnn_internals.cpp:36:                     int out_idx = b * (output.channels * output.height * output.width) +
	add	w4, w22, w4	// _35, _32, _215
	mov	w11, w3	// _237, _237
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	add	x19, x16, x9	// _83, MEM[(const struct vector *)bias_68(D)].D.54862._M_impl.D.54163._M_start, _164
	mov	w3, w25	// ivtmp.349, ivtmp.349
	str	x0, [sp, 168]	// ivtmp.346, %sfp
// src/cpp/cnn_internals.cpp:16:             for (int oh = 0; oh < output.height; ++oh) {
	mov	w9, 0	// oh,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	mov	w12, w26	// ivtmp.343, ivtmp.343
// src/cpp/cnn_internals.cpp:36:                     int out_idx = b * (output.channels * output.height * output.width) +
	str	w4, [sp, 148]	// _35, %sfp
	mov	x18, x17	// bias, bias
	mov	w25, w11	// _237, _237
	str	w7, [sp, 176]	// _54, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	mov	w4, 0	// ivtmp.341,
	mov	w7, w22	// _32, _32
	mov	x0, x6	// _239, _239
.L25:
	ldr	x22, [sp, 152]	// _93, %sfp
	mov	w11, w28	// ivtmp.335, ivtmp.335
	stp	w4, w9, [sp, 180]	// ivtmp.341, oh,
	mov	x9, x13	// output, output
	mov	x13, x0	// _239, _239
// src/cpp/cnn_internals.cpp:37:                                   oc * (output.height * output.width) +
	ldr	w6, [sp, 148]	//, %sfp
	str	w24, [sp, 188]	// _106, %sfp
	mov	w24, w25	// _237, _237
	add	w6, w4, w6	// _37, ivtmp.341, _35
	add	x17, x27, w6, sxtw	// _148, _162, _37
	add	x17, x22, x17, lsl 2	// _23, _93, _148,
	add	x6, x22, w6, sxtw 2	// ivtmp.332, _93, _37,
	mov	w4, w7	// _32, _32
	mov	w22, w26	// ivtmp.336, ivtmp.343
	str	x17, [sp, 136]	// _23, %sfp
	mov	w17, w30	// ivtmp.334, padding
.L20:
// src/cpp/cnn_internals.cpp:18:                     float pixel_value = bias.data[oc];
	ldr	s31, [x19]	//, *_83
// src/cpp/cnn_internals.cpp:19:                     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w0, w3	// ivtmp.327, ivtmp.349
// src/cpp/cnn_internals.cpp:19:                     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w25, 0	// ic,
// src/cpp/cnn_internals.cpp:19:                     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w5, 0	// in_channels,
	ble	.L6		//,
	stp	x19, x6, [sp, 120]	// _83, ivtmp.332,
	mov	w6, w3	// ivtmp.349, ivtmp.349
	mov	x3, x9	// output, output
	stp	w12, w21, [sp, 108]	// ivtmp.343, _27,
	str	w5, [sp, 116]	// in_channels, %sfp
	str	w2, [sp, 144]	// kernel_h, %sfp
.L5:
// src/cpp/cnn_internals.cpp:20:                         for (int kh = 0; kh < kernel_h; ++kh) {
	ldr	w2, [sp, 144]	//, %sfp
	cmp	w2, 0	// kernel_h,
	ble	.L14		//,
	mov	x9, x18	// bias, bias
	mov	x2, x3	// output, output
.L18:
// src/cpp/cnn_internals.cpp:24:                                 if (ih >= 0 && ih < input.height && iw >= 0 && iw < input.width) {
	ldr	w7, [sp, 108]	//, %sfp
	add	w18, w0, w17	// ivtmp.322, ivtmp.327, ivtmp.334
.L19:
// src/cpp/cnn_internals.cpp:21:                             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	w10, 0	// kernel_w,
	bgt	.L58		//,
// src/cpp/cnn_internals.cpp:20:                         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w7, w7, 1	// ivtmp.321, ivtmp.321,
	add	w18, w18, w10	// ivtmp.322, ivtmp.322, kernel_w
	cmp	w7, w15	// ivtmp.321, ivtmp.342
	bne	.L19		//,
// src/cpp/cnn_internals.cpp:19:                     for (int ic = 0; ic < in_channels; ++ic) {
	ldr	w3, [sp, 112]	//, %sfp
// src/cpp/cnn_internals.cpp:19:                     for (int ic = 0; ic < in_channels; ++ic) {
	add	w25, w25, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:19:                     for (int ic = 0; ic < in_channels; ++ic) {
	add	w0, w0, w3	// ivtmp.327, ivtmp.327, _27
	ldr	w3, [sp, 116]	//, %sfp
	cmp	w3, w25	// in_channels, ic
	bne	.L18		//,
.L60:
	mov	x5, x3	//,
	mov	x18, x9	// bias, bias
	ldp	w12, w21, [sp, 108]	//,,
	mov	w3, w6	// ivtmp.349, ivtmp.349
	mov	x9, x2	// output, output
	ldp	x19, x6, [sp, 120]	// _83, ivtmp.332,
	ldr	w2, [sp, 144]	//, %sfp
.L6:
// src/cpp/cnn_internals.cpp:17:                 for (int ow = 0; ow < output.width; ++ow) {
	sub	w17, w17, w23	// ivtmp.334, ivtmp.334, stride
	add	w11, w11, w23	// ivtmp.335, ivtmp.335, stride
// src/cpp/cnn_internals.cpp:39:                     output.data[out_idx] = pixel_value;
	str	s31, [x6], 4	//, MEM[(value_type &)_181]
// src/cpp/cnn_internals.cpp:17:                 for (int ow = 0; ow < output.width; ++ow) {
	ldr	x0, [sp, 136]	// _23, %sfp
	add	w22, w22, w23	// ivtmp.336, ivtmp.336, stride
	cmp	x0, x6	// _23, ivtmp.332
	bne	.L20		//,
// src/cpp/cnn_internals.cpp:16:             for (int oh = 0; oh < output.height; ++oh) {
	mov	w7, w4	// _32, _32
	mov	x0, x13	// _239, _239
	mov	x13, x9	// output, output
	ldp	w4, w9, [sp, 180]	//,,
	mov	w25, w24	// _237, _237
// src/cpp/cnn_internals.cpp:16:             for (int oh = 0; oh < output.height; ++oh) {
	add	w15, w15, w23	// ivtmp.342, ivtmp.342, stride
	add	w12, w12, w23	// ivtmp.343, ivtmp.343, stride
// src/cpp/cnn_internals.cpp:16:             for (int oh = 0; oh < output.height; ++oh) {
	ldr	w24, [sp, 188]	//, %sfp
	add	w9, w9, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:16:             for (int oh = 0; oh < output.height; ++oh) {
	add	w4, w4, w1	// ivtmp.341, ivtmp.341, _109
	cmp	w9, w24	// oh, _106
	bne	.L25		//,
// src/cpp/cnn_internals.cpp:15:         for (int oc = 0; oc < out_channels; ++oc) {
	mov	x6, x0	// _239, _239
	ldr	x0, [sp, 168]	// ivtmp.346, %sfp
	mov	w4, w25	// _237, _237
	mov	w25, w3	// ivtmp.349, ivtmp.349
	mov	w22, w7	// _32, _32
	mov	x17, x18	// bias, bias
	ldr	w7, [sp, 176]	//, %sfp
	mov	w3, w4	// _237, _237
	add	w25, w25, w4	// ivtmp.349, ivtmp.349, _237
	add	x0, x0, 1	// ivtmp.346, ivtmp.346,
	cmp	x0, x6	// ivtmp.346, _239
	beq	.L45		//,
// src/cpp/cnn_internals.cpp:37:                                   oc * (output.height * output.width) +
	mul	w4, w7, w0	// _215, _54, ivtmp.346
	lsl	x9, x0, 2	// _164, ivtmp.346,
	b	.L23		//
	.p2align 2,,3
.L58:
	mov	w19, w17	// ivtmp.334, ivtmp.334
	mov	w21, w6	// ivtmp.349, ivtmp.349
	.p2align 5,,15
.L11:
	tbnz	w7, #31, .L7	// ivtmp.321,
	ldr	w17, [x8, 32]	//, MEM[(int *)input_63(D) + 32B]
// src/cpp/cnn_internals.cpp:24:                                 if (ih >= 0 && ih < input.height && iw >= 0 && iw < input.width) {
	mov	w3, w22	// ivtmp.316, ivtmp.336
	stp	w10, w0, [sp, 96]	// kernel_w, ivtmp.327,
	cmp	w7, w17	// ivtmp.321, input__height_lsm0.308
	cset	w5, lt	// _141,
	b	.L9		//
	.p2align 2,,3
.L8:
// src/cpp/cnn_internals.cpp:21:                             for (int kw = 0; kw < kernel_w; ++kw) {
	add	w3, w3, 1	// ivtmp.316, ivtmp.316,
	cmp	w11, w3	// ivtmp.335, ivtmp.316
	beq	.L59		//,
.L9:
// src/cpp/cnn_internals.cpp:24:                                 if (ih >= 0 && ih < input.height && iw >= 0 && iw < input.width) {
	cmp	w3, 0	// ivtmp.316,
	ccmp	w5, 0, 4, ge	// _141,,,
	beq	.L8		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	add	w6, w18, w3	// _167, ivtmp.322, ivtmp.316
	mov	x0, x6	// _167, _167
// src/cpp/cnn_internals.cpp:24:                                 if (ih >= 0 && ih < input.height && iw >= 0 && iw < input.width) {
	ldr	w6, [x8, 36]	//, input_63(D)->width
// src/cpp/cnn_internals.cpp:26:                                                  ic * (input.height * input.width) +
	mul	w12, w6, w17	// _111, _139, input__height_lsm0.308
// src/cpp/cnn_internals.cpp:24:                                 if (ih >= 0 && ih < input.height && iw >= 0 && iw < input.width) {
	cmp	w6, w3	// _139, ivtmp.316
	ble	.L8		//,
// src/cpp/cnn_internals.cpp:25:                                     int in_idx = b * (input.channels * input.height * input.width) +
	ldr	w10, [x8, 28]	//, input_63(D)->channels
// src/cpp/cnn_internals.cpp:26:                                                  ic * (input.height * input.width) +
	mul	w12, w12, w25	// _110, _111, ic
// src/cpp/cnn_internals.cpp:25:                                     int in_idx = b * (input.channels * input.height * input.width) +
	mul	w10, w17, w10	// _114, input__height_lsm0.308, input_63(D)->channels
// src/cpp/cnn_internals.cpp:25:                                     int in_idx = b * (input.channels * input.height * input.width) +
	mul	w10, w10, w6	// _113, _114, _139
	str	w10, [sp, 104]	// _113, %sfp
// src/cpp/cnn_internals.cpp:31:                                     pixel_value += input.data[in_idx] * weight.data[w_idx];
	ldr	x10, [x14]	// tmp530, MEM[(const struct vector *)weight_58(D)].D.54862._M_impl.D.54163._M_start
	ldr	s29, [x10, x0, lsl 2]	//, *_170
// src/cpp/cnn_internals.cpp:25:                                     int in_idx = b * (input.channels * input.height * input.width) +
	ldr	w0, [sp, 104]	//, %sfp
	madd	w10, w0, w20, w12	// _107, _113, b, _110
// src/cpp/cnn_internals.cpp:31:                                     pixel_value += input.data[in_idx] * weight.data[w_idx];
	ldr	x0, [x8]	// tmp532, MEM[(const struct vector *)input_63(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:26:                                                  ic * (input.height * input.width) +
	madd	w6, w7, w6, w10	// _99, ivtmp.321, _139, _107
// src/cpp/cnn_internals.cpp:25:                                     int in_idx = b * (input.channels * input.height * input.width) +
	add	w6, w6, w3	// in_idx_97, _99, ivtmp.316
// src/cpp/cnn_internals.cpp:21:                             for (int kw = 0; kw < kernel_w; ++kw) {
	add	w3, w3, 1	// ivtmp.316, ivtmp.316,
// src/cpp/cnn_internals.cpp:31:                                     pixel_value += input.data[in_idx] * weight.data[w_idx];
	ldr	s30, [x0, w6, sxtw 2]	//, *_108
	fmadd	s31, s30, s29, s31	//,,,
// src/cpp/cnn_internals.cpp:21:                             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	w11, w3	// ivtmp.335, ivtmp.316
	bne	.L9		//,
	.p2align 5,,15
.L59:
	ldp	w10, w0, [sp, 96]	//,,
// src/cpp/cnn_internals.cpp:20:                         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w7, w7, 1	// ivtmp.321, ivtmp.321,
	add	w18, w18, w10	// ivtmp.322, ivtmp.322, kernel_w
	cmp	w15, w7	// ivtmp.342, ivtmp.321
	bne	.L11		//,
	mov	w17, w19	// ivtmp.334, ivtmp.334
	mov	w6, w21	// ivtmp.349, ivtmp.349
	mov	x18, x9	// bias, bias
	mov	x3, x2	// output, output
.L14:
// src/cpp/cnn_internals.cpp:19:                     for (int ic = 0; ic < in_channels; ++ic) {
	ldr	w2, [sp, 112]	//, %sfp
// src/cpp/cnn_internals.cpp:19:                     for (int ic = 0; ic < in_channels; ++ic) {
	add	w25, w25, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:19:                     for (int ic = 0; ic < in_channels; ++ic) {
	add	w0, w0, w2	// ivtmp.327, ivtmp.327, _27
	ldr	w2, [sp, 116]	//, %sfp
	cmp	w2, w25	// in_channels, ic
	bne	.L5		//,
	mov	x9, x3	// output, output
	ldp	w12, w21, [sp, 108]	//,,
	mov	w3, w6	// ivtmp.349, ivtmp.349
	ldp	x19, x6, [sp, 120]	// _83, ivtmp.332,
	ldr	w5, [sp, 116]	//, %sfp
	ldr	w2, [sp, 144]	//, %sfp
	b	.L6		//
	.p2align 2,,3
.L7:
// src/cpp/cnn_internals.cpp:20:                         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w7, w7, 1	// ivtmp.321, ivtmp.321,
	add	w18, w18, w10	// ivtmp.322, ivtmp.322, kernel_w
	cmp	w15, w7	// ivtmp.342, ivtmp.321
	bne	.L11		//,
// src/cpp/cnn_internals.cpp:19:                     for (int ic = 0; ic < in_channels; ++ic) {
	ldr	w3, [sp, 112]	//, %sfp
// src/cpp/cnn_internals.cpp:19:                     for (int ic = 0; ic < in_channels; ++ic) {
	add	w25, w25, 1	// ic, ic,
	mov	w17, w19	// ivtmp.334, ivtmp.334
	mov	w6, w21	// ivtmp.349, ivtmp.349
// src/cpp/cnn_internals.cpp:19:                     for (int ic = 0; ic < in_channels; ++ic) {
	add	w0, w0, w3	// ivtmp.327, ivtmp.327, _27
	ldr	w3, [sp, 116]	//, %sfp
	cmp	w3, w25	// in_channels, ic
	bne	.L18		//,
	b	.L60		//
.L45:
	mov	w12, w1	// _109, _109
	mov	x1, x6	// _239, _239
	mov	x6, x13	// output, output
	ldr	w13, [sp, 164]	//, %sfp
// src/cpp/cnn_internals.cpp:14:     for (int b = 0; b < input.batches; ++b) {
	add	w20, w20, 1	// b, b,
	mov	x15, x18	// bias, bias
// src/cpp/cnn_internals.cpp:14:     for (int b = 0; b < input.batches; ++b) {
	cmp	w20, w13	// b, _98
	bne	.L26		//,
	b	.L61		//
.L50:
	.cfi_def_cfa_offset 0
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 27
	.cfi_restore 28
	.cfi_restore 29
	.cfi_restore 30
	ret	
	.cfi_endproc
.LFE5444:
	.size	_Z14conv2d_forwardRK6TensorS1_S1_RS_ii, .-_Z14conv2d_forwardRK6TensorS1_S1_RS_ii
	.align	2
	.p2align 5,,15
	.global	_Z27conv2d_forward_restructuredRK6TensorS1_S1_RS_ii
	.type	_Z27conv2d_forward_restructuredRK6TensorS1_S1_RS_ii, %function
_Z27conv2d_forward_restructuredRK6TensorS1_S1_RS_ii:
.LFB5445:
	.cfi_startproc
// src/cpp/cnn_internals.cpp:56:     int input_batches = input.batches;
	ldr	w6, [x0, 24]	//, input_60(D)->batches
// src/cpp/cnn_internals.cpp:48:     int out_channels = weight.batches;
	ldr	w7, [x1, 24]	//, weight_56(D)->batches
// src/cpp/cnn_internals.cpp:58:     for (int b = 0; b < input_batches; ++b) {
	cmp	w6, 0	// input_batches,
	ble	.L92		//,
	cmp	w7, 0	// out_channels,
	ble	.L92		//,
// src/cpp/cnn_internals.cpp:47:                     int stride, int padding) {
	stp	x29, x30, [sp, -336]!	//,,,
	.cfi_def_cfa_offset 336
	.cfi_offset 29, -336
	.cfi_offset 30, -328
	mov	x29, sp	//,
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 25, -272
	.cfi_offset 26, -264
	mov	x26, x0	// input, tmp409
	mov	x0, x3	// output, tmp412
// src/cpp/cnn_internals.cpp:54:     int output_h     = output.height;
	ldr	w15, [x3, 32]	//, output_63(D)->height
	cmp	w15, 0	// output_h,
	ble	.L62		//,
// src/cpp/cnn_internals.cpp:55:     int output_w     = output.width;
	ldr	w16, [x3, 36]	//, output_63(D)->width
	cmp	w16, 0	// output_w,
	ble	.L62		//,
	sxtw	x3, w4	// _205, stride
	mov	x11, x1	// weight, tmp410
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 20, -312
	.cfi_offset 19, -320
// src/cpp/cnn_internals.cpp:91:                                   oc * (output_h * output_w) +
	mul	w18, w15, w16	// _137, output_h, output_w
	ptrue	p6.b, all	// tmp382
	ubfiz	x10, x7, 2, 32	// _307, out_channels,,
	mov	x1, x2	// bias, tmp411
	mov	w30, w4	// stride, tmp413
	mov	w25, w5	// padding, tmp414
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 22, -296
	.cfi_offset 21, -304
	neg	w17, w5	// ivtmp.444, padding
	sxtw	x22, w5	// _207, padding
// src/cpp/cnn_internals.cpp:58:     for (int b = 0; b < input_batches; ++b) {
	mov	w2, 0	// b,
	mov	x13, x22	// _207, _207
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -280
	.cfi_offset 23, -288
	sxtw	x23, w16	// _212, output_w
	stp	x27, x28, [sp, 80]	//,,
	.cfi_offset 28, -248
	.cfi_offset 27, -256
// src/cpp/cnn_internals.cpp:77:                             for (int kw = kw_lo; kw < kw_hi; ++kw) {
	cntw	x28		// tmp380
	mov	w27, 64	// tmp383,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	str	wzr, [sp, 164]	//, %sfp
	str	x3, [sp, 224]	// _205, %sfp
// src/cpp/cnn_internals.cpp:50:     int kernel_h     = weight.height;
	ldp	w7, w8, [x11, 28]	// in_channels, _1,
// src/cpp/cnn_internals.cpp:90:                     int out_idx = b * (output.channels * output_h * output_w) +
	ldr	w3, [x0, 28]	//, output_63(D)->channels
// src/cpp/cnn_internals.cpp:51:     int kernel_w     = weight.width;
	ldr	w12, [x11, 36]	//, weight_56(D)->width
// src/cpp/cnn_internals.cpp:50:     int kernel_h     = weight.height;
	str	w7, [sp, 136]	// in_channels, %sfp
// src/cpp/cnn_internals.cpp:83:                                 int w_idx  = oc * (in_channels * kernel_h * kernel_w) +
	mul	w5, w8, w7	// _45, _1, in_channels
	ldp	w19, w9, [x26, 32]	// _105, _41,
// src/cpp/cnn_internals.cpp:90:                     int out_idx = b * (output.channels * output_h * output_w) +
	mul	w3, w15, w3	// _32, output_h, output_63(D)->channels
	str	w30, [sp, 176]	// stride, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	ldr	x20, [x0]	// _100, MEM[(struct vector *)output_63(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:84:                                                 ic * (kernel_h * kernel_w) +
	mul	w0, w8, w12	// _110, _1, _2
	mov	w30, w12	// _2, _2
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	ldr	x14, [x1]	// _91, MEM[(const struct vector *)bias_74(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:90:                     int out_idx = b * (output.channels * output_h * output_w) +
	mul	w4, w3, w16	// _33, _32, output_w
	mul	w1, w9, w19	// _117, _41, _105
// src/cpp/cnn_internals.cpp:83:                                 int w_idx  = oc * (in_channels * kernel_h * kernel_w) +
	mul	w21, w5, w12	// _108, _45, _2
// src/cpp/cnn_internals.cpp:84:                                                 ic * (kernel_h * kernel_w) +
	mov	w5, w8	// _1, _1
	mov	w3, w4	// _33, _33
	mov	x4, x10	// _307, _307
// src/cpp/cnn_internals.cpp:80:                                 int in_idx = b * (in_channels * input_h * input_w) +
	mul	w7, w7, w1	// _53, in_channels, _117
	mov	w10, w18	// _137, _137
// src/cpp/cnn_internals.cpp:84:                                                 ic * (kernel_h * kernel_w) +
	stp	w0, w1, [sp, 208]	// _110, _117,
	mov	x1, x26	// input, input
	mov	w26, w25	// padding, padding
	mov	w25, w9	// _41, _41
	mov	w9, w17	// ivtmp.444, ivtmp.444
.L65:
// src/cpp/cnn_internals.cpp:90:                     int out_idx = b * (output.channels * output_h * output_w) +
	mov	w18, 0	// ivtmp.451,
	ldr	w12, [sp, 208]	//, %sfp
	mul	w8, w3, w2	// _34, _33, b
	mov	x17, x1	// input, input
	str	w6, [sp, 248]	// input_batches, %sfp
	mov	w6, w18	// ivtmp.451, ivtmp.451
	mov	x18, x11	// weight, weight
	ldr	w11, [sp, 212]	//, %sfp
	mov	w22, 0	// ivtmp.450,
	mov	x0, 0	// ivtmp.449,
	mov	w1, w10	// _137, _137
	stp	w3, w7, [sp, 232]	// _33, _53,
.L79:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	add	x3, x14, x0	// _93, _91, ivtmp.449
// src/cpp/cnn_internals.cpp:90:                     int out_idx = b * (output.channels * output_h * output_w) +
	mov	w7, w26	// ivtmp.441, padding
	str	w16, [sp, 180]	// output_w, %sfp
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	mov	w24, 0	// oh,
	mov	x16, x18	// weight, weight
	str	x23, [sp, 184]	// _212, %sfp
	mov	x23, x17	// input, input
	str	w26, [sp, 192]	// padding, %sfp
	mov	w26, w25	// _41, _41
	mov	w25, w30	// _2, _2
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	str	x3, [sp, 200]	// _93, %sfp
// src/cpp/cnn_internals.cpp:90:                     int out_idx = b * (output.channels * output_h * output_w) +
	add	w3, w22, w8	// _37, ivtmp.450, _34
	str	w8, [sp, 252]	// _34, %sfp
	mov	w8, w6	// ivtmp.451, ivtmp.451
	str	w2, [sp, 264]	// b, %sfp
	mov	w2, w5	// _1, _1
	stp	wzr, w9, [sp, 156]	//, ivtmp.444,
	str	w3, [sp, 196]	// _37, %sfp
	str	x0, [sp, 240]	// ivtmp.449, %sfp
	str	x14, [sp, 256]	// _91, %sfp
	str	w1, [sp, 268]	// _137, %sfp
	str	x4, [sp, 272]	// _307, %sfp
	str	w9, [sp, 280]	// ivtmp.444, %sfp
.L77:
// src/cpp/cnn_internals.cpp:91:                                   oc * (output_h * output_w) +
	ldp	w6, w3, [sp, 192]	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:262:       if (__a < __b)
	bic	w0, w7, w7, asr #31	// _5, ivtmp.441
	add	x9, sp, 312	//,,
	str	w2, [sp, 284]	// _1, %sfp
// src/cpp/cnn_internals.cpp:91:                                   oc * (output_h * output_w) +
	ldr	w1, [sp, 156]	//, %sfp
	str	w15, [sp, 288]	// output_h, %sfp
	str	w24, [sp, 292]	// oh, %sfp
	str	x20, [sp, 296]	// _100, %sfp
	add	w4, w1, w3	// _39, ivtmp.443, _37
	ldr	x1, [sp, 184]	// _212, %sfp
	add	x5, x1, w4, sxtw	// _213, _212, _39
	neg	x3, x13	// ivtmp.435, _207
	add	x1, x20, w4, sxtw 2	// ivtmp.433, _100, _39,
// src/cpp/cnn_internals.cpp:65:                 int kh_hi = std::min(kernel_h, input_h - c_h);
	add	w4, w19, w7	// _130, _105, ivtmp.441
	str	x1, [sp, 168]	// ivtmp.433, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:238:       if (__b < __a)
	cmp	w4, w2	// _130, _1
	csel	w30, w4, w2, le	// _103, _130, _1,
	mul	w4, w25, w0	// ivtmp.422, _2, _5
	mov	w2, w6	// ivtmp.431, ivtmp.431
	add	x1, x20, x5, lsl 2	// _216, _100, _213,
	str	w4, [sp, 152]	// ivtmp.422, %sfp
	stp	w19, w21, [x9, -8]	// _105, _108,
	stp	w7, w22, [x9]	// ivtmp.441, ivtmp.450,
	str	x1, [sp, 216]	// _216, %sfp
	ldr	w1, [sp, 160]	//, %sfp
	add	w5, w0, w1	// _187, _5, ivtmp.444
	mul	w5, w5, w26	// ivtmp.421, _187, _41
	str	w5, [sp, 148]	// ivtmp.421, %sfp
.L76:
// src/cpp/cnn_internals.cpp:68:                     float pixel_value = bias.data[oc];
	ldr	x1, [sp, 200]	// _93, %sfp
// src/cpp/cnn_internals.cpp:73:                     int kw_hi = std::min(kernel_w, input_w - c_w);
	add	w15, w26, w2	// _209, _41, ivtmp.431
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:262:       if (__a < __b)
	bic	w17, w2, w2, asr #31	// _77, ivtmp.431
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:238:       if (__b < __a)
	cmp	w15, w25	// _209, _2
	csel	w15, w15, w25, le	// _78, _209, _2,
// src/cpp/cnn_internals.cpp:68:                     float pixel_value = bias.data[oc];
	ldr	s31, [x1]	//, *_93
// src/cpp/cnn_internals.cpp:75:                     for (int ic = 0; ic < in_channels; ++ic) {
	ldr	w1, [sp, 136]	//, %sfp
	cmp	w1, 0	// in_channels,
	ble	.L66		//,
	sxtw	x4, w17	// _269, _77
	sub	w21, w15, w17	// _180, _78, _77
	str	w8, [sp, 124]	// ivtmp.451, %sfp
	add	x18, x3, x4	// _48, ivtmp.435, _269
	and	w5, w21, -16	// niters_vector_mult_vf.365, _180,
	stp	w12, w11, [sp, 140]	// _110, _117,
	str	x4, [sp, 96]	// _269, %sfp
// src/cpp/cnn_internals.cpp:75:                     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w1, 0	// ic,
	sub	w20, w21, #1	// _179, _180,
// src/cpp/cnn_internals.cpp:75:                     for (int ic = 0; ic < in_channels; ++ic) {
	ldr	w4, [sp, 164]	//, %sfp
	lsr	w24, w21, 4	// bnd.364_176, _180,
	and	w22, w21, 15	// _54, _180,
	str	w5, [sp, 116]	// niters_vector_mult_vf.365, %sfp
	mov	w5, w1	// ic, ic
	str	w2, [sp, 320]	// ivtmp.431, %sfp
	str	w8, [sp, 324]	// ivtmp.451, %sfp
	str	x13, [sp, 328]	// _207, %sfp
	str	w4, [sp, 120]	// ivtmp.455, %sfp
	.p2align 5,,15
.L67:
// src/cpp/cnn_internals.cpp:76:                         for (int kh = kh_lo; kh < kh_hi; ++kh) {
	cmp	w0, w30	// _5, _103
	bge	.L72		//,
	ldr	x2, [sp, 96]	// _269, %sfp
// src/cpp/cnn_internals.cpp:76:                         for (int kh = kh_lo; kh < kh_hi; ++kh) {
	mov	w7, w0	// kh, _5
	stp	w0, w5, [sp, 128]	// _5, ic,
	ldpsw	x14, x1, [sp, 120]	// _271, _283,
// src/cpp/cnn_internals.cpp:76:                         for (int kh = kh_lo; kh < kh_hi; ++kh) {
	ldp	w8, w9, [sp, 148]	//,,
	add	x19, x1, x2	// _119, _283, _269
	str	x1, [sp, 104]	// _283, %sfp
	.p2align 5,,15
.L75:
// src/cpp/cnn_internals.cpp:77:                             for (int kw = kw_lo; kw < kw_hi; ++kw) {
	cmp	w17, w15	// _77, _78
	bge	.L74		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	ldr	x10, [x16]	// _94, MEM[(const struct vector *)weight_56(D)].D.54862._M_impl.D.54163._M_start
	ldr	x11, [x23]	// _97, MEM[(const struct vector *)input_60(D)].D.54862._M_impl.D.54163._M_start
	cmp	w20, 14	// _179,
	bls	.L80		//,
	movi	v30.4s, 0	//
	sxtw	x5, w9	// _284, ivtmp.422
	sxtw	x6, w8	// _272, ivtmp.421
	ldp	x2, x12, [sp, 96]	// _269, _283,
	add	x1, x19, x5	// _141, _119, _284
	add	x0, x6, x14	// _79, _272, _271
	add	x1, x10, x1, lsl 2	// ivtmp.411, _94, _141,
	add	x0, x18, x0	// _157, _48, _79
	mov	x13, x14	// _271, _271
	add	x0, x11, x0, lsl 2	// ivtmp.415, _97, _157,
	mov	v0.16b, v30.16b	//,
	mov	v28.16b, v30.16b	//,
	mov	v1.16b, v30.16b	//,
	umaddl	x4, w24, w27, x1	// _87, bnd.364_176, tmp383, ivtmp.411
	.p2align 5,,15
.L69:
// src/cpp/cnn_internals.cpp:86:                                 pixel_value += input.data[in_idx] * weight.data[w_idx];
	ldp	q23, q22, [x1, 16]	//,,
// src/cpp/cnn_internals.cpp:86:                                 pixel_value += input.data[in_idx] * weight.data[w_idx];
	ldp	q25, q24, [x0, 32]	//,,
	ldp	q2, q26, [x0], 64	//,, ivtmp.415,
// src/cpp/cnn_internals.cpp:86:                                 pixel_value += input.data[in_idx] * weight.data[w_idx];
	ldr	q21, [x1, 48]	//, MEM <const vector(4) float> [(const value_type &)_96 + 48]
	ldr	q20, [x1], 64	//, MEM <const vector(4) float> [(const value_type &)_96]
// src/cpp/cnn_internals.cpp:86:                                 pixel_value += input.data[in_idx] * weight.data[w_idx];
	fmla	v0.4s, v22.4s, v25.4s	//,,
	fmla	v28.4s, v23.4s, v26.4s	//,,
	fmla	v30.4s, v24.4s, v21.4s	//,,
	fmla	v1.4s, v20.4s, v2.4s	//,,
	cmp	x4, x1	// _87, ivtmp.411
	bne	.L69		//,
	ldr	w0, [sp, 116]	//, %sfp
	fadd	v1.4s, v1.4s, v28.4s	//,,
	fadd	v0.4s, v0.4s, v30.4s	//,,
	fadd	v0.4s, v1.4s, v0.4s	//,,
	faddp	v0.4s, v0.4s, v0.4s	//,,
	faddp	v0.4s, v0.4s, v0.4s	//,,
	fadd	s31, s31, s0	//,,
	cbz	w22, .L74	// _54,
.L68:
	add	x1, x3, x2	// _20, ivtmp.435, _269
	add	x2, x12, x2	// _244, _283, _269
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	mov	z19.s, #0	//
	uxtw	x12, w0	// _275, niters_vector_mult_vf.365
	add	x6, x13, x6	// _19, _271, _272
	add	x1, x1, x12	// _14, _20, _275
	add	x5, x5, x12	// _296, _284, _275
	add	x1, x1, x6	// _276, _14, _19
	add	x2, x2, x5	// _289, _244, _296
	add	x1, x11, x1, lsl 2	// vectp.387, _97, _276,
	mov	x4, 0	// ivtmp_299,
	sub	w0, w21, w0	// bnd.384, _180, niters_vector_mult_vf.365
	whilelo	p7.s, wzr, w0	// loop_mask_280,, bnd.384
	add	x2, x10, x2, lsl 2	// vectp.390, _94, _289,
.L71:
// src/cpp/cnn_internals.cpp:86:                                 pixel_value += input.data[in_idx] * weight.data[w_idx];
	ld1w	z27.s, p7/z, [x1, x4, lsl 2]	//, loop_mask_280,* vectp.387
// src/cpp/cnn_internals.cpp:86:                                 pixel_value += input.data[in_idx] * weight.data[w_idx];
	ld1w	z29.s, p7/z, [x2, x4, lsl 2]	//, loop_mask_280,* vectp.390
// src/cpp/cnn_internals.cpp:77:                             for (int kw = kw_lo; kw < kw_hi; ++kw) {
	add	x4, x4, x28	// ivtmp_299, ivtmp_299, tmp380
// src/cpp/cnn_internals.cpp:86:                                 pixel_value += input.data[in_idx] * weight.data[w_idx];
	fmla	z19.s, p7/m, z27.s, z29.s	//, loop_mask_280,,
	whilelo	p7.s, w4, w0	// loop_mask_280, ivtmp_299, bnd.384
	b.any	.L71	//,
	faddv	s19, p6, z19.s	//, tmp382,
	fadd	s31, s31, s19	//,,
.L74:
// src/cpp/cnn_internals.cpp:76:                         for (int kh = kh_lo; kh < kh_hi; ++kh) {
	add	w7, w7, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:76:                         for (int kh = kh_lo; kh < kh_hi; ++kh) {
	add	w8, w8, w26	// ivtmp.421, ivtmp.421, _41
	add	w9, w9, w25	// ivtmp.422, ivtmp.422, _2
	cmp	w7, w30	// kh, _103
	bne	.L75		//,
	ldp	w0, w5, [sp, 128]	//,,
.L72:
// src/cpp/cnn_internals.cpp:75:                     for (int ic = 0; ic < in_channels; ++ic) {
	add	w5, w5, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:75:                     for (int ic = 0; ic < in_channels; ++ic) {
	ldr	w1, [sp, 120]	//, %sfp
	ldr	w2, [sp, 144]	//, %sfp
	add	w1, w1, w2	// ivtmp.426, ivtmp.426, _117
	ldr	w2, [sp, 140]	//, %sfp
	str	w1, [sp, 120]	// ivtmp.426, %sfp
	ldr	w1, [sp, 124]	//, %sfp
	add	w1, w1, w2	// ivtmp.427, ivtmp.427, _110
	str	w1, [sp, 124]	// ivtmp.427, %sfp
	ldr	w1, [sp, 136]	//, %sfp
	cmp	w1, w5	// in_channels, ic
	bne	.L67		//,
	ldr	x13, [sp, 328]	// _207, %sfp
	ldp	w12, w11, [sp, 140]	//,,
	ldr	w2, [sp, 320]	//, %sfp
	ldr	w8, [sp, 324]	//, %sfp
.L66:
	ldr	w1, [sp, 176]	//, %sfp
	sub	w2, w2, w1	// ivtmp.431, ivtmp.431, stride
// src/cpp/cnn_internals.cpp:67:                 for (int ow = 0; ow < output_w; ++ow) {
	ldp	x4, x1, [sp, 216]	// _216, _205,
	add	x3, x3, x1	// ivtmp.435, ivtmp.435, _205
// src/cpp/cnn_internals.cpp:93:                     output.data[out_idx] = pixel_value;
	ldr	x1, [sp, 168]	// ivtmp.433, %sfp
	str	s31, [x1], 4	//, MEM[(value_type &)_210]
	str	x1, [sp, 168]	// ivtmp.433, %sfp
// src/cpp/cnn_internals.cpp:67:                 for (int ow = 0; ow < output_w; ++ow) {
	cmp	x1, x4	// ivtmp.433, _216
	bne	.L76		//,
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	ldp	w0, w3, [sp, 176]	//,,
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	add	x4, sp, 300	//,,
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	ldr	w1, [sp, 156]	//, %sfp
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	ldp	w15, w24, [x4, -12]	//,,
	ldp	w19, w21, [x4, 4]	//,,
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	add	w1, w1, w3	// ivtmp.443, ivtmp.443, output_w
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	ldr	w7, [sp, 312]	//, %sfp
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	str	w1, [sp, 156]	// ivtmp.443, %sfp
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	add	w24, w24, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	ldr	w1, [sp, 160]	//, %sfp
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	ldr	x20, [sp, 296]	// _100, %sfp
	sub	w7, w7, w0	// ivtmp.441, ivtmp.441, stride
	ldr	w2, [sp, 284]	//, %sfp
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	add	w0, w1, w0	// ivtmp.444, ivtmp.444, stride
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	ldr	w22, [sp, 316]	//, %sfp
// src/cpp/cnn_internals.cpp:61:             for (int oh = 0; oh < output_h; ++oh) {
	str	w0, [sp, 160]	// ivtmp.444, %sfp
	cmp	w15, w24	// output_h, oh
	bne	.L77		//,
// src/cpp/cnn_internals.cpp:59:         for (int oc = 0; oc < out_channels; ++oc) {
	ldr	x0, [sp, 240]	// ivtmp.449, %sfp
	mov	w5, w2	// _1, _1
	mov	w30, w25	// _2, _2
	mov	x17, x23	// input, input
	mov	w25, w26	// _41, _41
	ldr	x4, [sp, 272]	// _307, %sfp
	mov	x18, x16	// weight, weight
	add	w6, w8, w21	// ivtmp.451, ivtmp.451, _108
	mov	x16, x3	//,
	ldr	w1, [sp, 268]	//, %sfp
	add	x0, x0, 4	// ivtmp.449, ivtmp.449,
	ldr	x23, [sp, 184]	// _212, %sfp
	ldr	x14, [sp, 256]	// _91, %sfp
	add	w22, w22, w1	// ivtmp.450, ivtmp.450, _137
	ldr	w26, [sp, 192]	//, %sfp
	ldr	w8, [sp, 252]	//, %sfp
	ldr	w2, [sp, 264]	//, %sfp
	ldr	w9, [sp, 280]	//, %sfp
	cmp	x0, x4	// ivtmp.449, _307
	bne	.L79		//,
// src/cpp/cnn_internals.cpp:58:     for (int b = 0; b < input_batches; ++b) {
	add	w2, w2, 1	// b, b,
	mov	w10, w1	// _137, _137
// src/cpp/cnn_internals.cpp:58:     for (int b = 0; b < input_batches; ++b) {
	ldr	w0, [sp, 164]	//, %sfp
// src/cpp/cnn_internals.cpp:58:     for (int b = 0; b < input_batches; ++b) {
	mov	x1, x17	// input, input
	mov	x11, x18	// weight, weight
	ldp	w3, w7, [sp, 232]	//,,
	ldr	w6, [sp, 248]	//, %sfp
// src/cpp/cnn_internals.cpp:58:     for (int b = 0; b < input_batches; ++b) {
	add	w0, w0, w7	// ivtmp.455, ivtmp.455, _53
	str	w0, [sp, 164]	// ivtmp.455, %sfp
	cmp	w6, w2	// input_batches, b
	bne	.L65		//,
	ldp	x19, x20, [sp, 16]	//,,
	.cfi_restore 20
	.cfi_restore 19
	ldp	x21, x22, [sp, 32]	//,,
	.cfi_restore 22
	.cfi_restore 21
	ldp	x23, x24, [sp, 48]	//,,
	.cfi_restore 24
	.cfi_restore 23
	ldp	x27, x28, [sp, 80]	//,,
	.cfi_restore 28
	.cfi_restore 27
.L62:
// src/cpp/cnn_internals.cpp:98: }
	ldp	x25, x26, [sp, 64]	//,,
	ldp	x29, x30, [sp], 336	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 25
	.cfi_restore 26
	.cfi_def_cfa_offset 0
	ret	
.L80:
	.cfi_def_cfa_offset 336
	.cfi_offset 19, -320
	.cfi_offset 20, -312
	.cfi_offset 21, -304
	.cfi_offset 22, -296
	.cfi_offset 23, -288
	.cfi_offset 24, -280
	.cfi_offset 25, -272
	.cfi_offset 26, -264
	.cfi_offset 27, -256
	.cfi_offset 28, -248
	.cfi_offset 29, -336
	.cfi_offset 30, -328
	ldr	x2, [sp, 96]	// _269, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	mov	w0, 0	// niters_vector_mult_vf.365,
	sxtw	x5, w9	// _284, ivtmp.422
	sxtw	x6, w8	// _272, ivtmp.421
	ldpsw	x13, x12, [sp, 120]	// _271, _283,
	b	.L68		//
.L92:
	.cfi_def_cfa_offset 0
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 27
	.cfi_restore 28
	.cfi_restore 29
	.cfi_restore 30
	ret	
	.cfi_endproc
.LFE5445:
	.size	_Z27conv2d_forward_restructuredRK6TensorS1_S1_RS_ii, .-_Z27conv2d_forward_restructuredRK6TensorS1_S1_RS_ii
	.align	2
	.p2align 5,,15
	.global	_Z29conv2d_forward_hoist_restrictRK6TensorS1_S1_RS_ii
	.type	_Z29conv2d_forward_hoist_restrictRK6TensorS1_S1_RS_ii, %function
_Z29conv2d_forward_hoist_restrictRK6TensorS1_S1_RS_ii:
.LFB5446:
	.cfi_startproc
// src/cpp/cnn_internals.cpp:115:     int input_batches = input.batches;
	ldr	w7, [x0, 24]	//, input_49(D)->batches
// src/cpp/cnn_internals.cpp:107:     int out_channels = weight.batches;
	ldr	w6, [x1, 24]	//, weight_52(D)->batches
// src/cpp/cnn_internals.cpp:124:     for (int b = 0; b < input_batches; ++b) {
	cmp	w7, 0	// input_batches,
	ble	.L126		//,
	cmp	w6, 0	// out_channels,
	ble	.L126		//,
// src/cpp/cnn_internals.cpp:113:     int output_h     = output.height;
	ldr	w11, [x3, 32]	//, output_54(D)->height
	mov	x18, x3	// output, tmp399
	cmp	w11, 0	// output_h,
	ble	.L126		//,
// src/cpp/cnn_internals.cpp:101:                     int stride, int padding) {
	stp	x29, x30, [sp, -336]!	//,,,
	.cfi_def_cfa_offset 336
	.cfi_offset 29, -336
	.cfi_offset 30, -328
	mov	x29, sp	//,
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -320
	.cfi_offset 20, -312
	mov	x20, x2	// bias, tmp398
// src/cpp/cnn_internals.cpp:114:     int output_w     = output.width;
	ldr	w2, [x3, 36]	//, output_54(D)->width
	cmp	w2, 0	// output_w,
	ble	.L96		//,
	mov	x19, x1	// weight, tmp397
	sxtw	x1, w4	// _152, stride
// src/cpp/cnn_internals.cpp:119:     int out_size = output_h * output_w;
	mul	w3, w11, w2	// out_size, output_h, output_w
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 22, -296
	.cfi_offset 21, -304
	ptrue	p6.b, all	// tmp363
	neg	w22, w5	// ivtmp.551, padding
	sxtw	x17, w5	// _154, padding
// src/cpp/cnn_internals.cpp:118:     int in_ch_size = in_channels * in_size;
	mov	x16, 0	// ivtmp.568,
	mov	w8, w4	// stride, tmp400
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -280
	.cfi_offset 23, -288
	sbfiz	x9, x3, 2, 32	// _317, out_size,,
	ubfiz	x15, x2, 2, 32	// _181, output_w,,
	mov	x10, 0	// ivtmp.567,
// src/cpp/cnn_internals.cpp:124:     for (int b = 0; b < input_batches; ++b) {
	mov	w2, 0	// b,
	mov	w4, 64	// tmp364,
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 26, -264
	.cfi_offset 25, -272
	mov	w26, w5	// padding, tmp401
// src/cpp/cnn_internals.cpp:120:     int out_ch_size = out_channels * out_size;
	mul	w5, w6, w3	// out_ch_size_66, out_channels, out_size
	stp	x27, x28, [sp, 80]	//,,
	.cfi_offset 28, -248
	.cfi_offset 27, -256
	sxtw	x12, w5	// _332, out_ch_size_66
// src/cpp/cnn_internals.cpp:150:                             for (int kw = kw_lo; kw < kw_hi; ++kw) {
	cntw	x27		// tmp361
	str	x1, [sp, 160]	// _152, %sfp
// src/cpp/cnn_internals.cpp:110:     int kernel_w     = weight.width;
	ldr	w1, [x19, 36]	//, weight_52(D)->width
// src/cpp/cnn_internals.cpp:112:     int input_w      = input.width;
	ldp	w13, w24, [x0, 32]	// input_h, input_w,
// src/cpp/cnn_internals.cpp:108:     int in_channels  = weight.channels;
	ldp	w21, w14, [x19, 28]	// in_channels, _1,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x28, [x19]	// _80, MEM[(const struct vector *)weight_52(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:117:     int in_size = input_h * input_w;
	mul	w5, w13, w24	// in_size, input_h, input_w
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x25, [x0]	// _79, MEM[(const struct vector *)input_49(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:121:     int weight_size = kernel_h * kernel_w;
	mul	w3, w14, w1	// weight_size, _1, _2
// src/cpp/cnn_internals.cpp:108:     int in_channels  = weight.channels;
	str	w21, [sp, 108]	// in_channels, %sfp
// src/cpp/cnn_internals.cpp:118:     int in_ch_size = in_channels * in_size;
	mul	w0, w21, w5	// in_ch_size_64, in_channels, in_size
	sxtw	x5, w5	// _145, in_size
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1263:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x18, [x18]	// _85, MEM[(struct vector *)output_54(D)].D.54862._M_impl.D.54163._M_start
	stp	x9, x17, [sp, 184]	// _317, _154,
	sxtw	x19, w3	// _244, weight_size
// src/cpp/cnn_internals.cpp:122:     int weight_ch_size = in_channels * weight_size;
	mul	w3, w21, w3	// weight_ch_size, in_channels, weight_size
	sxtw	x21, w0	// _335, in_ch_size_64
	mov	w0, w22	// ivtmp.551, ivtmp.551
	stp	x5, x19, [sp, 168]	// _145, _244,
	mov	w22, w13	// input_h, input_h
	mov	x5, x12	// _332, _332
	sxtw	x19, w3	// _324, weight_ch_size
	mov	w13, w0	// ivtmp.551, ivtmp.551
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x3, [x20]	// _84, MEM[(const struct vector *)bias_53(D)].D.54862._M_impl.D.54163._M_start
	add	x20, x3, w6, uxtw 2	// _328, _84, out_channels,
	mov	w6, w26	// padding, padding
	mov	w26, w24	// input_w, input_w
	mov	x24, x28	// _80, _80
	mov	w28, w1	// _2, _2
	mov	w1, w7	// input_batches, input_batches
	mov	x7, x16	// ivtmp.568, ivtmp.568
	mov	w16, w11	// output_h, output_h
	mov	w11, w6	// padding, padding
.L112:
// src/cpp/cnn_internals.cpp:124:     for (int b = 0; b < input_batches; ++b) {
	stp	w1, w2, [sp, 200]	// input_batches, b,
	mov	x2, x3	// ivtmp.558, ivtmp.558
	add	x23, x18, x10, lsl 2	// ivtmp.560, _85, ivtmp.567,
	mov	x17, x7	// ivtmp.568, ivtmp.568
	stp	x3, x18, [sp, 208]	// _84, _85,
	ldr	x3, [sp, 176]	// _244, %sfp
	str	xzr, [sp, 144]	//, %sfp
	str	x5, [sp, 224]	// _332, %sfp
.L111:
	ldr	x0, [sp, 168]	// _145, %sfp
// src/cpp/cnn_internals.cpp:131:             for (int oh = 0; oh < output_h; ++oh) {
	mov	w12, 0	// oh,
	mov	w18, w13	// ivtmp.551, ivtmp.551
// src/cpp/cnn_internals.cpp:124:     for (int b = 0; b < input_batches; ++b) {
	mov	w7, w11	// ivtmp.546, padding
	mov	x9, x2	// ivtmp.558, ivtmp.558
	str	x23, [sp, 136]	// ivtmp.560, %sfp
	stp	x23, x20, [sp, 232]	// ivtmp.560, _328,
	str	x10, [sp, 248]	// ivtmp.567, %sfp
	str	w13, [sp, 256]	// ivtmp.551, %sfp
.L110:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:262:       if (__a < __b)
	bic	w5, w7, w7, asr #31	// _99, ivtmp.546
	ldr	x1, [sp, 136]	// ivtmp.550, %sfp
// src/cpp/cnn_internals.cpp:135:                 int kh_hi = std::min(kernel_h, input_h - c_h);
	add	w13, w22, w7	// _303, input_h, ivtmp.546
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:238:       if (__b < __a)
	cmp	w13, w14	// _303, _1
	csel	w13, w13, w14, le	// _51, _303, _1,
	add	w6, w5, w18	// _134, _99, ivtmp.551
	ldr	x2, [sp, 192]	// _154, %sfp
	mul	w6, w6, w26	// ivtmp.524, _134, input_w
	str	x1, [sp, 152]	// ivtmp.550, %sfp
	add	x1, x1, x15	// ivtmp.550, ivtmp.550, _181
	str	w6, [sp, 132]	// ivtmp.524, %sfp
	add	x6, sp, 268	//,,
	neg	x30, x2	// ivtmp.540, _154
	mov	w2, w11	// ivtmp.536, padding
	stp	w14, w22, [x6, -8]	// _1, input_h,
	stp	w16, w12, [x6]	// output_h, oh,
	mov	w6, w5	// _99, _99
	str	x1, [sp, 136]	// ivtmp.550, %sfp
	mul	w1, w5, w28	// ivtmp.525, _99, _2
	str	w7, [sp, 276]	// ivtmp.546, %sfp
	mov	x7, x17	// ivtmp.568, ivtmp.568
	stp	x15, x21, [sp, 280]	// _181, _335,
	str	w18, [sp, 296]	// ivtmp.551, %sfp
	str	w11, [sp, 300]	// padding, %sfp
	str	x19, [sp, 304]	// _324, %sfp
.L109:
// src/cpp/cnn_internals.cpp:145:                     for (int ic = 0; ic < in_channels; ++ic) {
	ldr	w5, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:143:                     int kw_hi = std::min(kernel_w, input_w - c_w);
	add	w14, w2, w26	// _156, ivtmp.536, input_w
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:262:       if (__a < __b)
	bic	w15, w2, w2, asr #31	// _4, ivtmp.536
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:238:       if (__b < __a)
	cmp	w14, w28	// _156, _2
	csel	w14, w14, w28, le	// _3, _156, _2,
// src/cpp/cnn_internals.cpp:138:                     float pixel_value = bias_ptr[oc];
	ldr	s31, [x9]	//, MEM[(const float *)_329]
// src/cpp/cnn_internals.cpp:145:                     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w5, 0	// in_channels,
	ble	.L98		//,
	ldr	x12, [sp, 144]	// ivtmp.530, %sfp
	sub	w18, w14, w15	// _104, _3, _4
	sxtw	x16, w15	// _252, _4
// src/cpp/cnn_internals.cpp:145:                     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w5, 0	// ic,
	sub	w19, w18, #1	// _103, _104,
// src/cpp/cnn_internals.cpp:145:                     for (int ic = 0; ic < in_channels; ++ic) {
	stp	x0, x3, [sp, 112]	// _145, _244,
	lsr	w22, w18, 4	// bnd.467_100, _104,
	and	w21, w18, 15	// _336, _104,
	and	w23, w18, -16	// niters_vector_mult_vf.468, _104,
	add	x20, x16, x30	// _70, _252, ivtmp.540
	mov	x10, x7	// ivtmp.529, ivtmp.568
	str	w1, [sp, 128]	// ivtmp.525, %sfp
	stp	x9, x7, [sp, 312]	// ivtmp.558, ivtmp.568,
	str	w2, [sp, 328]	// ivtmp.536, %sfp
	str	w8, [sp, 332]	// stride, %sfp
	.p2align 5,,15
.L108:
// src/cpp/cnn_internals.cpp:149:                         for (int kh = kh_lo; kh < kh_hi; ++kh) {
	ldp	w9, w8, [sp, 128]	//,,
// src/cpp/cnn_internals.cpp:149:                         for (int kh = kh_lo; kh < kh_hi; ++kh) {
	mov	w7, w6	// kh, _99
	add	x17, x16, x12	// _113, _252, ivtmp.530
// src/cpp/cnn_internals.cpp:149:                         for (int kh = kh_lo; kh < kh_hi; ++kh) {
	cmp	w13, w6	// _51, _99
	ble	.L100		//,
	stp	w5, w6, [sp, 100]	// ic, _99,
	.p2align 5,,15
.L99:
// src/cpp/cnn_internals.cpp:150:                             for (int kw = kw_lo; kw < kw_hi; ++kw) {
	cmp	w14, w15	// _3, _4
	ble	.L107		//,
	cmp	w19, 14	// _103,
	bls	.L114		//,
	mov	x2, x16	// _252, _252
.L104:
	sxtw	x6, w8	// _260, ivtmp.524
// src/cpp/cnn_internals.cpp:145:                     for (int ic = 0; ic < in_channels; ++ic) {
	movi	v30.4s, 0	//
	sxtw	x5, w9	// _275, ivtmp.525
	add	x0, x6, x10	// _74, _260, ivtmp.529
	add	x1, x17, x5	// _211, _113, _275
	add	x0, x20, x0	// _193, _70, _74
	add	x1, x24, x1, lsl 2	// ivtmp.517, _80, _211,
	add	x0, x25, x0, lsl 2	// ivtmp.514, _79, _193,
	mov	v1.16b, v30.16b	//,
	mov	v28.16b, v30.16b	//,
	mov	v0.16b, v30.16b	//,
	umaddl	x3, w22, w4, x0	// _94, bnd.467_100, tmp364, ivtmp.514
	.p2align 5,,15
.L102:
// src/cpp/cnn_internals.cpp:156:                                 pixel_value += input_ptr[in_idx] * weight_ptr[w_idx];
	ldp	q23, q22, [x1, 16]	//,,
// src/cpp/cnn_internals.cpp:156:                                 pixel_value += input_ptr[in_idx] * weight_ptr[w_idx];
	ldp	q25, q24, [x0, 32]	//,,
	ldp	q20, q26, [x0], 64	//,, ivtmp.514,
// src/cpp/cnn_internals.cpp:156:                                 pixel_value += input_ptr[in_idx] * weight_ptr[w_idx];
	ldr	q21, [x1, 48]	//, MEM <const vector(4) float> [(const float *)_24 + 48B]
	ldr	q2, [x1], 64	//, MEM <const vector(4) float> [(const float *)_24]
// src/cpp/cnn_internals.cpp:156:                                 pixel_value += input_ptr[in_idx] * weight_ptr[w_idx];
	fmla	v1.4s, v25.4s, v22.4s	//,,
	fmla	v28.4s, v26.4s, v23.4s	//,,
	fmla	v30.4s, v24.4s, v21.4s	//,,
	fmla	v0.4s, v20.4s, v2.4s	//,,
	cmp	x3, x0	// _94, ivtmp.514
	bne	.L102		//,
	fadd	v1.4s, v1.4s, v30.4s	//,,
	fadd	v0.4s, v0.4s, v28.4s	//,,
	fadd	v0.4s, v1.4s, v0.4s	//,,
	faddp	v0.4s, v0.4s, v0.4s	//,,
	faddp	v0.4s, v0.4s, v0.4s	//,,
	fadd	s31, s31, s0	//,,
	cbz	w21, .L103	// _336,
	mov	w1, w23	// niters_vector_mult_vf.468, niters_vector_mult_vf.468
.L101:
	uxtw	x3, w1	// _263, niters_vector_mult_vf.468
	add	x11, x30, x2	// _128, ivtmp.540, _252
// src/cpp/cnn_internals.cpp:154:                                 int w_idx  = out_oc_ch_size + ic_weight_size + kh * kernel_w + kw;
	mov	z19.s, #0	//
	add	x5, x5, x3	// _286, _275, _263
	add	x6, x6, x10	// _188, _260, ivtmp.529
	add	x3, x11, x3	// _187, _128, _263
	add	x2, x12, x2	// _230, ivtmp.530, _252
	add	x3, x3, x6	// _264, _187, _188
	add	x2, x2, x5	// _279, _230, _286
	add	x3, x25, x3, lsl 2	// vectp.490, _79, _264,
	mov	x0, 0	// ivtmp_289,
	sub	w1, w18, w1	// bnd.487, _104, niters_vector_mult_vf.468
	whilelo	p7.s, wzr, w1	// loop_mask_268,, bnd.487
	add	x2, x24, x2, lsl 2	// vectp.493, _80, _279,
.L105:
// src/cpp/cnn_internals.cpp:156:                                 pixel_value += input_ptr[in_idx] * weight_ptr[w_idx];
	ld1w	z29.s, p7/z, [x3, x0, lsl 2]	//, loop_mask_268,* vectp.490
// src/cpp/cnn_internals.cpp:156:                                 pixel_value += input_ptr[in_idx] * weight_ptr[w_idx];
	ld1w	z27.s, p7/z, [x2, x0, lsl 2]	//, loop_mask_268,* vectp.493
// src/cpp/cnn_internals.cpp:150:                             for (int kw = kw_lo; kw < kw_hi; ++kw) {
	add	x0, x0, x27	// ivtmp_289, ivtmp_289, tmp361
// src/cpp/cnn_internals.cpp:156:                                 pixel_value += input_ptr[in_idx] * weight_ptr[w_idx];
	fmla	z19.s, p7/m, z29.s, z27.s	//, loop_mask_268,,
	whilelo	p7.s, w0, w1	// loop_mask_268, ivtmp_289, bnd.487
	b.any	.L105	//,
	faddv	s19, p6, z19.s	//, tmp363,
	fadd	s31, s31, s19	//,,
.L107:
// src/cpp/cnn_internals.cpp:149:                         for (int kh = kh_lo; kh < kh_hi; ++kh) {
	add	w7, w7, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:149:                         for (int kh = kh_lo; kh < kh_hi; ++kh) {
	add	w8, w8, w26	// ivtmp.524, ivtmp.524, input_w
	add	w9, w9, w28	// ivtmp.525, ivtmp.525, _2
	cmp	w13, w7	// _51, kh
	bne	.L99		//,
.L124:
	ldp	w5, w6, [sp, 100]	//,,
.L100:
// src/cpp/cnn_internals.cpp:145:                     for (int ic = 0; ic < in_channels; ++ic) {
	add	w5, w5, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:145:                     for (int ic = 0; ic < in_channels; ++ic) {
	ldr	x0, [sp, 112]	// _145, %sfp
	add	x10, x10, x0	// ivtmp.529, ivtmp.529, _145
	ldr	x0, [sp, 120]	// _244, %sfp
	add	x12, x12, x0	// ivtmp.530, ivtmp.530, _244
	ldr	w0, [sp, 108]	//, %sfp
	cmp	w0, w5	// in_channels, ic
	bne	.L108		//,
	ldr	w1, [sp, 128]	//, %sfp
	ldp	x0, x3, [sp, 112]	// _145, _244,
	ldp	x9, x7, [sp, 312]	// ivtmp.558, ivtmp.568,
	ldr	w2, [sp, 328]	//, %sfp
	ldr	w8, [sp, 332]	//, %sfp
.L98:
	sub	w2, w2, w8	// ivtmp.536, ivtmp.536, stride
// src/cpp/cnn_internals.cpp:137:                 for (int ow = 0; ow < output_w; ++ow) {
	ldr	x5, [sp, 160]	// _152, %sfp
	ldr	x10, [sp, 136]	// ivtmp.550, %sfp
	add	x30, x30, x5	// ivtmp.540, ivtmp.540, _152
// src/cpp/cnn_internals.cpp:161:                     out_ptr[out_idx] = pixel_value;
	ldr	x5, [sp, 152]	// ivtmp.538, %sfp
	str	s31, [x5], 4	//, MEM[(float *)_157]
	str	x5, [sp, 152]	// ivtmp.538, %sfp
// src/cpp/cnn_internals.cpp:137:                 for (int ow = 0; ow < output_w; ++ow) {
	cmp	x5, x10	// ivtmp.538, ivtmp.550
	bne	.L109		//,
// src/cpp/cnn_internals.cpp:131:             for (int oh = 0; oh < output_h; ++oh) {
	add	x1, sp, 268	//,,
	ldr	w18, [sp, 296]	//, %sfp
	mov	x17, x7	// ivtmp.568, ivtmp.568
	ldp	x15, x21, [sp, 280]	// _181, _335,
	ldp	w16, w12, [x1]	//,,
// src/cpp/cnn_internals.cpp:131:             for (int oh = 0; oh < output_h; ++oh) {
	add	w18, w18, w8	// ivtmp.551, ivtmp.551, stride
// src/cpp/cnn_internals.cpp:131:             for (int oh = 0; oh < output_h; ++oh) {
	ldr	w7, [sp, 276]	//, %sfp
	ldr	x19, [sp, 304]	// _324, %sfp
	add	w12, w12, 1	// oh, oh,
	ldp	w14, w22, [x1, -8]	//,,
	sub	w7, w7, w8	// ivtmp.546, ivtmp.546, stride
	ldr	w11, [sp, 300]	//, %sfp
// src/cpp/cnn_internals.cpp:131:             for (int oh = 0; oh < output_h; ++oh) {
	cmp	w16, w12	// output_h, oh
	bne	.L110		//,
// src/cpp/cnn_internals.cpp:127:         for (int oc = 0; oc < out_channels; ++oc) {
	ldp	x23, x20, [sp, 232]	// ivtmp.560, _328,
	add	x2, x9, 4	// ivtmp.558, ivtmp.558,
	ldr	x0, [sp, 184]	// _317, %sfp
	ldr	x10, [sp, 248]	// ivtmp.567, %sfp
	ldr	w13, [sp, 256]	//, %sfp
	add	x23, x23, x0	// ivtmp.560, ivtmp.560, _317
	ldr	x0, [sp, 144]	// ivtmp.561, %sfp
	add	x0, x0, x19	// ivtmp.561, ivtmp.561, _324
	str	x0, [sp, 144]	// ivtmp.561, %sfp
	cmp	x2, x20	// ivtmp.558, _328
	bne	.L111		//,
// src/cpp/cnn_internals.cpp:124:     for (int b = 0; b < input_batches; ++b) {
	ldp	w1, w2, [sp, 200]	//,,
// src/cpp/cnn_internals.cpp:124:     for (int b = 0; b < input_batches; ++b) {
	add	x7, x17, x21	// ivtmp.568, ivtmp.568, _335
// src/cpp/cnn_internals.cpp:124:     for (int b = 0; b < input_batches; ++b) {
	ldp	x3, x18, [sp, 208]	// _84, _85,
	ldr	x5, [sp, 224]	// _332, %sfp
	add	w2, w2, 1	// b, b,
// src/cpp/cnn_internals.cpp:124:     for (int b = 0; b < input_batches; ++b) {
	add	x10, x10, x5	// ivtmp.567, ivtmp.567, _332
	cmp	w1, w2	// input_batches, b
	bne	.L112		//,
	ldp	x21, x22, [sp, 32]	//,,
	.cfi_restore 22
	.cfi_restore 21
	ldp	x23, x24, [sp, 48]	//,,
	.cfi_restore 24
	.cfi_restore 23
	ldp	x25, x26, [sp, 64]	//,,
	.cfi_restore 26
	.cfi_restore 25
	ldp	x27, x28, [sp, 80]	//,,
	.cfi_restore 28
	.cfi_restore 27
.L96:
// src/cpp/cnn_internals.cpp:166: }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x29, x30, [sp], 336	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
.L114:
	.cfi_def_cfa_offset 336
	.cfi_offset 19, -320
	.cfi_offset 20, -312
	.cfi_offset 21, -304
	.cfi_offset 22, -296
	.cfi_offset 23, -288
	.cfi_offset 24, -280
	.cfi_offset 25, -272
	.cfi_offset 26, -264
	.cfi_offset 27, -256
	.cfi_offset 28, -248
	.cfi_offset 29, -336
	.cfi_offset 30, -328
// src/cpp/cnn_internals.cpp:154:                                 int w_idx  = out_oc_ch_size + ic_weight_size + kh * kernel_w + kw;
	mov	w1, 0	// niters_vector_mult_vf.468,
	mov	x2, x16	// _252, _252
	sxtw	x6, w8	// _260, ivtmp.524
	sxtw	x5, w9	// _275, ivtmp.525
	b	.L101		//
.L103:
// src/cpp/cnn_internals.cpp:149:                         for (int kh = kh_lo; kh < kh_hi; ++kh) {
	add	w7, w7, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:149:                         for (int kh = kh_lo; kh < kh_hi; ++kh) {
	add	w8, w8, w26	// ivtmp.524, ivtmp.524, input_w
	add	w9, w9, w28	// ivtmp.525, ivtmp.525, _2
	cmp	w13, w7	// _51, kh
	bne	.L104		//,
	b	.L124		//
.L126:
	.cfi_def_cfa_offset 0
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 27
	.cfi_restore 28
	.cfi_restore 29
	.cfi_restore 30
	ret	
	.cfi_endproc
.LFE5446:
	.size	_Z29conv2d_forward_hoist_restrictRK6TensorS1_S1_RS_ii, .-_Z29conv2d_forward_hoist_restrictRK6TensorS1_S1_RS_ii
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC0:
	.string	"cannot create std::vector larger than max_size()"
	.text
	.align	2
	.p2align 5,,15
	.global	_Z22conv2d_forward_reorderRK6TensorS1_S1_RS_ii
	.type	_Z22conv2d_forward_reorderRK6TensorS1_S1_RS_ii, %function
_Z22conv2d_forward_reorderRK6TensorS1_S1_RS_ii:
.LFB5447:
	.cfi_startproc
	stp	x29, x30, [sp, -384]!	//,,,
	.cfi_def_cfa_offset 384
	.cfi_offset 29, -384
	.cfi_offset 30, -376
	mov	x29, sp	//,
	mov	x6, x3	// output, tmp601
	mov	x3, x0	// input, tmp598
	mov	x0, x1	// weight, tmp599
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	mov	x1, 2305843009213693951	// tmp372,
// src/cpp/cnn_internals.cpp:169:                     int stride, int padding) {
	stp	x19, x20, [sp, 16]	//,,
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 19, -368
	.cfi_offset 20, -360
	.cfi_offset 21, -352
	.cfi_offset 22, -344
	mov	w21, w4	// stride, tmp602
	stp	x23, x24, [sp, 48]	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x2, [x2]	// _123, MEM[(const struct vector *)bias_75(D)].D.54862._M_impl.D.54163._M_start
	.cfi_offset 23, -336
	.cfi_offset 24, -328
// src/cpp/cnn_internals.cpp:181:     int output_h     = output.height;
	ldp	w7, w24, [x6, 32]	// output_h, _2,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x4, [x3]	// _108, MEM[(const struct vector *)input_72(D)].D.54862._M_impl.D.54163._M_start
	str	x2, [sp, 224]	// _123, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1263:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x2, [x6]	// _132, MEM[(struct vector *)output_76(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:181:     int output_h     = output.height;
	str	w7, [sp, 172]	// output_h, %sfp
// src/cpp/cnn_internals.cpp:192:     std::vector<float> acc(output_w); 
	sxtw	x19, w24	// _2, _2
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1263:       { return _M_data_ptr(this->_M_impl._M_start); }
	str	x2, [sp, 152]	// _132, %sfp
// src/cpp/cnn_internals.cpp:187:     int out_size = output_h * output_w;
	mul	w2, w24, w7	// out_size, _2, output_h
	str	w2, [sp, 184]	// out_size, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x2, [x0]	// _109, MEM[(const struct vector *)weight_74(D)].D.54862._M_impl.D.54163._M_start
	str	x2, [sp, 240]	// _109, %sfp
// src/cpp/cnn_internals.cpp:183:     int input_batches = input.batches;
	ldr	w2, [x3, 24]	//, input_72(D)->batches
	str	w2, [sp, 148]	// input_batches, %sfp
// src/cpp/cnn_internals.cpp:176:     int in_channels  = weight.channels;
	ldp	w23, w7, [x0, 24]	// out_channels, in_channels,
// src/cpp/cnn_internals.cpp:178:     int kernel_w     = weight.width;
	ldp	w0, w2, [x0, 32]	// _1, kernel_w,
	str	w2, [sp, 144]	// kernel_w, %sfp
	str	w0, [sp, 208]	// _1, %sfp
// src/cpp/cnn_internals.cpp:180:     int input_w      = input.width;
	ldp	w0, w2, [x3, 32]	// input_h, input_w,
	str	w2, [sp, 168]	// input_w, %sfp
	str	w0, [sp, 212]	// input_h, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	cmp	x19, x1	// _2, tmp372
	bhi	.L212		//,
	mov	w22, w5	// padding, tmp603
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	cbz	x19, .L213	// _2,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	lsl	x20, x19, 2	// _145, _2,
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 26, -312
	.cfi_offset 25, -320
	mov	x0, x20	//, _145
	str	w7, [sp, 104]	// in_channels, %sfp
	str	x4, [sp, 112]	// _108, %sfp
	str	x20, [sp, 136]	// _145, %sfp
	bl	_Znwm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	cmp	x19, 1	// _2,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	str	wzr, [x0]	//, *_152
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x26, x0	// iftmp.18_56, tmp604
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	ldr	x4, [sp, 112]	// _108, %sfp
	ldr	w7, [sp, 104]	//, %sfp
	beq	.L135		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	sub	x2, x20, #4	//, _145,
	mov	w1, 0	//,
	str	w7, [sp, 104]	// in_channels, %sfp
	add	x0, x0, 4	//, iftmp.18_56,
	bl	memset		//
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	ldr	w0, [sp, 148]	//, %sfp
	ldr	x4, [sp, 112]	// _108, %sfp
	ldr	w7, [sp, 104]	//, %sfp
	cmp	w0, 0	// input_batches,
	ble	.L137		//,
	cmp	w23, 0	// out_channels,
	ble	.L137		//,
	ldr	w0, [sp, 172]	//, %sfp
	cmp	w0, 0	// output_h,
	bgt	.L133		//,
.L137:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	ldr	x1, [sp, 136]	//, %sfp
	mov	x0, x26	//, iftmp.18_56
// src/cpp/cnn_internals.cpp:243: }
	ldp	x19, x20, [sp, 16]	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	ldp	x25, x26, [sp, 64]	//,,
	.cfi_restore 26
	.cfi_restore 25
// src/cpp/cnn_internals.cpp:243: }
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x29, x30, [sp], 384	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	b	_ZdlPvm		//
.L213:
	.cfi_def_cfa_offset 384
	.cfi_offset 19, -368
	.cfi_offset 20, -360
	.cfi_offset 21, -352
	.cfi_offset 22, -344
	.cfi_offset 23, -336
	.cfi_offset 24, -328
	.cfi_offset 29, -384
	.cfi_offset 30, -376
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	ldr	w0, [sp, 148]	//, %sfp
	cmp	w0, 0	// input_batches,
	ble	.L129		//,
	cmp	w23, 0	// out_channels,
	ble	.L129		//,
	ldr	w0, [sp, 172]	//, %sfp
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 26, -312
	.cfi_offset 25, -320
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	mov	x26, 0	// iftmp.18_56,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	str	xzr, [sp, 136]	//, %sfp
	cmp	w0, 0	// output_h,
	ble	.L214		//,
.L133:
// src/cpp/cnn_internals.cpp:185:     int in_size = input_h * input_w;
	ldp	w2, w0, [sp, 208]	//,,
	lsl	x11, x19, 2	// _254, _2,
	lsr	w9, w24, 3	// bnd.583_168, _2,
	index	z31.s, #0, w21	//,, stride
// src/cpp/cnn_internals.cpp:231:                                 acc_ptr[ow] += input_ptr[in_row_offset + iw] * w_val;
	ptrue	p6.b, all	// tmp571
	mvn	w5, w21	// tmp387, stride
	mov	w10, 32	// tmp585,
	stp	x27, x28, [sp, 80]	//,,
	.cfi_offset 28, -296
	.cfi_offset 27, -304
	lsr	w19, w5, 31	// _332, tmp387,
	sbfiz	x18, x21, 2, 32	// _463, stride,,
// src/cpp/cnn_internals.cpp:186:     int in_ch_size = in_channels * in_size;
	mov	w3, 0	// ivtmp.731,
	lsr	w20, w21, 31	// _370, stride,
// src/cpp/cnn_internals.cpp:185:     int in_size = input_h * input_w;
	ldr	w1, [sp, 168]	//, %sfp
// src/cpp/cnn_internals.cpp:186:     int in_ch_size = in_channels * in_size;
	mov	x16, 0	// ivtmp.732,
	sxtw	x25, w21	// _333, stride
	str	x11, [sp, 248]	// _254, %sfp
	sbfiz	x28, x21, 4, 32	// _459, stride,,
	sbfiz	x30, x21, 3, 32	// _552, stride,,
// src/cpp/cnn_internals.cpp:188:     int out_ch_size = out_channels * out_size;
	ldr	w15, [sp, 184]	//, %sfp
// src/cpp/cnn_internals.cpp:186:     int in_ch_size = in_channels * in_size;
	str	xzr, [sp, 176]	//, %sfp
	ldr	x5, [sp, 224]	// _123, %sfp
	add	x5, x5, w23, sxtw 2	// _407, _123, out_channels,
// src/cpp/cnn_internals.cpp:185:     int in_size = input_h * input_w;
	mul	w8, w0, w1	// in_size, input_h, input_w
	mov	w0, 12	// tmp394,
	str	x5, [sp, 104]	// _407, %sfp
	mov	w5, w3	// ivtmp.731, ivtmp.731
// src/cpp/cnn_internals.cpp:189:     int weight_size = kernel_h * kernel_w;
	ldr	w1, [sp, 144]	//, %sfp
	mov	x3, x18	// _463, _463
	mov	w18, w19	// _332, _332
	smull	x27, w21, w0	// _555, stride, tmp394
	mov	w19, w20	// _370, _370
	mov	w20, w24	// _2, _2
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	str	wzr, [sp, 216]	//, %sfp
// src/cpp/cnn_internals.cpp:186:     int in_ch_size = in_channels * in_size;
	mul	w11, w7, w8	// in_ch_size_87, in_channels, in_size
// src/cpp/cnn_internals.cpp:185:     int in_size = input_h * input_w;
	str	w8, [sp, 236]	// in_size, %sfp
	sub	w8, w24, #1	// _576, _2,
	umaddl	x0, w9, w10, x26	// _439, bnd.583_168, tmp585, iftmp.18_56
	str	w8, [sp, 232]	// _576, %sfp
// src/cpp/cnn_internals.cpp:189:     int weight_size = kernel_h * kernel_w;
	mul	w6, w2, w1	// weight_size, _1, kernel_w
	sxtw	x17, w1	// _129, kernel_w
	sbfiz	x8, x1, 2, 32	// _42, kernel_w,,
// src/cpp/cnn_internals.cpp:188:     int out_ch_size = out_channels * out_size;
	mul	w2, w23, w15	// out_ch_size, out_channels, out_size
	lsr	w1, w24, 3	// _251, _2,
	sxtw	x15, w15	// _411, out_size
	mov	x24, x3	// _463, _463
	ldr	x3, [sp, 104]	// _407, %sfp
// src/cpp/cnn_internals.cpp:222:                             int ow_lo = num <= 0 ? 0 : (num + stride - 1) / stride;
	sub	w23, w22, #1	// _222, padding,
// src/cpp/cnn_internals.cpp:190:     int weight_ch_size = in_channels * weight_size;
	mul	w12, w7, w6	// weight_ch_size, in_channels, weight_size
	sbfiz	x6, x6, 2, 32	// _515, weight_size,,
	sxtw	x13, w11	// _176, in_ch_size_87
	stp	x0, x15, [sp, 288]	// _439, _411,
	sxtw	x11, w22	// _321, padding
	neg	w0, w22	// ivtmp.715, padding
	mov	w10, w2	// out_ch_size, out_ch_size
	str	x6, [sp, 264]	// _515, %sfp
	sxtw	x6, w2	// _267, out_ch_size
	sxtw	x14, w12	// _209, weight_ch_size
	mov	x9, x6	// _267, _267
	mov	x12, x16	// ivtmp.732, ivtmp.732
	mov	w2, w0	// ivtmp.715, ivtmp.715
	mov	w16, w23	// _222, _222
	mov	x0, x11	// _321, _321
	mov	w23, w21	// stride, stride
	mov	x6, x1	// _251, _251
	mov	x21, x17	// _129, _129
	mov	w17, w22	// padding, padding
	mov	x22, x4	// _108, _108
.L176:
	ldr	w1, [sp, 168]	//, %sfp
	lsl	x4, x6, 5	// _250, _251,
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	str	w5, [sp, 132]	// ivtmp.731, %sfp
	str	w10, [sp, 312]	// out_ch_size, %sfp
	mov	x10, x6	// _251, _251
	str	w5, [sp, 316]	// ivtmp.731, %sfp
	mov	x5, x3	// _407, _407
	stp	x12, x9, [sp, 320]	// ivtmp.732, _267,
	mov	w9, w19	// _370, _370
	mov	x19, x21	// _129, _129
	add	w1, w1, w17	// _244, input_w, padding
	mov	w21, w17	// padding, padding
	sub	w15, w1, #1	// _115, _244,
	mov	w17, w16	// _222, _222
	ldr	x1, [sp, 224]	// ivtmp.720, %sfp
	mov	w11, w15	// _115, _115
	mov	x16, x12	// ivtmp.724, ivtmp.732
	str	x12, [sp, 104]	// ivtmp.732, %sfp
	str	xzr, [sp, 160]	//, %sfp
	str	x4, [sp, 280]	// _250, %sfp
	str	x13, [sp, 304]	// _176, %sfp
	str	x1, [sp, 200]	// ivtmp.720, %sfp
	mov	w1, w2	// ivtmp.715, ivtmp.715
.L164:
// src/cpp/cnn_internals.cpp:202:             const float b_val = bias_ptr[oc];
	ldr	x2, [sp, 200]	// ivtmp.720, %sfp
	mov	w4, w1	// ivtmp.715, ivtmp.715
	mov	w6, w21	// ivtmp.711, ivtmp.711
	mov	x13, x10	// _251, _251
	str	w1, [sp, 188]	// ivtmp.715, %sfp
	mov	w1, w16	// ivtmp.713, ivtmp.713
// src/cpp/cnn_internals.cpp:204:             for (int oh = 0; oh < output_h; ++oh) {
	str	wzr, [sp, 220]	//, %sfp
	str	x5, [sp, 336]	// _407, %sfp
// src/cpp/cnn_internals.cpp:202:             const float b_val = bias_ptr[oc];
	ldr	s25, [x2]	// b_val, MEM[(const float *)_575]
	dup	v30.4s, v25.s[0]	//, b_val
.L138:
// src/cpp/cnn_internals.cpp:209:                 int kh_hi = std::min(kernel_h, input_h - c_h);
	ldp	w3, w2, [sp, 208]	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:262:       if (__a < __b)
	bic	w12, w6, w6, asr #31	// _280, ivtmp.711
// src/cpp/cnn_internals.cpp:209:                 int kh_hi = std::min(kernel_h, input_h - c_h);
	add	w2, w2, w6	// _365, input_h, ivtmp.711
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:238:       if (__b < __a)
	cmp	w2, w3	// _365, _1
	csel	w10, w2, w3, le	// _283, _365, _1,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	cbz	w20, .L215	// _2,
	ldr	w2, [sp, 232]	//, %sfp
	cmp	w2, 7	// _576,
	bls	.L182		//,
	mov	x2, x26	// ivtmp.703, iftmp.18_56
.L140:
	ldr	x3, [sp, 288]	// _439, %sfp
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	stp	q30, q30, [x2], 32	//,, ivtmp.703,
	cmp	x3, x2	// _439, ivtmp.703
	bne	.L140		//,
	and	w2, w20, -8	// _202, _2,
	cmp	w20, w2	// _2, _202
	beq	.L141		//,
.L139:
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	uxtw	x3, w2	// _207, _202
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	add	w15, w2, 1	// ow_184, _202,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	lsl	x5, x3, 2	// _206, _207,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	str	s25, [x26, x5]	// b_val, *_205
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	cmp	w20, w15	// _2, ow_184
	ble	.L141		//,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	add	x3, x26, x5	// tmp407, iftmp.18_56, _206
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	add	w5, w2, 2	// ow_50, _202,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	str	s25, [x3, 4]	// b_val, *_48
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	cmp	w20, w5	// _2, ow_50
	ble	.L141		//,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	add	w5, w2, 3	// ow_486, _202,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	str	s25, [x3, 8]	// b_val, *_162
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	cmp	w20, w5	// _2, ow_486
	ble	.L141		//,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	add	w5, w2, 4	// ow_410, _202,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	str	s25, [x3, 12]	// b_val, *_408
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	cmp	w20, w5	// _2, ow_410
	ble	.L141		//,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	add	w5, w2, 5	// ow_417, _202,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	str	s25, [x3, 16]	// b_val, *_415
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	cmp	w20, w5	// _2, ow_417
	ble	.L141		//,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	add	w5, w2, 6	// ow_30, _202,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	str	s25, [x3, 20]	// b_val, *_423
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	cmp	w20, w5	// _2, ow_30
	ble	.L141		//,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	add	w2, w2, 7	// ow_36, _202,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	str	s25, [x3, 24]	// b_val, *_34
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	cmp	w20, w2	// _2, ow_36
	ble	.L141		//,
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	str	s25, [x3, 28]	// b_val, *_551
// src/cpp/cnn_internals.cpp:211:                 for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w7, 0	// in_channels,
	ble	.L216		//,
.L142:
	cmp	w12, w10	// _280, _283
	bge	.L146		//,
	ldr	w2, [sp, 144]	//, %sfp
	cmp	w2, 0	// kernel_w,
	ble	.L146		//,
.L177:
	ldr	w2, [sp, 144]	//, %sfp
// src/cpp/cnn_internals.cpp:211:                 for (int ic = 0; ic < in_channels; ++ic) {
	mov	w3, 0	// ic,
	str	x8, [sp, 120]	// _42, %sfp
	mov	w8, w7	// in_channels, in_channels
	mov	w7, w3	// ic, ic
	mov	x3, x14	// _209, _209
	str	w1, [sp, 356]	// ivtmp.713, %sfp
	mov	x1, 0	// ivtmp.696,
	str	x16, [sp, 344]	// ivtmp.724, %sfp
	str	w6, [sp, 352]	// ivtmp.711, %sfp
	mul	w5, w2, w12	// _520, kernel_w, _280
	ldr	x2, [sp, 160]	// ivtmp.726, %sfp
	add	x5, x2, w5, sxtw	// _502, ivtmp.726, _520
	ldr	w2, [sp, 188]	//, %sfp
	add	w15, w2, w12	// _12, ivtmp.715, _280
	ldr	x2, [sp, 240]	// _109, %sfp
	add	x2, x2, x5, lsl 2	// ivtmp.698, _109, _502,
	str	x2, [sp, 192]	// ivtmp.698, %sfp
	ldrsw	x2, [sp, 236]	// _539, %sfp
	str	x2, [sp, 272]	// _539, %sfp
	ldr	w2, [sp, 168]	//, %sfp
	mul	w15, w15, w2	// _11, _12, input_w
	sxtw	x5, w2	// _107, input_w
	ldr	x2, [sp, 176]	// ivtmp.733, %sfp
	add	x2, x2, w15, sxtw	// _449, ivtmp.733, _11
	str	x2, [sp, 256]	// _449, %sfp
	mov	x2, x13	// _251, _251
.L151:
	ldr	x6, [sp, 256]	// _449, %sfp
// src/cpp/cnn_internals.cpp:215:                     for (int kh = kh_lo; kh < kh_hi; ++kh) {
	mov	w15, w12	// kh, _280
	str	w7, [sp, 364]	// ic, %sfp
	mov	x7, x3	// _209, _209
// src/cpp/cnn_internals.cpp:211:                 for (int ic = 0; ic < in_channels; ++ic) {
	ldr	x13, [sp, 192]	// ivtmp.688, %sfp
	str	x1, [sp, 368]	// ivtmp.696, %sfp
	str	w10, [sp, 128]	// _283, %sfp
	add	x16, x6, x1	// ivtmp.691, _449, ivtmp.696
	mov	x1, x2	// _251, _251
	str	w8, [sp, 360]	// in_channels, %sfp
	str	w12, [sp, 376]	// _280, %sfp
	str	w4, [sp, 380]	// ivtmp.715, %sfp
	.p2align 5,,15
.L147:
// src/cpp/cnn_internals.cpp:223:                             int ow_hi = std::min(output_w, (input_w + padding - kw - 1) / stride + 1);
	mov	x8, 0	// ivtmp.673,
	sub	x14, x16, x0	// _286, ivtmp.691, _321
	str	x16, [sp, 104]	// ivtmp.691, %sfp
	str	w15, [sp, 112]	// kh, %sfp
	.p2align 5,,15
.L150:
// src/cpp/cnn_internals.cpp:222:                             int ow_lo = num <= 0 ? 0 : (num + stride - 1) / stride;
	sub	w4, w21, w8	// _130, padding, ivtmp.673
	mov	w2, w8	// _298, ivtmp.673
// src/cpp/cnn_internals.cpp:222:                             int ow_lo = num <= 0 ? 0 : (num + stride - 1) / stride;
	mov	w3, 0	// iftmp.14_62,
// src/cpp/cnn_internals.cpp:222:                             int ow_lo = num <= 0 ? 0 : (num + stride - 1) / stride;
	cmp	w4, 0	// _130,
	ble	.L153		//,
// src/cpp/cnn_internals.cpp:222:                             int ow_lo = num <= 0 ? 0 : (num + stride - 1) / stride;
	sub	w3, w23, w8	// _26, stride, ivtmp.673
	add	w3, w3, w17	// _538, _26, _222
// src/cpp/cnn_internals.cpp:222:                             int ow_lo = num <= 0 ? 0 : (num + stride - 1) / stride;
	sdiv	w3, w3, w23	// iftmp.14_62, _538, stride
.L153:
// src/cpp/cnn_internals.cpp:223:                             int ow_hi = std::min(output_w, (input_w + padding - kw - 1) / stride + 1);
	sub	w4, w11, w2	// _270, _115, _298
// src/cpp/cnn_internals.cpp:226:                             float w_val = weight_ptr[w_idx];
	ldr	s1, [x13, x8, lsl 2]	//, MEM[(const float *)_297 + ivtmp.673_560 * 4]
// src/cpp/cnn_internals.cpp:223:                             int ow_hi = std::min(output_w, (input_w + padding - kw - 1) / stride + 1);
	sdiv	w4, w4, w23	// _23, _270, stride
// src/cpp/cnn_internals.cpp:223:                             int ow_hi = std::min(output_w, (input_w + padding - kw - 1) / stride + 1);
	add	w4, w4, 1	// _24, _23,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:238:       if (__b < __a)
	cmp	w4, w20	// _24, _2
	csel	w4, w4, w20, le	// _3, _24, _2,
// src/cpp/cnn_internals.cpp:228:                             for (int ow = ow_lo; ow < ow_hi; ++ow) {
	cmp	w4, w3	// _3, iftmp.14_62
	ble	.L160		//,
	sub	w15, w4, w3	// _337, _3, iftmp.14_62
	sxtw	x6, w3	// ivtmp.654, iftmp.14_62
	mul	w12, w3, w23	// _319, iftmp.14_62, stride
	add	x12, x8, w12, sxtw	// _227, ivtmp.673, _319
	add	x2, x26, w3, sxtw 2	// _346, iftmp.18_56, iftmp.14_62,
	sub	x10, x15, #1	// _338, _337,
	add	x12, x12, x14	// _330, _227, _286
	sub	x16, x6, #1	// _188, ivtmp.654,
	cmp	w9, 0	// _370,
	mul	x10, x10, x24	// _340, _338, _463
	add	x16, x16, x15	// _354, _188, _337
	csel	x15, x10, xzr, ne	// _379, _340,,
	cmp	w18, 0	// _332,
	csel	x10, x10, xzr, ne	// _341, _340,,
	add	x15, x15, x12, lsl 2	// _380, _379, _330,
	sub	w3, w4, w3	// _336, _3, iftmp.14_62
	add	x16, x26, x16, lsl 2	// _356, iftmp.18_56, _354,
	add	x10, x10, x12, lsl 2	// _342, _341, _330,
	add	x15, x22, x15	// _381, _108, _380
	cmp	x16, x15	// _356, _381
	lsl	x15, x12, 2	// _331, _330,
	add	x10, x22, x10	// _343, _108, _342
	ccmp	x10, x2, 0, cs	// _343, _346,,
	bcs	.L156		//,
	sub	w4, w3, #1	// _429, _336,
	cmp	w4, 2	// _429,
	bls	.L184		//,
	dup	v26.4s, v1.s[0]	//,
	add	x15, x22, x15	// ivtmp.665, _108, _331
	lsr	w4, w3, 2	// bnd.600_432, _336,
	add	x4, x2, w4, uxtw 4	// _559, ivtmp.662, bnd.600_432,
	.p2align 5,,15
.L158:
// src/cpp/cnn_internals.cpp:231:                                 acc_ptr[ow] += input_ptr[in_row_offset + iw] * w_val;
	ldr	s28, [x15, x30]	// MEM[(const float *)_128 + _552 * 1], MEM[(const float *)_128 + _552 * 1]
	ldr	s3, [x15, x25, lsl 2]	//, MEM[(const float *)_128 + _333 * 4]
	ldr	s29, [x15, x27]	// MEM[(const float *)_128 + _555 * 1], MEM[(const float *)_128 + _555 * 1]
	ldr	s4, [x15]	//, MEM[(const float *)_128]
	add	x15, x15, x28	// ivtmp.665, ivtmp.665, _459
// src/cpp/cnn_internals.cpp:231:                                 acc_ptr[ow] += input_ptr[in_row_offset + iw] * w_val;
	ldr	q2, [x2]	//, MEM <vector(4) float> [(float *)_317]
// src/cpp/cnn_internals.cpp:231:                                 acc_ptr[ow] += input_ptr[in_row_offset + iw] * w_val;
	ins	v3.s[1], v29.s[0]	//, MEM[(const float *)_128 + _555 * 1]
	ins	v4.s[1], v28.s[0]	//, MEM[(const float *)_128 + _552 * 1]
	zip1	v3.4s, v4.4s, v3.4s	//,,
// src/cpp/cnn_internals.cpp:231:                                 acc_ptr[ow] += input_ptr[in_row_offset + iw] * w_val;
	fmla	v2.4s, v3.4s, v26.4s	//,,
	str	q2, [x2], 16	//, MEM <vector(4) float> [(float *)_317]
	cmp	x2, x4	// ivtmp.662, _559
	bne	.L158		//,
	and	w4, w3, -4	// niters_vector_mult_vf.601, _336,
	tst	x3, 3	// _336,
	beq	.L160		//,
.L157:
	uxtw	x2, w4	// _495, niters_vector_mult_vf.601
	sub	w3, w3, w4	// _542, _336, niters_vector_mult_vf.601
	mov	z1.s, s1	//,
	whilelo	p7.s, xzr, x3	// max_mask_545,, _542
	add	x6, x6, x2	// _496, ivtmp.654, _495
// src/cpp/cnn_internals.cpp:231:                                 acc_ptr[ow] += input_ptr[in_row_offset + iw] * w_val;
	ld1w	z27.s, p7/z, [x26, x6, lsl 2]	//, max_mask_545,* iftmp.18_56
	madd	x2, x25, x2, x12	// _526, _333, _495, _330
	add	x2, x22, x2, lsl 2	// vectp.616, _108, _526,
// src/cpp/cnn_internals.cpp:231:                                 acc_ptr[ow] += input_ptr[in_row_offset + iw] * w_val;
	ld1w	z28.s, p7/z, [x2, z31.s, sxtw 2]	//, max_mask_545, vectp.616,,
// src/cpp/cnn_internals.cpp:231:                                 acc_ptr[ow] += input_ptr[in_row_offset + iw] * w_val;
	fmad	z1.s, p6/m, z28.s, z27.s	//, tmp571,,
	st1w	z1.s, p7, [x26, x6, lsl 2]	//, max_mask_545,* iftmp.18_56
.L160:
// src/cpp/cnn_internals.cpp:220:                         for (int kw = 0; kw < kernel_w; ++kw) {
	add	x8, x8, 1	// ivtmp.673, ivtmp.673,
	cmp	x19, x8	// _129, ivtmp.673
	bne	.L150		//,
// src/cpp/cnn_internals.cpp:215:                     for (int kh = kh_lo; kh < kh_hi; ++kh) {
	ldr	x2, [sp, 120]	// _42, %sfp
// src/cpp/cnn_internals.cpp:215:                     for (int kh = kh_lo; kh < kh_hi; ++kh) {
	ldr	w15, [sp, 112]	//, %sfp
	ldr	x16, [sp, 104]	// ivtmp.691, %sfp
// src/cpp/cnn_internals.cpp:215:                     for (int kh = kh_lo; kh < kh_hi; ++kh) {
	add	x13, x13, x2	// ivtmp.688, ivtmp.688, _42
	ldr	w2, [sp, 128]	//, %sfp
// src/cpp/cnn_internals.cpp:215:                     for (int kh = kh_lo; kh < kh_hi; ++kh) {
	add	w15, w15, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:215:                     for (int kh = kh_lo; kh < kh_hi; ++kh) {
	add	x16, x16, x5	// ivtmp.691, ivtmp.691, _107
	cmp	w15, w2	// kh, _283
	bne	.L147		//,
	mov	x2, x1	// _251, _251
	add	x1, sp, 372	//,,
// src/cpp/cnn_internals.cpp:211:                 for (int ic = 0; ic < in_channels; ++ic) {
	ldp	x13, x6, [sp, 264]	// _515, _539,
	mov	x3, x7	// _209, _209
	ldp	w12, w4, [x1, 4]	//,,
	ldp	w8, w7, [x1, -12]	//,,
	ldr	x1, [sp, 368]	// ivtmp.696, %sfp
	ldr	w10, [sp, 128]	//, %sfp
// src/cpp/cnn_internals.cpp:211:                 for (int ic = 0; ic < in_channels; ++ic) {
	add	w7, w7, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:211:                 for (int ic = 0; ic < in_channels; ++ic) {
	add	x1, x1, x6	// ivtmp.696, ivtmp.696, _539
	ldr	x6, [sp, 192]	// ivtmp.698, %sfp
	add	x6, x6, x13	// ivtmp.698, ivtmp.698, _515
	str	x6, [sp, 192]	// ivtmp.698, %sfp
	cmp	w8, w7	// in_channels, ic
	bgt	.L151		//,
	ldr	w1, [sp, 356]	//, %sfp
	mov	x13, x2	// _251, _251
	mov	w7, w8	// in_channels, in_channels
	mov	x14, x3	// _209, _209
// src/cpp/cnn_internals.cpp:236:                 int out_row = out_b_ch_size + out_oc_size + oh * output_w;
	ldr	w2, [sp, 132]	//, %sfp
	ldr	x8, [sp, 120]	// _42, %sfp
	ldr	x16, [sp, 344]	// ivtmp.724, %sfp
	add	w2, w2, w1	// _39, ivtmp.722, ivtmp.713
	ldr	w6, [sp, 352]	//, %sfp
// src/cpp/cnn_internals.cpp:237:                 for (int ow = 0; ow < output_w; ++ow) {
	cbz	w20, .L149	// _2,
.L148:
	add	x2, x16, w2, sxtw	// _233, ivtmp.724, _39
	ldr	x3, [sp, 152]	// _132, %sfp
	add	x5, x3, x2, lsl 2	// _231, _132, _233,
	add	x3, x26, 4	// _230, iftmp.18_56,
	sub	x3, x5, x3	// _229, _231, _230
	cmp	x3, 24	// _229,
	bls	.L166		//,
	ldr	w3, [sp, 232]	//, %sfp
	cmp	w3, 6	// _576,
	bls	.L185		//,
.L143:
// src/cpp/cnn_internals.cpp:228:                             for (int ow = ow_lo; ow < ow_hi; ++ow) {
	mov	x3, 0	// ivtmp.645,
	add	x12, x26, 16	// _249, iftmp.18_56,
	add	x10, x5, 16	// _248, _231,
.L168:
// src/cpp/cnn_internals.cpp:238:                     out_ptr[out_row + ow] = acc_ptr[ow];
	ldr	q28, [x26, x3]	// tmp995, MEM <vector(4) float> [(float *)iftmp.18_563 + ivtmp.645_253 * 1]
// src/cpp/cnn_internals.cpp:238:                     out_ptr[out_row + ow] = acc_ptr[ow];
	ldr	q29, [x12, x3]	// MEM <vector(4) float> [(float *)_249 + ivtmp.645_253 * 1], MEM <vector(4) float> [(float *)_249 + ivtmp.645_253 * 1]
	ldr	x15, [sp, 280]	// _250, %sfp
// src/cpp/cnn_internals.cpp:238:                     out_ptr[out_row + ow] = acc_ptr[ow];
	str	q28, [x5, x3]	// tmp995, MEM <vector(4) float> [(float *)_300 + ivtmp.645_253 * 1]
	str	q29, [x10, x3]	// MEM <vector(4) float> [(float *)_249 + ivtmp.645_253 * 1], MEM <vector(4) float> [(float *)_248 + ivtmp.645_253 * 1]
	add	x3, x3, 32	// ivtmp.645, ivtmp.645,
	cmp	x15, x3	// _250, ivtmp.645
	bne	.L168		//,
	and	w3, w20, -8	// niters_vector_mult_vf.584, _2,
	cmp	w3, w20	// niters_vector_mult_vf.584, _2
	beq	.L149		//,
.L167:
	uxtw	x5, w3	// _54, niters_vector_mult_vf.584
	sub	w3, w20, w3	// bnd.593, _2, niters_vector_mult_vf.584
	whilelo	p5.s, wzr, w3	// max_mask_293,, bnd.593
// src/cpp/cnn_internals.cpp:238:                     out_ptr[out_row + ow] = acc_ptr[ow];
	ld1w	z0.s, p5/z, [x26, x5, lsl 2]	//, max_mask_293,* iftmp.18_56
	cntw	x10		// tmp997
	whilelo	p7.s, w10, w3	// next_mask_210, tmp997, bnd.593
	ldr	x3, [sp, 152]	// _132, %sfp
	add	x2, x5, x2	// _4, _54, _233
	add	x5, x26, x5, lsl 2	// vectp_iftmp.595, iftmp.18_56, _54,
	add	x2, x3, x2, lsl 2	// vectp.598, _132, _4,
// src/cpp/cnn_internals.cpp:238:                     out_ptr[out_row + ow] = acc_ptr[ow];
	st1w	z0.s, p5, [x2]	//, max_mask_293,* vectp.598
	b.none	.L149	//,
// src/cpp/cnn_internals.cpp:238:                     out_ptr[out_row + ow] = acc_ptr[ow];
	ld1w	z29.s, p7/z, [x5, #1, mul vl]	//, next_mask_210,
// src/cpp/cnn_internals.cpp:238:                     out_ptr[out_row + ow] = acc_ptr[ow];
	st1w	z29.s, p7, [x2, #1, mul vl]	//, next_mask_210,
.L149:
// src/cpp/cnn_internals.cpp:204:             for (int oh = 0; oh < output_h; ++oh) {
	ldr	w3, [sp, 188]	//, %sfp
	sub	w6, w6, w23	// ivtmp.711, ivtmp.711, stride
	add	w1, w1, w20	// ivtmp.713, ivtmp.713, _2
// src/cpp/cnn_internals.cpp:204:             for (int oh = 0; oh < output_h; ++oh) {
	ldr	w2, [sp, 220]	//, %sfp
// src/cpp/cnn_internals.cpp:204:             for (int oh = 0; oh < output_h; ++oh) {
	add	w3, w3, w23	// ivtmp.715, ivtmp.715, stride
	str	w3, [sp, 188]	// ivtmp.715, %sfp
// src/cpp/cnn_internals.cpp:204:             for (int oh = 0; oh < output_h; ++oh) {
	add	w2, w2, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:204:             for (int oh = 0; oh < output_h; ++oh) {
	ldr	w3, [sp, 172]	//, %sfp
// src/cpp/cnn_internals.cpp:204:             for (int oh = 0; oh < output_h; ++oh) {
	str	w2, [sp, 220]	// oh, %sfp
// src/cpp/cnn_internals.cpp:204:             for (int oh = 0; oh < output_h; ++oh) {
	cmp	w3, w2	// output_h, oh
	bne	.L138		//,
// src/cpp/cnn_internals.cpp:199:         for (int oc = 0; oc < out_channels; ++oc) {
	ldr	w3, [sp, 132]	//, %sfp
	mov	w1, w4	// ivtmp.715, ivtmp.715
	mov	x10, x13	// _251, _251
	ldr	w4, [sp, 184]	//, %sfp
	ldr	x2, [sp, 200]	// ivtmp.720, %sfp
	ldr	x5, [sp, 336]	// _407, %sfp
	sub	w3, w3, w4	// ivtmp.722, ivtmp.722, out_size
	str	w3, [sp, 132]	// ivtmp.722, %sfp
	add	x2, x2, 4	// ivtmp.720, ivtmp.720,
	ldr	x3, [sp, 296]	// _411, %sfp
	str	x2, [sp, 200]	// ivtmp.720, %sfp
	add	x16, x16, x3	// ivtmp.724, ivtmp.724, _411
	ldr	x3, [sp, 160]	// ivtmp.726, %sfp
	add	x3, x3, x14	// ivtmp.726, ivtmp.726, _209
	str	x3, [sp, 160]	// ivtmp.726, %sfp
	cmp	x2, x5	// ivtmp.720, _407
	bne	.L164		//,
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	ldr	x4, [sp, 176]	// ivtmp.733, %sfp
	mov	x6, x13	// _251, _251
	mov	w2, w1	// ivtmp.715, ivtmp.715
	mov	x3, x5	// _407, _407
	mov	w16, w17	// _222, _222
	ldr	x13, [sp, 304]	// _176, %sfp
	mov	w17, w21	// padding, padding
	mov	x21, x19	// _129, _129
	mov	w19, w9	// _370, _370
	ldp	x12, x9, [sp, 320]	// ivtmp.732, _267,
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	ldr	w1, [sp, 216]	//, %sfp
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	add	x4, x4, x13	// ivtmp.733, ivtmp.733, _176
	add	x12, x12, x9	// ivtmp.732, ivtmp.732, _267
	ldr	w10, [sp, 312]	//, %sfp
	str	x4, [sp, 176]	// ivtmp.733, %sfp
	ldr	w5, [sp, 316]	//, %sfp
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	add	w1, w1, 1	// b, b,
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	ldr	w4, [sp, 148]	//, %sfp
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	str	w1, [sp, 216]	// b, %sfp
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	sub	w5, w5, w10	// ivtmp.731, ivtmp.731, out_ch_size
	cmp	w4, w1	// input_batches, b
	bgt	.L176		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x26, .L217	// iftmp.18_56,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	ldr	x1, [sp, 136]	//, %sfp
	mov	x0, x26	//, iftmp.18_56
	ldp	x27, x28, [sp, 80]	//,,
	.cfi_remember_state
	.cfi_restore 28
	.cfi_restore 27
	ldp	x25, x26, [sp, 64]	//,,
	.cfi_restore 26
	.cfi_restore 25
// src/cpp/cnn_internals.cpp:243: }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x29, x30, [sp], 384	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	b	_ZdlPvm		//
	.p2align 2,,3
.L156:
	.cfi_restore_state
	add	x15, x22, x15	// ivtmp.657, _108, _331
	.p2align 5,,15
.L162:
// src/cpp/cnn_internals.cpp:231:                                 acc_ptr[ow] += input_ptr[in_row_offset + iw] * w_val;
	ldr	s5, [x26, x6, lsl 2]	//, MEM[(float *)iftmp.18_563 + ivtmp.654_247 * 4]
	ldr	s6, [x15]	//, MEM[(const float *)_309]
// src/cpp/cnn_internals.cpp:228:                             for (int ow = ow_lo; ow < ow_hi; ++ow) {
	add	x15, x15, x24	// ivtmp.657, ivtmp.657, _463
// src/cpp/cnn_internals.cpp:231:                                 acc_ptr[ow] += input_ptr[in_row_offset + iw] * w_val;
	fmadd	s5, s1, s6, s5	//,,,
	str	s5, [x26, x6, lsl 2]	//, MEM[(float *)iftmp.18_563 + ivtmp.654_247 * 4]
// src/cpp/cnn_internals.cpp:228:                             for (int ow = ow_lo; ow < ow_hi; ++ow) {
	add	x6, x6, 1	// ivtmp.654, ivtmp.654,
	cmp	w4, w6	// _3, ivtmp.654
	bgt	.L162		//,
	b	.L160		//
.L184:
	mov	w4, 0	// niters_vector_mult_vf.601,
	b	.L157		//
.L141:
// src/cpp/cnn_internals.cpp:211:                 for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w7, 0	// in_channels,
	bgt	.L142		//,
.L146:
// src/cpp/cnn_internals.cpp:236:                 int out_row = out_b_ch_size + out_oc_size + oh * output_w;
	ldr	w2, [sp, 132]	//, %sfp
	add	w2, w2, w1	// _39, ivtmp.722, ivtmp.713
	b	.L148		//
.L216:
	ldr	x3, [sp, 152]	// _132, %sfp
	ldr	w2, [sp, 132]	//, %sfp
	add	w2, w2, w1	// _214, ivtmp.722, ivtmp.713
	add	x2, x16, w2, sxtw	// _233, ivtmp.724, _214
	add	x5, x3, x2, lsl 2	// _231, _132, _233,
	add	x3, x26, 4	// _158, iftmp.18_56,
	sub	x3, x5, x3	// _323, _231, _158
	cmp	x3, 24	// _323,
	bhi	.L143		//,
.L166:
// src/cpp/cnn_internals.cpp:228:                             for (int ow = ow_lo; ow < ow_hi; ++ow) {
	mov	x2, 0	// ivtmp.640,
	ldr	x3, [sp, 152]	// _132, %sfp
	add	x3, x3, w1, sxtw 2	// _256, _132, ivtmp.713,
.L172:
// src/cpp/cnn_internals.cpp:238:                     out_ptr[out_row + ow] = acc_ptr[ow];
	ldr	s29, [x26, x2]	// tmp1000, MEM[(float *)iftmp.18_563 + ivtmp.640_260 * 1]
// src/cpp/cnn_internals.cpp:237:                 for (int ow = 0; ow < output_w; ++ow) {
	ldr	x5, [sp, 248]	// _254, %sfp
// src/cpp/cnn_internals.cpp:238:                     out_ptr[out_row + ow] = acc_ptr[ow];
	str	s29, [x3, x2]	// tmp1000, MEM[(float *)_256 + ivtmp.640_260 * 1]
// src/cpp/cnn_internals.cpp:237:                 for (int ow = 0; ow < output_w; ++ow) {
	add	x2, x2, 4	// ivtmp.640, ivtmp.640,
	cmp	x5, x2	// _254, ivtmp.640
	bne	.L172		//,
	b	.L149		//
.L182:
// src/cpp/cnn_internals.cpp:205:                 for (int ow = 0; ow < output_w; ++ow) acc_ptr[ow] = b_val;
	mov	w2, 0	// _202,
	b	.L139		//
.L215:
// src/cpp/cnn_internals.cpp:211:                 for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w7, 0	// in_channels,
	ble	.L149		//,
	cmp	w12, w10	// _280, _283
	bge	.L149		//,
	ldr	w2, [sp, 144]	//, %sfp
	cmp	w2, 0	// kernel_w,
	bgt	.L177		//,
	b	.L149		//
.L185:
// src/cpp/cnn_internals.cpp:228:                             for (int ow = ow_lo; ow < ow_hi; ++ow) {
	mov	w3, 0	// niters_vector_mult_vf.584,
	b	.L167		//
.L217:
	ldp	x25, x26, [sp, 64]	//,,
	.cfi_restore 26
	.cfi_restore 25
	ldp	x27, x28, [sp, 80]	//,,
	.cfi_restore 28
	.cfi_restore 27
.L129:
// src/cpp/cnn_internals.cpp:243: }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x29, x30, [sp], 384	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
.L135:
	.cfi_def_cfa_offset 384
	.cfi_offset 19, -368
	.cfi_offset 20, -360
	.cfi_offset 21, -352
	.cfi_offset 22, -344
	.cfi_offset 23, -336
	.cfi_offset 24, -328
	.cfi_offset 25, -320
	.cfi_offset 26, -312
	.cfi_offset 29, -384
	.cfi_offset 30, -376
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	ldr	w0, [sp, 148]	//, %sfp
	cmp	w0, 0	// input_batches,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	mov	x0, 4	// _145,
	str	x0, [sp, 136]	// _145, %sfp
// src/cpp/cnn_internals.cpp:195:     for (int b = 0; b < input_batches; ++b) {
	ble	.L137		//,
	cmp	w23, 0	// out_channels,
	ble	.L137		//,
	ldr	w0, [sp, 184]	//, %sfp
	cmp	w0, 0	// out_size,
	bgt	.L133		//,
	b	.L137		//
.L214:
	ldp	x25, x26, [sp, 64]	//,,
	.cfi_restore 26
	.cfi_restore 25
	b	.L129		//
.L212:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1920: 	  __throw_length_error(
	adrp	x0, .LC0	// tmp374,
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 26, -312
	.cfi_offset 25, -320
	add	x0, x0, :lo12:.LC0	//, tmp374,
	stp	x27, x28, [sp, 80]	//,,
	.cfi_offset 28, -296
	.cfi_offset 27, -304
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1920: 	  __throw_length_error(
	bl	_ZSt20__throw_length_errorPKc		//
	.cfi_endproc
.LFE5447:
	.size	_Z22conv2d_forward_reorderRK6TensorS1_S1_RS_ii, .-_Z22conv2d_forward_reorderRK6TensorS1_S1_RS_ii
	.align	2
	.p2align 5,,15
	.global	_Z26conv2d_forward_specializedRK6TensorS1_S1_RS_
	.type	_Z26conv2d_forward_specializedRK6TensorS1_S1_RS_, %function
_Z26conv2d_forward_specializedRK6TensorS1_S1_RS_:
.LFB5448:
	.cfi_startproc
	sub	sp, sp, #528	//,,
	.cfi_def_cfa_offset 528
	mov	x5, x0	// input, tmp499
	mov	x0, x1	// weight, tmp500
	stp	x29, x30, [sp]	//,,
	.cfi_offset 29, -528
	.cfi_offset 30, -520
	mov	x29, sp	//,
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 23, -480
	.cfi_offset 24, -472
// src/cpp/cnn_internals.cpp:259:     int input_batches = input.batches;
	ldr	w23, [x5, 24]	//, input_45(D)->batches
// src/cpp/cnn_internals.cpp:251:     int out_channels = weight.batches;
	ldr	w6, [x1, 24]	//, weight_48(D)->batches
// src/cpp/cnn_internals.cpp:271:     for (int b = 0; b < input_batches; ++b) {
	cmp	w23, 0	// input_batches,
	ble	.L218		//,
	cmp	w6, 0	// out_channels,
	ble	.L218		//,
// src/cpp/cnn_internals.cpp:257:     int output_h     = output.height;
	ldr	w8, [x3, 32]	//, output_52(D)->height
	mov	x1, x3	// output, tmp502
	cmp	w8, 0	// output_h,
	ble	.L218		//,
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 20, -504
	.cfi_offset 19, -512
	mov	x7, x2	// bias, tmp501
// src/cpp/cnn_internals.cpp:304:                     out_ptr[out_row + ow] = acc[ow];
	mov	x11, 0	// ivtmp.910,
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 22, -488
	.cfi_offset 21, -496
	mov	w22, 0	// ivtmp.909,
// src/cpp/cnn_internals.cpp:271:     for (int b = 0; b < input_batches; ++b) {
	mov	w21, 0	// b,
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 26, -456
	.cfi_offset 25, -464
	stp	x27, x28, [sp, 80]	//,,
	.cfi_offset 28, -440
	.cfi_offset 27, -448
// src/cpp/cnn_internals.cpp:254:     int kernel_w     = weight.width;
	ldr	w15, [x0, 36]	//, weight_48(D)->width
// src/cpp/cnn_internals.cpp:258:     int output_w     = output.width;
	ldr	w25, [x1, 36]	//, output_52(D)->width
// src/cpp/cnn_internals.cpp:252:     int in_channels  = weight.channels;
	ldp	w4, w18, [x0, 28]	// in_channels, kernel_h,
	sub	w2, w15, #3	// _566, kernel_w,
	sxtw	x19, w15	// _310, kernel_w
// src/cpp/cnn_internals.cpp:256:     int input_w      = input.width;
	ldp	w3, w10, [x5, 32]	// input_45(D)->height, input_w,
	lsr	w2, w2, 1	// _564, _566,
	sbfiz	x14, x15, 2, 32	// _102, kernel_w,,
	add	w2, w2, 2	// _561, _564,
	mov	x13, x14	// _102, _102
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1263:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x1, [x1]	// _53, MEM[(struct vector *)output_52(D)].D.54862._M_impl.D.54163._M_start
	lsl	x20, x2, 1	// _560, _561,
// src/cpp/cnn_internals.cpp:304:                     out_ptr[out_row + ow] = acc[ow];
	sub	w2, w25, #1	// _146, output_w,
// src/cpp/cnn_internals.cpp:265:     int weight_size = kernel_h * kernel_w;
	mul	w9, w18, w15	// weight_size, kernel_h, kernel_w
// src/cpp/cnn_internals.cpp:304:                     out_ptr[out_row + ow] = acc[ow];
	add	x2, x2, 1	// _139, _146,
	mov	x14, x19	// _310, _310
	lsl	x2, x2, 2	// _147, _139,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x24, [x0]	// _49, MEM[(const struct vector *)weight_48(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:263:     int out_size = output_h * output_w;
	mul	w0, w8, w25	// out_size, output_h, output_w
	mov	w19, w18	// kernel_h, kernel_h
// src/cpp/cnn_internals.cpp:261:     int in_size = input_h * input_w;
	mul	w3, w10, w3	// in_size, input_w, input_45(D)->height
	mov	w18, w15	// kernel_w, kernel_w
	sub	w28, w25, #1	// _451, output_w,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x26, [x5]	// _47, MEM[(const struct vector *)input_45(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:266:     int weight_ch_size = in_channels * weight_size;
	mul	w5, w4, w9	// weight_ch_size, in_channels, weight_size
	and	w27, w25, -4	// niters_vector_mult_vf.750, output_w,
// src/cpp/cnn_internals.cpp:304:                     out_ptr[out_row + ow] = acc[ow];
	str	x2, [sp, 176]	// _147, %sfp
// src/cpp/cnn_internals.cpp:261:     int in_size = input_h * input_w;
	stp	w3, w9, [sp, 168]	// in_size, weight_size,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x2, [x7]	// _51, MEM[(const struct vector *)bias_50(D)].D.54862._M_impl.D.54163._M_start
	sxtw	x5, w5	// _516, weight_ch_size
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1263:       { return _M_data_ptr(this->_M_impl._M_start); }
	str	x1, [sp, 160]	// _53, %sfp
// src/cpp/cnn_internals.cpp:262:     int in_ch_size = in_channels * in_size;
	mul	w1, w4, w3	// in_ch_size_64, in_channels, in_size
// src/cpp/cnn_internals.cpp:264:     int out_ch_size = out_channels * out_size;
	mul	w3, w6, w0	// _246, out_channels, out_size
	add	x6, x2, w6, uxtw 2	// _190, _51, out_channels,
	sxtw	x7, w1	// _250, in_ch_size_64
	mov	x1, x7	// _250, _250
	mov	w7, w8	// output_h, output_h
	mov	w8, w23	// input_batches, input_batches
	mov	x23, x20	// _560, _560
	mov	x20, x2	// _51, _51
	mov	w2, w0	// out_size, out_size
.L263:
// src/cpp/cnn_internals.cpp:271:     for (int b = 0; b < input_batches; ++b) {
	mov	x0, x20	// ivtmp.901, _51
	mov	x12, x20	// _51, _51
	str	xzr, [sp, 136]	//, %sfp
	mov	w20, w18	// kernel_w, kernel_w
	mov	x18, x13	// _102, _102
	mov	x13, x1	// _250, _250
	mov	x1, x14	// _310, _310
	mov	w9, w22	// ivtmp.905, ivtmp.909
	mov	w16, w8	// input_batches, input_batches
	mov	w14, w3	// _246, _246
	mov	w8, w21	// b, b
	mov	w3, w22	// ivtmp.909, ivtmp.909
	mov	w15, w4	// in_channels, in_channels
	mov	w21, w19	// kernel_h, kernel_h
	mov	x22, x1	// _310, _310
.L221:
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	cmp	w25, 0	// output_w,
	ble	.L264		//,
	str	d15, [sp, 96]	//,
	.cfi_offset 79, -432
// src/cpp/cnn_internals.cpp:278:             const float b_val = bias_ptr[oc];
	mov	w4, 0	// ivtmp.897,
	add	x19, sp, 272	// tmp459,,
	stp	wzr, w7, [sp, 144]	//, output_h,
	stp	w16, w2, [sp, 192]	// input_batches, out_size,
	stp	w8, w9, [sp, 208]	// b, ivtmp.905,
	mov	x8, x0	// ivtmp.901, ivtmp.901
	stp	w3, w14, [sp, 216]	// ivtmp.909, _246,
	mov	x3, x5	// _516, _516
	mov	x5, x22	// _310, _310
	ldr	s15, [x0]	// b_val, MEM[(const float *)_212]
	mov	x22, x18	// _102, _102
	str	w9, [sp, 132]	// ivtmp.905, %sfp
	str	x11, [sp, 152]	// ivtmp.910, %sfp
	str	x12, [sp, 184]	// _51, %sfp
	str	x6, [sp, 200]	// _190, %sfp
	str	x13, [sp, 224]	// _250, %sfp
.L261:
	cmp	w28, 7	// _451,
	bls	.L357		//,
	dup	v31.4s, v15.s[0]	//, b_val
	lsr	w0, w25, 3	// bnd.815, output_w,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	stp	q31, q31, [sp, 272]	//,,
	cmp	w0, 1	// bnd.815,
	beq	.L266		//,
	stp	q31, q31, [sp, 304]	//,,
	cmp	w0, 2	// bnd.815,
	beq	.L229		//,
	stp	q31, q31, [sp, 336]	//,,
	cmp	w0, 3	// bnd.815,
	beq	.L229		//,
	stp	q31, q31, [sp, 368]	//,,
	cmp	w0, 4	// bnd.815,
	beq	.L229		//,
	stp	q31, q31, [sp, 400]	//,,
	cmp	w0, 5	// bnd.815,
	beq	.L229		//,
	stp	q31, q31, [sp, 432]	//,,
	cmp	w0, 6	// bnd.815,
	beq	.L229		//,
	stp	q31, q31, [sp, 464]	//,,
	cmp	w0, 7	// bnd.815,
	beq	.L229		//,
	stp	q31, q31, [sp, 496]	//,,
.L229:
	and	w0, w25, -8	// tmp.817, output_w,
	cmp	w25, w0	// output_w, tmp.817
	beq	.L230		//,
.L228:
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	add	w1, w0, 1	// ow, tmp.817,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	str	s15, [x19, w0, sxtw 2]	// b_val, acc[tmp.817_525]
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	cmp	w25, w1	// output_w, ow
	ble	.L230		//,
.L225:
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	str	s15, [x19, w1, sxtw 2]	// b_val, acc[ow_202]
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	add	w1, w0, 2	// ow, tmp.817,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	cmp	w25, w1	// output_w, ow
	ble	.L230		//,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	str	s15, [x19, w1, sxtw 2]	// b_val, acc[ow_182]
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	add	w1, w0, 3	// ow, tmp.817,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	cmp	w25, w1	// output_w, ow
	ble	.L230		//,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	str	s15, [x19, w1, sxtw 2]	// b_val, acc[ow_108]
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	add	w1, w0, 4	// ow, tmp.817,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	cmp	w25, w1	// output_w, ow
	ble	.L230		//,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	str	s15, [x19, w1, sxtw 2]	// b_val, acc[ow_461]
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	add	w1, w0, 5	// ow, tmp.817,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	cmp	w25, w1	// output_w, ow
	ble	.L230		//,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	str	s15, [x19, w1, sxtw 2]	// b_val, acc[ow_519]
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	add	w1, w0, 6	// ow, tmp.817,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	cmp	w25, w1	// output_w, ow
	ble	.L230		//,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	add	w0, w0, 7	// ow, tmp.817,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	str	s15, [x19, w1, sxtw 2]	// b_val, acc[ow_588]
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	cmp	w25, w0	// output_w, ow
	ble	.L230		//,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	str	s15, [x19, w0, sxtw 2]	// b_val, acc[ow_243]
.L230:
// src/cpp/cnn_internals.cpp:283:                 for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w15, 0	// in_channels,
	ble	.L227		//,
	cmp	w21, 0	// kernel_h,
	ble	.L227		//,
	cmp	w20, 0	// kernel_w,
	ble	.L227		//,
	ldpsw	x0, x16, [sp, 168]	// _256, _465,
	mov	x18, x8	// ivtmp.901, ivtmp.901
	mov	x11, x3	// _516, _516
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ptrue	p6.b, all	// tmp480
// src/cpp/cnn_internals.cpp:283:                 for (int ic = 0; ic < in_channels; ++ic) {
	mov	w9, 0	// ic,
	sxtw	x14, w10	// _327, input_w
	str	w10, [sp, 232]	// input_w, %sfp
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	ldr	x2, [sp, 136]	// ivtmp.888, %sfp
	sub	x30, x24, #4	// _173, _49,
	lsr	w7, w25, 2	// bnd.775_332, output_w,
	mov	w8, w15	// in_channels, in_channels
	ldr	x1, [sp, 152]	// ivtmp.910, %sfp
	add	x17, x1, w4, sxtw	// ivtmp.886, ivtmp.910, ivtmp.897
	mov	x3, x0	// _256, _256
	mov	x0, x16	// _465, _465
.L242:
	add	x10, x30, x2, lsl 2	// ivtmp.875, _173, ivtmp.888,
// src/cpp/cnn_internals.cpp:283:                 for (int ic = 0; ic < in_channels; ++ic) {
	mov	x1, x17	// ivtmp.870, ivtmp.886
	mov	w15, w2	// ivtmp.872, ivtmp.888
	stp	x17, x2, [sp, 240]	// ivtmp.886, ivtmp.888,
	mov	x13, x2	// ivtmp.874, ivtmp.888
// src/cpp/cnn_internals.cpp:287:                     for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w12, 0	// kh,
	mov	w16, w4	// ivtmp.897, ivtmp.897
	mov	x17, x0	// _465, _465
	str	w8, [sp, 256]	// in_channels, %sfp
	str	w9, [sp, 260]	// ic, %sfp
	str	x3, [sp, 264]	// _256, %sfp
	.p2align 5,,15
.L244:
// src/cpp/cnn_internals.cpp:294:                             float w_val = weight_ptr[w_idx];
	mov	w2, 0	// kw,
	cmp	w20, 2	// kernel_w,
	ble	.L248		//,
	add	x8, x26, x1, lsl 2	// ivtmp.859, _47, ivtmp.870,
// src/cpp/cnn_internals.cpp:287:                     for (int kh = 0; kh < kernel_h; ++kh) {
	mov	x4, 2	// ivtmp.852,
	sub	x9, x10, #4	// _330, ivtmp.875,
// src/cpp/cnn_internals.cpp:293:                             int w_idx  = out_oc_ch_size + ic_weight_size + kh_kernel_w + kw;
	stp	w21, w20, [sp, 124]	// kernel_h, kernel_w,
	sub	x6, x1, #2	// _137, ivtmp.870,
	sub	w3, w15, #1	// _326, ivtmp.872,
	.p2align 5,,15
.L253:
// src/cpp/cnn_internals.cpp:294:                             float w_val = weight_ptr[w_idx];
	ldr	s24, [x9, x4, lsl 2]	// w_val, MEM[(const float *)_571 + ivtmp.852_595 * 4]
	cmp	w28, 2	// _451,
	bls	.L267		//,
	str	w27, [sp, 120]	// niters_vector_mult_vf.776, %sfp
.L250:
	ldr	s29, [x10, x4, lsl 2]	// D__lsm0.825, MEM[(const float *)_158 + ivtmp.852_338 * 4]
	dup	v30.4s, v24.s[0]	//, w_val
	add	x20, x8, 4	// ivtmp.844, ivtmp.859,
	mov	x0, x19	// ivtmp.839, tmp459
	add	x21, x8, w7, uxtw 4	// _596, ivtmp.859, bnd.775_332,
	mov	x2, x8	// ivtmp.842, ivtmp.859
	dup	v29.4s, v29.s[0]	//, D__lsm0.825
	.p2align 5,,15
.L259:
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q0, [x2], 16	//, MEM <const vector(4) float> [(const float *)_463]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q27, [x0]	//, MEM <vector(4) float> [(float *)_464]
	fmla	v27.4s, v0.4s, v30.4s	//,,
	str	q27, [x0]	//, MEM <vector(4) float> [(float *)_464]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q28, [x20], 16	//, MEM <const vector(4) float> [(const float *)_458]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	fmla	v27.4s, v29.4s, v28.4s	//,,
	str	q27, [x0], 16	//, MEM <vector(4) float> [(float *)_464]
	cmp	x2, x21	// ivtmp.842, _596
	bne	.L259		//,
	cmp	w27, w25	// niters_vector_mult_vf.750, output_w
	beq	.L358		//,
	ldr	w0, [sp, 120]	//, %sfp
	mov	x2, x4	// ivtmp.852, ivtmp.852
.L249:
	uxtw	x21, w0	// _385, niters_vector_mult_vf.776
	sub	w0, w25, w0	// bnd.796_382, output_w, niters_vector_mult_vf.776
	whilelo	p7.s, wzr, w0	// max_mask_441,, bnd.796_382
	mov	z24.s, s24	//, w_val
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ld1w	z2.s, p7/z, [x19, x21, lsl 2]	//, max_mask_441,
	add	x20, x4, x21	// _134, ivtmp.852, _385
	add	x0, x19, x21, lsl 2	// vectp_acc.798, tmp459, _385,
// src/cpp/cnn_internals.cpp:293:                             int w_idx  = out_oc_ch_size + ic_weight_size + kh_kernel_w + kw;
	add	w21, w3, w4	// _602, _326, ivtmp.852
// src/cpp/cnn_internals.cpp:294:                             float w_val = weight_ptr[w_idx];
	add	x21, x24, w21, sxtw 2	// tmp388, _49, _602,
	add	x20, x20, x6	// _400, _134, _137
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ld1w	z1.s, p7/z, [x26, x20, lsl 2]	//, max_mask_441,* _47
	add	x4, x4, 2	// ivtmp.852, ivtmp.852,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	fmad	z24.s, p6/m, z1.s, z2.s	//, tmp480,,
	st1w	z24.s, p7, [x0]	//, max_mask_441,* vectp_acc.798
	add	x20, x26, x20, lsl 2	// _426, _47, _400,
// src/cpp/cnn_internals.cpp:294:                             float w_val = weight_ptr[w_idx];
	ld1rw	z25.s, p6/z, [x21]	//, tmp480, *_315
	add	x8, x8, 8	// ivtmp.859, ivtmp.859,
	add	x20, x20, 4	// vectp.808, _426,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ld1w	z26.s, p7/z, [x20]	//, max_mask_441,* vectp.808
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	fmad	z25.s, p6/m, z26.s, z24.s	//, tmp480,,
	st1w	z25.s, p7, [x0]	//, max_mask_441,* vectp_acc.798
	cmp	x23, x4	// _560, ivtmp.852
	bne	.L253		//,
// src/cpp/cnn_internals.cpp:292:                         for (int kw = 0; kw < kernel_w; ++kw) {
	ldp	w21, w20, [sp, 124]	//,,
.L248:
	sxtw	x0, w2	// ivtmp.830, kw
	add	x8, x24, x13, lsl 2	// _322, _49, ivtmp.874,
	add	x2, x0, x1	// _18, ivtmp.830, ivtmp.870
	add	x2, x26, x2, lsl 2	// ivtmp.835, _47, _18,
.L258:
// src/cpp/cnn_internals.cpp:294:                             float w_val = weight_ptr[w_idx];
	ldr	s23, [x8, x0, lsl 2]	//, MEM[(const float *)_322 + ivtmp.830_161 * 4]
	cmp	w28, 2	// _451,
	bls	.L269		//,
	mov	w3, w27	// niters_vector_mult_vf.750, niters_vector_mult_vf.750
	ldr	q22, [sp, 272]	//, MEM <vector(4) float> [(float *)&acc]
.L257:
	dup	v21.4s, v23.s[0]	//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q0, [x2]	//, MEM <const vector(4) float> [(const float *)_301]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	fmla	v22.4s, v0.4s, v23.s[0]	//,,
	str	q22, [sp, 272]	//, MEM <vector(4) float> [(float *)&acc]
	cmp	w7, 1	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q31, [x2, 16]	//, MEM <const vector(4) float> [(const float *)_301 + 16B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q18, [sp, 288]	//, MEM <vector(4) float> [(float *)&acc + 16B]
	fmla	v18.4s, v31.4s, v21.4s	//,,
	str	q18, [sp, 288]	//, MEM <vector(4) float> [(float *)&acc + 16B]
	cmp	w7, 2	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q17, [x2, 32]	//, MEM <const vector(4) float> [(const float *)_301 + 32B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q16, [sp, 304]	//, MEM <vector(4) float> [(float *)&acc + 32B]
	fmla	v16.4s, v17.4s, v21.4s	//,,
	str	q16, [sp, 304]	//, MEM <vector(4) float> [(float *)&acc + 32B]
	cmp	w7, 3	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q7, [x2, 48]	//, MEM <const vector(4) float> [(const float *)_301 + 48B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q6, [sp, 320]	//, MEM <vector(4) float> [(float *)&acc + 48B]
	fmla	v6.4s, v7.4s, v21.4s	//,,
	str	q6, [sp, 320]	//, MEM <vector(4) float> [(float *)&acc + 48B]
	cmp	w7, 4	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q5, [x2, 64]	//, MEM <const vector(4) float> [(const float *)_301 + 64B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q4, [sp, 336]	//, MEM <vector(4) float> [(float *)&acc + 64B]
	fmla	v4.4s, v5.4s, v21.4s	//,,
	str	q4, [sp, 336]	//, MEM <vector(4) float> [(float *)&acc + 64B]
	cmp	w7, 5	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q3, [x2, 80]	//, MEM <const vector(4) float> [(const float *)_301 + 80B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q20, [sp, 352]	//, MEM <vector(4) float> [(float *)&acc + 80B]
	fmla	v20.4s, v21.4s, v3.4s	//,,
	str	q20, [sp, 352]	//, MEM <vector(4) float> [(float *)&acc + 80B]
	cmp	w7, 6	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q19, [x2, 96]	//, MEM <const vector(4) float> [(const float *)_301 + 96B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q26, [sp, 368]	//, MEM <vector(4) float> [(float *)&acc + 96B]
	fmla	v26.4s, v21.4s, v19.4s	//,,
	str	q26, [sp, 368]	//, MEM <vector(4) float> [(float *)&acc + 96B]
	cmp	w7, 7	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q25, [x2, 112]	//, MEM <const vector(4) float> [(const float *)_301 + 112B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q24, [sp, 384]	//, MEM <vector(4) float> [(float *)&acc + 112B]
	fmla	v24.4s, v21.4s, v25.4s	//,,
	str	q24, [sp, 384]	//, MEM <vector(4) float> [(float *)&acc + 112B]
	cmp	w7, 8	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q2, [x2, 128]	//, MEM <const vector(4) float> [(const float *)_301 + 128B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q1, [sp, 400]	//, MEM <vector(4) float> [(float *)&acc + 128B]
	fmla	v1.4s, v21.4s, v2.4s	//,,
	str	q1, [sp, 400]	//, MEM <vector(4) float> [(float *)&acc + 128B]
	cmp	w7, 9	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q30, [x2, 144]	//, MEM <const vector(4) float> [(const float *)_301 + 144B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q29, [sp, 416]	//, MEM <vector(4) float> [(float *)&acc + 144B]
	fmla	v29.4s, v21.4s, v30.4s	//,,
	str	q29, [sp, 416]	//, MEM <vector(4) float> [(float *)&acc + 144B]
	cmp	w7, 10	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q28, [x2, 160]	//, MEM <const vector(4) float> [(const float *)_301 + 160B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q27, [sp, 432]	//, MEM <vector(4) float> [(float *)&acc + 160B]
	fmla	v27.4s, v28.4s, v21.4s	//,,
	str	q27, [sp, 432]	//, MEM <vector(4) float> [(float *)&acc + 160B]
	cmp	w7, 11	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q0, [x2, 176]	//, MEM <const vector(4) float> [(const float *)_301 + 176B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q31, [sp, 448]	//, MEM <vector(4) float> [(float *)&acc + 176B]
	fmla	v31.4s, v21.4s, v0.4s	//,,
	str	q31, [sp, 448]	//, MEM <vector(4) float> [(float *)&acc + 176B]
	cmp	w7, 12	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q18, [x2, 192]	//, MEM <const vector(4) float> [(const float *)_301 + 192B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q17, [sp, 464]	//, MEM <vector(4) float> [(float *)&acc + 192B]
	fmla	v17.4s, v21.4s, v18.4s	//,,
	str	q17, [sp, 464]	//, MEM <vector(4) float> [(float *)&acc + 192B]
	cmp	w7, 13	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q16, [x2, 208]	//, MEM <const vector(4) float> [(const float *)_301 + 208B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q7, [sp, 480]	//, MEM <vector(4) float> [(float *)&acc + 208B]
	fmla	v7.4s, v21.4s, v16.4s	//,,
	str	q7, [sp, 480]	//, MEM <vector(4) float> [(float *)&acc + 208B]
	cmp	w7, 14	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q6, [x2, 224]	//, MEM <const vector(4) float> [(const float *)_301 + 224B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q5, [sp, 496]	//, MEM <vector(4) float> [(float *)&acc + 224B]
	fmla	v5.4s, v21.4s, v6.4s	//,,
	str	q5, [sp, 496]	//, MEM <vector(4) float> [(float *)&acc + 224B]
	cmp	w7, 15	// bnd.775_332,
	beq	.L255		//,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q4, [x2, 240]	//, MEM <const vector(4) float> [(const float *)_301 + 240B]
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ldr	q3, [sp, 512]	//, MEM <vector(4) float> [(float *)&acc + 240B]
	fmla	v3.4s, v4.4s, v21.4s	//,,
	str	q3, [sp, 512]	//, MEM <vector(4) float> [(float *)&acc + 240B]
	.p2align 5,,15
.L255:
	cmp	w27, w25	// niters_vector_mult_vf.750, output_w
	beq	.L359		//,
.L254:
	uxtw	x4, w3	// _220, niters_vector_mult_vf.750
	add	x6, x1, x0	// _233, ivtmp.870, ivtmp.830
	mov	z23.s, s23	//,
	sub	w3, w25, w3	// bnd.763_217, output_w, niters_vector_mult_vf.750
	add	x6, x6, x4	// _235, _233, _220
	whilelo	p7.s, wzr, w3	// max_mask_252,, bnd.763_217
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ld1w	z20.s, p7/z, [x26, x6, lsl 2]	//, max_mask_252,* _47
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	ld1w	z19.s, p7/z, [x19, x4, lsl 2]	//, max_mask_252,
// src/cpp/cnn_internals.cpp:292:                         for (int kw = 0; kw < kernel_w; ++kw) {
	add	x0, x0, 1	// ivtmp.830, ivtmp.830,
// src/cpp/cnn_internals.cpp:297:                                 acc[ow] += input_ptr[in_row_offset + ow + kw] * w_val;
	fmad	z23.s, p6/m, z20.s, z19.s	//, tmp480,,
	st1w	z23.s, p7, [x19, x4, lsl 2]	//, max_mask_252,
// src/cpp/cnn_internals.cpp:292:                         for (int kw = 0; kw < kernel_w; ++kw) {
	add	x2, x2, 4	// ivtmp.835, ivtmp.835,
	cmp	w20, w0	// kernel_w, ivtmp.830
	bgt	.L258		//,
.L256:
// src/cpp/cnn_internals.cpp:287:                     for (int kh = 0; kh < kernel_h; ++kh) {
	add	w12, w12, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:287:                     for (int kh = 0; kh < kernel_h; ++kh) {
	add	x1, x1, x14	// ivtmp.870, ivtmp.870, _327
	add	w15, w15, w20	// ivtmp.872, ivtmp.872, kernel_w
	add	x13, x13, x5	// ivtmp.874, ivtmp.874, _310
	add	x10, x10, x22	// ivtmp.875, ivtmp.875, _102
	cmp	w21, w12	// kernel_h, kh
	bne	.L244		//,
	ldr	w9, [sp, 260]	//, %sfp
	mov	x0, x17	// _465, _465
	mov	w4, w16	// ivtmp.897, ivtmp.897
	ldp	x17, x2, [sp, 240]	// ivtmp.886, ivtmp.888,
	ldr	x3, [sp, 264]	// _256, %sfp
// src/cpp/cnn_internals.cpp:283:                 for (int ic = 0; ic < in_channels; ++ic) {
	add	w9, w9, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:283:                 for (int ic = 0; ic < in_channels; ++ic) {
	add	x2, x2, x0	// ivtmp.888, ivtmp.888, _465
	ldr	w8, [sp, 256]	//, %sfp
	add	x17, x17, x3	// ivtmp.886, ivtmp.886, _256
	cmp	w8, w9	// in_channels, ic
	bgt	.L242		//,
	ldr	w10, [sp, 232]	//, %sfp
	mov	w15, w8	// in_channels, in_channels
	mov	x3, x11	// _516, _516
	mov	x8, x18	// ivtmp.901, ivtmp.901
.L227:
	str	x3, [sp, 248]	// _516, %sfp
// src/cpp/cnn_internals.cpp:304:                     out_ptr[out_row + ow] = acc[ow];
	mov	x1, x19	//, tmp459
	ldr	x3, [sp, 160]	// _53, %sfp
	stp	w15, w10, [sp, 120]	// in_channels, input_w,
	ldr	x2, [sp, 176]	//, %sfp
	str	w4, [sp, 128]	// ivtmp.897, %sfp
	ldr	w0, [sp, 132]	//, %sfp
	add	x0, x3, w0, sxtw 2	//, _53, ivtmp.896,
	stp	x5, x8, [sp, 232]	// _310, ivtmp.901,
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	ldp	w4, w0, [sp, 128]	//,,
	ldp	x5, x8, [sp, 232]	// _310, ivtmp.901,
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	ldr	w1, [sp, 144]	//, %sfp
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	add	w0, w0, w25	// ivtmp.896, ivtmp.896, output_w
	ldp	w15, w10, [sp, 120]	//,,
	str	w0, [sp, 132]	// ivtmp.896, %sfp
	ldr	w0, [sp, 148]	//, %sfp
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	add	w1, w1, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	ldr	x3, [sp, 248]	// _516, %sfp
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	str	w1, [sp, 144]	// oh, %sfp
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	add	w4, w4, w10	// ivtmp.897, ivtmp.897, input_w
	cmp	w0, w1	// output_h, oh
	bne	.L261		//,
	ldr	d15, [sp, 96]	//,
	.cfi_restore 79
	mov	x7, x0	//,
	mov	x18, x22	// _102, _102
	mov	x0, x8	// ivtmp.901, ivtmp.901
	mov	x22, x5	// _310, _310
	ldr	x11, [sp, 152]	// ivtmp.910, %sfp
	mov	x5, x3	// _516, _516
	ldr	x12, [sp, 184]	// _51, %sfp
	ldr	x6, [sp, 200]	// _190, %sfp
	ldr	x13, [sp, 224]	// _250, %sfp
	ldp	w16, w2, [sp, 192]	//,,
	ldp	w8, w9, [sp, 208]	//,,
	ldp	w3, w14, [sp, 216]	//,,
.L237:
// src/cpp/cnn_internals.cpp:275:         for (int oc = 0; oc < out_channels; ++oc) {
	add	x0, x0, 4	// ivtmp.901, ivtmp.901,
	add	w9, w9, w2	// ivtmp.905, ivtmp.905, out_size
	ldr	x1, [sp, 136]	// ivtmp.904, %sfp
	add	x1, x1, x5	// ivtmp.904, ivtmp.904, _516
	str	x1, [sp, 136]	// ivtmp.904, %sfp
	cmp	x6, x0	// _190, ivtmp.901
	bne	.L221		//,
	mov	w19, w21	// kernel_h, kernel_h
	mov	w21, w8	// b, b
	mov	x1, x13	// _250, _250
	mov	x0, x22	// _310, _310
// src/cpp/cnn_internals.cpp:271:     for (int b = 0; b < input_batches; ++b) {
	add	w21, w21, 1	// b, b,
	mov	w22, w3	// ivtmp.909, ivtmp.909
	mov	w3, w14	// _246, _246
	mov	x13, x18	// _102, _102
	mov	w4, w15	// in_channels, in_channels
	mov	w18, w20	// kernel_w, kernel_w
	mov	w8, w16	// input_batches, input_batches
	mov	x20, x12	// _51, _51
	mov	x14, x0	// _310, _310
// src/cpp/cnn_internals.cpp:271:     for (int b = 0; b < input_batches; ++b) {
	add	w22, w22, w3	// ivtmp.909, ivtmp.909, _246
	add	x11, x11, x1	// ivtmp.910, ivtmp.910, _250
	cmp	w16, w21	// input_batches, b
	bne	.L263		//,
	ldp	x19, x20, [sp, 16]	//,,
	.cfi_restore 20
	.cfi_restore 19
	ldp	x21, x22, [sp, 32]	//,,
	.cfi_restore 22
	.cfi_restore 21
	ldp	x25, x26, [sp, 64]	//,,
	.cfi_restore 26
	.cfi_restore 25
	ldp	x27, x28, [sp, 80]	//,,
	.cfi_restore 28
	.cfi_restore 27
.L218:
// src/cpp/cnn_internals.cpp:309: }
	ldp	x29, x30, [sp]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	add	sp, sp, 528	//,,
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 29
	.cfi_restore 30
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L358:
	.cfi_def_cfa_offset 528
	.cfi_offset 19, -512
	.cfi_offset 20, -504
	.cfi_offset 21, -496
	.cfi_offset 22, -488
	.cfi_offset 23, -480
	.cfi_offset 24, -472
	.cfi_offset 25, -464
	.cfi_offset 26, -456
	.cfi_offset 27, -448
	.cfi_offset 28, -440
	.cfi_offset 29, -528
	.cfi_offset 30, -520
	.cfi_offset 79, -432
	add	x0, x4, 2	// ivtmp.852, ivtmp.852,
	add	x8, x8, 8	// ivtmp.859, ivtmp.859,
	cmp	x0, x23	// ivtmp.852, _560
	beq	.L360		//,
// src/cpp/cnn_internals.cpp:294:                             float w_val = weight_ptr[w_idx];
	ldr	s24, [x9, x0, lsl 2]	// w_val, MEM[(const float *)_556 + ivtmp.852_345 * 4]
	mov	x4, x0	// ivtmp.852, ivtmp.852
	b	.L250		//
.L267:
	mov	x2, x4	// ivtmp.852, ivtmp.852
	mov	w0, 0	// niters_vector_mult_vf.776,
	b	.L249		//
.L359:
// src/cpp/cnn_internals.cpp:292:                         for (int kw = 0; kw < kernel_w; ++kw) {
	add	x0, x0, 1	// ivtmp.830, ivtmp.830,
	cmp	w20, w0	// kernel_w, ivtmp.830
	ble	.L256		//,
// src/cpp/cnn_internals.cpp:294:                             float w_val = weight_ptr[w_idx];
	ldr	s23, [x8, x0, lsl 2]	//, MEM[(const float *)_322 + ivtmp.830_23 * 4]
// src/cpp/cnn_internals.cpp:292:                         for (int kw = 0; kw < kernel_w; ++kw) {
	add	x2, x2, 4	// ivtmp.835, ivtmp.835,
	b	.L257		//
.L269:
// src/cpp/cnn_internals.cpp:294:                             float w_val = weight_ptr[w_idx];
	mov	w3, 0	// niters_vector_mult_vf.750,
	b	.L254		//
.L360:
// src/cpp/cnn_internals.cpp:292:                         for (int kw = 0; kw < kernel_w; ++kw) {
	ldp	w21, w20, [sp, 124]	//,,
	mov	w2, w4	// kw, ivtmp.852
	b	.L248		//
.L266:
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	mov	w0, 8	// tmp.817,
	b	.L228		//
.L264:
	.cfi_restore 79
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	mov	w1, 0	// oh,
.L223:
// src/cpp/cnn_internals.cpp:283:                 for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w15, 0	// in_channels,
	bgt	.L246		//,
	b	.L361		//
.L238:
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	add	w1, w1, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	cmp	w7, w1	// output_h, oh
	beq	.L237		//,
.L246:
	cmp	w21, 0	// kernel_h,
	ble	.L237		//,
.L245:
	cmp	w20, 0	// kernel_w,
	bgt	.L238		//,
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	add	w1, w1, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	cmp	w7, w1	// output_h, oh
	bne	.L245		//,
	b	.L237		//
.L361:
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	add	w1, w1, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:280:             for (int oh = 0; oh < output_h; ++oh) {
	cmp	w7, w1	// output_h, oh
	bne	.L223		//,
	b	.L237		//
.L357:
	.cfi_offset 79, -432
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	mov	w1, 1	// ow,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	str	s15, [sp, 272]	// b_val, acc[0]
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	mov	w0, 0	// tmp.817,
// src/cpp/cnn_internals.cpp:281:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = b_val;
	cmp	w25, w1	// output_w,
	bne	.L225		//,
	b	.L230		//
	.cfi_endproc
.LFE5448:
	.size	_Z26conv2d_forward_specializedRK6TensorS1_S1_RS_, .-_Z26conv2d_forward_specializedRK6TensorS1_S1_RS_
	.align	2
	.p2align 5,,15
	.global	_Z12relu_forwardR6Tensor
	.type	_Z12relu_forwardR6Tensor, %function
_Z12relu_forwardR6Tensor:
.LFB5472:
	.cfi_startproc
// src/cpp/cnn_internals.cpp:499:     int n = tensor.batches * tensor.channels * tensor.height * tensor.width;
	ldp	w2, w4, [x0, 24]	// tensor_13(D)->batches, tensor_13(D)->channels,
// src/cpp/cnn_internals.cpp:499:     int n = tensor.batches * tensor.channels * tensor.height * tensor.width;
	ldp	w3, w1, [x0, 32]	// tensor_13(D)->height, tensor_13(D)->width,
// src/cpp/cnn_internals.cpp:499:     int n = tensor.batches * tensor.channels * tensor.height * tensor.width;
	mul	w2, w2, w4	// _3, tensor_13(D)->batches, tensor_13(D)->channels
// src/cpp/cnn_internals.cpp:499:     int n = tensor.batches * tensor.channels * tensor.height * tensor.width;
	mul	w2, w2, w3	// _5, _3, tensor_13(D)->height
// src/cpp/cnn_internals.cpp:499:     int n = tensor.batches * tensor.channels * tensor.height * tensor.width;
	mul	w2, w2, w1	// n, _5, tensor_13(D)->width
// src/cpp/cnn_internals.cpp:500:     for (int i = 0; i < n; ++i)
	cmp	w2, 0	// n,
	ble	.L362		//,
	sub	w1, w2, #1	// _59, n,
	cnth	x3		// tmp143
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	ldr	x6, [x0]	// _16, MEM[(struct vector *)tensor_13(D)].D.54862._M_impl.D.54163._M_start
	sub	w4, w3, #1	// tmp142, tmp143,
	mov	w0, w1	// _59, _59
	mov	w1, 0	// ivtmp_81,
	cmp	w0, w4	// _59, tmp142
	bcc	.L364		//,
	sub	w5, w2, w3	// _80, n, tmp143
	mov	x0, x6	// ivtmp.938, _16
// src/cpp/cnn_internals.cpp:501:         if (tensor.data[i] < 0.0f) tensor.data[i] = 0.0f;
	ptrue	p7.b, all	// tmp146
// src/cpp/cnn_internals.cpp:501:         if (tensor.data[i] < 0.0f) tensor.data[i] = 0.0f;
	mov	z31.s, #0	//
	cntb	x4, all, mul #2	// tmp159
	.p2align 5,,15
.L365:
// src/cpp/cnn_internals.cpp:501:         if (tensor.data[i] < 0.0f) tensor.data[i] = 0.0f;
	ld1w	z0.s, p7/z, [x0, #1, mul vl]	//, tmp146, MEM <vector([4,4]) float> [(value_type &)_58 + POLY_INT_CST [16, 16]]
	ld1w	z30.s, p7/z, [x0]	//, tmp146, MEM <vector([4,4]) float> [(value_type &)_58]
	fcmlt	p5.s, p7/z, z0.s, #0.0	// mask__21.922, tmp146,
	fcmlt	p6.s, p7/z, z30.s, #0.0	// mask__21.922_72, tmp146,
// src/cpp/cnn_internals.cpp:501:         if (tensor.data[i] < 0.0f) tensor.data[i] = 0.0f;
	st1w	z31.s, p5, [x0, #1, mul vl]	//, mask__21.922,
// src/cpp/cnn_internals.cpp:500:     for (int i = 0; i < n; ++i)
	add	w1, w1, w3	// ivtmp_81, ivtmp_81, tmp143
// src/cpp/cnn_internals.cpp:501:         if (tensor.data[i] < 0.0f) tensor.data[i] = 0.0f;
	st1w	z31.s, p6, [x0]	//, mask__21.922_72,* ivtmp.938
	add	x0, x0, x4	// ivtmp.938, ivtmp.938, tmp159
	cmp	w5, w1	// _80, ivtmp_81
	bcs	.L365		//,
	cmp	w2, w1	// n, ivtmp_81
	beq	.L362		//,
.L364:
	sub	w2, w2, w1	// bnd.926, n, ivtmp_81
	mov	z29.s, #0	//
	whilelo	p7.s, wzr, w2	// max_mask_108,, bnd.926
	add	x1, x6, w1, uxtw 2	// vectp.928, _16, ivtmp_81,,
// src/cpp/cnn_internals.cpp:501:         if (tensor.data[i] < 0.0f) tensor.data[i] = 0.0f;
	ld1w	z27.s, p7/z, [x1]	//, max_mask_108,* vectp.928
// src/cpp/cnn_internals.cpp:501:         if (tensor.data[i] < 0.0f) tensor.data[i] = 0.0f;
	fcmlt	p7.s, p7/z, z27.s, z29.s	// vec_mask_and_19, max_mask_108,,
	st1w	z29.s, p7, [x1]	//, vec_mask_and_19,* vectp.928
	cntw	x0		// tmp167
	whilelo	p7.s, w0, w2	// next_mask_50, tmp167, bnd.926
	b.none	.L362	//,
// src/cpp/cnn_internals.cpp:501:         if (tensor.data[i] < 0.0f) tensor.data[i] = 0.0f;
	ld1w	z28.s, p7/z, [x1, #1, mul vl]	//, next_mask_50,
// src/cpp/cnn_internals.cpp:501:         if (tensor.data[i] < 0.0f) tensor.data[i] = 0.0f;
	fcmlt	p7.s, p7/z, z28.s, z29.s	// vec_mask_and_103, next_mask_50,,
	st1w	z29.s, p7, [x1, #1, mul vl]	//, vec_mask_and_103,
.L362:
// src/cpp/cnn_internals.cpp:502: }
	ret	
	.cfi_endproc
.LFE5472:
	.size	_Z12relu_forwardR6Tensor, .-_Z12relu_forwardR6Tensor
	.align	2
	.p2align 5,,15
	.global	_Z14linear_forwardRK6TensorS1_S1_RS_
	.type	_Z14linear_forwardRK6TensorS1_S1_RS_, %function
_Z14linear_forwardRK6TensorS1_S1_RS_:
.LFB5473:
	.cfi_startproc
	stp	x29, x30, [sp, -64]!	//,,,
	.cfi_def_cfa_offset 64
	.cfi_offset 29, -64
	.cfi_offset 30, -56
	mov	x29, sp	//,
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -48
	.cfi_offset 20, -40
// src/cpp/cnn_internals.cpp:509:     for (int b = 0; b < input.batches; ++b) {
	ldr	w20, [x0, 24]	//, input_24(D)->batches
// src/cpp/cnn_internals.cpp:506:     int out_features = weight.height;
	ldr	w10, [x1, 32]	//, weight_21(D)->height
// src/cpp/cnn_internals.cpp:509:     for (int b = 0; b < input.batches; ++b) {
	cmp	w20, 0	// _47,
	ble	.L370		//,
	cmp	w10, 0	// out_features,
	ble	.L370		//,
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 22, -24
	.cfi_offset 21, -32
	mov	x19, x1	// weight, tmp236
	mov	x21, x0	// input, tmp235
	ptrue	p6.b, all	// tmp224
	ubfiz	x10, x10, 2, 32	// _181, out_features,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	mov	w18, 0	// ivtmp.1018,
	mov	x22, 0	// ivtmp.1017,
// src/cpp/cnn_internals.cpp:512:             for (int in_f = 0; in_f < in_features; ++in_f)
	cntw	x5		// tmp222
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -8
	.cfi_offset 23, -16
// src/cpp/cnn_internals.cpp:509:     for (int b = 0; b < input.batches; ++b) {
	mov	w30, 0	// b,
	mov	w17, 64	// tmp226,
// src/cpp/cnn_internals.cpp:505:     int in_features  = weight.width;
	ldr	w6, [x1, 36]	//, weight_21(D)->width
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	ldr	x12, [x2]	// _30, MEM[(const struct vector *)bias_26(D)].D.54862._M_impl.D.54163._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	ldr	x23, [x3]	// _43, MEM[(struct vector *)output_29(D)].D.54862._M_impl.D.54163._M_start
	sub	w15, w6, #1	// _49, in_features,
	lsr	w16, w6, 4	// bnd.952_18, in_features,
	and	w9, w6, -16	// niters_vector_mult_vf.953, in_features,
	.p2align 5,,15
.L372:
	sxtw	x13, w18	// _122, ivtmp.1018
	sbfiz	x14, x18, 2, 32	// _123, ivtmp.1018,,
// src/cpp/cnn_internals.cpp:504: void linear_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
	mov	x3, 0	// ivtmp.1012,
	mov	w4, 0	// ivtmp.1011,
	add	x11, x23, x22	// _39, _43, ivtmp.1017
.L380:
// src/cpp/cnn_internals.cpp:511:             float val = bias.data[out_f];
	ldr	s31, [x12, x3]	//, MEM[(const value_type &)_30 + ivtmp.1012_5 * 1]
// src/cpp/cnn_internals.cpp:512:             for (int in_f = 0; in_f < in_features; ++in_f)
	cmp	w6, 0	// in_features,
	bgt	.L387		//,
// src/cpp/cnn_internals.cpp:514:             output.data[b * out_features + out_f] = val;
	str	s31, [x11, x3]	//, MEM[(value_type &)_39 + ivtmp.1012_5 * 1]
// src/cpp/cnn_internals.cpp:510:         for (int out_f = 0; out_f < out_features; ++out_f) {
	add	x3, x3, 4	// ivtmp.1012, ivtmp.1012,
	add	w4, w4, w6	// ivtmp.1011, ivtmp.1011, in_features
	cmp	x3, x10	// ivtmp.1012, _181
	bne	.L380		//,
	.p2align 5,,15
.L378:
// src/cpp/cnn_internals.cpp:509:     for (int b = 0; b < input.batches; ++b) {
	add	w30, w30, 1	// b, b,
// src/cpp/cnn_internals.cpp:509:     for (int b = 0; b < input.batches; ++b) {
	add	x22, x22, x10	// ivtmp.1017, ivtmp.1017, _181
	add	w18, w18, w6	// ivtmp.1018, ivtmp.1018, in_features
	cmp	w30, w20	// b, _47
	bne	.L372		//,
	ldp	x21, x22, [sp, 32]	//,,
	.cfi_restore 22
	.cfi_restore 21
	ldp	x23, x24, [sp, 48]	//,,
	.cfi_restore 24
	.cfi_restore 23
.L370:
// src/cpp/cnn_internals.cpp:517: }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x29, x30, [sp], 64	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L387:
	.cfi_def_cfa_offset 64
	.cfi_offset 19, -48
	.cfi_offset 20, -40
	.cfi_offset 21, -32
	.cfi_offset 22, -24
	.cfi_offset 23, -16
	.cfi_offset 24, -8
	.cfi_offset 29, -64
	.cfi_offset 30, -56
	ldr	x7, [x19]	// _27, MEM[(const struct vector *)weight_21(D)].D.54862._M_impl.D.54163._M_start
	ldr	x8, [x21]	// _40, MEM[(const struct vector *)input_24(D)].D.54862._M_impl.D.54163._M_start
	cmp	w15, 14	// _49,
	bls	.L381		//,
	.p2align 5,,15
.L388:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	movi	v30.4s, 0	//
	add	x0, x8, x14	// ivtmp.999, _40, _123
	add	x1, x7, w4, uxtw 2	// ivtmp.1002, _27, ivtmp.1011,,
	umaddl	x2, w16, w17, x0	// _76, bnd.952_18, tmp226, ivtmp.999
	mov	v1.16b, v30.16b	//,
	mov	v28.16b, v30.16b	//,
	mov	v0.16b, v30.16b	//,
	.p2align 5,,15
.L375:
// src/cpp/cnn_internals.cpp:513:                 val += input.data[b * in_features + in_f] * weight.data[out_f * in_features + in_f];
	ldp	q23, q22, [x1, 16]	//,,
// src/cpp/cnn_internals.cpp:513:                 val += input.data[b * in_features + in_f] * weight.data[out_f * in_features + in_f];
	ldp	q25, q24, [x0, 32]	//,,
	ldp	q20, q26, [x0], 64	//,, ivtmp.999,
// src/cpp/cnn_internals.cpp:513:                 val += input.data[b * in_features + in_f] * weight.data[out_f * in_features + in_f];
	ldr	q21, [x1, 48]	//, MEM <const vector(4) float> [(const value_type &)_70 + 48]
	ldr	q2, [x1], 64	//, MEM <const vector(4) float> [(const value_type &)_70]
// src/cpp/cnn_internals.cpp:513:                 val += input.data[b * in_features + in_f] * weight.data[out_f * in_features + in_f];
	fmla	v1.4s, v25.4s, v22.4s	//,,
	fmla	v28.4s, v26.4s, v23.4s	//,,
	fmla	v30.4s, v24.4s, v21.4s	//,,
	fmla	v0.4s, v20.4s, v2.4s	//,,
	cmp	x0, x2	// ivtmp.999, _76
	bne	.L375		//,
	mov	w1, w9	// niters_vector_mult_vf.953, niters_vector_mult_vf.953
	fadd	v1.4s, v1.4s, v30.4s	//,,
	fadd	v0.4s, v0.4s, v28.4s	//,,
	fadd	v0.4s, v1.4s, v0.4s	//,,
	faddp	v0.4s, v0.4s, v0.4s	//,,
	faddp	v0.4s, v0.4s, v0.4s	//,,
	fadd	s31, s31, s0	//,,
	cmp	w9, w6	// niters_vector_mult_vf.953, in_features
	beq	.L376		//,
.L374:
	uxtw	x2, w1	// _172, niters_vector_mult_vf.953
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	mov	x0, 0	// ivtmp_192,
	mov	z19.s, #0	//
	add	x24, x13, x2	// _173, _122, _172
	add	w2, w4, w2	// _182, ivtmp.1011, _172
	add	x24, x8, x24, lsl 2	// vectp.975, _40, _173,
	sub	w1, w6, w1	// bnd.972, in_features, niters_vector_mult_vf.953
	whilelo	p7.s, wzr, w1	// loop_mask_177,, bnd.972
	add	x2, x7, x2, lsl 2	// vectp.978, _27, _182,
.L377:
// src/cpp/cnn_internals.cpp:513:                 val += input.data[b * in_features + in_f] * weight.data[out_f * in_features + in_f];
	ld1w	z29.s, p7/z, [x24, x0, lsl 2]	//, loop_mask_177,* vectp.975
// src/cpp/cnn_internals.cpp:513:                 val += input.data[b * in_features + in_f] * weight.data[out_f * in_features + in_f];
	ld1w	z27.s, p7/z, [x2, x0, lsl 2]	//, loop_mask_177,* vectp.978
// src/cpp/cnn_internals.cpp:512:             for (int in_f = 0; in_f < in_features; ++in_f)
	add	x0, x0, x5	// ivtmp_192, ivtmp_192, tmp222
// src/cpp/cnn_internals.cpp:513:                 val += input.data[b * in_features + in_f] * weight.data[out_f * in_features + in_f];
	fmla	z19.s, p7/m, z29.s, z27.s	//, loop_mask_177,,
	whilelo	p7.s, w0, w1	// loop_mask_177, ivtmp_192, bnd.972
	b.any	.L377	//,
	faddv	s19, p6, z19.s	//, tmp224,
	fadd	s31, s31, s19	//,,
.L376:
// src/cpp/cnn_internals.cpp:514:             output.data[b * out_features + out_f] = val;
	str	s31, [x11, x3]	//, MEM[(value_type &)_9 + ivtmp.1012_68 * 1]
// src/cpp/cnn_internals.cpp:510:         for (int out_f = 0; out_f < out_features; ++out_f) {
	add	x3, x3, 4	// ivtmp.1012, ivtmp.1012,
	add	w4, w4, w6	// ivtmp.1011, ivtmp.1011, in_features
	cmp	x3, x10	// ivtmp.1012, _181
	beq	.L378		//,
// src/cpp/cnn_internals.cpp:511:             float val = bias.data[out_f];
	ldr	s31, [x12, x3]	//, MEM[(const value_type &)_30 + ivtmp.1012_13 * 1]
	cmp	w15, 14	// _49,
	bhi	.L388		//,
.L381:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	mov	w1, 0	// niters_vector_mult_vf.953,
	b	.L374		//
	.cfi_endproc
.LFE5473:
	.size	_Z14linear_forwardRK6TensorS1_S1_RS_, .-_Z14linear_forwardRK6TensorS1_S1_RS_
	.align	2
	.p2align 5,,15
	.global	_Z17maxpool2d_forwardRK6TensorRS_ii
	.type	_Z17maxpool2d_forwardRK6TensorRS_ii, %function
_Z17maxpool2d_forwardRK6TensorRS_ii:
.LFB5474:
	.cfi_startproc
// src/cpp/cnn_internals.cpp:521:     for (int b = 0; b < input.batches; ++b) {
	ldr	w6, [x0, 24]	//, input_44(D)->batches
// src/cpp/cnn_internals.cpp:521:     for (int b = 0; b < input.batches; ++b) {
	cmp	w6, 0	// _73,
	ble	.L418		//,
// src/cpp/cnn_internals.cpp:522:         for (int c = 0; c < input.channels; ++c) {
	ldr	w17, [x0, 28]	//, input_44(D)->channels
	mov	x7, x0	// input, tmp204
	cmp	w17, 0	// _76,
	ble	.L418		//,
// src/cpp/cnn_internals.cpp:523:             for (int oh = 0; oh < output.height; ++oh) {
	ldr	w16, [x1, 32]	//, output_46(D)->height
	mov	w14, w2	// pool_size, tmp206
	mov	x5, x1	// output, tmp205
	mov	w2, w3	// stride, tmp207
// src/cpp/cnn_internals.cpp:521:     for (int b = 0; b < input.batches; ++b) {
	mov	w11, 0	// b,
.L392:
	cmp	w16, 0	// _79,
	bgt	.L421		//,
// src/cpp/cnn_internals.cpp:521:     for (int b = 0; b < input.batches; ++b) {
	add	w11, w11, 1	// b, b,
// src/cpp/cnn_internals.cpp:521:     for (int b = 0; b < input.batches; ++b) {
	cmp	w11, w6	// b, _73
	bne	.L392		//,
.L418:
	ret	
.L421:
// src/cpp/cnn_internals.cpp:541:                     output.data[out_idx] = max_val;
	mov	w0, 27432	// tmp208,
// src/cpp/cnn_internals.cpp:519: void maxpool2d_forward(const Tensor& input, Tensor& output, int pool_size, int stride) {
	stp	x29, x30, [sp, -112]!	//,,,
	.cfi_def_cfa_offset 112
	.cfi_offset 29, -112
	.cfi_offset 30, -104
	mov	x29, sp	//,
// src/cpp/cnn_internals.cpp:541:                     output.data[out_idx] = max_val;
	movk	w0, 0xce6e, lsl 16	// tmp208,,
	mov	w4, w6	// _73, _73
	fmov	s29, w0	// tmp200, tmp208
// src/cpp/cnn_internals.cpp:519: void maxpool2d_forward(const Tensor& input, Tensor& output, int pool_size, int stride) {
	stp	x19, x20, [sp, 16]	//,,
	stp	x21, x22, [sp, 32]	//,,
	ldr	w15, [x5, 36]	//, output_46(D)->width
// src/cpp/cnn_internals.cpp:539:                                   c * (output.height * output.width) +
	mul	w12, w16, w15	// _86, _79, _82
	.cfi_offset 19, -96
	.cfi_offset 20, -88
	.cfi_offset 21, -80
	.cfi_offset 22, -72
	sxtw	x21, w15	// _67, _82
.L407:
// src/cpp/cnn_internals.cpp:522:         for (int c = 0; c < input.channels; ++c) {
	mov	w19, 0	// c,
.L409:
	cmp	w15, 0	// _82,
	bgt	.L422		//,
// src/cpp/cnn_internals.cpp:522:         for (int c = 0; c < input.channels; ++c) {
	add	w19, w19, 1	// c, c,
// src/cpp/cnn_internals.cpp:522:         for (int c = 0; c < input.channels; ++c) {
	cmp	w19, w17	// c, _76
	bne	.L409		//,
// src/cpp/cnn_internals.cpp:521:     for (int b = 0; b < input.batches; ++b) {
	add	w11, w11, 1	// b, b,
// src/cpp/cnn_internals.cpp:521:     for (int b = 0; b < input.batches; ++b) {
	cmp	w4, w11	// _73, b
	bne	.L407		//,
.L424:
// src/cpp/cnn_internals.cpp:546: }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x29, x30, [sp], 112	//,,,
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L422:
	.cfi_restore_state
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -56
	.cfi_offset 23, -64
// src/cpp/cnn_internals.cpp:539:                                   c * (output.height * output.width) +
	mul	w22, w19, w12	// _80, c, _86
// src/cpp/cnn_internals.cpp:538:                     int out_idx = b * (output.channels * output.height * output.width) +
	mov	w10, w4	// _73, _73
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 26, -40
	.cfi_offset 25, -48
	stp	x27, x28, [sp, 80]	//,,
	.cfi_offset 28, -24
	.cfi_offset 27, -32
	ldr	w0, [x5, 28]	//, output_46(D)->channels
	ldr	x6, [x5]	// _68, MEM[(struct vector *)output_46(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:538:                     int out_idx = b * (output.channels * output.height * output.width) +
	mul	w0, w16, w0	// _16, _79, output_46(D)->channels
// src/cpp/cnn_internals.cpp:538:                     int out_idx = b * (output.channels * output.height * output.width) +
	mul	w0, w0, w15	// _17, _16, _82
// src/cpp/cnn_internals.cpp:538:                     int out_idx = b * (output.channels * output.height * output.width) +
	mul	w8, w0, w11	// _18, _17, b
.L404:
// src/cpp/cnn_internals.cpp:538:                     int out_idx = b * (output.channels * output.height * output.width) +
	add	w24, w8, w22	// _21, _18, _80
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	mov	w20, w14	// ivtmp.1052, pool_size
	mov	w9, 0	// ivtmp.1053,
	mov	w4, 0	// ivtmp.1051,
// src/cpp/cnn_internals.cpp:523:             for (int oh = 0; oh < output.height; ++oh) {
	mov	w30, 0	// oh,
.L406:
// src/cpp/cnn_internals.cpp:539:                                   c * (output.height * output.width) +
	add	w0, w4, w24	// _23, ivtmp.1051, _21
	add	x3, x21, w0, sxtw	// _62, _67, _23
	add	x3, x6, x3, lsl 2	// _127, _68, _62,
	mov	w18, w14	// ivtmp.1046, pool_size
	add	x0, x6, w0, sxtw 2	// ivtmp.1044, _68, _23,
	mov	w23, 0	// ivtmp.1047,
.L401:
// src/cpp/cnn_internals.cpp:526:                     for (int ph = 0; ph < pool_size; ++ph) {
	cmp	w14, 0	// pool_size,
	bgt	.L423		//,
// src/cpp/cnn_internals.cpp:524:                 for (int ow = 0; ow < output.width; ++ow) {
	add	w18, w18, w2	// ivtmp.1046, ivtmp.1046, stride
	add	w23, w23, w2	// ivtmp.1047, ivtmp.1047, stride
// src/cpp/cnn_internals.cpp:541:                     output.data[out_idx] = max_val;
	str	s29, [x0], 4	// tmp200, MEM[(value_type &)_153]
// src/cpp/cnn_internals.cpp:524:                 for (int ow = 0; ow < output.width; ++ow) {
	cmp	x3, x0	// _127, ivtmp.1044
	bne	.L401		//,
// src/cpp/cnn_internals.cpp:523:             for (int oh = 0; oh < output.height; ++oh) {
	add	w30, w30, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:523:             for (int oh = 0; oh < output.height; ++oh) {
	add	w4, w4, w15	// ivtmp.1051, ivtmp.1051, _82
	add	w20, w20, w2	// ivtmp.1052, ivtmp.1052, stride
	add	w9, w9, w2	// ivtmp.1053, ivtmp.1053, stride
	cmp	w30, w16	// oh, _79
	bne	.L406		//,
.L402:
// src/cpp/cnn_internals.cpp:522:         for (int c = 0; c < input.channels; ++c) {
	add	w19, w19, 1	// c, c,
// src/cpp/cnn_internals.cpp:522:         for (int c = 0; c < input.channels; ++c) {
	cmp	w19, w17	// c, _76
	beq	.L415		//,
// src/cpp/cnn_internals.cpp:539:                                   c * (output.height * output.width) +
	mul	w22, w19, w12	// _80, c, _86
	b	.L404		//
.L423:
	ldr	w22, [x7, 32]	//, input_44(D)->height
// src/cpp/cnn_internals.cpp:531:                                 int in_idx = b * (input.channels * input.height * input.width) +
	mov	x1, x0	// ivtmp.1044, ivtmp.1044
	mul	w25, w17, w22	// _27, _76, _3
	.p2align 5,,15
.L399:
	mov	w13, 27432	// tmp209,
	mov	w0, w9	// ivtmp.1040, ivtmp.1053
	movk	w13, 0xce6e, lsl 16	// tmp209,,
	mov	w28, w9	// ivtmp.1053, ivtmp.1053
	fmov	s31, w13	// prephitmp_74, tmp209
	.p2align 5,,15
.L394:
	cmp	w22, w0	// _3, ivtmp.1040
	ble	.L398		//,
	ldr	w26, [x7, 36]	//, MEM[(int *)input_44(D) + 36B]
// src/cpp/cnn_internals.cpp:532:                                              c * (input.height * input.width) +
	mov	w9, w23	// ivtmp.1036, ivtmp.1047
	str	x1, [sp, 104]	// ivtmp.1044, %sfp
// src/cpp/cnn_internals.cpp:531:                                 int in_idx = b * (input.channels * input.height * input.width) +
	mul	w27, w25, w26	// _51, _27, input__width_lsm0.1029
// src/cpp/cnn_internals.cpp:532:                                              c * (input.height * input.width) +
	mul	w13, w22, w26	// _55, _3, input__width_lsm0.1029
// src/cpp/cnn_internals.cpp:531:                                 int in_idx = b * (input.channels * input.height * input.width) +
	mul	w27, w27, w11	// _110, _51, b
// src/cpp/cnn_internals.cpp:531:                                 int in_idx = b * (input.channels * input.height * input.width) +
	madd	w13, w13, w19, w27	// _70, _55, c, _110
	mov	w27, w0	// ivtmp.1040, ivtmp.1040
// src/cpp/cnn_internals.cpp:532:                                              c * (input.height * input.width) +
	madd	w13, w0, w26, w13	// _85, ivtmp.1040, input__width_lsm0.1029, _70
	.p2align 5,,15
.L396:
// src/cpp/cnn_internals.cpp:531:                                 int in_idx = b * (input.channels * input.height * input.width) +
	add	w0, w9, w13	// in_idx_84, ivtmp.1036, _85
// src/cpp/cnn_internals.cpp:530:                             if (ih < input.height && iw < input.width) {
	cmp	w9, w26	// ivtmp.1036, input__width_lsm0.1029
	bge	.L395		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:262:       if (__a < __b)
	ldr	x1, [x7]	// tmp380, MEM[(const struct vector *)input_44(D)].D.54862._M_impl.D.54163._M_start
	ldr	s30, [x1, w0, sxtw 2]	// MEM[(const float &)_109], MEM[(const float &)_109]
	fmaxnm	s31, s31, s30	// prephitmp_74, prephitmp_74, MEM[(const float &)_109]
.L395:
// src/cpp/cnn_internals.cpp:527:                         for (int pw = 0; pw < pool_size; ++pw) {
	add	w9, w9, 1	// ivtmp.1036, ivtmp.1036,
	cmp	w18, w9	// ivtmp.1046, ivtmp.1036
	bne	.L396		//,
	ldr	x1, [sp, 104]	// ivtmp.1044, %sfp
	mov	w0, w27	// ivtmp.1040, ivtmp.1040
.L398:
// src/cpp/cnn_internals.cpp:526:                     for (int ph = 0; ph < pool_size; ++ph) {
	add	w0, w0, 1	// ivtmp.1040, ivtmp.1040,
	cmp	w0, w20	// ivtmp.1040, ivtmp.1052
	bne	.L394		//,
// src/cpp/cnn_internals.cpp:524:                 for (int ow = 0; ow < output.width; ++ow) {
	mov	w9, w28	// ivtmp.1053, ivtmp.1053
	add	w18, w18, w2	// ivtmp.1046, ivtmp.1046, stride
// src/cpp/cnn_internals.cpp:541:                     output.data[out_idx] = max_val;
	str	s31, [x1], 4	// prephitmp_74, MEM[(value_type &)_64]
// src/cpp/cnn_internals.cpp:524:                 for (int ow = 0; ow < output.width; ++ow) {
	add	w23, w23, w2	// ivtmp.1047, ivtmp.1047, stride
	cmp	x1, x3	// ivtmp.1044, _127
	bne	.L399		//,
// src/cpp/cnn_internals.cpp:523:             for (int oh = 0; oh < output.height; ++oh) {
	add	w30, w30, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:523:             for (int oh = 0; oh < output.height; ++oh) {
	add	w4, w4, w15	// ivtmp.1051, ivtmp.1051, _82
	add	w20, w20, w2	// ivtmp.1052, ivtmp.1052, stride
	add	w9, w9, w2	// ivtmp.1053, ivtmp.1053, stride
	cmp	w30, w16	// oh, _79
	bne	.L406		//,
	b	.L402		//
.L415:
	mov	w4, w10	// _73, _73
// src/cpp/cnn_internals.cpp:521:     for (int b = 0; b < input.batches; ++b) {
	add	w11, w11, 1	// b, b,
	ldp	x23, x24, [sp, 48]	//,,
	.cfi_restore 24
	.cfi_restore 23
	ldp	x25, x26, [sp, 64]	//,,
	.cfi_restore 26
	.cfi_restore 25
	ldp	x27, x28, [sp, 80]	//,,
	.cfi_restore 28
	.cfi_restore 27
// src/cpp/cnn_internals.cpp:521:     for (int b = 0; b < input.batches; ++b) {
	cmp	w4, w11	// _73, b
	bne	.L407		//,
	b	.L424		//
	.cfi_endproc
.LFE5474:
	.size	_Z17maxpool2d_forwardRK6TensorRS_ii, .-_Z17maxpool2d_forwardRK6TensorRS_ii
	.align	2
	.p2align 5,,15
	.global	_Z29maxpool2d_forward_specializedRK6TensorRS_
	.type	_Z29maxpool2d_forward_specializedRK6TensorRS_, %function
_Z29maxpool2d_forward_specializedRK6TensorRS_:
.LFB5475:
	.cfi_startproc
// src/cpp/cnn_internals.cpp:553:     int in_channels   = input.channels;
	ldp	w11, w2, [x0, 24]	// input_batches, in_channels,
// src/cpp/cnn_internals.cpp:569:     for (int b = 0; b < input_batches; ++b) {
	cmp	w11, 0	// input_batches,
	ble	.L490		//,
	cmp	w2, 0	// in_channels,
	ble	.L490		//,
// src/cpp/cnn_internals.cpp:556:     int output_h      = output.height;
	ldr	w5, [x1, 32]	//, output_37(D)->height
	cmp	w5, 0	// output_h,
	ble	.L490		//,
	mov	x4, x0	// input, tmp249
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	mov	w0, 27432	// tmp252,
// src/cpp/cnn_internals.cpp:548: void maxpool2d_forward_specialized(const Tensor& input, Tensor& output) {
	stp	x29, x30, [sp, -480]!	//,,,
	.cfi_def_cfa_offset 480
	.cfi_offset 29, -480
	.cfi_offset 30, -472
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	movk	w0, 0xce6e, lsl 16	// tmp252,,
// src/cpp/cnn_internals.cpp:548: void maxpool2d_forward_specialized(const Tensor& input, Tensor& output) {
	mov	x29, sp	//,
// src/cpp/cnn_internals.cpp:594:                     out_ptr[out_row + ow] = acc[ow];
	mov	x8, 0	// ivtmp.1121,
// src/cpp/cnn_internals.cpp:569:     for (int b = 0; b < input_batches; ++b) {
	mov	w7, 0	// b,
// src/cpp/cnn_internals.cpp:548: void maxpool2d_forward_specialized(const Tensor& input, Tensor& output) {
	stp	x19, x20, [sp, 16]	//,,
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 19, -464
	.cfi_offset 20, -456
	.cfi_offset 21, -448
	.cfi_offset 22, -440
	lsl	w21, w5, 1	// _200, output_h,
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 23, -432
	.cfi_offset 24, -424
	add	x24, sp, 224	// tmp230,,
	stp	x25, x26, [sp, 64]	//,,
	stp	x27, x28, [sp, 80]	//,,
	.cfi_offset 25, -416
	.cfi_offset 26, -408
	.cfi_offset 27, -400
	.cfi_offset 28, -392
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x25, [x4]	// _38, MEM[(const struct vector *)input_34(D)].D.54862._M_impl.D.54163._M_start
	ldr	x6, [x1]	// ivtmp.1120, MEM[(struct vector *)output_37(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:557:     int output_w      = output.width;
	ldr	w23, [x1, 36]	//, output_37(D)->width
// src/cpp/cnn_internals.cpp:564:     int out_ch_size  = out_channels * out_size;
	ldr	w12, [x1, 28]	//, output_37(D)->channels
// src/cpp/cnn_internals.cpp:548: void maxpool2d_forward_specialized(const Tensor& input, Tensor& output) {
	str	d15, [sp, 96]	//,
	.cfi_offset 79, -384
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	fmov	s15, w0	// tmp240, tmp252
// src/cpp/cnn_internals.cpp:555:     int input_w       = input.width;
	ldp	w27, w9, [x4, 32]	// input_34(D)->height, input_w,
// src/cpp/cnn_internals.cpp:563:     int out_size     = output_h * output_w;
	mul	w0, w5, w23	// out_size, output_h, output_w
// src/cpp/cnn_internals.cpp:594:                     out_ptr[out_row + ow] = acc[ow];
	sub	w20, w23, #1	// _88, output_w,
	add	x20, x20, 1	// _87, _88,
// src/cpp/cnn_internals.cpp:564:     int out_ch_size  = out_channels * out_size;
	mul	w4, w0, w12	// out_ch_size_49, out_size, output_37(D)->channels
	sbfiz	x0, x0, 2, 32	// _209, out_size,,
// src/cpp/cnn_internals.cpp:561:     int in_size      = input_h * input_w;
	mul	w1, w9, w27	// in_size, input_w, input_34(D)->height
	lsl	w19, w9, 1	// _194, input_w,
	sbfiz	x27, x23, 2, 32	// _65, output_w,,
	sxtw	x26, w9	// _185, input_w
	sxtw	x19, w19	// _195, _194
	add	x28, x24, x27	// _20, tmp230, _65
	sbfiz	x12, x4, 2, 32	// _225, out_ch_size_49,,
// src/cpp/cnn_internals.cpp:562:     int in_ch_size   = in_channels * in_size;
	mul	w5, w2, w1	// in_ch_size_47, in_channels, in_size
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	dup	v31.4s, v15.s[0]	// tmp242, tmp240
	sxtw	x1, w1	// _216, in_size
	stp	x0, x1, [sp, 168]	// _209, _216,
// src/cpp/cnn_internals.cpp:594:                     out_ptr[out_row + ow] = acc[ow];
	lsl	x1, x20, 2	// _86, _87,
	sub	w0, w23, #1	// _133, output_w,
	mov	x20, x26	// _185, _185
	sxtw	x5, w5	// _229, in_ch_size_47
	mov	w26, w23	// output_w, output_w
	mov	w23, w21	// _200, _200
	mov	x21, x19	// _195, _195
	mov	x19, x27	// _65, _65
	str	x1, [sp, 128]	// _86, %sfp
	str	w0, [sp, 140]	// _133, %sfp
.L447:
// src/cpp/cnn_internals.cpp:569:     for (int b = 0; b < input_batches; ++b) {
	stp	x6, x8, [sp, 144]	// ivtmp.1120, ivtmp.1121,
// src/cpp/cnn_internals.cpp:573:         for (int c = 0; c < in_channels; ++c) {
	stp	wzr, w2, [sp, 160]	//, in_channels,
// src/cpp/cnn_internals.cpp:569:     for (int b = 0; b < input_batches; ++b) {
	stp	w11, w7, [sp, 184]	// input_batches, b,
	stp	x6, x12, [sp, 192]	// ivtmp.1120, _225,
	stp	x8, x5, [sp, 208]	// ivtmp.1121, _229,
.L428:
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	mov	w22, 0	// ivtmp.1108,
	ldp	x8, x27, [sp, 144]	// ivtmp.1105, ivtmp.1107,
	cmp	w26, 0	// output_w,
	ble	.L443		//,
	.p2align 5,,15
.L445:
	ldr	w0, [sp, 140]	//, %sfp
	cmp	w0, 7	// _133,
	bls	.L449		//,
	lsr	w0, w26, 3	// bnd.1073, output_w,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	stp	q31, q31, [sp, 224]	// tmp242, tmp242,
	cmp	w0, 1	// bnd.1073,
	beq	.L450		//,
	stp	q31, q31, [sp, 256]	// tmp242, tmp242,
	cmp	w0, 2	// bnd.1073,
	beq	.L432		//,
	stp	q31, q31, [sp, 288]	// tmp242, tmp242,
	cmp	w0, 3	// bnd.1073,
	beq	.L432		//,
	stp	q31, q31, [sp, 320]	// tmp242, tmp242,
	cmp	w0, 4	// bnd.1073,
	beq	.L432		//,
	stp	q31, q31, [sp, 352]	// tmp242, tmp242,
	cmp	w0, 5	// bnd.1073,
	beq	.L432		//,
	stp	q31, q31, [sp, 384]	// tmp242, tmp242,
	cmp	w0, 6	// bnd.1073,
	beq	.L432		//,
	stp	q31, q31, [sp, 416]	// tmp242, tmp242,
	cmp	w0, 7	// bnd.1073,
	beq	.L432		//,
	stp	q31, q31, [sp, 448]	// tmp242, tmp242,
.L432:
	and	w0, w26, -8	// ow, output_w,
	cmp	w26, w0	// output_w, ow
	beq	.L443		//,
.L431:
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	add	w1, w0, 1	// ow, ow,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	str	s15, [x24, w0, sxtw 2]	// tmp240, acc[ow_134]
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	cmp	w26, w1	// output_w, ow
	ble	.L443		//,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	add	w2, w0, 2	// ow, ow,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	str	s15, [x24, w1, sxtw 2]	// tmp240, acc[ow_105]
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	cmp	w2, w26	// ow, output_w
	bge	.L443		//,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	add	w1, w0, 3	// ow, ow,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	str	s15, [x24, w2, sxtw 2]	// tmp240, acc[ow_4]
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	cmp	w1, w26	// ow, output_w
	bge	.L443		//,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	add	w2, w0, 4	// ow, ow,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	str	s15, [x24, w1, sxtw 2]	// tmp240, acc[ow_8]
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	cmp	w26, w2	// output_w, ow
	ble	.L443		//,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	add	w1, w0, 5	// ow, ow,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	str	s15, [x24, w2, sxtw 2]	// tmp240, acc[ow_104]
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	cmp	w26, w1	// output_w, ow
	ble	.L443		//,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	add	w2, w0, 6	// ow, ow,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	str	s15, [x24, w1, sxtw 2]	// tmp240, acc[ow_115]
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	cmp	w26, w2	// output_w, ow
	ble	.L443		//,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	add	w0, w0, 7	// ow, ow,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	str	s15, [x24, w2, sxtw 2]	// tmp240, acc[ow_68]
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	cmp	w26, w0	// output_w, ow
	ble	.L443		//,
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	str	s15, [x24, w0, sxtw 2]	// tmp240, acc[ow_79]
.L443:
	mov	x1, x27	// ivtmp.1101, ivtmp.1107
	mov	w4, 2	// ivtmp_118,
.L441:
	add	x3, x25, x1, lsl 2	// ivtmp.1092, _38, ivtmp.1101,
// src/cpp/cnn_internals.cpp:585:                         for (int ow = 0; ow < output_w; ++ow) {
	mov	w5, 0	// pw,
	cmp	w26, 0	// output_w,
	ble	.L440		//,
.L434:
	mov	x2, x3	// ivtmp.1084, ivtmp.1092
	mov	x0, x24	// ivtmp.1083, tmp230
	.p2align 5,,15
.L439:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:262:       if (__a < __b)
	ldr	s30, [x0]	// MEM[(float *)_150], MEM[(float *)_150]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:262:       if (__a < __b)
	ldr	s29, [x2]	// _77, MEM[(const float &)_27]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:262:       if (__a < __b)
	fcmpe	s29, s30	// _77, MEM[(float *)_150]
	bgt	.L436		//,
// src/cpp/cnn_internals.cpp:585:                         for (int ow = 0; ow < output_w; ++ow) {
	add	x0, x0, 4	// ivtmp.1083, ivtmp.1083,
	add	x2, x2, 8	// ivtmp.1084, ivtmp.1084,
	cmp	x28, x0	// _20, ivtmp.1083
	bne	.L439		//,
// src/cpp/cnn_internals.cpp:584:                     for (int pw = 0; pw < 2; ++pw) {
	cbnz	w5, .L440	// pw,
.L493:
// src/cpp/cnn_internals.cpp:585:                         for (int ow = 0; ow < output_w; ++ow) {
	cmp	w26, 0	// output_w,
	ble	.L440		//,
// src/cpp/cnn_internals.cpp:584:                     for (int pw = 0; pw < 2; ++pw) {
	add	x3, x3, 4	// ivtmp.1092, ivtmp.1092,
	mov	w5, 1	// pw,
	b	.L434		//
	.p2align 2,,3
.L436:
// src/cpp/cnn_internals.cpp:585:                         for (int ow = 0; ow < output_w; ++ow) {
	add	x2, x2, 8	// ivtmp.1084, ivtmp.1084,
// src/cpp/cnn_internals.cpp:587:                             acc[ow] = std::max(acc[ow], input_ptr[in_row_offset + iw]);
	str	s29, [x0], 4	// _77, MEM[(float *)_150]
// src/cpp/cnn_internals.cpp:585:                         for (int ow = 0; ow < output_w; ++ow) {
	cmp	x28, x0	// _20, ivtmp.1083
	bne	.L439		//,
// src/cpp/cnn_internals.cpp:584:                     for (int pw = 0; pw < 2; ++pw) {
	cbz	w5, .L493	// pw,
.L440:
// src/cpp/cnn_internals.cpp:580:                 for (int ph = 0; ph < 2; ++ph) {
	add	x1, x1, x20	// ivtmp.1101, ivtmp.1101, _185
	cmp	w4, 1	// ivtmp_118,
	bne	.L452		//,
// src/cpp/cnn_internals.cpp:593:                 for (int ow = 0; ow < output_w; ++ow) {
	cmp	w26, 0	// output_w,
	ble	.L494		//,
// src/cpp/cnn_internals.cpp:594:                     out_ptr[out_row + ow] = acc[ow];
	ldr	x2, [sp, 128]	//, %sfp
	mov	x0, x8	//, ivtmp.1105
	mov	x1, x24	//, tmp230
// src/cpp/cnn_internals.cpp:577:             for (int oh = 0; oh < output_h; ++oh) {
	add	w22, w22, 2	// ivtmp.1108, ivtmp.1108,
	str	q31, [sp, 112]	// tmp242, %sfp
	add	x27, x27, x21	// ivtmp.1107, ivtmp.1107, _195
// src/cpp/cnn_internals.cpp:594:                     out_ptr[out_row + ow] = acc[ow];
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:577:             for (int oh = 0; oh < output_h; ++oh) {
	cmp	w22, w23	// ivtmp.1108, _200
	ldr	q31, [sp, 112]	// tmp242, %sfp
	add	x8, x0, x19	// ivtmp.1105, ivtmp.1105, _65
	bne	.L445		//,
.L446:
// src/cpp/cnn_internals.cpp:573:         for (int c = 0; c < in_channels; ++c) {
	ldr	x1, [sp, 144]	// ivtmp.1114, %sfp
	ldr	x2, [sp, 168]	// _209, %sfp
// src/cpp/cnn_internals.cpp:573:         for (int c = 0; c < in_channels; ++c) {
	ldr	w0, [sp, 160]	//, %sfp
// src/cpp/cnn_internals.cpp:573:         for (int c = 0; c < in_channels; ++c) {
	add	x1, x1, x2	// ivtmp.1114, ivtmp.1114, _209
	ldr	x2, [sp, 176]	// _216, %sfp
	str	x1, [sp, 144]	// ivtmp.1114, %sfp
// src/cpp/cnn_internals.cpp:573:         for (int c = 0; c < in_channels; ++c) {
	add	w0, w0, 1	// c, c,
// src/cpp/cnn_internals.cpp:573:         for (int c = 0; c < in_channels; ++c) {
	ldr	x1, [sp, 152]	// ivtmp.1115, %sfp
// src/cpp/cnn_internals.cpp:573:         for (int c = 0; c < in_channels; ++c) {
	str	w0, [sp, 160]	// c, %sfp
// src/cpp/cnn_internals.cpp:573:         for (int c = 0; c < in_channels; ++c) {
	add	x1, x1, x2	// ivtmp.1115, ivtmp.1115, _216
	str	x1, [sp, 152]	// ivtmp.1115, %sfp
	ldr	w1, [sp, 164]	//, %sfp
	cmp	w1, w0	// in_channels, c
	bne	.L428		//,
	ldp	w11, w7, [sp, 184]	//,,
	ldp	x6, x12, [sp, 192]	// ivtmp.1120, _225,
	ldp	x8, x5, [sp, 208]	// ivtmp.1121, _229,
// src/cpp/cnn_internals.cpp:569:     for (int b = 0; b < input_batches; ++b) {
	add	w7, w7, 1	// b, b,
// src/cpp/cnn_internals.cpp:569:     for (int b = 0; b < input_batches; ++b) {
	add	x6, x6, x12	// ivtmp.1120, ivtmp.1120, _225
	ldr	w2, [sp, 164]	//, %sfp
	add	x8, x8, x5	// ivtmp.1121, ivtmp.1121, _229
	cmp	w11, w7	// input_batches, b
	bne	.L447		//,
// src/cpp/cnn_internals.cpp:599: }
	ldr	d15, [sp, 96]	//,
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x25, x26, [sp, 64]	//,,
	ldp	x27, x28, [sp, 80]	//,,
	ldp	x29, x30, [sp], 480	//,,,
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 27
	.cfi_restore 28
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 79
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L452:
	.cfi_restore_state
	mov	w4, 1	// ivtmp_118,
	b	.L441		//
.L494:
// src/cpp/cnn_internals.cpp:577:             for (int oh = 0; oh < output_h; ++oh) {
	add	w22, w22, 2	// ivtmp.1108, ivtmp.1108,
	add	x8, x8, x19	// ivtmp.1105, ivtmp.1105, _65
	add	x27, x27, x21	// ivtmp.1107, ivtmp.1107, _195
	cmp	w22, w23	// ivtmp.1108, _200
	bne	.L443		//,
	b	.L446		//
.L450:
// src/cpp/cnn_internals.cpp:578:                 for (int ow = 0; ow < output_w; ++ow) acc[ow] = -1e9f;
	mov	w0, 8	// ow,
	b	.L431		//
.L449:
	mov	w0, 0	// ow,
	b	.L431		//
.L490:
	.cfi_def_cfa_offset 0
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 27
	.cfi_restore 28
	.cfi_restore 29
	.cfi_restore 30
	.cfi_restore 79
	ret	
	.cfi_endproc
.LFE5475:
	.size	_Z29maxpool2d_forward_specializedRK6TensorRS_, .-_Z29maxpool2d_forward_specializedRK6TensorRS_
	.align	2
	.p2align 5,,15
	.global	_Z26adaptive_avgpool2d_forwardRK6TensorRS_
	.type	_Z26adaptive_avgpool2d_forwardRK6TensorRS_, %function
_Z26adaptive_avgpool2d_forwardRK6TensorRS_:
.LFB5476:
	.cfi_startproc
	stp	x29, x30, [sp, -48]!	//,,,
	.cfi_def_cfa_offset 48
	.cfi_offset 29, -48
	.cfi_offset 30, -40
	mov	x29, sp	//,
	mov	x18, x0	// input, tmp230
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -32
	.cfi_offset 20, -24
// src/cpp/cnn_internals.cpp:604:     for (int b = 0; b < input.batches; ++b) {
	ldr	w19, [x0, 24]	//, input_23(D)->batches
// src/cpp/cnn_internals.cpp:602:     int spatial_size = input.height * input.width;
	ldp	w6, w0, [x0, 32]	// input_23(D)->height, input_23(D)->width,
	mul	w6, w6, w0	// spatial_size, input_23(D)->height, input_23(D)->width
// src/cpp/cnn_internals.cpp:604:     for (int b = 0; b < input.batches; ++b) {
	cmp	w19, 0	// _39,
	ble	.L495		//,
// src/cpp/cnn_internals.cpp:609:             output.data[b * input.channels + c] = sum / spatial_size;
	scvtf	s5, w6	//, spatial_size
// src/cpp/cnn_internals.cpp:605:         for (int c = 0; c < input.channels; ++c) {
	ldr	w17, [x18, 28]	//, input_23(D)->channels
	fmov	s30, 1.0e+0	//,
// src/cpp/cnn_internals.cpp:608:                 sum += input.data[b * (input.channels * spatial_size) + c * spatial_size + i];
	mul	w20, w17, w6	// _42, _43, spatial_size
	fdiv	s30, s30, s5	//,,
	cmp	w17, 0	// _43,
	ble	.L495		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	ldr	x30, [x1]	// _35, MEM[(struct vector *)output_26(D)].D.54862._M_impl.D.54163._M_start
	ubfiz	x17, x17, 2, 32	// _68, _43,,
	sub	w11, w6, #1	// _20, spatial_size,
	ptrue	p6.b, all	// tmp219
	mov	w16, 0	// ivtmp.1179,
	mov	x14, 0	// ivtmp.1176,
	str	x21, [sp, 32]	//,
	.cfi_offset 21, -16
// src/cpp/cnn_internals.cpp:607:             for (int i = 0; i < spatial_size; ++i)
	cntw	x3		// tmp217
	lsr	w13, w6, 4	// bnd.1129_17, spatial_size,
// src/cpp/cnn_internals.cpp:604:     for (int b = 0; b < input.batches; ++b) {
	mov	w15, 0	// b,
	mov	w12, 64	// tmp221,
	and	w9, w6, -16	// niters_vector_mult_vf.1130, spatial_size,
	add	x21, x30, x17	// _113, _35, _68
	.p2align 5,,15
.L498:
	add	x4, x30, x14	// ivtmp.1170, _35, ivtmp.1176
	add	x10, x21, x14	// _64, _113, ivtmp.1176
// src/cpp/cnn_internals.cpp:606:             float sum = 0.0f;
	mov	w5, 0	// ivtmp.1172,
.L506:
// src/cpp/cnn_internals.cpp:607:             for (int i = 0; i < spatial_size; ++i)
	cmp	w6, 0	// spatial_size,
	bgt	.L517		//,
// src/cpp/cnn_internals.cpp:605:         for (int c = 0; c < input.channels; ++c) {
	add	w5, w5, w6	// ivtmp.1172, ivtmp.1172, spatial_size
// src/cpp/cnn_internals.cpp:609:             output.data[b * input.channels + c] = sum / spatial_size;
	str	wzr, [x4], 4	//, MEM[(value_type &)_120]
// src/cpp/cnn_internals.cpp:605:         for (int c = 0; c < input.channels; ++c) {
	cmp	x10, x4	// _64, ivtmp.1170
	bne	.L506		//,
	.p2align 5,,15
.L504:
// src/cpp/cnn_internals.cpp:604:     for (int b = 0; b < input.batches; ++b) {
	add	w15, w15, 1	// b, b,
// src/cpp/cnn_internals.cpp:604:     for (int b = 0; b < input.batches; ++b) {
	add	x14, x14, x17	// ivtmp.1176, ivtmp.1176, _68
	add	w16, w16, w20	// ivtmp.1179, ivtmp.1179, _42
	cmp	w15, w19	// b, _39
	bne	.L498		//,
	ldr	x21, [sp, 32]	//,
	.cfi_restore 21
.L495:
// src/cpp/cnn_internals.cpp:612: }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x29, x30, [sp], 48	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L517:
	.cfi_def_cfa_offset 48
	.cfi_offset 19, -32
	.cfi_offset 20, -24
	.cfi_offset 21, -16
	.cfi_offset 29, -48
	.cfi_offset 30, -40
	ldr	x8, [x18]	// _27, MEM[(const struct vector *)input_23(D)].D.54862._M_impl.D.54163._M_start
	sxtw	x7, w16	// _127, ivtmp.1179
	cmp	w11, 14	// _20,
	bls	.L507		//,
	.p2align 5,,15
.L518:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	movi	v29.4s, 0	//
	sxtw	x2, w5	// _128, ivtmp.1172
	add	x0, x7, x2	// _95, _127, _128
	add	x0, x8, x0, lsl 2	// ivtmp.1163, _27, _95,
	mov	v0.16b, v29.16b	//,
	mov	v27.16b, v29.16b	//,
	mov	v26.16b, v29.16b	//,
	umaddl	x1, w13, w12, x0	// _62, bnd.1129_17, tmp221, ivtmp.1163
	.p2align 5,,15
.L500:
// src/cpp/cnn_internals.cpp:608:                 sum += input.data[b * (input.channels * spatial_size) + c * spatial_size + i];
	ldp	q4, q3, [x0]	//,,* ivtmp.1163
	ldp	q2, q1, [x0, 32]	//,,
	add	x0, x0, 64	// ivtmp.1163, ivtmp.1163,
	fadd	v26.4s, v26.4s, v4.4s	//,,
	fadd	v27.4s, v27.4s, v3.4s	//,,
	fadd	v0.4s, v0.4s, v2.4s	//,,
	fadd	v29.4s, v29.4s, v1.4s	//,,
	cmp	x0, x1	// ivtmp.1163, _62
	bne	.L500		//,
	fadd	v0.4s, v0.4s, v29.4s	//,,
	mov	w1, w9	// niters_vector_mult_vf.1130, niters_vector_mult_vf.1130
	fadd	v26.4s, v26.4s, v27.4s	//,,
	fadd	v26.4s, v0.4s, v26.4s	//,,
	faddp	v26.4s, v26.4s, v26.4s	//,,
	faddp	v26.4s, v26.4s, v26.4s	//,,
	cmp	w9, w6	// niters_vector_mult_vf.1130, spatial_size
	beq	.L501		//,
.L499:
	add	x2, x7, x2	// _129, _127, _128
	add	x2, x2, w1, uxtw	// _131, _129, niters_vector_mult_vf.1130
	add	x2, x8, x2, lsl 2	// vectp.1145, _27, _131,
// src/cpp/cnn_internals.cpp:606:             float sum = 0.0f;
	mov	x0, 0	// ivtmp_141,
	mov	z31.s, #0	//
	sub	w1, w6, w1	// bnd.1142, spatial_size, niters_vector_mult_vf.1130
	whilelo	p7.s, wzr, w1	// loop_mask_135,, bnd.1142
.L502:
// src/cpp/cnn_internals.cpp:608:                 sum += input.data[b * (input.channels * spatial_size) + c * spatial_size + i];
	ld1w	z28.s, p7/z, [x2, x0, lsl 2]	//, loop_mask_135,* vectp.1145
// src/cpp/cnn_internals.cpp:607:             for (int i = 0; i < spatial_size; ++i)
	add	x0, x0, x3	// ivtmp_141, ivtmp_141, tmp217
// src/cpp/cnn_internals.cpp:608:                 sum += input.data[b * (input.channels * spatial_size) + c * spatial_size + i];
	fadd	z31.s, p7/m, z31.s, z28.s	//, loop_mask_135,
	whilelo	p7.s, w0, w1	// loop_mask_135, ivtmp_141, bnd.1142
	b.any	.L502	//,
	faddv	s31, p6, z31.s	//, tmp219,
	fadd	s26, s26, s31	//,,
.L501:
// src/cpp/cnn_internals.cpp:609:             output.data[b * input.channels + c] = sum / spatial_size;
	fmul	s26, s30, s26	//,,
// src/cpp/cnn_internals.cpp:605:         for (int c = 0; c < input.channels; ++c) {
	add	w5, w5, w6	// ivtmp.1172, ivtmp.1172, spatial_size
// src/cpp/cnn_internals.cpp:609:             output.data[b * input.channels + c] = sum / spatial_size;
	str	s26, [x4], 4	//, MEM[(value_type &)_71]
// src/cpp/cnn_internals.cpp:605:         for (int c = 0; c < input.channels; ++c) {
	cmp	x10, x4	// _64, ivtmp.1170
	beq	.L504		//,
	cmp	w11, 14	// _20,
	bhi	.L518		//,
.L507:
// src/cpp/cnn_internals.cpp:606:             float sum = 0.0f;
	movi	v26.2s, #0	//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	mov	w1, 0	// niters_vector_mult_vf.1130,
	sxtw	x2, w5	// _128, ivtmp.1172
	b	.L499		//
	.cfi_endproc
.LFE5476:
	.size	_Z26adaptive_avgpool2d_forwardRK6TensorRS_, .-_Z26adaptive_avgpool2d_forwardRK6TensorRS_
	.section	.rodata._ZNSt6vectorIfSaIfEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPfS1_EEmRKf.str1.8,"aMS",@progbits,1
	.align	3
.LC1:
	.string	"vector::_M_fill_insert"
	.section	.text._ZNSt6vectorIfSaIfEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPfS1_EEmRKf,"axG",@progbits,_ZNSt6vectorIfSaIfEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPfS1_EEmRKf,comdat
	.align	2
	.p2align 5,,15
	.weak	_ZNSt6vectorIfSaIfEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPfS1_EEmRKf
	.type	_ZNSt6vectorIfSaIfEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPfS1_EEmRKf, %function
_ZNSt6vectorIfSaIfEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPfS1_EEmRKf:
.LFB5544:
	.cfi_startproc
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:703:       if (__n != 0)
	cbz	x2, .L586	// __n,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:700:     vector<_Tp, _Alloc>::
	stp	x29, x30, [sp, -112]!	//,,,
	.cfi_def_cfa_offset 112
	.cfi_offset 29, -112
	.cfi_offset 30, -104
	mov	x29, sp	//,
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -96
	.cfi_offset 20, -88
	mov	x19, x2	// __n, tmp350
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 21, -80
	.cfi_offset 22, -72
	mov	x22, x0	// this, tmp348
	mov	x21, x1	// __position, tmp349
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 23, -64
	.cfi_offset 24, -56
	mov	x24, x1	// tmp.1198, __position
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:706: 			- this->_M_impl._M_finish) >= __n)
	ldp	x23, x2, [x0, 8]	// _2, this_21(D)->D.54862._M_impl.D.54163._M_end_of_storage,
	sub	x2, x2, x23	// _3, this_21(D)->D.54862._M_impl.D.54163._M_end_of_storage, _2
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:705: 	  if (size_type(this->_M_impl._M_end_of_storage
	cmp	x19, x2, asr 2	// __n, _3,
	bhi	.L522		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_iterator.h:1328:     { return __lhs.base() - __rhs.base(); }
	sub	x20, x23, x1	// _35, _2, __position
	str	d15, [sp, 88]	//,
	.cfi_offset 79, -24
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_iterator.h:1328:     { return __lhs.base() - __rhs.base(); }
	asr	x0, x20, 2	// _33, _35,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:191: 	{ ::new((void *)__p) _Up(std::forward<_Args>(__args)...); }
	ldr	s15, [x3]	// _46, MEM[(const type &)__x_32(D)]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:716: 	      if (__elems_after > __n)
	cmp	x19, x0	// __n, _33
	bcs	.L523		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:719: 		  std::__uninitialized_move_a(__old_finish - __n,
	lsl	x20, x19, 2	// _6, __n,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:719: 		  std::__uninitialized_move_a(__old_finish - __n,
	sub	x19, x23, x19, lsl 2	// _8, _2, __n,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:451: 	  if (__builtin_expect(_Num > 1, true))
	cmp	x20, 4	// _6,
	ble	.L524		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:452: 	    __builtin_memmove(__result, __first, sizeof(_Tp) * _Num);
	mov	x2, x20	//, _6
	mov	x1, x19	//, _8
	mov	x0, x23	//, _2
	bl	memmove		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:723: 		  this->_M_impl._M_finish += __n;
	ldr	x0, [x22, 8]	// prephitmp_126, this_21(D)->D.54862._M_impl.D.54163._M_finish
.L525:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:723: 		  this->_M_impl._M_finish += __n;
	add	x0, x0, x20	// _10, prephitmp_126, _6
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:764: 	  const ptrdiff_t _Num = __last - __first;
	sub	x2, x19, x21	// _114, _8, __position
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:723: 		  this->_M_impl._M_finish += __n;
	str	x0, [x22, 8]	// _10, this_21(D)->D.54862._M_impl.D.54163._M_finish
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:765: 	  if (__builtin_expect(_Num > 1, true))
	cmp	x2, 4	// _114,
	ble	.L528		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:766: 	    __builtin_memmove(__result - _Num, __first, sizeof(_Tp) * _Num);
	sub	x0, x23, x2	//, _2, _114
	mov	x1, x21	//, __position
	bl	memmove		//
.L529:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:727: 		  std::fill(__position.base(), __position.base() + __n,
	add	x2, x21, x20	// _11, __position, _6
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x21	// _11, __position
	beq	.L606		//,
	sub	x20, x20, #4	// _49, _6,
	lsr	x3, x20, 2	// _135, _49,
	add	x3, x3, 1	// niters.1189, _135,
	cmp	x20, 28	// _49,
	bls	.L531		//,
	lsr	x1, x3, 3	// bnd.1190_31, niters.1189,
	dup	v28.4s, v15.s[0]	//, _46
	mov	x0, x21	// ivtmp.1222, __position
	add	x1, x21, x1, lsl 5	// _82, __position, bnd.1190_31,
	.p2align 5,,15
.L532:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	stp	q28, q28, [x0], 32	//,, ivtmp.1222,
	cmp	x1, x0	// _82, ivtmp.1222
	bne	.L532		//,
	and	x24, x3, -8	// niters_vector_mult_vf.1191_83, niters.1189,
	add	x24, x21, x24, lsl 2	// tmp.1198, __position, niters_vector_mult_vf.1191_83,
	tst	x3, 7	// niters.1189,
	beq	.L606		//,
.L531:
	mov	x0, x24	// __first_48, tmp.1198
	str	s15, [x0], 4	// _46, *__first_70
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x0	// _11, __first_48
	beq	.L606		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x0, x24, 8	// __first_196, tmp.1198,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 4]	// _46, MEM[(float *)__first_70 + 4B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x0	// _11, __first_196
	beq	.L606		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x0, x24, 12	// __first_144, tmp.1198,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 8]	// _46, MEM[(float *)__first_70 + 8B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x0	// _11, __first_144
	beq	.L606		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x0, x24, 16	// __first_100, tmp.1198,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 12]	// _46, MEM[(float *)__first_70 + 12B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x0	// _11, __first_100
	beq	.L606		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x0, x24, 20	// __first_250, tmp.1198,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 16]	// _46, MEM[(float *)__first_70 + 16B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x0	// _11, __first_250
	beq	.L606		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x0, x24, 24	// __first_254, tmp.1198,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 20]	// _46, MEM[(float *)__first_70 + 20B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x0	// _11, __first_254
	beq	.L606		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x0, x24, 28	// __first_258, tmp.1198,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 24]	// _46, MEM[(float *)__first_70 + 24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x0	// _11, __first_258
	beq	.L606		//,
.L607:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 28]	// _46,
	ldr	d15, [sp, 88]	//,
	.cfi_restore 79
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:799:     }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x29, x30, [sp], 112	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L586:
	ret	
	.p2align 2,,3
.L523:
	.cfi_def_cfa_offset 112
	.cfi_offset 19, -96
	.cfi_offset 20, -88
	.cfi_offset 21, -80
	.cfi_offset 22, -72
	.cfi_offset 23, -64
	.cfi_offset 24, -56
	.cfi_offset 29, -112
	.cfi_offset 30, -104
	.cfi_offset 79, -24
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	subs	x1, x19, x0	// _12, __n, _33
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1147: 	return __first;
	mov	x0, x23	// _63, _2
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	beq	.L535		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1151:       std::__fill_a(__first, __first + __n, __value);
	add	x2, x23, x1, lsl 2	// _63, _2, _12,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1151:       std::__fill_a(__first, __first + __n, __value);
	lsl	x1, x1, 2	// _59, _12,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1151:       std::__fill_a(__first, __first + __n, __value);
	mov	x0, x2	// _63, _63
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x23, x2	// _2, _63
	beq	.L535		//,
	sub	x0, x1, #4	// _189, _59,
	lsr	x1, x0, 2	// _190, _189,
	add	x1, x1, 1	// niters.1201, _190,
	cmp	x0, 28	// _189,
	bls	.L561		//,
	lsr	x0, x1, 3	// bnd.1202_203, niters.1201,
	dup	v30.4s, v15.s[0]	//, _46
	mov	x3, x23	// ivtmp.1234, _2
	add	x0, x23, x0, lsl 5	// _221, _2, bnd.1202_203,
	.p2align 5,,15
.L537:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	stp	q30, q30, [x3], 32	//,, ivtmp.1234,
	cmp	x3, x0	// ivtmp.1234, _221
	bne	.L537		//,
	and	x0, x1, -8	// niters_vector_mult_vf.1203_204, niters.1201,
	add	x0, x23, x0, lsl 2	// tmp.1204, _2, niters_vector_mult_vf.1203_204,
	tst	x1, 7	// niters.1201,
	beq	.L538		//,
.L536:
	mov	x1, x0	// __first_290, tmp.1204
	str	s15, [x1], 4	// _46, *__first_201
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x1	// _63, __first_290
	beq	.L538		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x1, x0, 8	// __first_294, tmp.1204,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x0, 4]	// _46, MEM[(float *)__first_201 + 4B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x1	// _63, __first_294
	beq	.L538		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x1, x0, 12	// __first_298, tmp.1204,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x0, 8]	// _46, MEM[(float *)__first_201 + 8B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x1	// _63, __first_298
	beq	.L538		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x1, x0, 16	// __first_302, tmp.1204,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x0, 12]	// _46, MEM[(float *)__first_201 + 12B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x1	// _63, __first_302
	beq	.L538		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x1, x0, 20	// __first_306, tmp.1204,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x0, 16]	// _46, MEM[(float *)__first_201 + 16B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x1	// _63, __first_306
	beq	.L538		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x1, x0, 24	// __first_310, tmp.1204,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x0, 20]	// _46, MEM[(float *)__first_201 + 20B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x1	// _63, __first_310
	beq	.L538		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x1, x0, 28	// __first_314, tmp.1204,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x0, 24]	// _46, MEM[(float *)__first_201 + 24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x2, x1	// _63, __first_314
	beq	.L538		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x0, 28]	// _46, MEM[(float *)__first_201 + 28B]
.L538:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1152:       return __first + __n;
	mov	x0, x2	// _63, _63
.L535:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:733: 		  this->_M_impl._M_finish =
	str	x0, [x22, 8]	// _63, this_21(D)->D.54862._M_impl.D.54163._M_finish
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:451: 	  if (__builtin_expect(_Num > 1, true))
	cmp	x20, 4	// _35,
	ble	.L539		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:452: 	    __builtin_memmove(__result, __first, sizeof(_Tp) * _Num);
	mov	x2, x20	//, _35
	mov	x1, x21	//, __position
	bl	memmove		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:742: 		  this->_M_impl._M_finish += __elems_after;
	ldr	x0, [x22, 8]	// _63, this_21(D)->D.54862._M_impl.D.54163._M_finish
.L540:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:742: 		  this->_M_impl._M_finish += __elems_after;
	add	x0, x0, x20	// _14, _63, _35
	str	x0, [x22, 8]	// _14, this_21(D)->D.54862._M_impl.D.54163._M_finish
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x23, x21	// _2, __position
	beq	.L606		//,
	sub	x1, x23, x21	// _195, _2, __position
	mov	x0, x21	// ivtmp.1227, __position
	sub	x1, x1, #4	// _159, _195,
	lsr	x2, x1, 2	// _160, _159,
	add	x2, x2, 1	// niters.1195, _160,
	cmp	x1, 28	// _159,
	bls	.L543		//,
	lsr	x1, x2, 3	// bnd.1196_173, niters.1195,
	dup	v29.4s, v15.s[0]	//, _46
	add	x1, x21, x1, lsl 5	// _299, __position, bnd.1196_173,
	.p2align 5,,15
.L544:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	stp	q29, q29, [x0], 32	//,, ivtmp.1227,
	cmp	x0, x1	// ivtmp.1227, _299
	bne	.L544		//,
	and	x24, x2, -8	// niters_vector_mult_vf.1197_174, niters.1195,
	add	x24, x21, x24, lsl 2	// tmp.1198, __position, niters_vector_mult_vf.1197_174,
	tst	x2, 7	// niters.1195,
	beq	.L606		//,
.L543:
	mov	x0, x24	// __first_262, tmp.1198
	str	s15, [x0], 4	// _46, *__first_171
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x23, x0	// _2, __first_262
	beq	.L606		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x0, x24, 8	// __first_266, tmp.1198,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 4]	// _46, MEM[(float *)__first_171 + 4B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x23, x0	// _2, __first_266
	beq	.L606		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x0, x24, 12	// __first_270, tmp.1198,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 8]	// _46, MEM[(float *)__first_171 + 8B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x23, x0	// _2, __first_270
	beq	.L606		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x0, x24, 16	// __first_274, tmp.1198,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 12]	// _46, MEM[(float *)__first_171 + 12B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x23, x0	// _2, __first_274
	beq	.L606		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x0, x24, 20	// __first_278, tmp.1198,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 16]	// _46, MEM[(float *)__first_171 + 16B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x23, x0	// _2, __first_278
	beq	.L606		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x0, x24, 24	// __first_282, tmp.1198,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 20]	// _46, MEM[(float *)__first_171 + 20B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x23, x0	// _2, __first_282
	beq	.L606		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x0, x24, 28	// __first_286, tmp.1198,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s15, [x24, 24]	// _46, MEM[(float *)__first_171 + 24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x23, x0	// _2, __first_286
	bne	.L607		//,
.L606:
	ldr	d15, [sp, 88]	//,
	.cfi_restore 79
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:799:     }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x29, x30, [sp], 112	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L522:
	.cfi_def_cfa_offset 112
	.cfi_offset 19, -96
	.cfi_offset 20, -88
	.cfi_offset 21, -80
	.cfi_offset 22, -72
	.cfi_offset 23, -64
	.cfi_offset 24, -56
	.cfi_offset 29, -112
	.cfi_offset 30, -104
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 26, -40
	.cfi_offset 25, -48
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1908: 	if (max_size() - size() < __n)
	mov	x1, 2305843009213693951	// tmp305,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:751: 	      pointer __old_start = this->_M_impl._M_start;
	ldr	x26, [x0]	// __old_start, this_21(D)->D.54862._M_impl.D.54163._M_start
	str	x27, [sp, 80]	//,
	.cfi_offset 27, -32
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:993:       { return size_type(this->_M_impl._M_finish - this->_M_impl._M_start); }
	sub	x4, x23, x26	// _127, _2, __old_start
	asr	x4, x4, 2	// _128, _127,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1908: 	if (max_size() - size() < __n)
	sub	x0, x1, x4	// _130, tmp305, _128
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1908: 	if (max_size() - size() < __n)
	cmp	x19, x0	// __n, _130
	bhi	.L608		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:262:       if (__a < __b)
	cmp	x19, x4	// __n, _128
	csel	x0, x19, x4, cs	// _132, __n, _128,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:757: 	      const size_type __elems_before = __pos - __old_start;
	sub	x24, x21, x26	// _23, __position, __old_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1912: 	return (__len < size() || __len > max_size()) ? max_size() : __len;
	adds	x4, x4, x0	// tmp309, _128, _132
	bcs	.L562		//,
	mov	x20, 0	// _66,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	mov	x25, 0	// iftmp.18_84,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	cbnz	x4, .L609	// tmp309,
.L551:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1151:       std::__fill_a(__first, __first + __n, __value);
	lsl	x27, x19, 2	// _77, __n,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:763: 		  std::__uninitialized_fill_n_a(__new_start + __elems_before,
	add	x4, x25, x24	// tmp.1210, iftmp.18_84, _23
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:950:       const _Tp __tmp = __value;
	ldr	s30, [x3]	// __tmp, MEM[(const float &)__x_32(D)]
	sub	x1, x27, #4	// _219, _77,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1151:       std::__fill_a(__first, __first + __n, __value);
	add	x0, x4, x19, lsl 2	// _78, tmp.1210, __n,
	and	x19, x19, 4611686018427387903	// _248, __n,
	cmp	x1, 28	// _219,
	bls	.L552		//,
	dup	v31.4s, v30.s[0]	//, __tmp
	and	x1, x27, -32	// _319, _77,
	mov	x3, x4	// ivtmp.1241, tmp.1210
	add	x1, x1, x4	// _239, _319, tmp.1210
	.p2align 5,,15
.L553:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	stp	q31, q31, [x3], 32	//,, ivtmp.1241,
	cmp	x1, x3	// _239, ivtmp.1241
	bne	.L553		//,
	and	x1, x19, -8	// niters_vector_mult_vf.1209_234, _248,
	add	x4, x4, x1, lsl 2	// tmp.1210, tmp.1210, niters_vector_mult_vf.1209_234,
	tst	x19, 7	// _248,
	beq	.L554		//,
.L552:
	mov	x1, x4	// __first_318, tmp.1210
	str	s30, [x1], 4	// __tmp, *__first_231
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x0, x1	// _78, __first_318
	beq	.L554		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x1, x4, 8	// __first_322, tmp.1210,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s30, [x4, 4]	// __tmp, MEM[(float *)__first_231 + 4B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x0, x1	// _78, __first_322
	beq	.L554		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x1, x4, 12	// __first_326, tmp.1210,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s30, [x4, 8]	// __tmp, MEM[(float *)__first_231 + 8B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x0, x1	// _78, __first_326
	beq	.L554		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x1, x4, 16	// __first_330, tmp.1210,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s30, [x4, 12]	// __tmp, MEM[(float *)__first_231 + 12B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x0, x1	// _78, __first_330
	beq	.L554		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x1, x4, 20	// __first_334, tmp.1210,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s30, [x4, 16]	// __tmp, MEM[(float *)__first_231 + 16B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x0, x1	// _78, __first_334
	beq	.L554		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x1, x4, 24	// __first_338, tmp.1210,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s30, [x4, 20]	// __tmp, MEM[(float *)__first_231 + 20B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x0, x1	// _78, __first_338
	beq	.L554		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	add	x1, x4, 28	// __first_342, tmp.1210,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s30, [x4, 24]	// __tmp, MEM[(float *)__first_231 + 24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	cmp	x0, x1	// _78, __first_342
	beq	.L554		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	str	s30, [x4, 28]	// __tmp, MEM[(float *)__first_231 + 28B]
.L554:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:451: 	  if (__builtin_expect(_Num > 1, true))
	cmp	x24, 4	// _23,
	ble	.L555		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:452: 	    __builtin_memmove(__result, __first, sizeof(_Tp) * _Num);
	mov	x2, x24	//, _23
	mov	x1, x26	//, __old_start
	mov	x0, x25	//, iftmp.18_84
	bl	memmove		//
.L556:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:772: 		  __new_finish += __n;
	add	x24, x24, x27	// _27, _23, _77
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:450: 	  const ptrdiff_t _Num = __last - __first;
	sub	x23, x23, x21	// _105, _2, __position
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:772: 		  __new_finish += __n;
	add	x24, x25, x24	// __new_finish, iftmp.18_84, _27
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:451: 	  if (__builtin_expect(_Num > 1, true))
	cmp	x23, 4	// _105,
	ble	.L557		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:452: 	    __builtin_memmove(__result, __first, sizeof(_Tp) * _Num);
	mov	x1, x21	//, __position
	mov	x2, x23	//, _105
	mov	x0, x24	//, __new_finish
	bl	memcpy		//
.L558:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:456: 	  return __result + _Num;
	add	x24, x24, x23	// _71, __new_finish, _105
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x26, .L559	// __old_start,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:793: 			    this->_M_impl._M_end_of_storage - __old_start);
	ldr	x1, [x22, 16]	// this_21(D)->D.54862._M_impl.D.54163._M_end_of_storage, this_21(D)->D.54862._M_impl.D.54163._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x0, x26	//, __old_start
	sub	x1, x1, x26	//, this_21(D)->D.54862._M_impl.D.54163._M_end_of_storage, __old_start
	bl	_ZdlPvm		//
.L559:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:794: 	      this->_M_impl._M_start = __new_start;
	stp	x25, x24, [x22]	// iftmp.18_84, _71,* this
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:796: 	      this->_M_impl._M_end_of_storage = __new_start + __len;
	ldr	x27, [sp, 80]	//,
	.cfi_remember_state
	.cfi_restore 27
	str	x20, [x22, 16]	// _66, this_21(D)->D.54862._M_impl.D.54163._M_end_of_storage
	ldp	x25, x26, [sp, 64]	//,,
	.cfi_restore 26
	.cfi_restore 25
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:799:     }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x29, x30, [sp], 112	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L562:
	.cfi_restore_state
	mov	x20, 9223372036854775804	// prephitmp_44,
.L550:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x0, x20	//, prephitmp_44
	str	x3, [sp, 104]	// __x, %sfp
	bl	_Znwm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:796: 	      this->_M_impl._M_end_of_storage = __new_start + __len;
	ldr	x3, [sp, 104]	// __x, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x25, x0	// iftmp.18_84, tmp352
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:796: 	      this->_M_impl._M_end_of_storage = __new_start + __len;
	add	x20, x0, x20	// _66, iftmp.18_84, prephitmp_44
	b	.L551		//
	.p2align 2,,3
.L609:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1912: 	return (__len < size() || __len > max_size()) ? max_size() : __len;
	cmp	x4, x1	// tmp309, tmp305
	csel	x4, x4, x1, ls	// _134, tmp309, tmp305,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	lsl	x20, x4, 2	// prephitmp_44, _134,
	b	.L550		//
	.p2align 2,,3
.L528:
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 27
	.cfi_offset 79, -24
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:767: 	  else if (_Num == 1)
	bne	.L529		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:438: 	{ *__to = std::move(*__from); }
	ldr	s31, [x21]	// tmp388, MEM[(type &)__position_76]
	str	s31, [x23, -4]	// tmp388, MEM[(float *)_2 + -4B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:438: 	{ *__to = std::move(*__from); }
	b	.L529		//
	.p2align 2,,3
.L524:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:453: 	  else if (_Num == 1)
	beq	.L526		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:700:     vector<_Tp, _Alloc>::
	mov	x0, x23	// prephitmp_126, _2
	b	.L525		//
	.p2align 2,,3
.L526:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:438: 	{ *__to = std::move(*__from); }
	ldr	s31, [x19]	// tmp387, MEM[(type &)_8]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:700:     vector<_Tp, _Alloc>::
	mov	x0, x23	// prephitmp_126, _2
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:438: 	{ *__to = std::move(*__from); }
	str	s31, [x23]	// tmp387, *_2
	b	.L525		//
	.p2align 2,,3
.L539:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:453: 	  else if (_Num == 1)
	bne	.L540		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:438: 	{ *__to = std::move(*__from); }
	ldr	s31, [x21]	// tmp389, MEM[(type &)__position_76]
	str	s31, [x0]	// tmp389, *_63
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:438: 	{ *__to = std::move(*__from); }
	b	.L540		//
	.p2align 2,,3
.L557:
	.cfi_offset 25, -48
	.cfi_offset 26, -40
	.cfi_offset 27, -32
	.cfi_restore 79
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:453: 	  else if (_Num == 1)
	bne	.L558		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:438: 	{ *__to = std::move(*__from); }
	ldr	s31, [x21]	// tmp391, MEM[(type &)__position_76]
	str	s31, [x24]	// tmp391, *__new_finish_34
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:438: 	{ *__to = std::move(*__from); }
	b	.L558		//
	.p2align 2,,3
.L555:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:453: 	  else if (_Num == 1)
	bne	.L556		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:438: 	{ *__to = std::move(*__from); }
	ldr	s31, [x26]	// tmp390, MEM[(type &)__old_start_22]
	str	s31, [x25]	// tmp390, *iftmp.18_84
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:438: 	{ *__to = std::move(*__from); }
	b	.L556		//
.L561:
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 27
	.cfi_offset 79, -24
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:951:       for (; __first != __last; ++__first)
	mov	x0, x23	// tmp.1204, _2
	b	.L536		//
.L608:
	.cfi_offset 25, -48
	.cfi_offset 26, -40
	.cfi_offset 27, -32
	.cfi_restore 79
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1909: 	  __throw_length_error(__N(__s));
	adrp	x0, .LC1	// tmp307,
	str	d15, [sp, 88]	//,
	.cfi_offset 79, -24
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1909: 	  __throw_length_error(__N(__s));
	add	x0, x0, :lo12:.LC1	//, tmp307,
	bl	_ZSt20__throw_length_errorPKc		//
	.cfi_endproc
.LFE5544:
	.size	_ZNSt6vectorIfSaIfEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPfS1_EEmRKf, .-_ZNSt6vectorIfSaIfEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPfS1_EEmRKf
	.text
	.align	2
	.p2align 5,,15
	.global	_Z34conv2d_forward_specialized_blockedRK6TensorS1_S1_RS_
	.type	_Z34conv2d_forward_specialized_blockedRK6TensorS1_S1_RS_, %function
_Z34conv2d_forward_specialized_blockedRK6TensorS1_S1_RS_:
.LFB5452:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA5452
	sub	sp, sp, #896	//,,
	.cfi_def_cfa_offset 896
	stp	x29, x30, [sp]	//,,
	.cfi_offset 29, -896
	.cfi_offset 30, -888
	mov	x29, sp	//,
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 21, -864
	.cfi_offset 22, -856
	mov	x21, x0	// input, tmp3863
// src/cpp/cnn_internals.cpp:438:     const int OH = output.height,  OW = output.width;
	add	x0, x3, 32	// tmp5802, output,
// src/cpp/cnn_internals.cpp:434: {
	mov	x22, x1	// weight, tmp3864
	stp	x19, x20, [sp, 16]	//,,
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 19, -880
	.cfi_offset 20, -872
	.cfi_offset 25, -832
	.cfi_offset 26, -824
	mov	x26, x2	// bias, tmp3865
	stp	x27, x28, [sp, 80]	//,,
	str	x3, [sp, 216]	// output, %sfp
	.cfi_offset 27, -816
	.cfi_offset 28, -808
// src/cpp/cnn_internals.cpp:438:     const int OH = output.height,  OW = output.width;
	ldp	w0, w28, [x0]	// _7, _8,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x27, [x21]	// in_ptr, MEM[(const struct vector *)input_29(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:438:     const int OH = output.height,  OW = output.width;
	str	w0, [sp, 180]	// _7, %sfp
// src/cpp/cnn_internals.cpp:447:     const int NV = (OW + 3) / 4;
	adds	w0, w28, 3	// _9, _8,
	add	w7, w28, 6	// tmp2166, _8,
	csel	w7, w7, w0, mi	// _9, tmp2166, _9,
// src/cpp/cnn_internals.cpp:435:     const int OC = weight.batches, IC = weight.channels;
	ldp	w1, w20, [x1, 24]	// _1, _2,
// src/cpp/cnn_internals.cpp:451:     const std::size_t slack = (std::size_t)(NV * 4 - OW);
	and	w0, w7, -4	// _17, _9,
// src/cpp/cnn_internals.cpp:447:     const int NV = (OW + 3) / 4;
	asr	w25, w7, 2	// NV, _9,
// src/cpp/cnn_internals.cpp:451:     const std::size_t slack = (std::size_t)(NV * 4 - OW);
	sub	w0, w0, w28	// _18, _17, _8
// src/cpp/cnn_internals.cpp:435:     const int OC = weight.batches, IC = weight.channels;
	str	w1, [sp, 108]	// _1, %sfp
// src/cpp/cnn_internals.cpp:450:     const std::size_t numel = (std::size_t)input.batches * IC * IH * IW;
	sxtw	x1, w20	// _12, _2
	str	x1, [sp, 248]	// _12, %sfp
// src/cpp/cnn_internals.cpp:437:     const int IH = input.height,   IW = input.width;
	ldp	w3, w2, [x21, 32]	// _5, _6,
// src/cpp/cnn_internals.cpp:450:     const std::size_t numel = (std::size_t)input.batches * IC * IH * IW;
	sxtw	x4, w2	// _6, _6
// src/cpp/cnn_internals.cpp:437:     const int IH = input.height,   IW = input.width;
	str	w3, [sp, 184]	// _5, %sfp
	str	w2, [sp, 224]	// _6, %sfp
// src/cpp/cnn_internals.cpp:450:     const std::size_t numel = (std::size_t)input.batches * IC * IH * IW;
	str	x4, [sp, 208]	// _6, %sfp
// src/cpp/cnn_internals.cpp:436:     const int KH = weight.height,  KW = weight.width;
	ldp	w19, w4, [x22, 32]	// _3, _4,
	str	w4, [sp, 176]	// _4, %sfp
// src/cpp/cnn_internals.cpp:454:     if (slack && input.data.size() < numel + slack) {
	cbz	w0, .L611	// _18,
	sxtw	x1, w0	// slack, _18
// src/cpp/cnn_internals.cpp:450:     const std::size_t numel = (std::size_t)input.batches * IC * IH * IW;
	smull	x0, w3, w2	// _2133, _5, _6
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -840
	.cfi_offset 23, -848
	ldr	w3, [x21, 24]	//, input_29(D)->batches
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:993:       { return size_type(this->_M_impl._M_finish - this->_M_impl._M_start); }
	ldr	x2, [x21, 8]	// MEM[(const struct vector *)input_29(D)].D.54862._M_impl.D.54163._M_finish, MEM[(const struct vector *)input_29(D)].D.54862._M_impl.D.54163._M_finish
// src/cpp/cnn_internals.cpp:450:     const std::size_t numel = (std::size_t)input.batches * IC * IH * IW;
	smull	x3, w20, w3	// _988, _2, input_29(D)->batches
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:993:       { return size_type(this->_M_impl._M_finish - this->_M_impl._M_start); }
	sub	x2, x2, x27	// _40, MEM[(const struct vector *)input_29(D)].D.54862._M_impl.D.54163._M_finish, in_ptr
// src/cpp/cnn_internals.cpp:454:     if (slack && input.data.size() < numel + slack) {
	madd	x23, x3, x0, x1	// _19, _988, _2133, slack
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:993:       { return size_type(this->_M_impl._M_finish - this->_M_impl._M_start); }
	asr	x0, x2, 2	// _84, _40,
	str	x0, [sp, 192]	// _84, %sfp
// src/cpp/cnn_internals.cpp:454:     if (slack && input.data.size() < numel + slack) {
	cmp	x23, x0	// _19, _84
	bhi	.L1311		//,
	ldp	x23, x24, [sp, 48]	//,,
	.cfi_restore 24
	.cfi_restore 23
.L611:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x0, [x26]	// _87, MEM[(const struct vector *)bias_46(D)].D.54862._M_impl.D.54163._M_start
	str	x0, [sp, 200]	// _87, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1263:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x0, [sp, 216]	// output, %sfp
	ldr	x0, [x0]	// _86, MEM[(struct vector *)output_32(D)].D.54862._M_impl.D.54163._M_start
	str	x0, [sp, 216]	// _86, %sfp
// src/cpp/cnn_internals.cpp:482:     switch (NV) {
	cmp	w25, 4	// NV,
	beq	.L1080		//,
	cmp	w28, 16	// _8,
	bgt	.L1029		//,
	cmp	w25, 2	// NV,
	beq	.L1081		//,
	cmp	w25, 3	// NV,
	beq	.L1082		//,
	cmp	w25, 1	// NV,
	bne	.L610		//,
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -840
	.cfi_offset 23, -848
	mov	x23, 0	// in_pad$_M_end_of_storage,
	str	xzr, [sp, 192]	//, %sfp
	str	xzr, [sp, 240]	//, %sfp
.L622:
	ldr	w1, [sp, 176]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	mov	x0, 2305843009213693951	// tmp2188,
// src/cpp/cnn_internals.cpp:468:         std::vector<float> packed((std::size_t)OC * IC * KH * KW);
	ldr	w3, [sp, 108]	//, %sfp
	smull	x2, w19, w1	// _5960, _3, _4
	sxtw	x24, w1	// _4, _4
	sbfiz	x1, x1, 2, 32	// _197, _4,,
	smull	x25, w20, w3	// _5938, _2, _1
	stp	x2, x1, [sp, 272]	// _5960, _197,
	mul	x25, x25, x2	// _200, _5938, _5960
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	cmp	x25, x0	// _200, tmp2188
	bhi	.L1312		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	cbz	x25, .L1045	// _200,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	lsl	x7, x25, 2	// _256, _200,
	mov	x0, x7	//, _256
	str	x7, [sp, 256]	// _256, %sfp
.LEHB0:
	bl	_Znwm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	str	wzr, [x0]	//, *_257
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x3, x0	// _257, tmp3869
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	cmp	x25, 1	// _200,
	beq	.L1046		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	ldr	x7, [sp, 256]	// _256, %sfp
	mov	w1, 0	//,
	add	x0, x0, 4	//, _257,
	sub	x2, x7, #4	//, _256,
	stp	x3, x7, [sp, 256]	// _257, _256,
	bl	memset		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	ldp	x3, x7, [sp, 256]	// _257, _256,
	mov	x25, x7	// _200, _256
	b	.L628		//
.L1029:
	.cfi_restore 23
	.cfi_restore 24
// src/cpp/cnn_internals.cpp:482:     switch (NV) {
	cmp	w25, 6	// NV,
	beq	.L1084		//,
	cmp	w25, 7	// NV,
	beq	.L1085		//,
	cmp	w25, 5	// NV,
	bne	.L610		//,
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -840
	.cfi_offset 23, -848
	mov	x23, 0	// in_pad$_M_end_of_storage,
	str	xzr, [sp, 192]	//, %sfp
	str	xzr, [sp, 240]	//, %sfp
.L626:
	ldr	w0, [sp, 176]	//, %sfp
// src/cpp/cnn_internals.cpp:468:         std::vector<float> packed((std::size_t)OC * IC * KH * KW);
	ldr	w1, [sp, 108]	//, %sfp
	smull	x24, w19, w0	// _5871, _3, _4
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	mov	x0, 2305843009213693951	// tmp3066,
// src/cpp/cnn_internals.cpp:468:         std::vector<float> packed((std::size_t)OC * IC * KH * KW);
	smull	x25, w20, w1	// _6084, _2, _1
	mul	x25, x25, x24	// _1524, _6084, _5871
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	cmp	x25, x0	// _1524, tmp3066
	bhi	.L1313		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	cbz	x25, .L1066	// _1524,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	lsl	x7, x25, 2	// _1580, _1524,
	mov	x0, x7	//, _1580
	str	x7, [sp, 112]	// _1580, %sfp
	bl	_Znwm		//
.LEHE0:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	str	wzr, [x0]	//, *_1581
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x3, x0	// _1581, tmp3873
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	cmp	x25, 1	// _1524,
	beq	.L1067		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	ldr	x7, [sp, 112]	// _1580, %sfp
	mov	w1, 0	//,
	add	x0, x0, 4	//, _1581,
	str	x3, [sp, 112]	// _1581, %sfp
	sub	x2, x7, #4	//, _1580,
	str	x7, [sp, 256]	// _1580, %sfp
	bl	memset		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	ldr	x7, [sp, 256]	// _1580, %sfp
	ldr	x3, [sp, 112]	// _1581, %sfp
	mov	x25, x7	// _1524, _1580
	b	.L869		//
.L1311:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:325: 	if (__len > capacity())
	cbz	x2, .L1037	// _40,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	mov	x0, 9223372036854775804	// tmp2176,
	cmp	x2, x0	// _40, tmp2176
	bhi	.L1314		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x0, x2	//, _40
	str	x2, [sp, 200]	// _40, %sfp
.LEHB1:
	bl	_Znwm		//
.LEHE1:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:451: 	  if (__builtin_expect(_Num > 1, true))
	ldr	x2, [sp, 200]	// _40, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x24, x0	// _167, tmp3868
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:451: 	  if (__builtin_expect(_Num > 1, true))
	cmp	x2, 4	// _40,
	beq	.L616		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:452: 	    __builtin_memmove(__result, __first, sizeof(_Tp) * _Num);
	mov	x1, x27	//, in_ptr
	bl	memcpy		//
	ldr	x2, [sp, 200]	// _40, %sfp
.L617:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/vector.tcc:339: 	    this->_M_impl._M_finish = this->_M_impl._M_start + __len;
	add	x1, x24, x2	// in_pad$_M_finish, _167, _40
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1037: 	  _M_fill_insert(end(), __new_size - size(), __x);
	ldr	x0, [sp, 192]	// _84, %sfp
	sub	x23, x23, x0	// _19, _19, _84
.L612:
	mov	x2, x23	//, _19
	add	x3, sp, 672	//,,
	str	x24, [sp, 640]	// _167, MEM <float *> [(struct vector *)&in_pad]
	add	x0, sp, 640	//,,
	str	x1, [sp, 648]	// in_pad$_M_finish, MEM <float *> [(struct vector *)&in_pad + 8B]
	str	x1, [sp, 656]	// in_pad$_M_finish, MEM <float *> [(struct vector *)&in_pad + 16B]
// src/cpp/cnn_internals.cpp:456:         in_pad.resize(numel + slack, 0.0f);
	str	wzr, [sp, 672]	//, MEM[(float *)_2907]
.LEHB2:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1037: 	  _M_fill_insert(end(), __new_size - size(), __x);
	bl	_ZNSt6vectorIfSaIfEE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPfS1_EEmRKf		//
.LEHE2:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x1, [x26]	// _87, MEM[(const struct vector *)bias_46(D)].D.54862._M_impl.D.54163._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1037: 	  _M_fill_insert(end(), __new_size - size(), __x);
	ldr	x0, [sp, 640]	// in_pad$_M_start, MEM <float *> [(struct vector *)&in_pad]
	ldr	x23, [sp, 656]	// in_pad$_M_end_of_storage, MEM <float *> [(struct vector *)&in_pad + 16B]
	stp	x0, x1, [sp, 192]	// in_pad$_M_start, _87,
// src/cpp/cnn_internals.cpp:482:     switch (NV) {
	mov	x27, x0	// in_ptr, in_pad$_M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1263:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x1, [sp, 216]	// output, %sfp
	ldr	x1, [x1]	// _86, MEM[(struct vector *)output_32(D)].D.54862._M_impl.D.54163._M_start
	str	x1, [sp, 216]	// _86, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	sub	x1, x23, x0	// _1379, in_pad$_M_end_of_storage, in_pad$_M_start
	str	x1, [sp, 240]	// _1379, %sfp
// src/cpp/cnn_internals.cpp:482:     switch (NV) {
	cmp	w25, 4	// NV,
	beq	.L618		//,
	cmp	w28, 16	// _8,
	bgt	.L619		//,
	cmp	w25, 2	// NV,
	beq	.L620		//,
	cmp	w25, 3	// NV,
	beq	.L621		//,
	cmp	w25, 1	// NV,
	beq	.L622		//,
	b	.L623		//
	.p2align 2,,3
.L1080:
	.cfi_restore 23
	.cfi_restore 24
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -840
	.cfi_offset 23, -848
	mov	x23, 0	// in_pad$_M_end_of_storage,
	str	xzr, [sp, 192]	//, %sfp
	str	xzr, [sp, 240]	//, %sfp
.L618:
	ldr	w1, [sp, 176]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	mov	x0, 2305843009213693951	// tmp2849,
	smull	x24, w19, w1	// _2355, _3, _4
	sbfiz	x1, x1, 2, 32	// _5700, _4,,
	str	x1, [sp, 360]	// _5700, %sfp
// src/cpp/cnn_internals.cpp:468:         std::vector<float> packed((std::size_t)OC * IC * KH * KW);
	ldr	w1, [sp, 108]	//, %sfp
	smull	x7, w20, w1	// _3022, _2, _1
	mul	x7, x7, x24	// _1192, _3022, _2355
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	cmp	x7, x0	// _1192, tmp2849
	bhi	.L1315		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	cbz	x7, .L1060	// _1192,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	lsl	x25, x7, 2	// _1248, _1192,
	str	x7, [sp, 112]	// _1192, %sfp
	mov	x0, x25	//, _1248
.LEHB3:
	bl	_Znwm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	ldr	x7, [sp, 112]	// _1192, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x23, x0	// _1249, tmp3872
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	str	wzr, [x0]	//, *_1249
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	cmp	x7, 1	// _1192,
	beq	.L1061		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	sub	x2, x25, #4	//, _1248,
	mov	w1, 0	//,
	add	x0, x0, 4	//, _1249,
	bl	memset		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	mov	x7, x25	// _1192, _1248
	b	.L810		//
.L1081:
	.cfi_restore 23
	.cfi_restore 24
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -840
	.cfi_offset 23, -848
// src/cpp/cnn_internals.cpp:482:     switch (NV) {
	mov	x23, 0	// in_pad$_M_end_of_storage,
	str	xzr, [sp, 192]	//, %sfp
	str	xzr, [sp, 240]	//, %sfp
.L620:
	ldr	w1, [sp, 176]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	mov	x0, 2305843009213693951	// tmp2390,
// src/cpp/cnn_internals.cpp:468:         std::vector<float> packed((std::size_t)OC * IC * KH * KW);
	ldr	w3, [sp, 108]	//, %sfp
	smull	x2, w19, w1	// _954, _3, _4
	sxtw	x24, w1	// _4, _4
	sbfiz	x1, x1, 2, 32	// _563, _4,,
	smull	x3, w20, w3	// _4154, _2, _1
	stp	x1, x2, [sp, 256]	// _563, _954,
	mul	x3, x3, x2	// _528, _4154, _954
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	cmp	x3, x0	// _528, tmp2390
	bhi	.L1316		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	cbz	x3, .L1048	// _528,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	lsl	x7, x3, 2	// _584, _528,
	mov	x0, x7	//, _584
	stp	x7, x3, [sp, 272]	// _584, _528,
	bl	_Znwm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	ldr	x3, [sp, 280]	// _528, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	str	wzr, [x0]	//, *_585
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	str	x0, [sp, 112]	// _585, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	cmp	x3, 1	// _528,
	beq	.L1049		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	ldr	x7, [sp, 272]	// _584, %sfp
	mov	w1, 0	//,
	add	x0, x0, 4	//, _585,
	sub	x2, x7, #4	//, _584,
	bl	memset		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	ldr	x7, [sp, 272]	// _584, %sfp
	mov	x3, x7	// _528, _584
	b	.L686		//
.L619:
// src/cpp/cnn_internals.cpp:482:     switch (NV) {
	cmp	w25, 6	// NV,
	beq	.L624		//,
	cmp	w25, 7	// NV,
	beq	.L625		//,
	cmp	w25, 5	// NV,
	beq	.L626		//,
	b	.L623		//
.L1060:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	mov	x23, 0	// _1249,
.L810:
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	ldr	w0, [sp, 176]	//, %sfp
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	mov	w3, 21846	// tmp2859,
	movk	w3, 0x5555, lsl 16	// tmp2859,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x8, [x22]	// _1194, MEM[(const struct vector *)weight_24(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	mul	w25, w19, w0	// ksz, _3, _4
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	ldr	w0, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:318:     const int wcsz  = in_channels * ksz;              // per-oc stride in OIHW
	mul	w22, w20, w25	// wcsz, _2, ksz
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	smull	x3, w0, w3	// tmp2858, _1, tmp2859
	asr	w16, w0, 31	// tmp3581, _1,
	lsr	x3, x3, 32	// tmp2860, tmp2858,
	sub	w3, w3, w16	// tiles, tmp2860, tmp3581
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w0, 2	// _1,
	ble	.L811		//,
	cmp	w20, 0	// _2,
	ble	.L811		//,
	sub	w0, w25, #1	// _5761, ksz,
	sxtw	x9, w22	// _1544, wcsz
	add	x15, x9, w22, sxtw 1	// _1546, _1544, wcsz,
	add	x0, x0, 1	// _414, _5761,
	lsr	w1, w25, 2	// _1533, ksz,
	add	x0, x0, x0, lsl 1	// _3844, _414, _414,
	mov	x12, 0	// ivtmp.2175,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w14, 0	// tb,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	x11, x9	// ivtmp.2176, _1544
	lsl	x30, x1, 4	// _1534, _1533,
	sub	w6, w25, #1	// _4314, ksz,
	mov	x10, x23	// packed, _1249
	lsl	x5, x0, 2	// tmp3856, _3844,
	mov	w13, 2	// ivtmp.2173,
	and	w18, w25, -4	// niters_vector_mult_vf.1571, ksz,
.L812:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	uxtw	x0, w13	// _3786, ivtmp.2173
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	w2, 0	// ivtmp.2169,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w1, 0	// ic,
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mul	x17, x0, x9	// _3788, _3786, _1544
.L819:
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w25, 0	// ksz,
	bgt	.L1317		//,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w1, w1, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w2, w2, w25	// ivtmp.2169, ivtmp.2169, ksz
	cmp	w20, w1	// _2, ic
	bne	.L819		//,
.L817:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	w14, w14, 1	// tb, tb,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	w13, w13, 3	// ivtmp.2173, ivtmp.2173,
	add	x12, x12, x15	// ivtmp.2175, ivtmp.2175, _1546
	add	x11, x11, x15	// ivtmp.2176, ivtmp.2176, _1546
	cmp	w3, w14	// tiles, tb
	bgt	.L812		//,
.L811:
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	ldr	w1, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:471:         const int n_full = (OC / OC_TILE) * OC_TILE;
	add	w0, w3, w3, lsl 1	// n_full, tiles, tiles,
	str	w0, [sp, 256]	// n_full, %sfp
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	cmp	w1, w0	// _1, n_full
	bgt	.L1318		//,
.L820:
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w1, [sp, 184]	//, %sfp
	ldr	w2, [sp, 224]	//, %sfp
// src/cpp/cnn_internals.cpp:478:             input.batches, IC, OC, IH, IW, KH, KW, OH, OW);
	ldr	w0, [x21, 24]	//, input_29(D)->batches
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	ldr	w4, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	mul	w17, w1, w2	// in_size, _5, _6
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	ldr	w1, [sp, 180]	//, %sfp
// src/cpp/cnn_internals.cpp:404:     const int tile_wsz    = in_channels * ksz * OC_TILE;   // packed floats/tile
	add	w2, w22, w22, lsl 1	// tile_wsz_1318, wcsz, wcsz,
// src/cpp/cnn_internals.cpp:478:             input.batches, IC, OC, IH, IW, KH, KW, OH, OW);
	str	w0, [sp, 376]	// _1224, %sfp
// src/cpp/cnn_internals.cpp:400:     const int in_ch_size  = in_channels * in_size;
	mul	w13, w20, w17	// in_ch_size, _2, in_size
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	mul	w1, w1, w28	// out_size, _7, _8
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	mul	w11, w4, w1	// out_ch_size, _1, out_size
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	cmp	w0, 0	// _1224,
	ble	.L829		//,
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, 0	// _7,
	ble	.L829		//,
	ldr	w4, [sp, 176]	//, %sfp
	smull	x14, w3, w2	// _418, tiles, tile_wsz_1318
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	sub	w0, w28, #4	// _5070, _8,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	mov	w9, 0	// b,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	lsr	w15, w0, 2	// _5058, _5070,
	and	w0, w0, -4	// _5796, _5070,
	ldr	x5, [sp, 248]	// _12, %sfp
	add	w0, w0, 4	// col, _5796,
	mov	x16, x27	// ivtmp.2141, in_ptr
	add	x6, x23, x14, lsl 2	// ivtmp.2095, _1249, _418,
	sub	w8, w4, #1	// _5786, _4,
	add	x22, x8, 1	// _430, _5786,
	mov	w8, 12	// tmp2927,
	mul	x12, x5, x24	// _856, _12, _2355
	add	w5, w15, 1	// _415, _5058,
	smull	x4, w4, w8	// _1347, _4, tmp2927
	add	x10, x22, x22, lsl 1	// _3973, _430, _430,
	str	w5, [sp, 368]	// _415, %sfp
	stp	x6, x4, [sp, 392]	// ivtmp.2095, _1347,
	sbfiz	x6, x13, 2, 32	// _1507, in_ch_size,,
	sbfiz	x4, x11, 2, 32	// _1513, out_ch_size,,
	stp	x6, x4, [sp, 424]	// _1507, _1513,
	smull	x6, w1, w8	// _1435, out_size, tmp2927
	lsl	x4, x10, 2	// tmp3841, _3973,
	ldr	x8, [sp, 216]	// ivtmp.2142, %sfp
	str	w0, [sp, 216]	// col, %sfp
	mov	x0, x5	// tmp6151, _415
	ubfiz	x0, x0, 4, 31	// _5054, tmp6151,,
	str	x6, [sp, 408]	// _1435, %sfp
	lsl	x6, x12, 2	// _1275, _856,
	str	x0, [sp, 352]	// _5054, %sfp
	sbfiz	x0, x2, 2, 32	// _1455, tile_wsz_1318,,
	str	x6, [sp, 384]	// _1275, %sfp
	str	x0, [sp, 416]	// _1455, %sfp
// src/cpp/cnn_internals.cpp:376:                             + (std::size_t)(oc0 + t) * out_size
	sxtw	x0, w1	// _2835, out_size
	stp	x0, x4, [sp, 440]	// _2835, tmp3841,
	sbfiz	x0, x1, 2, 32	// _1295, out_size,,
	str	x0, [sp, 112]	// _1295, %sfp
	sxtw	x0, w28	// _2604, _8
	str	x0, [sp, 184]	// _2604, %sfp
.L867:
// src/cpp/cnn_internals.cpp:410:         float*       __restrict out_b = out_ptr   + (std::size_t)b * out_ch_size;
	mov	x1, 0	// ivtmp.2136,
	mov	x26, 0	// ivtmp.2135,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	mov	w27, 0	// oh,
.L830:
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	w0, [sp, 256]	//, %sfp
	cmp	w0, 0	// n_full,
	ble	.L866		//,
	add	x25, sp, 672	// tmp3576,,
	mov	w10, w20	// _2, _2
	mov	x21, x25	// tmp3576, tmp3576
	mov	w25, w19	// _3, _3
	mov	x19, x26	// ivtmp.2135, ivtmp.2135
	ldr	x26, [sp, 208]	// _6, %sfp
	mov	w20, w27	// oh, oh
	mov	x24, x23	// _1249, _1249
	mov	x5, x22	// _430, _430
.L847:
	ldr	x22, [sp, 200]	// ivtmp.2129, %sfp
	add	x4, x8, x19, lsl 2	// ivtmp.2130, ivtmp.2142, ivtmp.2135,
// src/cpp/cnn_internals.cpp:413:             int oc0 = 0, tile = 0;
	mov	w23, 0	// oc0,
	mov	x15, x24	// _1249, _1249
	mov	x13, x19	// ivtmp.2135, ivtmp.2135
	mov	w24, w10	// _2, _2
	mov	x27, x4	// ivtmp.2130, ivtmp.2130
	mov	x10, x26	// _6, _6
	mov	w14, w20	// oh, oh
	mov	w26, w23	// oc0, oc0
	mov	x4, x15	// ivtmp.2131, ivtmp.2131
	mov	x23, x22	// ivtmp.2129, ivtmp.2129
	mov	x22, x21	// tmp3576, tmp3576
.L845:
// src/cpp/cnn_internals.cpp:347:         const float32x4_t bv = vdupq_n_f32(bias_ptr[oc0 + t]);
	ldp	s18, s19, [x23, 4]	// _4769, _4783,
// src/cpp/cnn_internals.cpp:415:                 conv_row_tile<OC_TILE, NV>(
	mov	x18, x4	// wp, ivtmp.2131
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v17.4s}, [x23]	//, MEM[(const float *)_1457]
	dup	v18.4s, v18.s[0]	//, _4769
	dup	v19.4s, v19.s[0]	//, _4783
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q18, q18, [sp, 736]	//,,
	stp	q17, q17, [sp, 672]	//,,
	stp	q17, q17, [sp, 704]	//,,
	stp	q18, q18, [sp, 768]	//,,
	stp	q19, q19, [sp, 800]	//,,
	stp	q19, q19, [sp, 832]	//,,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w24, 0	// _2,
	ble	.L833		//,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	sxtw	x19, w17	// _1902, in_size
	cmp	w25, 0	// _3,
	ble	.L833		//,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w12, 0	// ic,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	mov	x2, x1	// ivtmp.2121, ivtmp.2136
	.p2align 5,,15
.L839:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w11, 0	// kh,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x20, x2	// ivtmp.2117, ivtmp.2121
.L838:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w0, [sp, 176]	//, %sfp
	cmp	w0, 0	// _4,
	bgt	.L836		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w11, w11, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x20, x20, x10	// ivtmp.2117, ivtmp.2117, _6
	cmp	w25, w11	// _3, kh
	bne	.L838		//,
	.p2align 5,,15
.L837:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w12, w12, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x2, x2, x19	// ivtmp.2121, ivtmp.2121, _1902
	cmp	w24, w12	// _2, ic
	bne	.L839		//,
.L833:
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	mov	w19, 0	// t,
	mov	x21, x27	// ivtmp.2102, ivtmp.2102
	str	x23, [sp, 288]	// ivtmp.2129, %sfp
	mov	w20, w19	// t, t
	mov	x23, x22	// ivtmp.2103, ivtmp.2103
	str	w9, [sp, 208]	// b, %sfp
	mov	w19, w28	// _8, _8
	mov	x28, x5	// _430, _430
	str	x13, [sp, 224]	// ivtmp.2135, %sfp
	str	w14, [sp, 248]	// oh, %sfp
	stp	x7, x15, [sp, 264]	// _1192, _1249,
	str	w3, [sp, 280]	// tiles, %sfp
	str	w17, [sp, 304]	// in_size, %sfp
	stp	x4, x16, [sp, 312]	// ivtmp.2131, ivtmp.2141,
	stp	x8, x1, [sp, 328]	// ivtmp.2142, ivtmp.2136,
	str	x10, [sp, 344]	// _6, %sfp
.L832:
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w19, 3	// _8,
	ble	.L840		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	x2, [sp, 352]	//, %sfp
	mov	x1, x23	//, ivtmp.2103
	mov	x0, x21	//, ivtmp.2102
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	ldr	w0, [sp, 216]	//, %sfp
	cmp	w19, w0	// _8, col
	bgt	.L841		//,
.L1319:
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	w20, w20, 1	// t, t,
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	cmp	w20, 3	// t,
	beq	.L1289		//,
	ldr	x0, [sp, 112]	// _1295, %sfp
	add	x23, x23, 64	// ivtmp.2103, ivtmp.2103,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	mov	x1, x23	//, ivtmp.2103
	ldr	x2, [sp, 352]	//, %sfp
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	x21, x21, x0	// ivtmp.2102, ivtmp.2102, _1295
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	mov	x0, x21	//, ivtmp.2102
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	ldr	w0, [sp, 216]	//, %sfp
	cmp	w19, w0	// _8, col
	ble	.L1319		//,
.L841:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldrsw	x0, [sp, 368]	// v, %sfp
	add	x0, x0, w20, sxtw 2	// tmp2975, v, t,
	lsl	x0, x0, 4	// tmp2976, tmp2975,
	ldr	q31, [x22, x0]	// _1426, MEM[(__Float32x4_t[3][4] *)_2907][t_5940][v_3666]
	ldr	w7, [sp, 216]	//, %sfp
.L844:
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w19, #1	// _2060, _8,
	uxtw	x0, w7	// _5075, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// _1426, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w2, w7	// _5089, _2060, col
	cmp	w19, w7	// _8, col
	add	x0, x21, x0, lsl 2	//, ivtmp.2102, _5075,
	mov	x7, 4	// tmp2987,
	add	x1, sp, 624	//,,
	add	x2, x2, 1	// _5088, _5089,
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	w20, w20, 1	// t, t,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	lsl	x2, x2, 2	// _5085, _5088,
	csel	x2, x2, x7, gt	//, _5085, tmp2987,
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	x23, x23, 64	// ivtmp.2103, ivtmp.2103,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	ldr	x0, [sp, 112]	// _1295, %sfp
	add	x21, x21, x0	// ivtmp.2102, ivtmp.2102, _1295
	cmp	w20, 3	// t,
	bne	.L832		//,
.L1289:
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	x0, [sp, 408]	// _1435, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	w26, w26, 3	// oc0, oc0,
	mov	x5, x28	// _430, _430
	ldp	x4, x16, [sp, 312]	// ivtmp.2131, ivtmp.2141,
	mov	w28, w19	// _8, _8
	ldp	x7, x15, [sp, 264]	// _1192, _1249,
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	x27, x27, x0	// ivtmp.2130, ivtmp.2130, _1435
	ldp	x8, x1, [sp, 328]	// ivtmp.2142, ivtmp.2136,
	ldr	x0, [sp, 416]	// _1455, %sfp
	ldr	x23, [sp, 288]	// ivtmp.2129, %sfp
	ldr	x13, [sp, 224]	// ivtmp.2135, %sfp
	add	x4, x4, x0	// ivtmp.2131, ivtmp.2131, _1455
	ldr	w0, [sp, 256]	//, %sfp
	add	x23, x23, 12	// ivtmp.2129, ivtmp.2129,
	ldr	x10, [sp, 344]	// _6, %sfp
	ldr	w9, [sp, 208]	//, %sfp
	ldr	w14, [sp, 248]	//, %sfp
	ldr	w3, [sp, 280]	//, %sfp
	ldr	w17, [sp, 304]	//, %sfp
	cmp	w0, w26	// n_full, oc0
	bgt	.L845		//,
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	mov	w23, w26	// oc0, oc0
	mov	x21, x22	// tmp3576, tmp3576
	ldr	w0, [sp, 108]	//, %sfp
	mov	x26, x10	// _6, _6
	mov	w10, w24	// _2, _2
	mov	x24, x15	// _1249, _1249
	cmp	w0, w23	// _1, oc0
	ble	.L1320		//,
	mov	w19, w25	// _3, _3
	mov	w27, w14	// oh, oh
	str	x26, [sp, 208]	// _6, %sfp
	mov	x25, x21	// tmp3576, tmp3576
	mov	w20, w10	// _2, _2
	mov	w21, w23	// oc0, oc0
	mov	x22, x5	// _430, _430
	mov	x26, x13	// ivtmp.2135, ivtmp.2135
	mov	x23, x15	// _1249, _1249
.L846:
	str	x23, [sp, 248]	// _1249, %sfp
	mov	w23, w20	// _2, _2
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	sxtw	x0, w17	// _5496, in_size
	ldr	x20, [sp, 208]	// _6, %sfp
	mov	x4, x26	// ivtmp.2135, ivtmp.2135
	mov	w5, w27	// oh, oh
	mov	x26, x16	// ivtmp.2141, ivtmp.2141
	mov	w27, w19	// _3, _3
	str	x0, [sp, 280]	// _5496, %sfp
	lsl	x0, x22, 2	// _5731, _430,
	str	w9, [sp, 224]	// b, %sfp
	str	x22, [sp, 264]	// _430, %sfp
	str	x0, [sp, 304]	// _5731, %sfp
.L850:
	ldr	w2, [sp, 108]	//, %sfp
	sxtw	x0, w21	// _1260, oc0
	str	x1, [sp, 272]	// ivtmp.2136, %sfp
	ldr	x6, [sp, 440]	// _2835, %sfp
	str	x4, [sp, 288]	// ivtmp.2135, %sfp
// src/cpp/cnn_internals.cpp:413:             int oc0 = 0, tile = 0;
	ldr	x22, [sp, 392]	// ivtmp.2095, %sfp
	sub	w19, w2, w21	// _1288, _1, oc0
	str	w5, [sp, 312]	// oh, %sfp
	ldr	x2, [sp, 200]	// _87, %sfp
	add	x19, x19, x0	// _1289, _1288, _1260
	add	x24, x2, w21, uxtw 2	// ivtmp.2091, _87, oc0,,
	madd	x0, x0, x6, x4	// _1270, _1260, _2835, ivtmp.2135
	str	x7, [sp, 320]	// _1192, %sfp
	str	w3, [sp, 328]	// tiles, %sfp
	str	w17, [sp, 336]	// in_size, %sfp
	add	x21, x8, x0, lsl 2	// ivtmp.2094, ivtmp.2142, _1270,
	mov	x0, x25	// tmp3576, tmp3576
	mov	w25, w28	// _8, _8
	str	x8, [sp, 344]	// ivtmp.2142, %sfp
	add	x19, x2, x19, lsl 2	// _1292, _87, _1289,
	mov	x28, x21	// ivtmp.2094, ivtmp.2094
	mov	x21, x0	// tmp3576, tmp3576
.L864:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v22.4s}, [x24]	//, MEM[(const float *)_1283]
// src/cpp/cnn_internals.cpp:421:                 conv_row_tile<1, NV>(
	mov	x8, x22	// wp, ivtmp.2095
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q22, q22, [x21]	//,,
	stp	q22, q22, [x21, 32]	//,,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w23, 0	// _2,
	ble	.L854		//,
	cmp	w27, 0	// _3,
	ble	.L854		//,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	ldp	x0, x1, [sp, 272]	// ivtmp.2085, _5496,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w7, 0	// ic,
	.p2align 5,,15
.L860:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w2, 0	// kh,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x4, x0	// ivtmp.2081, ivtmp.2085
.L859:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w3, [sp, 176]	//, %sfp
	cmp	w3, 0	// _4,
	bgt	.L857		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w2, w2, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x20	// ivtmp.2081, ivtmp.2081, _6
	cmp	w27, w2	// _3, kh
	bne	.L859		//,
	.p2align 5,,15
.L858:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w7, w7, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x0, x0, x1	// ivtmp.2085, ivtmp.2085, _5496
	cmp	w23, w7	// _2, ic
	bne	.L860		//,
.L854:
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w25, 3	// _8,
	ble	.L1321		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	x2, [sp, 352]	//, %sfp
	mov	x1, x21	//, tmp3576
	mov	x0, x28	//, ivtmp.2094
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	ldr	w0, [sp, 216]	//, %sfp
	ldr	w1, [sp, 368]	//, %sfp
	mov	w4, w0	// col, col
	cmp	w25, w0	// _8, col
	ble	.L863		//,
.L861:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ubfiz	x1, x1, 4, 31	// tmp3047, v,,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w25, #1	// _3110, _8,
	sub	w2, w2, w4	// _5030, _3110, col
	uxtw	x0, w4	// _5018, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [x21, x1]	// tmp6182, MEM[(__Float32x4_t[1][4] *)_2907][0][v_5992]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	cmp	w25, w4	// _8, col
	add	x2, x2, 1	// _5028, _5030,
	add	x0, x28, x0, lsl 2	//, ivtmp.2094, _5018,
	mov	x4, 4	// tmp3059,
	add	x1, sp, 624	//,,
	lsl	x2, x2, 2	// _5027, _5028,
	csel	x2, x2, x4, gt	//, _5027, tmp3059,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// tmp6182, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	bl	memcpy		//
.L863:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	x0, [sp, 112]	// _1295, %sfp
	add	x24, x24, 4	// ivtmp.2091, ivtmp.2091,
	add	x28, x28, x0	// ivtmp.2094, ivtmp.2094, _1295
	ldr	x0, [sp, 384]	// _1275, %sfp
	add	x22, x22, x0	// ivtmp.2095, ivtmp.2095, _1275
	cmp	x24, x19	// ivtmp.2091, _1292
	bne	.L864		//,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	ldr	x0, [sp, 184]	// _2604, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	mov	w28, w25	// _8, _8
	mov	x25, x21	// tmp3576, tmp3576
	ldr	x4, [sp, 288]	// ivtmp.2135, %sfp
	ldr	w5, [sp, 312]	//, %sfp
	ldr	x1, [sp, 272]	// ivtmp.2136, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	x4, x4, x0	// ivtmp.2135, ivtmp.2135, _2604
	ldr	w0, [sp, 180]	//, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w5, w5, 1	// oh, oh,
	ldr	x7, [sp, 320]	// _1192, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	x1, x1, x20	// ivtmp.2136, ivtmp.2136, _6
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	ldr	x8, [sp, 344]	// ivtmp.2142, %sfp
	ldr	w3, [sp, 328]	//, %sfp
	ldr	w17, [sp, 336]	//, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	cmp	w0, w5	// _7, oh
	beq	.L1291		//,
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	cbz	w3, .L1322	// tiles,
	ldr	x24, [sp, 248]	// _1249, %sfp
	mov	x16, x26	// ivtmp.2141, ivtmp.2141
	mov	x21, x25	// tmp3576, tmp3576
	mov	x26, x20	// _6, _6
	mov	w10, w23	// _2, _2
	mov	w20, w5	// oh, oh
	ldr	x5, [sp, 264]	// _430, %sfp
	mov	x19, x4	// ivtmp.2135, ivtmp.2135
	mov	w25, w27	// _3, _3
	ldr	w9, [sp, 224]	//, %sfp
	b	.L847		//
.L1048:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	str	xzr, [sp, 112]	//, %sfp
.L686:
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	ldr	w0, [sp, 176]	//, %sfp
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	ldr	w1, [sp, 108]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x22, [x22]	// _530, MEM[(const struct vector *)weight_24(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	mul	w23, w19, w0	// ksz, _3, _4
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	cmp	w1, 0	// _1,
	add	w0, w1, 3	// tmp2400, _1,
	csel	w0, w0, w1, lt	// _1, tmp2400, _1,
// src/cpp/cnn_internals.cpp:318:     const int wcsz  = in_channels * ksz;              // per-oc stride in OIHW
	mul	w8, w20, w23	// wcsz, _2, ksz
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	asr	w12, w0, 2	// tiles, _1,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w1, 3	// _1,
	ble	.L687		//,
	cmp	w20, 0	// _2,
	ble	.L687		//,
	ldr	x11, [sp, 112]	// packed, %sfp
	sub	w0, w23, #1	// _5933, ksz,
	lsr	w2, w23, 2	// _645, ksz,
	add	x0, x0, 1	// _491, _5933,
	mov	x10, 0	// ivtmp.1955,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w1, 0	// tb,
	lsl	x6, x2, 4	// _646, _645,
	lsl	x5, x0, 4	// _529, _491,
	sxtw	x9, w8	// ivtmp.1956, wcsz
	sbfiz	x13, x8, 2, 32	// _673, wcsz,,
	sbfiz	x14, x8, 1, 32	// _702, wcsz,,
	sub	w4, w23, #1	// _311, ksz,
	and	w30, w23, -4	// niters_vector_mult_vf.1497, ksz,
.L688:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	add	x17, x10, x14	// _3414, ivtmp.1955, _702
	add	x18, x9, x14	// _3433, ivtmp.1956, _702
	mov	w7, 0	// ivtmp.1951,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w2, 0	// ic,
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w23, 0	// ksz,
	bgt	.L692		//,
.L1258:
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w2, w2, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w7, w7, w23	// ivtmp.1951, ivtmp.1951, ksz
	cmp	w20, w2	// _2, ic
	beq	.L693		//,
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w23, 0	// ksz,
	ble	.L1258		//,
.L692:
	cmp	w4, 2	// _311,
	bls	.L1050		//,
	sxtw	x0, w7	// _2022, ivtmp.1951
	mov	x26, x11	// ivtmp.1939, packed
	add	x16, x10, x0	// _1833, ivtmp.1955, _2022
	add	x25, x0, x17	// _1756, _2022, _3414
	add	x16, x22, x16, lsl 2	// vectp.1501, _530, _1833,
	add	x15, x0, x18	// _1762, _2022, _3433
	str	x0, [sp, 304]	// _2022, %sfp
	add	x25, x22, x25, lsl 2	// vectp.1507, _530, _1756,
	str	x16, [sp, 280]	// vectp.1501, %sfp
	add	x16, x9, x0	// _828, ivtmp.1956, _2022
	add	x16, x22, x16, lsl 2	// vectp.1504, _530, _828,
	str	x16, [sp, 272]	// vectp.1504, %sfp
	add	x16, x22, x15, lsl 2	// vectp.1510, _530, _1762,
	mov	x15, 0	// ivtmp.1936,
.L690:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	ldr	x0, [sp, 280]	// vectp.1501, %sfp
	ldr	q27, [x25, x15]	//, MEM <const vector(4) float> [(const float *)vectp.1507_2995 + ivtmp.1936_636 * 1]
	ldr	q28, [x16, x15]	//, MEM <const vector(4) float> [(const float *)vectp.1510_1760 + ivtmp.1936_636 * 1]
	ldr	q25, [x0, x15]	//, MEM <const vector(4) float> [(const float *)vectp.1501_1831 + ivtmp.1936_636 * 1]
	ldr	x0, [sp, 272]	// vectp.1504, %sfp
	ldr	q26, [x0, x15]	//, MEM <const vector(4) float> [(const float *)vectp.1504_1136 + ivtmp.1936_636 * 1]
	add	x15, x15, 16	// ivtmp.1936, ivtmp.1936,
	st4	{v25.4s - v28.4s}, [x26]	//, MEM <float[16]> [(float *)vectp_packed.1512_1402]
	add	x26, x26, 64	// ivtmp.1939, ivtmp.1939,
	cmp	x15, x6	// ivtmp.1936, _646
	bne	.L690		//,
	ldr	x0, [sp, 304]	// _2022, %sfp
	mov	w16, w30	// niters_vector_mult_vf.1497, niters_vector_mult_vf.1497
	cmp	w23, w30	// ksz, niters_vector_mult_vf.1497
	beq	.L691		//,
.L689:
	uxtw	x15, w16	// _1489, niters_vector_mult_vf.1497
	add	x25, x9, x0	// _1941, ivtmp.1956, _2022
	add	x25, x25, x15	// _1499, _1941, _1489
	add	x26, x10, x0	// _2127, ivtmp.1955, _2022
	sub	w16, w23, w16	// bnd.1516_2985, ksz, niters_vector_mult_vf.1497
	whilelo	p7.s, wzr, w16	// max_mask_3742,, bnd.1516_2985
	ld1w	z18.s, p7/z, [x22, x25, lsl 2]	//, max_mask_3742,* _530
	add	x25, x0, x17	// _3639, _2022, _3414
	add	x0, x0, x18	// _1424, _2022, _3433
	add	x16, x11, x15, lsl 4	// vectp_packed.1530, packed, _1489,
	add	x26, x26, x15	// _1490, _2127, _1489
	ld1w	z17.s, p7/z, [x22, x26, lsl 2]	//, max_mask_3742,* _530
	add	x25, x25, x15	// _237, _3639, _1489
	add	x0, x0, x15	// _3641, _1424, _1489
	ld1w	z19.s, p7/z, [x22, x25, lsl 2]	//, max_mask_3742,* _530
	ld1w	z20.s, p7/z, [x22, x0, lsl 2]	//, max_mask_3742,* _530
	st4w	{z17.s - z20.s}, p7, [x16]	//, max_mask_3742,* vectp_packed.1530
.L691:
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w2, w2, 1	// ic, ic,
	add	x11, x11, x5	// packed, packed, _529
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w7, w7, w23	// ivtmp.1951, ivtmp.1951, ksz
	cmp	w20, w2	// _2, ic
	bne	.L692		//,
.L693:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	w1, w1, 1	// tb, tb,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	x10, x10, x13	// ivtmp.1955, ivtmp.1955, _673
	add	x9, x9, x13	// ivtmp.1956, ivtmp.1956, _673
	cmp	w12, w1	// tiles, tb
	bgt	.L688		//,
.L687:
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	ldr	w0, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:471:         const int n_full = (OC / OC_TILE) * OC_TILE;
	lsl	w7, w12, 2	// n_full, tiles,
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	cmp	w0, w7	// _1, n_full
	bgt	.L1323		//,
.L696:
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w0, [sp, 184]	//, %sfp
// src/cpp/cnn_internals.cpp:404:     const int tile_wsz    = in_channels * ksz * OC_TILE;   // packed floats/tile
	lsl	w8, w8, 2	// tile_wsz, wcsz,
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w1, [sp, 224]	//, %sfp
// src/cpp/cnn_internals.cpp:478:             input.batches, IC, OC, IH, IW, KH, KW, OH, OW);
	ldr	w23, [x21, 24]	//, input_29(D)->batches
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	mul	w9, w0, w1	// in_size, _5, _6
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	ldr	w0, [sp, 180]	//, %sfp
// src/cpp/cnn_internals.cpp:400:     const int in_ch_size  = in_channels * in_size;
	mul	w13, w20, w9	// in_ch_size, _2, in_size
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	mul	w2, w0, w28	// out_size, _7, _8
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	ldr	w0, [sp, 108]	//, %sfp
	mul	w11, w0, w2	// out_ch_size, _1, out_size
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	cmp	w23, 0	// _560,
	ble	.L705		//,
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, 0	// _7,
	ble	.L705		//,
	ldr	x0, [sp, 248]	// _12, %sfp
	cmp	w7, 4	// n_full,
	mov	w1, 4	// tmp2467,
	csel	w1, w7, w1, ge	// oc0, n_full, tmp2467,
	smull	x15, w12, w8	// _492, tiles, tile_wsz
	cmp	w7, 0	// n_full,
	csel	w1, w1, wzr, gt	// oc0, oc0,,
	uxtw	x12, w1	// _279, oc0
	ldr	x4, [sp, 264]	// _954, %sfp
	sbfiz	x21, x2, 2, 32	// _489, out_size,,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	mov	w26, 0	// b,
	sbfiz	x22, x13, 2, 32	// _620, in_ch_size,,
	mul	x14, x0, x4	// _1188, _12, _954
	ldr	w4, [sp, 108]	//, %sfp
	ldr	w0, [sp, 176]	//, %sfp
	sub	w10, w4, w1	// _319, _1, oc0
	smull	x4, w1, w2	// _300, oc0, out_size
	add	x10, x10, w1, sxtw	// _329, _319, oc0
	sub	w0, w0, #1	// _5957, _4,
	str	x4, [sp, 320]	// _300, %sfp
	add	x0, x0, 1	// _553, _5957,
	ldr	x4, [sp, 112]	// _585, %sfp
	add	x4, x4, x15, lsl 2	// ivtmp.1865, _585, _492,
	mov	x15, x27	// ivtmp.1920, in_ptr
	sbfiz	x27, x11, 2, 32	// _626, out_ch_size,,
	str	x4, [sp, 336]	// ivtmp.1865, %sfp
	ldr	x4, [sp, 200]	// _87, %sfp
	add	x5, x4, x10, lsl 2	// _339, _87, _329,
	add	x4, x4, x12, lsl 2	// ivtmp.1861, _87, _279,
	ldr	x12, [sp, 216]	// ivtmp.1921, %sfp
	str	x5, [sp, 344]	// _339, %sfp
	lsl	x5, x14, 2	// _306, _1188,
	str	x4, [sp, 312]	// ivtmp.1861, %sfp
	sbfiz	x4, x8, 2, 32	// _461, tile_wsz,,
	str	x5, [sp, 328]	// _306, %sfp
	str	x4, [sp, 368]	// _461, %sfp
	lsl	x4, x24, 4	// _387, _4,
	str	x4, [sp, 352]	// _387, %sfp
	lsl	x4, x0, 4	// _5799, _553,
	str	x4, [sp, 400]	// _5799, %sfp
	add	x4, x21, 16	// _509, _489,
	str	x4, [sp, 376]	// _509, %sfp
	sbfiz	x4, x2, 4, 32	// _427, out_size,,
	sbfiz	x2, x2, 3, 32	// _597, out_size,,
	str	x4, [sp, 360]	// _427, %sfp
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	sxtw	x4, w9	// _5885, in_size
	str	x4, [sp, 392]	// _5885, %sfp
	sbfiz	x4, x9, 2, 32	// _252, in_size,,
	str	x4, [sp, 288]	// _252, %sfp
	ldr	x4, [sp, 208]	// _6, %sfp
	lsl	x5, x4, 2	// _170, _6,
	sxtw	x4, w28	// _2604, _8
	str	x4, [sp, 184]	// _2604, %sfp
.L751:
// src/cpp/cnn_internals.cpp:410:         float*       __restrict out_b = out_ptr   + (std::size_t)b * out_ch_size;
	mov	x14, 0	// ivtmp.1909,
	mov	x10, 0	// ivtmp.1908,
	str	w1, [sp, 272]	// oc0, %sfp
	add	x4, x2, x12	// _601, _597, ivtmp.1921
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	mov	w9, 0	// oh,
	str	x3, [sp, 408]	// _528, %sfp
	mov	x1, x10	// ivtmp.1908, ivtmp.1908
	mov	x3, x5	// _170, _170
	stp	x2, x22, [sp, 424]	// _597, _620,
	mov	w22, w28	// _8, _8
	mov	x28, x15	// ivtmp.1920, ivtmp.1920
	str	x27, [sp, 440]	// _626, %sfp
	mov	x27, x14	// ivtmp.1909, ivtmp.1909
	str	w26, [sp, 448]	// b, %sfp
	mov	x26, x21	// _489, _489
	str	w7, [sp, 248]	// n_full, %sfp
	str	x0, [sp, 280]	// _553, %sfp
	str	x4, [sp, 384]	// _601, %sfp
	str	w23, [sp, 416]	// _560, %sfp
.L706:
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	w0, [sp, 248]	//, %sfp
	cmp	w0, 0	// n_full,
	ble	.L750		//,
	ldr	x0, [sp, 384]	// _601, %sfp
	add	x25, x12, x1, lsl 2	// ivtmp.1898, ivtmp.1921, ivtmp.1908,
	str	x27, [sp, 304]	// ivtmp.1909, %sfp
	mov	x27, x26	// _489, _489
	ldr	x21, [sp, 200]	// ivtmp.1893, %sfp
	mov	x26, x25	// ivtmp.1898, ivtmp.1898
// src/cpp/cnn_internals.cpp:413:             int oc0 = 0, tile = 0;
	str	wzr, [sp, 264]	//, %sfp
	stp	x3, x1, [sp, 456]	// _170, ivtmp.1908,
	add	x24, x0, x1, lsl 2	// ivtmp.1896, _601, ivtmp.1908,
	ldr	x0, [sp, 112]	// _585, %sfp
	mov	x25, x24	// ivtmp.1896, ivtmp.1896
	mov	x24, x21	// ivtmp.1893, ivtmp.1893
	mov	w21, w20	// _2, _2
	str	x12, [sp, 472]	// ivtmp.1921, %sfp
	ldr	x20, [sp, 376]	// _509, %sfp
	str	w9, [sp, 496]	// oh, %sfp
	str	x0, [sp, 216]	// _585, %sfp
.L731:
// src/cpp/cnn_internals.cpp:347:         const float32x4_t bv = vdupq_n_f32(bias_ptr[oc0 + t]);
	ldr	s29, [x24, 12]	// _4434, MEM[(const float *)_464 + 12B]
	ldp	s28, s30, [x24, 4]	// _4410, _4422,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v31.4s}, [x24]	//, MEM[(const float *)_464]
	dup	v29.4s, v29.s[0]	// _2097, _4434
// src/cpp/cnn_internals.cpp:415:                 conv_row_tile<OC_TILE, NV>(
	ldr	x10, [sp, 216]	// wp, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	dup	v28.4s, v28.s[0]	// _4113, _4410
	dup	v30.4s, v30.s[0]	// _193, _4422
	str	q29, [sp, 224]	// _2097, %sfp
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q29, q29, [sp, 768]	// _2097, _2097,
	stp	q31, q31, [sp, 672]	//,,
	stp	q28, q28, [sp, 704]	// _4113, _4113,
	stp	q30, q30, [sp, 736]	// _193, _193,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w21, 0	// _2,
	ble	.L722		//,
	cmp	w19, 0	// _3,
	ble	.L722		//,
	ldr	x6, [sp, 208]	// _6, %sfp
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w9, 0	// ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	ldr	x1, [sp, 304]	// ivtmp.1886, %sfp
	ldr	x2, [sp, 392]	// _5885, %sfp
	.p2align 5,,15
.L728:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w3, 0	// kh,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x4, x1	// ivtmp.1882, ivtmp.1886
.L727:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w0, [sp, 176]	//, %sfp
	cmp	w0, 0	// _4,
	bgt	.L725		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w3, w3, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x6	// ivtmp.1882, ivtmp.1882, _6
	cmp	w19, w3	// _3, kh
	bne	.L727		//,
	.p2align 5,,15
.L726:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w9, w9, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x1, x1, x2	// ivtmp.1886, ivtmp.1886, _5885
	cmp	w21, w9	// _2, ic
	bne	.L728		//,
.L722:
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w22, 3	// _8,
	ble	.L1324		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [sp, 672]	// tmp5986, MEM[(__Float32x4_t[4][2] *)_2907][0][0]
	str	q31, [x26]	// tmp5986, MEM <__Float32x4_t> [(float * {ref-all})o_4117]
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w22, 7	// _8,
	ble	.L712		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [sp, 688]	// tmp5987, MEM[(__Float32x4_t[4][2] *)_2907][0][1]
	str	q31, [x26, 16]	// tmp5987, MEM <__Float32x4_t> [(float * {ref-all})o_4117 + 16B]
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w22, 8	// _8,
	bne	.L1051		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [sp, 704]	// tmp5988, MEM[(__Float32x4_t[4][2] *)_2907][1][0]
	ldp	q28, q30, [sp, 752]	// MEM[(__Float32x4_t[4][2] *)_2907][2][1], MEM[(__Float32x4_t[4][2] *)_2907][3][0],
	str	q31, [x26, x27]	// tmp5988, MEM <__Float32x4_t> [(float * {ref-all})o_4117 + _489 * 1]
	ldp	q31, q29, [sp, 720]	// tmp5989, MEM[(__Float32x4_t[4][2] *)_2907][2][0],
	str	q31, [x26, x20]	// tmp5989, MEM <__Float32x4_t> [(float * {ref-all})o_4117 + _509 * 1]
	ldr	q31, [sp, 784]	// MEM[(__Float32x4_t[4][2] *)_2907][3][1], MEM[(__Float32x4_t[4][2] *)_2907][3][1]
	stp	q29, q28, [x25]	// MEM[(__Float32x4_t[4][2] *)_2907][2][0], MEM[(__Float32x4_t[4][2] *)_2907][2][1],* ivtmp.1896
	str	q30, [x25, x27]	// MEM[(__Float32x4_t[4][2] *)_2907][3][0], MEM <__Float32x4_t> [(float * {ref-all})o_89 + _489 * 1]
	str	q31, [x25, x20]	// MEM[(__Float32x4_t[4][2] *)_2907][3][1], MEM <__Float32x4_t> [(float * {ref-all})o_89 + _509 * 1]
.L730:
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldp	x1, x2, [sp, 360]	// _427, _461,
	add	x24, x24, 16	// ivtmp.1893, ivtmp.1893,
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	w0, [sp, 264]	//, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	x25, x25, x1	// ivtmp.1896, ivtmp.1896, _427
	add	x26, x26, x1	// ivtmp.1898, ivtmp.1898, _427
	ldr	x1, [sp, 216]	// ivtmp.1899, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	w0, w0, 4	// oc0, oc0,
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	x1, x1, x2	// ivtmp.1899, ivtmp.1899, _461
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	str	w0, [sp, 264]	// oc0, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	str	x1, [sp, 216]	// ivtmp.1899, %sfp
	ldr	w1, [sp, 248]	//, %sfp
	cmp	w1, w0	// n_full, oc0
	bgt	.L731		//,
	ldr	x12, [sp, 472]	// ivtmp.1921, %sfp
	mov	x26, x27	// _489, _489
	mov	w20, w21	// _2, _2
	ldp	x3, x1, [sp, 456]	// _170, ivtmp.1908,
	ldr	x27, [sp, 304]	// ivtmp.1909, %sfp
	ldr	w9, [sp, 496]	//, %sfp
.L750:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	w0, [sp, 108]	//, %sfp
	ldr	w2, [sp, 272]	//, %sfp
	cmp	w0, w2	// _1, oc0
	ble	.L748		//,
	ldp	x24, x0, [sp, 312]	// ivtmp.1861, _300,
	add	x21, sp, 672	// tmp3576,,
	str	x27, [sp, 304]	// ivtmp.1909, %sfp
	ldr	x2, [sp, 336]	// ivtmp.1865, %sfp
	add	x4, x0, x1	// _301, _300, ivtmp.1908
	add	x0, x28, 16	// _1395, ivtmp.1920,
	str	x1, [sp, 264]	// ivtmp.1908, %sfp
	add	x0, x0, x27, lsl 2	// _592, _1395, ivtmp.1909,
	mov	x27, x21	// tmp3576, tmp3576
	mov	w21, w20	// _2, _2
	str	x12, [sp, 456]	// ivtmp.1921, %sfp
	add	x25, x12, x4, lsl 2	// ivtmp.1864, ivtmp.1921, _301,
	mov	w20, w22	// _8, _8
	mov	x22, x3	// _170, _170
	str	w9, [sp, 464]	// oh, %sfp
	stp	x2, x0, [sp, 216]	// ivtmp.1865, _592,
	ldr	x0, [sp, 280]	// _553, %sfp
	lsl	x23, x0, 2	// _5810, _553,
.L747:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v30.4s}, [x24]	//, MEM[(const float *)_310]
// src/cpp/cnn_internals.cpp:421:                 conv_row_tile<1, NV>(
	ldr	x13, [sp, 216]	// wp, %sfp
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q30, q30, [x27]	//,,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w21, 0	// _2,
	ble	.L737		//,
	cmp	w19, 0	// _3,
	ble	.L737		//,
	ldr	x11, [sp, 224]	// ivtmp.1852, %sfp
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w12, 0	// ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	ldr	x4, [sp, 256]	// _563, %sfp
	ldr	x5, [sp, 288]	// _252, %sfp
	.p2align 5,,15
.L743:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w3, 0	// kh,
	sub	x2, x11, #16	// ivtmp.1842, ivtmp.1852,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x1, x11	// ivtmp.1843, ivtmp.1852
.L742:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w0, [sp, 176]	//, %sfp
	cmp	w0, 0	// _4,
	bgt	.L740		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w3, w3, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x2, x2, x22	// ivtmp.1842, ivtmp.1842, _170
	add	x1, x1, x22	// ivtmp.1843, ivtmp.1843, _170
	cmp	w19, w3	// _3, kh
	bne	.L742		//,
	.p2align 5,,15
.L741:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w12, w12, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x11, x11, x5	// ivtmp.1852, ivtmp.1852, _252
	cmp	w21, w12	// _2, ic
	bne	.L743		//,
.L737:
	cmp	w20, 3	// _8,
	ble	.L1325		//,
	cmp	w20, 7	// _8,
	ble	.L1326		//,
	ldp	q31, q30, [sp, 672]	// MEM[(__Float32x4_t *)_2907], acc_I_I_lsm0.1689,
	mov	w2, 8	// col,
	mov	w0, 2	// v,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	stp	q31, q30, [x25]	// MEM[(__Float32x4_t *)_2907], acc_I_I_lsm0.1689,* ivtmp.1864
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w20, w2	// _8,
	beq	.L744		//,
.L745:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ubfiz	x1, x0, 4, 2	// tmp2615, v,,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w5, w20, w2	// _5289, _8, col
	uxtw	x0, w2	// _5285, col
	cmp	w20, w2	// _8, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [x27, x1]	// tmp6020, MEM[(__Float32x4_t[1][2] *)_2907][0][v_2317]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	lsl	x2, x5, 2	// _5288, _5289,
	add	x0, x25, x0, lsl 2	//, ivtmp.1864, _5285,
	mov	x5, 4	// tmp2628,
	csel	x2, x2, x5, gt	//, _5288, tmp2628,
	add	x1, sp, 624	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// tmp6020, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	bl	memcpy		//
.L744:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	x0, [sp, 216]	// ivtmp.1865, %sfp
	add	x24, x24, 4	// ivtmp.1861, ivtmp.1861,
	add	x25, x25, x26	// ivtmp.1864, ivtmp.1864, _489
	ldr	x1, [sp, 328]	// _306, %sfp
	add	x0, x0, x1	// ivtmp.1865, ivtmp.1865, _306
	str	x0, [sp, 216]	// ivtmp.1865, %sfp
	ldr	x0, [sp, 344]	// _339, %sfp
	cmp	x24, x0	// ivtmp.1861, _339
	bne	.L747		//,
	ldr	x1, [sp, 264]	// ivtmp.1908, %sfp
	mov	x3, x22	// _170, _170
	mov	w22, w20	// _8, _8
	mov	w20, w21	// _2, _2
	ldr	x27, [sp, 304]	// ivtmp.1909, %sfp
	ldr	x12, [sp, 456]	// ivtmp.1921, %sfp
	ldr	w9, [sp, 464]	//, %sfp
.L748:
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w9, w9, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	ldr	x0, [sp, 184]	// _2604, %sfp
	add	x1, x1, x0	// ivtmp.1908, ivtmp.1908, _2604
	ldr	x0, [sp, 208]	// _6, %sfp
	add	x27, x27, x0	// ivtmp.1909, ivtmp.1909, _6
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, w9	// _7, oh
	bne	.L706		//,
	mov	x21, x26	// _489, _489
	ldr	w26, [sp, 448]	//, %sfp
	mov	x15, x28	// ivtmp.1920, ivtmp.1920
	mov	w28, w22	// _8, _8
	ldp	x2, x22, [sp, 424]	// _597, _620,
	mov	x5, x3	// _170, _170
	ldr	x27, [sp, 440]	// _626, %sfp
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	w26, w26, 1	// b, b,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	x15, x15, x22	// ivtmp.1920, ivtmp.1920, _620
	ldr	w23, [sp, 416]	//, %sfp
	ldr	x0, [sp, 280]	// _553, %sfp
	add	x12, x12, x27	// ivtmp.1921, ivtmp.1921, _626
	ldr	x3, [sp, 408]	// _528, %sfp
	ldr	w7, [sp, 248]	//, %sfp
	ldr	w1, [sp, 272]	//, %sfp
	cmp	w23, w26	// _560, b
	bne	.L751		//,
.L705:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	ldr	x0, [sp, 112]	// _585, %sfp
	cbz	x0, .L623	// _585,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x1, x3	//, _528
	bl	_ZdlPvm		//
.L623:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	ldr	x0, [sp, 192]	// in_pad$_M_start, %sfp
	cbz	x0, .L1309	// in_pad$_M_start,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	ldr	x1, [sp, 240]	//, %sfp
	ldp	x23, x24, [sp, 48]	//,,
	.cfi_remember_state
	.cfi_restore 24
	.cfi_restore 23
// src/cpp/cnn_internals.cpp:496: }
	ldp	x29, x30, [sp]	//,,
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x25, x26, [sp, 64]	//,,
	ldp	x27, x28, [sp, 80]	//,,
	add	sp, sp, 896	//,,
	.cfi_restore 27
	.cfi_restore 28
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 29
	.cfi_restore 30
	.cfi_def_cfa_offset 0
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	b	_ZdlPvm		//
	.p2align 2,,3
.L1327:
	.cfi_restore_state
	ldr	x0, [sp, 400]	// _5799, %sfp
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w3, w3, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x6	// ivtmp.1882, ivtmp.1882, _6
	add	x10, x10, x0	// wp, wp, _5799
	cmp	w19, w3	// _3, kh
	beq	.L726		//,
.L725:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldp	q22, q23, [sp, 672]	// _4350, _4376,
	add	x5, x28, x4, lsl 2	// ivtmp.1876, ivtmp.1920, ivtmp.1882,
	ldp	q24, q25, [sp, 704]	// _4354, _4380,
	ldp	q26, q27, [sp, 736]	// _4358, _4384,
	ldp	q16, q29, [sp, 768]	// _4362, _4388,
	ldr	x0, [sp, 352]	// _387, %sfp
	add	x7, x10, x0	// _388, wp, _387
	mov	x0, x10	// wp, wp
	.p2align 5,,15
.L724:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldp	q17, q31, [x5]	// MEM <__Float32x4_t> [(float * {ref-all})_379], MEM <__Float32x4_t> [(float * {ref-all})_379 + 16B],* ivtmp.1876
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x5, x5, 4	// ivtmp.1876, ivtmp.1876,
// src/cpp/cnn_internals.cpp:360:                 for (int t = 0; t < OC_T; ++t) wv[t] = wp[t];
	ldp	s18, s19, [x0]	// _3317, _3325,* wp
	ldp	s20, s21, [x0, 8]	// _3333, _3341,
// src/cpp/cnn_internals.cpp:361:                 wp += OC_T;
	add	x0, x0, 16	// wp, wp,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v22.4s, v17.4s, v18.s[0]	// _4350, MEM <__Float32x4_t> [(float * {ref-all})_379], _3317
	fmla	v24.4s, v17.4s, v19.s[0]	// _4354, MEM <__Float32x4_t> [(float * {ref-all})_379], _3325
	fmla	v26.4s, v17.4s, v20.s[0]	// _4358, MEM <__Float32x4_t> [(float * {ref-all})_379], _3333
	fmla	v23.4s, v31.4s, v18.s[0]	// _4376, MEM <__Float32x4_t> [(float * {ref-all})_379 + 16B], _3317
	fmla	v25.4s, v31.4s, v19.s[0]	// _4380, MEM <__Float32x4_t> [(float * {ref-all})_379 + 16B], _3325
	fmla	v27.4s, v31.4s, v20.s[0]	// _4384, MEM <__Float32x4_t> [(float * {ref-all})_379 + 16B], _3333
	fmla	v16.4s, v17.4s, v21.s[0]	// _4362, MEM <__Float32x4_t> [(float * {ref-all})_379], _3341
	fmla	v29.4s, v31.4s, v21.s[0]	// _4388, MEM <__Float32x4_t> [(float * {ref-all})_379 + 16B], _3341
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q22, q23, [sp, 672]	// _4350, _4376,
	stp	q24, q25, [sp, 704]	// _4354, _4380,
	stp	q26, q27, [sp, 736]	// _4358, _4384,
	stp	q16, q29, [sp, 768]	// _4362, _4388,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x7, x0	// _388, wp
	bne	.L724		//,
	b	.L1327		//
	.p2align 2,,3
.L1328:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w3, w3, 1	// kh, kh,
	add	x13, x13, x23	// wp, wp, _5810
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x2, x2, x22	// ivtmp.1842, ivtmp.1842, _170
	add	x1, x1, x22	// ivtmp.1843, ivtmp.1843, _170
	cmp	w19, w3	// _3, kh
	beq	.L741		//,
.L740:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldp	q29, q16, [x27]	//,,
	mov	x0, 0	// ivtmp.1831,
	.p2align 5,,15
.L739:
// src/cpp/cnn_internals.cpp:360:                 for (int t = 0; t < OC_T; ++t) wv[t] = wp[t];
	ldr	s3, [x13, x0]	//, MEM[(const float *)wp_5303 + ivtmp.1831_146 * 1]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldr	q0, [x2, x0]	//, MEM <__Float32x4_t> [(float * {ref-all})_152 + ivtmp.1831_146 * 1]
	ldr	q2, [x1, x0]	//, MEM <__Float32x4_t> [(float * {ref-all})_158 + ivtmp.1831_146 * 1]
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x0, x0, 4	// ivtmp.1831, ivtmp.1831,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v29.4s, v0.4s, v3.s[0]	//,,
	fmla	v16.4s, v2.4s, v3.s[0]	//,,
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q29, q16, [x27]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x0, x4	// ivtmp.1831, _563
	bne	.L739		//,
	b	.L1328		//
.L1326:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [sp, 672]	// tmp6021, MEM[(__Float32x4_t *)_2907]
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	mov	w2, 4	// col,
	mov	w0, 1	// v,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [x25]	// tmp6021, MEM <__Float32x4_t> [(float * {ref-all})o_823]
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w20, w2	// _8,
	bne	.L745		//,
	b	.L744		//
.L1324:
	cmp	w22, 0	// _8,
	ble	.L730		//,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q28, [sp, 672]	// _4113, MEM[(__Float32x4_t[4][2] *)_2907][0][0]
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	mov	w2, 0	// col,
.L713:
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	uxtw	x0, w2	// _3932, col
	sub	w6, w22, w2	// _3914, _8, col
	str	q30, [sp, 480]	// _193, %sfp
	cmp	w22, w2	// _8, col
	add	x0, x26, x0, lsl 2	//, ivtmp.1898, _3932,
	lsl	x2, x6, 2	// _3930, _3914,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q28, [sp, 624]	// _4113, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	mov	x6, 4	// tmp2504,
	add	x1, sp, 624	// tmp5985,,
	csel	x2, x2, x6, gt	//, _3930, tmp2504,
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w22, 3	// _8,
	ldr	q30, [sp, 480]	// _193, %sfp
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w23, w22, #1	// _1734, _8,
// src/cpp/cnn_internals.cpp:375:         float* __restrict o = out_batch
	add	x0, x26, x27	// o, ivtmp.1898, _489
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	mov	x6, 4	// tmp2504,
	bgt	.L1329		//,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q30, [sp, 704]	// _193, MEM[(__Float32x4_t[4][2] *)_2907][1][0]
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	mov	w2, 0	// col,
.L718:
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w6, w23, w2	// _3794, _1734, col
	uxtw	x1, w2	// _2948, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q30, [sp, 624]	// _193, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	add	w6, w6, 1	// _3796, _3794,
	cmp	w22, w2	// _8, col
	add	x0, x0, x1, lsl 2	//, o, _2948,
	lsl	x2, x6, 2	// _3797, _3796,
	add	x1, sp, 624	// tmp5993,,
	mov	x6, 4	// tmp2531,
	csel	x2, x2, x6, gt	//, _3797, tmp2531,
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w22, 3	// _8,
	mov	x6, 4	// tmp2531,
	bgt	.L1330		//,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [sp, 736]	// _2097, MEM[(__Float32x4_t[4][2] *)_2907][2][0]
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	mov	w2, 0	// col,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	str	q31, [sp, 224]	// _2097, %sfp
.L721:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [sp, 224]	// _2097, %sfp
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w6, w23, w2	// _3611, _1734, col
	uxtw	x0, w2	// _3631, col
	add	w6, w6, 1	// _3613, _3611,
	cmp	w22, w2	// _8, col
	add	x0, x25, x0, lsl 2	//, ivtmp.1896, _3631,
	lsl	x2, x6, 2	// _3629, _3613,
	add	x1, sp, 624	// tmp5995,,
	mov	x6, 4	// tmp2547,
	csel	x2, x2, x6, gt	//, _3629, tmp2547,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// _2097, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w22, 3	// _8,
// src/cpp/cnn_internals.cpp:375:         float* __restrict o = out_batch
	add	x0, x25, x27	// o, ivtmp.1896, _489
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	mov	x6, 4	// tmp2547,
	bgt	.L1331		//,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [sp, 768]	// _1070, MEM[(__Float32x4_t[4][2] *)_2907][3][0]
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	mov	w6, 0	// col,
.L729:
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w23, w6	// _5334, _1734, col
	uxtw	x1, w6	// _5320, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// _1070, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	cmp	w22, w6	// _8, col
	add	x0, x0, x1, lsl 2	//, o, _5320,
	mov	x6, 4	// tmp2580,
	add	w2, w2, 1	// _5324, _5334,
	add	x1, sp, 624	// tmp6000,,
	lsl	x2, x2, 2	// _5323, _5324,
	csel	x2, x2, x6, gt	//, _5323, tmp2580,
	bl	memcpy		//
	b	.L730		//
.L712:
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w22, 4	// _8,
	bne	.L1332		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [sp, 704]	// tmp5990, MEM[(__Float32x4_t[4][2] *)_2907][1][0]
	str	q31, [x26, x27]	// tmp5990, MEM <__Float32x4_t> [(float * {ref-all})o_4117 + _489 * 1]
	ldr	q31, [sp, 736]	// tmp5991, MEM[(__Float32x4_t[4][2] *)_2907][2][0]
	str	q31, [x25]	// tmp5991, MEM <__Float32x4_t> [(float * {ref-all})o_5908]
	ldr	q31, [sp, 768]	// tmp5992, MEM[(__Float32x4_t[4][2] *)_2907][3][0]
	str	q31, [x25, x27]	// tmp5992, MEM <__Float32x4_t> [(float * {ref-all})o_5908 + _489 * 1]
	b	.L730		//
.L1051:
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	mov	w2, 8	// col,
	b	.L713		//
.L1332:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q28, [sp, 688]	// _4113, MEM[(__Float32x4_t[4][2] *)_2907][0][1]
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	mov	w2, 4	// col,
	b	.L713		//
.L1331:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [sp, 768]	// tmp6420, MEM[(__Float32x4_t[4][2] *)_2907][3][0]
	str	q31, [x25, x27]	// tmp6420, MEM <__Float32x4_t> [(float * {ref-all})o_3572 + _489 * 1]
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [sp, 784]	// _1070, MEM[(__Float32x4_t[4][2] *)_2907][3][1]
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w22, 7	// _8,
	ble	.L729		//,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	mov	w6, 8	// col,
	ldp	q30, q31, [sp, 784]	// tmp6419, _1070,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q30, [x25, x20]	// tmp6419, MEM <__Float32x4_t> [(float * {ref-all})o_3572 + _509 * 1]
	b	.L729		//
.L1330:
	ldr	q31, [sp, 736]	// tmp6422, MEM[(__Float32x4_t[4][2] *)_2907][2][0]
	str	q31, [x25]	// tmp6422, MEM <__Float32x4_t> [(float * {ref-all})o_3572]
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w22, 7	// _8,
	ble	.L1333		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [sp, 752]	// tmp6421, MEM[(__Float32x4_t[4][2] *)_2907][2][1]
	mov	w2, 8	// col,
	str	q31, [x25, 16]	// tmp6421, MEM <__Float32x4_t> [(float * {ref-all})o_3572 + 16B]
	b	.L721		//
.L1329:
	ldr	q31, [sp, 704]	// tmp6425, MEM[(__Float32x4_t[4][2] *)_2907][1][0]
	str	q31, [x26, x27]	// tmp6425, MEM <__Float32x4_t> [(float * {ref-all})o_4117 + _489 * 1]
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w22, 7	// _8,
	ble	.L1334		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [sp, 720]	// tmp6424, MEM[(__Float32x4_t[4][2] *)_2907][1][1]
	mov	w2, 8	// col,
	str	q31, [x26, x20]	// tmp6424, MEM <__Float32x4_t> [(float * {ref-all})o_4117 + _509 * 1]
	b	.L718		//
.L1333:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [sp, 752]	// _2097, MEM[(__Float32x4_t[4][2] *)_2907][2][1]
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	mov	w2, w6	// col, tmp2531
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	str	q31, [sp, 224]	// _2097, %sfp
	b	.L721		//
.L1334:
	ldr	q30, [sp, 720]	// _193, MEM[(__Float32x4_t[4][2] *)_2907][1][1]
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	mov	w2, w6	// col, tmp2504
	b	.L718		//
.L1309:
	ldp	x23, x24, [sp, 48]	//,,
	.cfi_restore 24
	.cfi_restore 23
.L610:
// src/cpp/cnn_internals.cpp:496: }
	ldp	x29, x30, [sp]	//,,
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x25, x26, [sp, 64]	//,,
	ldp	x27, x28, [sp, 80]	//,,
	add	sp, sp, 896	//,,
	.cfi_restore 27
	.cfi_restore 28
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 29
	.cfi_restore 30
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L1325:
	.cfi_def_cfa_offset 896
	.cfi_offset 19, -880
	.cfi_offset 20, -872
	.cfi_offset 21, -864
	.cfi_offset 22, -856
	.cfi_offset 23, -848
	.cfi_offset 24, -840
	.cfi_offset 25, -832
	.cfi_offset 26, -824
	.cfi_offset 27, -816
	.cfi_offset 28, -808
	.cfi_offset 29, -896
	.cfi_offset 30, -888
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w20, 0	// _8,
	ble	.L744		//,
	mov	w2, 0	// col,
	mov	w0, 0	// v,
	b	.L745		//
.L1050:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	w16, 0	// niters_vector_mult_vf.1497,
	sxtw	x0, w7	// _2022, ivtmp.1951
	b	.L689		//
	.p2align 2,,3
.L1335:
	ldr	x0, [sp, 448]	// tmp3841, %sfp
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w11, w11, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x20, x20, x10	// ivtmp.2117, ivtmp.2117, _6
	add	x18, x18, x0	// wp, wp, tmp3841
	cmp	w25, w11	// _3, kh
	beq	.L837		//,
.L836:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldp	q16, q5, [sp, 672]	//,,
	add	x6, x16, x20, lsl 2	// ivtmp.2111, ivtmp.2141, ivtmp.2117,
	ldp	q2, q31, [sp, 704]	//,,
	ldp	q7, q4, [sp, 736]	//,,
	ldp	q1, q30, [sp, 768]	//,,
	ldp	q6, q3, [sp, 800]	//,,
	ldp	q0, q29, [sp, 832]	//,,
	ldr	x0, [sp, 400]	// _1347, %sfp
	add	x21, x18, x0	// _1353, wp, _1347
	mov	x0, x18	// wp, wp
	.p2align 5,,15
.L835:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldp	q20, q24, [x6]	//,,* ivtmp.2111
	ldp	q23, q25, [x6, 32]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x6, x6, 4	// ivtmp.2111, ivtmp.2111,
// src/cpp/cnn_internals.cpp:360:                 for (int t = 0; t < OC_T; ++t) wv[t] = wp[t];
	ldr	s26, [x0, 8]	//, MEM[(const float *)wp_233 + 8B]
	ldp	s28, s27, [x0]	//,,* wp
// src/cpp/cnn_internals.cpp:361:                 wp += OC_T;
	add	x0, x0, 12	// wp, wp,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v6.4s, v20.4s, v26.s[0]	//,,
	fmla	v16.4s, v20.4s, v28.s[0]	//,,
	fmla	v7.4s, v20.4s, v27.s[0]	//,,
	fmla	v5.4s, v24.4s, v28.s[0]	//,,
	fmla	v4.4s, v24.4s, v27.s[0]	//,,
	fmla	v2.4s, v23.4s, v28.s[0]	//,,
	fmla	v1.4s, v23.4s, v27.s[0]	//,,
	fmla	v31.4s, v25.4s, v28.s[0]	//,,
	fmla	v30.4s, v25.4s, v27.s[0]	//,,
	fmla	v3.4s, v24.4s, v26.s[0]	//,,
	fmla	v0.4s, v23.4s, v26.s[0]	//,,
	fmla	v29.4s, v25.4s, v26.s[0]	//,,
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q16, q5, [sp, 672]	//,,
	stp	q2, q31, [sp, 704]	//,,
	stp	q7, q4, [sp, 736]	//,,
	stp	q1, q30, [sp, 768]	//,,
	stp	q6, q3, [sp, 800]	//,,
	stp	q0, q29, [sp, 832]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x21, x0	// _1353, wp
	bne	.L835		//,
	b	.L1335		//
	.p2align 2,,3
.L1336:
	ldr	x3, [sp, 304]	// _5731, %sfp
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w2, w2, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x20	// ivtmp.2081, ivtmp.2081, _6
	add	x8, x8, x3	// wp, wp, _5731
	cmp	w27, w2	// _3, kh
	beq	.L858		//,
.L857:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldp	q21, q19, [x21]	//,,
	add	x3, x26, x4, lsl 2	// ivtmp.2075, ivtmp.2141, ivtmp.2081,
	ldp	q18, q17, [x21, 32]	//,,
	ldr	x5, [sp, 360]	// _5700, %sfp
	add	x6, x8, x5	// _1222, wp, _5700
	mov	x5, x8	// wp, wp
	.p2align 5,,15
.L856:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v31.4s}, [x5], 4	//, MEM[(const float *)wp_382]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldp	q2, q1, [x3]	//,,* ivtmp.2075
	ldp	q0, q30, [x3, 32]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x3, x3, 4	// ivtmp.2075, ivtmp.2075,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v21.4s, v2.4s, v31.4s	//,,
	fmla	v19.4s, v1.4s, v31.4s	//,,
	fmla	v18.4s, v0.4s, v31.4s	//,,
	fmla	v17.4s, v30.4s, v31.4s	//,,
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q21, q19, [x21]	//,,
	stp	q18, q17, [x21, 32]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x6, x5	// _1222, wp
	bne	.L856		//,
	b	.L1336		//
.L1321:
	mov	w4, 0	// col,
	mov	w1, 0	// v,
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w25, 0	// _8,
	bgt	.L861		//,
	b	.L863		//
.L840:
	cmp	w19, 0	// _8,
	ble	.L1289		//,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [x23]	// _1426, MEM[(__Float32x4_t *)_1309]
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	mov	w7, 0	// col,
	b	.L844		//
.L866:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	w0, [sp, 108]	//, %sfp
	cmp	w0, 0	// _1,
	bgt	.L1337		//,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	ldr	x0, [sp, 184]	// _2604, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w27, w27, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	x26, x26, x0	// ivtmp.2135, ivtmp.2135, _2604
	ldr	x0, [sp, 208]	// _6, %sfp
	add	x1, x1, x0	// ivtmp.2136, ivtmp.2136, _6
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, w27	// _7, oh
	bne	.L830		//,
	b	.L848		//
	.p2align 2,,3
.L1320:
	ldr	x0, [sp, 184]	// _2604, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w20, w14, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	x1, x1, x26	// ivtmp.2136, ivtmp.2136, _6
	add	x19, x13, x0	// ivtmp.2135, ivtmp.2135, _2604
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, w20	// _7, oh
	bne	.L847		//,
	mov	w20, w10	// _2, _2
	mov	w19, w25	// _3, _3
	str	x26, [sp, 208]	// _6, %sfp
	mov	x23, x15	// _1249, _1249
	mov	x22, x5	// _430, _430
.L848:
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	ldr	x0, [sp, 424]	// _1507, %sfp
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	w9, w9, 1	// b, b,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	x16, x16, x0	// ivtmp.2141, ivtmp.2141, _1507
	ldr	x0, [sp, 432]	// _1513, %sfp
	add	x8, x8, x0	// ivtmp.2142, ivtmp.2142, _1513
	ldr	w0, [sp, 376]	//, %sfp
	cmp	w0, w9	// _1224, b
	bne	.L867		//,
.L829:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x23, .L623	// _1249,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x1, x7	//, _1192
	mov	x0, x23	//, _1249
	bl	_ZdlPvm		//
// src/cpp/cnn_internals.cpp:479:     };
	b	.L623		//
	.p2align 2,,3
.L1317:
	str	w19, [sp, 112]	// _3, %sfp
	str	w28, [sp, 256]	// _8, %sfp
	str	x27, [sp, 264]	// in_ptr, %sfp
.L816:
	cmp	w6, 2	// _4314,
	bls	.L1062		//,
	sxtw	x0, w2	// _4198, ivtmp.2169
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	x19, 0	// ivtmp.2160,
	add	x26, x12, x0	// _4269, ivtmp.2175, _4198
	add	x27, x11, x0	// _4260, ivtmp.2176, _4198
	mov	x4, x26	// _4269, _4269
	add	x26, x17, x0	// _4251, _3788, _4198
	add	x27, x8, x27, lsl 2	// vectp.1578, _1194, _4260,
	add	x4, x8, x4, lsl 2	// vectp.1575, _1194, _4269,
	mov	x28, x10	// ivtmp.2158, packed
	add	x26, x8, x26, lsl 2	// vectp.1581, _1194, _4251,
.L814:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	ldr	q29, [x4, x19]	//, MEM <const vector(4) float> [(const float *)vectp.1575_4271 + ivtmp.2160_1531 * 1]
	ldr	q30, [x27, x19]	//, MEM <const vector(4) float> [(const float *)vectp.1578_4264 + ivtmp.2160_1531 * 1]
	ldr	q31, [x26, x19]	//, MEM <const vector(4) float> [(const float *)vectp.1581_4254 + ivtmp.2160_1531 * 1]
	add	x19, x19, 16	// ivtmp.2160, ivtmp.2160,
	st3	{v29.4s - v31.4s}, [x28]	//, MEM <float[12]> [(float *)vectp_packed.1583_4244]
	add	x28, x28, 48	// ivtmp.2158, ivtmp.2158,
	cmp	x19, x30	// ivtmp.2160, _1534
	bne	.L814		//,
	mov	w19, w18	// niters_vector_mult_vf.1571, niters_vector_mult_vf.1571
	cmp	w18, w25	// niters_vector_mult_vf.1571, ksz
	beq	.L815		//,
.L813:
	mov	w4, 12	// tmp6620,
	uxtw	x26, w19	// _4189, niters_vector_mult_vf.1571
	add	x27, x12, x0	// _4197, ivtmp.2175, _4198
	add	x28, x11, x0	// _4176, ivtmp.2176, _4198
	add	x0, x17, x0	// _4164, _3788, _4198
	umaddl	x4, w19, w4, x10	// vectp_packed.1598, niters_vector_mult_vf.1571, tmp6620, packed
	add	x27, x27, x26	// _4185, _4197, _4189
	sub	w19, w25, w19	// bnd.1587_4216, ksz, niters_vector_mult_vf.1571
	add	x28, x28, x26	// _4173, _4176, _4189
	whilelo	p7.s, wzr, w19	// max_mask_4136,, bnd.1587_4216
	ld1w	z23.s, p7/z, [x8, x27, lsl 2]	//, max_mask_4136,* _1194
	ld1w	z24.s, p7/z, [x8, x28, lsl 2]	//, max_mask_4136,* _1194
	add	x0, x0, x26	// _4161, _4164, _4189
	ld1w	z25.s, p7/z, [x8, x0, lsl 2]	//, max_mask_4136,* _1194
	st3w	{z23.s - z25.s}, p7, [x4]	//, max_mask_4136,* vectp_packed.1598
.L815:
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w1, w1, 1	// ic, ic,
	add	x10, x10, x5	// packed, packed, tmp3856
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w2, w2, w25	// ivtmp.2169, ivtmp.2169, ksz
	cmp	w20, w1	// _2, ic
	bne	.L816		//,
	ldr	x27, [sp, 264]	// in_ptr, %sfp
	ldr	w19, [sp, 112]	//, %sfp
	ldr	w28, [sp, 256]	//, %sfp
	b	.L817		//
.L1062:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	w19, 0	// niters_vector_mult_vf.1571,
	sxtw	x0, w2	// _4198, ivtmp.2169
	b	.L813		//
.L1322:
// src/cpp/cnn_internals.cpp:413:             int oc0 = 0, tile = 0;
	mov	w21, 0	// oc0,
	b	.L850		//
.L1291:
	ldr	x22, [sp, 264]	// _430, %sfp
	mov	w20, w23	// _2, _2
	mov	w19, w27	// _3, _3
	mov	x16, x26	// ivtmp.2141, ivtmp.2141
	ldr	x23, [sp, 248]	// _1249, %sfp
	ldr	w9, [sp, 224]	//, %sfp
	b	.L848		//
.L1085:
	.cfi_restore 23
	.cfi_restore 24
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -840
	.cfi_offset 23, -848
// src/cpp/cnn_internals.cpp:482:     switch (NV) {
	mov	x23, 0	// in_pad$_M_end_of_storage,
	str	xzr, [sp, 192]	//, %sfp
	str	xzr, [sp, 240]	//, %sfp
.L625:
	ldr	w0, [sp, 176]	//, %sfp
// src/cpp/cnn_internals.cpp:468:         std::vector<float> packed((std::size_t)OC * IC * KH * KW);
	ldr	w1, [sp, 108]	//, %sfp
	smull	x24, w19, w0	// _3070, _3, _4
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	mov	x0, 2305843009213693951	// tmp3387,
// src/cpp/cnn_internals.cpp:468:         std::vector<float> packed((std::size_t)OC * IC * KH * KW);
	smull	x25, w20, w1	// _1364, _2, _1
	mul	x26, x25, x24	// _1847, _1364, _3070
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	cmp	x26, x0	// _1847, tmp3387
	bhi	.L1338		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	cbz	x26, .L1076	// _1847,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	lsl	x7, x26, 2	// _2244, _1847,
	mov	x0, x7	//, _2244
	str	x7, [sp, 112]	// _2244, %sfp
	bl	_Znwm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	str	wzr, [x0]	//, *_2245
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x3, x0	// iftmp.18_1846, tmp3875
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	cmp	x26, 1	// _1847,
	beq	.L1077		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	ldr	x7, [sp, 112]	// _2244, %sfp
	mov	w1, 0	//,
	add	x0, x0, 4	//, iftmp.18_1846,
	str	x3, [sp, 112]	// iftmp.18_1846, %sfp
	sub	x2, x7, #4	//, _2244,
	str	x7, [sp, 256]	// _2244, %sfp
	bl	memset		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	ldr	x7, [sp, 256]	// _2244, %sfp
	ldr	x3, [sp, 112]	// iftmp.18_1846, %sfp
	mov	x26, x7	// _1847, _2244
	b	.L975		//
.L1084:
	.cfi_restore 23
	.cfi_restore 24
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -840
	.cfi_offset 23, -848
// src/cpp/cnn_internals.cpp:482:     switch (NV) {
	mov	x23, 0	// in_pad$_M_end_of_storage,
	str	xzr, [sp, 192]	//, %sfp
	str	xzr, [sp, 240]	//, %sfp
.L624:
	ldr	w0, [sp, 176]	//, %sfp
// src/cpp/cnn_internals.cpp:468:         std::vector<float> packed((std::size_t)OC * IC * KH * KW);
	ldr	w1, [sp, 108]	//, %sfp
	smull	x25, w19, w0	// _5895, _3, _4
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	mov	x0, 2305843009213693951	// tmp3225,
// src/cpp/cnn_internals.cpp:468:         std::vector<float> packed((std::size_t)OC * IC * KH * KW);
	smull	x7, w20, w1	// _1135, _2, _1
	mul	x7, x7, x25	// _1845, _1135, _5895
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	cmp	x7, x0	// _1845, tmp3225
	bhi	.L1339		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	cbz	x7, .L1071	// _1845,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	lsl	x24, x7, 2	// _1912, _1845,
	str	x7, [sp, 112]	// _1845, %sfp
	mov	x0, x24	//, _1912
	bl	_Znwm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	ldr	x7, [sp, 112]	// _1845, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x3, x0	// iftmp.18_1843, tmp3874
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	str	wzr, [x0]	//, *_1913
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	cmp	x7, 1	// _1845,
	beq	.L1072		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	sub	x2, x24, #4	//, _1912,
	mov	w1, 0	//,
	str	x3, [sp, 112]	// iftmp.18_1843, %sfp
	add	x0, x0, 4	//, iftmp.18_1843,
	bl	memset		//
	ldr	x3, [sp, 112]	// iftmp.18_1843, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	mov	x7, x24	// _1845, _1912
	b	.L922		//
.L1076:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	mov	x3, 0	// iftmp.18_1846,
.L975:
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	ldr	w0, [sp, 176]	//, %sfp
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	ldr	w2, [sp, 108]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x1, [x22]	// _2190, MEM[(const struct vector *)weight_24(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	mul	w0, w19, w0	// ksz, _3, _4
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	add	w22, w2, w2, lsr 31	// tmp3397, _1, _1,
// src/cpp/cnn_internals.cpp:318:     const int wcsz  = in_channels * ksz;              // per-oc stride in OIHW
	mul	w23, w20, w0	// wcsz, _2, ksz
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	asr	w22, w22, 1	// tiles, tmp3397,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w2, 1	// _1,
	ble	.L976		//,
	cmp	w20, 0	// _2,
	ble	.L976		//,
	sub	w2, w0, #1	// _5511, ksz,
	lsr	w7, w0, 2	// _2647, ksz,
	add	x2, x2, 1	// _454, _5511,
	mov	x12, 0	// ivtmp.2490,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w9, 0	// tb,
	lsl	x18, x7, 4	// _2648, _2647,
	lsl	x30, x2, 3	// _5625, _454,
	sxtw	x11, w23	// ivtmp.2491, wcsz
	sbfiz	x13, x23, 1, 32	// _2657, wcsz,,
	sub	w4, w0, #1	// _3673, ksz,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	x10, x3	// packed, iftmp.18_1846
	and	w17, w0, -4	// niters_vector_mult_vf.1652, ksz,
.L977:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	w8, 0	// ivtmp.2486,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w7, 0	// ic,
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w0, 0	// ksz,
	bgt	.L981		//,
.L1279:
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w7, w7, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w8, w8, w0	// ivtmp.2486, ivtmp.2486, ksz
	cmp	w20, w7	// _2, ic
	beq	.L982		//,
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w0, 0	// ksz,
	ble	.L1279		//,
.L981:
	cmp	w4, 2	// _3673,
	bls	.L1078		//,
	sxtw	x2, w8	// _3542, ivtmp.2486
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	x5, 0	// ivtmp.2479,
	add	x14, x12, x2	// _3651, ivtmp.2490, _3542
	add	x6, x11, x2	// _3642, ivtmp.2491, _3542
	add	x14, x1, x14, lsl 2	// vectp.1656, _2190, _3651,
	mov	x15, x10	// ivtmp.2477, packed
	add	x6, x1, x6, lsl 2	// vectp.1659, _2190, _3642,
.L979:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	ldr	q25, [x14, x5]	//, MEM <const vector(4) float> [(const float *)vectp.1656_3654 + ivtmp.2479_2645 * 1]
	ldr	q26, [x6, x5]	//, MEM <const vector(4) float> [(const float *)vectp.1659_3644 + ivtmp.2479_2645 * 1]
	add	x5, x5, 16	// ivtmp.2479, ivtmp.2479,
	st2	{v25.4s - v26.4s}, [x15], 32	//, MEM <float[8]> [(float *)vectp_packed.1661_3600]
	cmp	x5, x18	// ivtmp.2479, _2648
	bne	.L979		//,
	mov	w5, w17	// niters_vector_mult_vf.1652, niters_vector_mult_vf.1652
	cmp	w17, w0	// niters_vector_mult_vf.1652, ksz
	beq	.L980		//,
.L978:
	uxtw	x14, w5	// _3539, niters_vector_mult_vf.1652
	add	x6, x12, x2	// _3540, ivtmp.2490, _3542
	sub	w5, w0, w5	// bnd.1665_3545, ksz, niters_vector_mult_vf.1652
	add	x2, x11, x2	// _3526, ivtmp.2491, _3542
	whilelo	p7.s, wzr, w5	// max_mask_3492,, bnd.1665_3545
	add	x15, x10, x14, lsl 3	// vectp_packed.1673, packed, _3539,
	add	x5, x6, x14	// _3538, _3540, _3539
	add	x2, x2, x14	// _3523, _3526, _3539
	ld1w	z18.s, p7/z, [x1, x5, lsl 2]	//, max_mask_3492,* _2190
	ld1w	z19.s, p7/z, [x1, x2, lsl 2]	//, max_mask_3492,* _2190
	st2w	{z18.s - z19.s}, p7, [x15]	//, max_mask_3492,* vectp_packed.1673
.L980:
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w7, w7, 1	// ic, ic,
	add	x10, x10, x30	// packed, packed, _5625
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w8, w8, w0	// ivtmp.2486, ivtmp.2486, ksz
	cmp	w20, w7	// _2, ic
	bne	.L981		//,
.L982:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	w9, w9, 1	// tb, tb,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	x12, x12, x13	// ivtmp.2490, ivtmp.2490, _2657
	add	x11, x11, x13	// ivtmp.2491, ivtmp.2491, _2657
	cmp	w22, w9	// tiles, tb
	bgt	.L977		//,
.L976:
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	ldr	w2, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:471:         const int n_full = (OC / OC_TILE) * OC_TILE;
	lsl	w8, w22, 1	// n_full, tiles,
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	cmp	w2, w8	// _1, n_full
	bgt	.L1340		//,
.L985:
// src/cpp/cnn_internals.cpp:478:             input.batches, IC, OC, IH, IW, KH, KW, OH, OW);
	ldr	w21, [x21, 24]	//, input_29(D)->batches
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	cmp	w21, 0	// _2220,
	ble	.L990		//,
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w0, [sp, 184]	//, %sfp
// src/cpp/cnn_internals.cpp:404:     const int tile_wsz    = in_channels * ksz * OC_TILE;   // packed floats/tile
	lsl	w7, w23, 1	// tile_wsz, wcsz,
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w1, [sp, 224]	//, %sfp
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	ldr	w5, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	mul	w9, w0, w1	// in_size, _5, _6
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	ldr	w0, [sp, 180]	//, %sfp
// src/cpp/cnn_internals.cpp:400:     const int in_ch_size  = in_channels * in_size;
	mul	w12, w20, w9	// in_ch_size, _2, in_size
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	mul	w2, w0, w28	// out_size, _7, _8
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	mul	w11, w5, w2	// out_ch_size, _1, out_size
	cmp	w0, 0	// _7,
	ble	.L990		//,
	cmp	w8, 2	// n_full,
	smull	x15, w22, w7	// _457, tiles, tile_wsz
	mov	w13, 2	// tmp3443,
	csel	w13, w8, w13, ge	// oc0, n_full, tmp3443,
	ldr	x1, [sp, 248]	// _12, %sfp
	cmp	w8, 0	// n_full,
	csel	w13, w13, wzr, gt	// oc0, oc0,,
	smull	x6, w13, w2	// _2502, oc0, out_size
	uxtw	x16, w13	// _2453, oc0
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	sub	w0, w28, #4	// _4625, _8,
	ldr	w4, [sp, 176]	//, %sfp
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	mov	w22, 0	// b,
	sbfiz	x23, x12, 2, 32	// _2615, in_ch_size,,
	sbfiz	x11, x11, 2, 32	// _2620, out_ch_size,,
	str	x6, [sp, 424]	// _2502, %sfp
	add	x6, x3, x15, lsl 2	// ivtmp.2413, iftmp.18_1846, _457,
	sub	w15, w5, w13	// _2525, _1, oc0
	add	x15, x15, w13, sxtw	// _2526, _2525, oc0
	ldr	x5, [sp, 200]	// _87, %sfp
	mul	x14, x1, x24	// _1520, _12, _3070
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	lsr	w1, w0, 2	// _4624, _4625,
	str	x6, [sp, 440]	// ivtmp.2413, %sfp
	sub	w10, w4, #1	// _5541, _4,
	add	x10, x10, 1	// _468, _5541,
	add	x6, x5, x16, lsl 2	// ivtmp.2409, _87, _2453,
	mov	x16, x27	// ivtmp.2461, in_ptr
	str	x6, [sp, 416]	// ivtmp.2409, %sfp
	add	w6, w1, 1	// _455, _4624,
	and	w1, w0, -4	// _5552, _4625,
	add	x0, x5, x15, lsl 2	// _2529, _87, _2526,
	add	w27, w1, 4	// col, _5552,
	str	w6, [sp, 256]	// _455, %sfp
	str	x0, [sp, 448]	// _2529, %sfp
	lsl	x0, x14, 2	// _2512, _1520,
	ldr	x14, [sp, 216]	// ivtmp.2462, %sfp
	str	x0, [sp, 432]	// _2512, %sfp
	sbfiz	x0, x7, 2, 32	// _2598, tile_wsz,,
	str	x0, [sp, 472]	// _2598, %sfp
	mov	x0, x6	// tmp6367, _455
	ubfiz	x0, x0, 4, 31	// _4613, tmp6367,,
	str	x0, [sp, 112]	// _4613, %sfp
	lsl	x0, x10, 3	// _5602, _468,
	str	x0, [sp, 496]	// _5602, %sfp
	sbfiz	x0, x2, 3, 32	// _2592, out_size,,
	str	x0, [sp, 464]	// _2592, %sfp
	sbfiz	x0, x2, 2, 32	// _2532, out_size,,
	str	x0, [sp, 248]	// _2532, %sfp
	sbfiz	x0, x4, 3, 32	// _2563, _4,,
	str	x0, [sp, 456]	// _2563, %sfp
	sbfiz	x0, x4, 2, 32	// _2364, _4,,
	str	x0, [sp, 408]	// _2364, %sfp
	sxtw	x0, w28	// _2604, _8
	str	x0, [sp, 184]	// _2604, %sfp
.L1025:
	stp	x26, x23, [sp, 376]	// _1847, _2615,
	mov	w26, w21	// _2220, _2220
	mov	w21, w20	// _2, _2
// src/cpp/cnn_internals.cpp:410:         float*       __restrict out_b = out_ptr   + (std::size_t)b * out_ch_size;
	mov	x15, 0	// ivtmp.2456,
	mov	w23, w28	// _8, _8
	mov	x7, 0	// ivtmp.2455,
	mov	w28, w22	// b, b
	ldr	x20, [sp, 208]	// _6, %sfp
	mov	w22, w19	// _3, _3
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	mov	w2, 0	// oh,
	mov	w1, w9	// in_size, in_size
	mov	x19, x15	// ivtmp.2456, ivtmp.2456
	str	x3, [sp, 264]	// iftmp.18_1846, %sfp
	mov	w3, w8	// n_full, n_full
	str	x16, [sp, 216]	// ivtmp.2461, %sfp
	str	w13, [sp, 272]	// oc0, %sfp
	str	x10, [sp, 280]	// _468, %sfp
	str	x11, [sp, 392]	// _2620, %sfp
.L991:
	add	x0, sp, 672	// tmp3576,,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	sxtw	x18, w1	// _1047, in_size
	str	x0, [sp, 400]	// tmp3576, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	cmp	w3, 0	// n_full,
	ble	.L1341		//,
	mov	w11, w28	// b, b
	mov	x24, x7	// ivtmp.2455, ivtmp.2455
	mov	x28, x20	// _6, _6
	mov	x7, x19	// ivtmp.2456, ivtmp.2456
	mov	w25, w3	// n_full, n_full
	mov	w19, w21	// _2, _2
	mov	w20, w2	// oh, oh
	mov	w21, w26	// _2220, _2220
	mov	w26, w22	// _3, _3
.L1008:
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	ldr	x22, [sp, 200]	// ivtmp.2446, %sfp
	add	x8, x14, x24, lsl 2	// ivtmp.2448, ivtmp.2462, ivtmp.2455,
	mov	w12, w20	// oh, oh
	mov	x4, x28	// _6, _6
	mov	w20, w21	// _2220, _2220
// src/cpp/cnn_internals.cpp:413:             int oc0 = 0, tile = 0;
	mov	w9, 0	// oc0,
	ldr	x6, [sp, 264]	// ivtmp.2449, %sfp
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	mov	w21, w25	// n_full, n_full
	mov	x13, x24	// ivtmp.2455, ivtmp.2455
	mov	w28, w19	// _2, _2
	mov	x25, x22	// ivtmp.2446, ivtmp.2446
.L1006:
// src/cpp/cnn_internals.cpp:347:         const float32x4_t bv = vdupq_n_f32(bias_ptr[oc0 + t]);
	ldr	s24, [x25, 4]	// _5434, MEM[(const float *)_2600 + 4B]
// src/cpp/cnn_internals.cpp:415:                 conv_row_tile<OC_TILE, NV>(
	mov	x17, x6	// wp, ivtmp.2449
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v5.4s}, [x25]	//, MEM[(const float *)_2600]
	dup	v24.4s, v24.s[0]	//, _5434
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q24, q24, [sp, 800]	//,,
	stp	q5, q5, [sp, 672]	//,,
	stp	q5, q5, [sp, 704]	//,,
	stp	q5, q5, [sp, 736]	//,,
	stp	q5, q24, [sp, 768]	//,,
	stp	q24, q24, [sp, 832]	//,,
	stp	q24, q24, [sp, 864]	//,,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w28, 0	// _2,
	ble	.L994		//,
	cmp	w26, 0	// _3,
	ble	.L994		//,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w16, 0	// ic,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	mov	x0, x7	// ivtmp.2439, ivtmp.2456
	.p2align 5,,15
.L1000:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w5, 0	// kh,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x10, x0	// ivtmp.2435, ivtmp.2439
.L999:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w2, [sp, 176]	//, %sfp
	cmp	w2, 0	// _4,
	bgt	.L997		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w5, w5, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x10, x10, x4	// ivtmp.2435, ivtmp.2435, _6
	cmp	w26, w5	// _3, kh
	bne	.L999		//,
	.p2align 5,,15
.L998:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w16, w16, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x0, x0, x18	// ivtmp.2439, ivtmp.2439, _1047
	cmp	w28, w16	// _2, ic
	bne	.L1000		//,
.L994:
	ldr	x22, [sp, 400]	// ivtmp.2421, %sfp
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	mov	w24, 0	// t,
	mov	w19, w23	// _8, _8
	mov	x23, x4	// _6, _6
	str	w21, [sp, 224]	// n_full, %sfp
	str	w20, [sp, 304]	// _2220, %sfp
	mov	x20, x8	// ivtmp.2420, ivtmp.2420
	str	w1, [sp, 288]	// in_size, %sfp
	str	w9, [sp, 312]	// oc0, %sfp
	mov	x21, x22	// ivtmp.2421, ivtmp.2421
	mov	w22, w24	// t, t
	str	w11, [sp, 320]	// b, %sfp
	mov	x24, x18	// _1047, _1047
	str	w12, [sp, 328]	// oh, %sfp
	stp	x8, x6, [sp, 336]	// ivtmp.2448, ivtmp.2449,
	stp	x13, x7, [sp, 352]	// ivtmp.2455, ivtmp.2456,
	str	x14, [sp, 368]	// ivtmp.2462, %sfp
.L993:
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w19, 3	// _8,
	ble	.L1001		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	x2, [sp, 112]	//, %sfp
	mov	x1, x21	//, ivtmp.2421
	mov	x0, x20	//, ivtmp.2420
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w19, w27	// _8, col
	bgt	.L1002		//,
.L1342:
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	cbnz	w22, .L1302	// t,
	ldr	x0, [sp, 248]	// _2532, %sfp
	add	x21, x21, 112	// ivtmp.2421, ivtmp.2421,
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	mov	w22, 1	// t,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	mov	x1, x21	//, ivtmp.2421
	ldr	x2, [sp, 112]	//, %sfp
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	x20, x20, x0	// ivtmp.2420, ivtmp.2420, _2532
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	mov	x0, x20	//, ivtmp.2420
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w19, w27	// _8, col
	ble	.L1342		//,
.L1002:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	sxtw	x1, w22	// t, t
	ubfiz	x0, x22, 3, 1	// tmp3487, t,,
	sub	x0, x0, x1	// tmp3488, tmp3487, t
	ldrsw	x1, [sp, 256]	// v, %sfp
	mov	w7, w27	// col, col
	add	x0, x0, x1	// tmp3489, tmp3488, v
	ldr	x1, [sp, 400]	// tmp3576, %sfp
	lsl	x0, x0, 4	// tmp3490, tmp3489,
	ldr	q31, [x1, x0]	// _371, MEM[(__Float32x4_t[2][7] *)_2907][t_4236][v_6061]
.L1005:
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w19, #1	// _2361, _8,
	uxtw	x0, w7	// _4637, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// _371, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w2, w7	// _4648, _2361, col
	cmp	w19, w7	// _8, col
	add	x0, x20, x0, lsl 2	//, ivtmp.2420, _4637,
	add	x1, sp, 624	//,,
	mov	x7, 4	// tmp3501,
	add	x2, x2, 1	// _4647, _4648,
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	x21, x21, 112	// ivtmp.2421, ivtmp.2421,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	lsl	x2, x2, 2	// _4644, _4647,
	csel	x2, x2, x7, gt	//, _4644, tmp3501,
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	ldr	x1, [sp, 248]	// _2532, %sfp
	add	w0, w22, 1	// _466, t,
	mov	w22, 1	// t,
	add	x20, x20, x1	// ivtmp.2420, ivtmp.2420, _2532
	cmp	w0, 2	// _466,
	bne	.L993		//,
.L1302:
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	x0, [sp, 464]	// _2592, %sfp
	mov	x4, x23	// _6, _6
	mov	x18, x24	// _1047, _1047
	ldp	x8, x6, [sp, 336]	// ivtmp.2448, ivtmp.2449,
	mov	w23, w19	// _8, _8
	add	x25, x25, 8	// ivtmp.2446, ivtmp.2446,
	ldp	x13, x7, [sp, 352]	// ivtmp.2455, ivtmp.2456,
	add	x8, x8, x0	// ivtmp.2448, ivtmp.2448, _2592
	ldr	w9, [sp, 312]	//, %sfp
	ldr	x0, [sp, 472]	// _2598, %sfp
	ldr	w21, [sp, 224]	//, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	w9, w9, 2	// oc0, oc0,
	ldr	x14, [sp, 368]	// ivtmp.2462, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	x6, x6, x0	// ivtmp.2449, ivtmp.2449, _2598
	ldr	w1, [sp, 288]	//, %sfp
	ldr	w20, [sp, 304]	//, %sfp
	ldr	w11, [sp, 320]	//, %sfp
	ldr	w12, [sp, 328]	//, %sfp
	cmp	w21, w9	// n_full, oc0
	bgt	.L1006		//,
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	mov	w19, w28	// _2, _2
	mov	w25, w21	// n_full, n_full
	ldr	w0, [sp, 108]	//, %sfp
	mov	w21, w20	// _2220, _2220
	mov	x28, x4	// _6, _6
	ldr	w2, [sp, 272]	//, %sfp
	cmp	w0, w2	// _1, oc0
	ble	.L1343		//,
	mov	w22, w26	// _3, _3
	mov	w3, w25	// n_full, n_full
	mov	w26, w21	// _2220, _2220
	mov	w2, w12	// oh, oh
	mov	w21, w19	// _2, _2
	mov	x20, x4	// _6, _6
	mov	x19, x7	// ivtmp.2456, ivtmp.2456
	mov	w28, w11	// b, b
	mov	x7, x13	// ivtmp.2455, ivtmp.2455
.L1007:
	ldp	x25, x0, [sp, 416]	// ivtmp.2409, _2502,
	str	w3, [sp, 312]	// n_full, %sfp
	str	w26, [sp, 320]	// _2220, %sfp
	add	x24, x0, x7	// _2503, _2502, ivtmp.2455
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	ldr	x0, [sp, 440]	// ivtmp.2413, %sfp
	add	x24, x14, x24, lsl 2	// ivtmp.2412, ivtmp.2462, _2503,
	str	w1, [sp, 328]	// in_size, %sfp
	str	w2, [sp, 336]	// oh, %sfp
	stp	x7, x14, [sp, 344]	// ivtmp.2455, ivtmp.2462,
	str	x0, [sp, 224]	// ivtmp.2413, %sfp
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	sxtw	x0, w1	// _2848, in_size
	str	x0, [sp, 304]	// _2848, %sfp
	ldr	x0, [sp, 280]	// _468, %sfp
	lsl	x0, x0, 2	// _3533, _468,
	str	x0, [sp, 288]	// _3533, %sfp
.L1022:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v5.4s}, [x25]	//, MEM[(const float *)_2520]
// src/cpp/cnn_internals.cpp:421:                 conv_row_tile<1, NV>(
	ldr	x7, [sp, 224]	// wp, %sfp
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q5, q5, [sp, 672]	//,,
	stp	q5, q5, [sp, 704]	//,,
	stp	q5, q5, [sp, 736]	//,,
	str	q5, [sp, 768]	//, MEM[(__Float32x4_t[1][7] *)_2907][0][6]
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w21, 0	// _2,
	ble	.L1013		//,
	cmp	w22, 0	// _3,
	ble	.L1013		//,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	ldr	x1, [sp, 304]	// _2848, %sfp
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w6, 0	// ic,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	mov	x0, x19	// ivtmp.2403, ivtmp.2456
	.p2align 5,,15
.L1019:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w2, 0	// kh,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x4, x0	// ivtmp.2399, ivtmp.2403
.L1018:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w3, [sp, 176]	//, %sfp
	cmp	w3, 0	// _4,
	bgt	.L1016		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w2, w2, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x20	// ivtmp.2399, ivtmp.2399, _6
	cmp	w22, w2	// _3, kh
	bne	.L1018		//,
	.p2align 5,,15
.L1017:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w6, w6, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x0, x0, x1	// ivtmp.2403, ivtmp.2403, _2848
	cmp	w21, w6	// _2, ic
	bne	.L1019		//,
.L1013:
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w23, 3	// _8,
	ble	.L1344		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	add	x26, sp, 672	// tmp3576,,
	ldr	x2, [sp, 112]	//, %sfp
	mov	x0, x24	//, ivtmp.2412
	mov	x1, x26	//, tmp3576
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	ldr	w0, [sp, 256]	//, %sfp
	mov	w3, w27	// col, col
	cmp	w23, w27	// _8, col
	ble	.L1021		//,
.L1020:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ubfiz	x0, x0, 4, 31	// tmp3531, v,,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w23, #1	// _5612, _8,
	sub	w2, w2, w3	// _4583, _5612, col
	uxtw	x6, w3	// _4569, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [x26, x0]	// tmp6401, MEM[(__Float32x4_t[1][7] *)_2907][0][v_1571]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	cmp	w23, w3	// _8, col
	add	x2, x2, 1	// _4582, _4583,
	add	x0, x24, x6, lsl 2	//, ivtmp.2412, _4569,
	mov	x3, 4	// tmp3543,
	add	x1, sp, 624	//,,
	lsl	x2, x2, 2	// _4581, _4582,
	csel	x2, x2, x3, gt	//, _4581, tmp3543,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// tmp6401, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	bl	memcpy		//
.L1021:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	x0, [sp, 248]	// _2532, %sfp
	add	x25, x25, 4	// ivtmp.2409, ivtmp.2409,
	ldr	x1, [sp, 432]	// _2512, %sfp
	add	x24, x24, x0	// ivtmp.2412, ivtmp.2412, _2532
	ldr	x0, [sp, 224]	// ivtmp.2413, %sfp
	add	x0, x0, x1	// ivtmp.2413, ivtmp.2413, _2512
	str	x0, [sp, 224]	// ivtmp.2413, %sfp
	ldr	x0, [sp, 448]	// _2529, %sfp
	cmp	x25, x0	// ivtmp.2409, _2529
	bne	.L1022		//,
	ldr	w3, [sp, 312]	//, %sfp
	ldp	x7, x14, [sp, 344]	// ivtmp.2455, ivtmp.2462,
	ldr	w26, [sp, 320]	//, %sfp
	ldr	w1, [sp, 328]	//, %sfp
	ldr	w2, [sp, 336]	//, %sfp
.L1023:
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w2, w2, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	x19, x19, x20	// ivtmp.2456, ivtmp.2456, _6
	ldr	x0, [sp, 184]	// _2604, %sfp
	add	x7, x7, x0	// ivtmp.2455, ivtmp.2455, _2604
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, w2	// _7, oh
	bne	.L991		//,
	ldr	x16, [sp, 216]	// ivtmp.2461, %sfp
	mov	w20, w21	// _2, _2
	mov	w19, w22	// _3, _3
	mov	w8, w3	// n_full, n_full
	mov	w21, w26	// _2220, _2220
	ldr	x3, [sp, 264]	// iftmp.18_1846, %sfp
	mov	w22, w28	// b, b
	mov	w28, w23	// _8, _8
	ldp	x26, x23, [sp, 376]	// _1847, _2615,
	mov	w9, w1	// in_size, in_size
	ldr	x10, [sp, 280]	// _468, %sfp
	ldr	x11, [sp, 392]	// _2620, %sfp
	ldr	w13, [sp, 272]	//, %sfp
.L1009:
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	w22, w22, 1	// b, b,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	x16, x16, x23	// ivtmp.2461, ivtmp.2461, _2615
	add	x14, x14, x11	// ivtmp.2462, ivtmp.2462, _2620
	cmp	w21, w22	// _2220, b
	bne	.L1025		//,
.L990:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x3, .L623	// iftmp.18_1846,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x1, x26	//, _1847
	mov	x0, x3	//, iftmp.18_1846
	bl	_ZdlPvm		//
// src/cpp/cnn_internals.cpp:479:     };
	b	.L623		//
	.p2align 2,,3
.L1345:
	ldr	x2, [sp, 496]	// _5602, %sfp
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w5, w5, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x10, x10, x4	// ivtmp.2435, ivtmp.2435, _6
	add	x17, x17, x2	// wp, wp, _5602
	cmp	w26, w5	// _3, kh
	beq	.L998		//,
.L997:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldp	q7, q30, [sp, 672]	//,,
	ldp	q28, q23, [sp, 704]	//,,
	ldp	q6, q3, [sp, 736]	//,,
	ldp	q1, q31, [sp, 768]	//,,
	ldp	q29, q27, [sp, 800]	//,,
	ldp	q20, q4, [sp, 832]	//,,
	ldp	q2, q0, [sp, 864]	//,,
	ldr	x2, [sp, 216]	// ivtmp.2461, %sfp
	ldr	x3, [sp, 456]	// _2563, %sfp
	add	x2, x2, x10, lsl 2	// ivtmp.2429, ivtmp.2461, ivtmp.2435,
	add	x15, x17, x3	// _2564, wp, _2563
	mov	x3, x17	// wp, wp
	.p2align 5,,15
.L996:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldp	q24, q17, [x2]	//,,* ivtmp.2429
	ldp	q16, q19, [x2, 32]	//,,
	ldp	q18, q21, [x2, 64]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x2, x2, 4	// ivtmp.2429, ivtmp.2429,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldr	q22, [x2, 92]	//, MEM <__Float32x4_t> [(float * {ref-all})_2555 + 96B]
// src/cpp/cnn_internals.cpp:360:                 for (int t = 0; t < OC_T; ++t) wv[t] = wp[t];
	ldp	s26, s25, [x3], 8	//,, wp,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v7.4s, v24.4s, v26.s[0]	//,,
	fmla	v31.4s, v24.4s, v25.s[0]	//,,
	fmla	v30.4s, v17.4s, v26.s[0]	//,,
	fmla	v29.4s, v17.4s, v25.s[0]	//,,
	fmla	v28.4s, v16.4s, v26.s[0]	//,,
	fmla	v27.4s, v16.4s, v25.s[0]	//,,
	fmla	v23.4s, v19.4s, v26.s[0]	//,,
	fmla	v20.4s, v19.4s, v25.s[0]	//,,
	fmla	v6.4s, v18.4s, v26.s[0]	//,,
	fmla	v4.4s, v18.4s, v25.s[0]	//,,
	fmla	v3.4s, v21.4s, v26.s[0]	//,,
	fmla	v2.4s, v21.4s, v25.s[0]	//,,
	fmla	v1.4s, v22.4s, v26.s[0]	//,,
	fmla	v0.4s, v22.4s, v25.s[0]	//,,
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q7, q30, [sp, 672]	//,,
	stp	q28, q23, [sp, 704]	//,,
	stp	q6, q3, [sp, 736]	//,,
	stp	q1, q31, [sp, 768]	//,,
	stp	q29, q27, [sp, 800]	//,,
	stp	q20, q4, [sp, 832]	//,,
	stp	q2, q0, [sp, 864]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x3, x15	// wp, _2564
	bne	.L996		//,
	b	.L1345		//
	.p2align 2,,3
.L1346:
	ldr	x3, [sp, 288]	// _3533, %sfp
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w2, w2, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x20	// ivtmp.2399, ivtmp.2399, _6
	add	x7, x7, x3	// wp, wp, _3533
	cmp	w22, w2	// _3, kh
	beq	.L1017		//,
.L1016:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldp	q31, q30, [sp, 672]	//,,
	ldp	q29, q28, [sp, 704]	//,,
	ldp	q27, q26, [sp, 736]	//,,
	ldr	x3, [sp, 216]	// ivtmp.2461, %sfp
	ldr	x5, [sp, 408]	// _2364, %sfp
	ldr	q25, [sp, 768]	//, MEM[(__Float32x4_t[1][7] *)_2907][0][6]
	add	x3, x3, x4, lsl 2	// ivtmp.2393, ivtmp.2461, ivtmp.2399,
	add	x8, x7, x5	// _2374, wp, _2364
	mov	x5, x7	// wp, wp
	.p2align 5,,15
.L1015:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v24.4s}, [x5], 4	//, MEM[(const float *)wp_4305]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldp	q6, q4, [x3]	//,,* ivtmp.2393
	ldp	q3, q2, [x3, 32]	//,,
	ldp	q1, q0, [x3, 64]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x3, x3, 4	// ivtmp.2393, ivtmp.2393,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldr	q23, [x3, 92]	//, MEM <__Float32x4_t> [(float * {ref-all})_2301 + 96B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v31.4s, v6.4s, v24.4s	//,,
	fmla	v30.4s, v4.4s, v24.4s	//,,
	fmla	v29.4s, v3.4s, v24.4s	//,,
	fmla	v28.4s, v2.4s, v24.4s	//,,
	fmla	v27.4s, v1.4s, v24.4s	//,,
	fmla	v26.4s, v0.4s, v24.4s	//,,
	fmla	v25.4s, v23.4s, v24.4s	//,,
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q31, q30, [sp, 672]	//,,
	stp	q29, q28, [sp, 704]	//,,
	stp	q27, q26, [sp, 736]	//,,
	str	q25, [sp, 768]	//, MEM[(__Float32x4_t[1][7] *)_2907][0][6]
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x8, x5	// _2374, wp
	bne	.L1015		//,
	b	.L1346		//
.L1344:
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w23, 0	// _8,
	ble	.L1021		//,
	mov	w3, 0	// col,
	mov	w0, 0	// v,
	add	x26, sp, 672	// tmp3576,,
	b	.L1020		//
.L1001:
	cmp	w19, 0	// _8,
	ble	.L1302		//,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [x21]	// _371, MEM[(__Float32x4_t *)_2544]
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	mov	w7, 0	// col,
	b	.L1005		//
.L1341:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	w0, [sp, 108]	//, %sfp
	cmp	w0, 0	// _1,
	bgt	.L1007		//,
	b	.L1023		//
.L1343:
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	ldr	x0, [sp, 184]	// _2604, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w20, w12, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	x7, x7, x4	// ivtmp.2456, ivtmp.2456, _6
	add	x24, x13, x0	// ivtmp.2455, ivtmp.2455, _2604
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, w20	// _7, oh
	bne	.L1008		//,
	ldr	x16, [sp, 216]	// ivtmp.2461, %sfp
	mov	w20, w19	// _2, _2
	mov	w28, w23	// _8, _8
	mov	w19, w26	// _3, _3
	mov	w22, w11	// b, b
	ldp	x26, x23, [sp, 376]	// _1847, _2615,
	mov	w8, w25	// n_full, n_full
	mov	w9, w1	// in_size, in_size
	mov	x13, x2	//,
	ldr	x3, [sp, 264]	// iftmp.18_1846, %sfp
	ldr	x10, [sp, 280]	// _468, %sfp
	ldr	x11, [sp, 392]	// _2620, %sfp
	b	.L1009		//
.L1078:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	w5, 0	// niters_vector_mult_vf.1652,
	sxtw	x2, w8	// _3542, ivtmp.2486
	b	.L978		//
.L1071:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	mov	x3, 0	// iftmp.18_1843,
.L922:
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	ldr	w0, [sp, 176]	//, %sfp
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	ldr	w2, [sp, 108]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x1, [x22]	// _1858, MEM[(const struct vector *)weight_24(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	mul	w0, w19, w0	// ksz, _3, _4
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	add	w22, w2, w2, lsr 31	// tmp3235, _1, _1,
// src/cpp/cnn_internals.cpp:318:     const int wcsz  = in_channels * ksz;              // per-oc stride in OIHW
	mul	w23, w20, w0	// wcsz, _2, ksz
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	asr	w22, w22, 1	// tiles, tmp3235,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w2, 1	// _1,
	ble	.L923		//,
	cmp	w20, 0	// _2,
	ble	.L923		//,
	sub	w2, w0, #1	// _5585, ksz,
	lsr	w8, w0, 2	// _2272, ksz,
	add	x2, x2, 1	// _431, _5585,
	mov	x13, 0	// ivtmp.2385,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w10, 0	// tb,
	lsl	x30, x8, 4	// _2273, _2272,
	lsl	x5, x2, 3	// _1594, _431,
	sxtw	x12, w23	// ivtmp.2386, wcsz
	sbfiz	x14, x23, 1, 32	// _2287, wcsz,,
	sub	w4, w0, #1	// _3879, ksz,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	x11, x3	// packed, iftmp.18_1843
	and	w18, w0, -4	// niters_vector_mult_vf.1627, ksz,
.L924:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	w9, 0	// ivtmp.2381,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w8, 0	// ic,
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w0, 0	// ksz,
	bgt	.L928		//,
.L1276:
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w8, w8, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w9, w9, w0	// ivtmp.2381, ivtmp.2381, ksz
	cmp	w20, w8	// _2, ic
	beq	.L929		//,
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w0, 0	// ksz,
	ble	.L1276		//,
.L928:
	cmp	w4, 2	// _3879,
	bls	.L1073		//,
	sxtw	x2, w9	// _3804, ivtmp.2381
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	x6, 0	// ivtmp.2374,
	add	x16, x13, x2	// _3860, ivtmp.2385, _3804
	add	x15, x12, x2	// _3846, ivtmp.2386, _3804
	add	x16, x1, x16, lsl 2	// vectp.1631, _1858, _3860,
	mov	x17, x11	// ivtmp.2372, packed
	add	x15, x1, x15, lsl 2	// vectp.1634, _1858, _3846,
.L926:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	ldr	q28, [x16, x6]	//, MEM <const vector(4) float> [(const float *)vectp.1631_3863 + ivtmp.2374_2270 * 1]
	ldr	q29, [x15, x6]	//, MEM <const vector(4) float> [(const float *)vectp.1634_3848 + ivtmp.2374_2270 * 1]
	add	x6, x6, 16	// ivtmp.2374, ivtmp.2374,
	st2	{v28.4s - v29.4s}, [x17], 32	//, MEM <float[8]> [(float *)vectp_packed.1636_3836]
	cmp	x6, x30	// ivtmp.2374, _2273
	bne	.L926		//,
	mov	w6, w18	// niters_vector_mult_vf.1627, niters_vector_mult_vf.1627
	cmp	w18, w0	// niters_vector_mult_vf.1627, ksz
	beq	.L927		//,
.L925:
	uxtw	x16, w6	// _3801, niters_vector_mult_vf.1627
	add	x15, x13, x2	// _3802, ivtmp.2385, _3804
	sub	w6, w0, w6	// bnd.1640_3808, ksz, niters_vector_mult_vf.1627
	add	x2, x12, x2	// _3782, ivtmp.2386, _3804
	whilelo	p7.s, wzr, w6	// max_mask_3719,, bnd.1640_3808
	add	x17, x11, x16, lsl 3	// vectp_packed.1648, packed, _3801,
	add	x6, x15, x16	// _3800, _3802, _3801
	add	x2, x2, x16	// _3780, _3782, _3801
	ld1w	z24.s, p7/z, [x1, x6, lsl 2]	//, max_mask_3719,* _1858
	ld1w	z25.s, p7/z, [x1, x2, lsl 2]	//, max_mask_3719,* _1858
	st2w	{z24.s - z25.s}, p7, [x17]	//, max_mask_3719,* vectp_packed.1648
.L927:
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w8, w8, 1	// ic, ic,
	add	x11, x11, x5	// packed, packed, _1594
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w9, w9, w0	// ivtmp.2381, ivtmp.2381, ksz
	cmp	w20, w8	// _2, ic
	bne	.L928		//,
.L929:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	w10, w10, 1	// tb, tb,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	x13, x13, x14	// ivtmp.2385, ivtmp.2385, _2287
	add	x12, x12, x14	// ivtmp.2386, ivtmp.2386, _2287
	cmp	w22, w10	// tiles, tb
	bgt	.L924		//,
.L923:
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	ldr	w2, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:471:         const int n_full = (OC / OC_TILE) * OC_TILE;
	lsl	w11, w22, 1	// n_full, tiles,
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	cmp	w2, w11	// _1, n_full
	bgt	.L1347		//,
.L932:
// src/cpp/cnn_internals.cpp:478:             input.batches, IC, OC, IH, IW, KH, KW, OH, OW);
	ldr	w21, [x21, 24]	//, input_29(D)->batches
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	cmp	w21, 0	// _1888,
	ble	.L937		//,
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w0, [sp, 184]	//, %sfp
// src/cpp/cnn_internals.cpp:404:     const int tile_wsz    = in_channels * ksz * OC_TILE;   // packed floats/tile
	lsl	w9, w23, 1	// tile_wsz, wcsz,
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w1, [sp, 224]	//, %sfp
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	ldr	w5, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	mul	w13, w0, w1	// in_size, _5, _6
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	ldr	w0, [sp, 180]	//, %sfp
// src/cpp/cnn_internals.cpp:400:     const int in_ch_size  = in_channels * in_size;
	mul	w15, w20, w13	// in_ch_size, _2, in_size
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	mul	w2, w0, w28	// out_size, _7, _8
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	mul	w12, w5, w2	// out_ch_size, _1, out_size
	cmp	w0, 0	// _7,
	ble	.L937		//,
	cmp	w11, 2	// n_full,
	smull	x14, w22, w9	// _442, tiles, tile_wsz
	mov	w8, 2	// tmp3281,
	csel	w8, w11, w8, ge	// oc0, n_full, tmp3281,
	ldr	x1, [sp, 248]	// _12, %sfp
	cmp	w11, 0	// n_full,
	csel	w8, w8, wzr, gt	// oc0, oc0,,
	smull	x6, w8, w2	// _1958, oc0, out_size
	uxtw	x17, w8	// _1950, oc0
	str	x27, [sp, 224]	// in_ptr, %sfp
	ldr	w4, [sp, 176]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	sub	w0, w28, #4	// _4789, _8,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	mov	w22, 0	// b,
	sbfiz	x23, x15, 2, 32	// _2230, in_ch_size,,
	str	x6, [sp, 384]	// _1958, %sfp
	add	x6, x3, x14, lsl 2	// ivtmp.2308, iftmp.18_1843, _442,
	sub	w14, w5, w8	// _1983, _1, oc0
	add	x14, x14, w8, sxtw	// _1984, _1983, oc0
	ldr	x5, [sp, 200]	// _87, %sfp
	mul	x16, x1, x25	// _852, _12, _5895
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	lsr	w1, w0, 2	// _4777, _4789,
	str	x6, [sp, 400]	// ivtmp.2308, %sfp
	sub	w10, w4, #1	// _5609, _4,
	add	x6, x5, x17, lsl 2	// ivtmp.2304, _87, _1950,
	str	x6, [sp, 376]	// ivtmp.2304, %sfp
	add	w6, w1, 1	// _436, _4777,
	and	w1, w0, -4	// _5621, _4789,
	add	x0, x5, x14, lsl 2	// _1995, _87, _1984,
	add	w27, w1, 4	// col, _5621,
	mov	x1, x6	// tmp6286, _436
	ubfiz	x1, x1, 4, 31	// _4763, tmp6286,,
	add	x14, x10, 1	// _453, _5609,
	str	w6, [sp, 280]	// _436, %sfp
	sbfiz	x5, x9, 2, 32	// _2196, tile_wsz,,
	str	x0, [sp, 408]	// _1995, %sfp
	lsl	x0, x16, 2	// _1964, _852,
	str	x1, [sp, 112]	// _4763, %sfp
	lsl	x1, x14, 3	// _6108, _453,
	str	x0, [sp, 392]	// _1964, %sfp
	sbfiz	x0, x12, 2, 32	// _2237, out_ch_size,,
	str	x1, [sp, 448]	// _6108, %sfp
	sbfiz	x1, x2, 3, 32	// _2181, out_size,,
	ldr	x12, [sp, 216]	// ivtmp.2357, %sfp
	stp	x1, x5, [sp, 424]	// _2181, _2196,
	sbfiz	x1, x2, 2, 32	// _2010, out_size,,
	str	x1, [sp, 264]	// _2010, %sfp
	sbfiz	x1, x4, 3, 32	// _2095, _4,,
	str	x1, [sp, 416]	// _2095, %sfp
	sbfiz	x1, x4, 2, 32	// _1928, _4,,
	str	x1, [sp, 368]	// _1928, %sfp
	sxtw	x1, w28	// _2604, _8
	str	x1, [sp, 184]	// _2604, %sfp
.L973:
	str	w21, [sp, 328]	// _1888, %sfp
// src/cpp/cnn_internals.cpp:410:         float*       __restrict out_b = out_ptr   + (std::size_t)b * out_ch_size;
	mov	x25, 0	// ivtmp.2351,
	mov	x10, 0	// ivtmp.2350,
	ldr	x21, [sp, 208]	// _6, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	mov	w9, 0	// oh,
	mov	w26, w28	// _8, _8
	mov	x2, x12	// ivtmp.2357, ivtmp.2357
	str	w8, [sp, 288]	// oc0, %sfp
	mov	w8, w9	// oh, oh
	str	x3, [sp, 304]	// iftmp.18_1843, %sfp
	mov	x3, x10	// ivtmp.2350, ivtmp.2350
	stp	x23, x0, [sp, 344]	// _2230, _2237,
	mov	w23, w19	// _3, _3
	mov	x19, x25	// ivtmp.2351, ivtmp.2351
	str	w11, [sp, 216]	// n_full, %sfp
	str	w13, [sp, 248]	// in_size, %sfp
	str	x7, [sp, 320]	// _1845, %sfp
	str	w22, [sp, 336]	// b, %sfp
.L938:
	add	x0, sp, 672	// tmp3576,,
	str	x0, [sp, 360]	// tmp3576, %sfp
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	ldrsw	x0, [sp, 248]	// _3937, %sfp
	str	x0, [sp, 440]	// _3937, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	w0, [sp, 216]	//, %sfp
	cmp	w0, 0	// n_full,
	ble	.L1348		//,
	mov	w22, w23	// _3, _3
	mov	x24, x21	// _6, _6
	mov	w23, w27	// col, col
	mov	w21, w20	// _2, _2
	mov	w27, w26	// _8, _8
	mov	w25, w8	// oh, oh
	mov	x26, x3	// ivtmp.2350, ivtmp.2350
	mov	x20, x19	// ivtmp.2351, ivtmp.2351
.L955:
	add	x0, x2, x26, lsl 2	// ivtmp.2343, ivtmp.2357, ivtmp.2350,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	ldr	x28, [sp, 200]	// ivtmp.2341, %sfp
	stp	x26, x2, [sp, 456]	// ivtmp.2350, ivtmp.2357,
	mov	x26, x24	// _6, _6
	mov	w24, w27	// _8, _8
	str	x0, [sp, 256]	// ivtmp.2343, %sfp
	ldr	x0, [sp, 304]	// iftmp.18_1843, %sfp
	str	w25, [sp, 472]	// oh, %sfp
	mov	x25, x28	// ivtmp.2341, ivtmp.2341
// src/cpp/cnn_internals.cpp:413:             int oc0 = 0, tile = 0;
	str	wzr, [sp, 312]	//, %sfp
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	str	x14, [sp, 496]	// _453, %sfp
	str	x0, [sp, 272]	// iftmp.18_1843, %sfp
.L953:
// src/cpp/cnn_internals.cpp:347:         const float32x4_t bv = vdupq_n_f32(bias_ptr[oc0 + t]);
	ldr	s23, [x25, 4]	// _5184, MEM[(const float *)_2198 + 4B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v20.4s}, [x25]	//, MEM[(const float *)_2198]
// src/cpp/cnn_internals.cpp:415:                 conv_row_tile<OC_TILE, NV>(
	ldr	x9, [sp, 272]	// wp, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	dup	v23.4s, v23.s[0]	//, _5184
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q23, q23, [sp, 768]	//,,
	stp	q20, q20, [sp, 672]	//,,
	stp	q20, q20, [sp, 704]	//,,
	stp	q20, q20, [sp, 736]	//,,
	stp	q23, q23, [sp, 800]	//,,
	stp	q23, q23, [sp, 832]	//,,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w21, 0	// _2,
	ble	.L941		//,
	cmp	w22, 0	// _3,
	ble	.L941		//,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	ldr	x1, [sp, 440]	// _3937, %sfp
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w5, 0	// ic,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	mov	x0, x20	// ivtmp.2334, ivtmp.2351
	.p2align 5,,15
.L947:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w3, 0	// kh,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x4, x0	// ivtmp.2330, ivtmp.2334
.L946:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w2, [sp, 176]	//, %sfp
	cmp	w2, 0	// _4,
	bgt	.L944		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w3, w3, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x26	// ivtmp.2330, ivtmp.2330, _6
	cmp	w22, w3	// _3, kh
	bne	.L946		//,
	.p2align 5,,15
.L945:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w5, w5, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x0, x0, x1	// ivtmp.2334, ivtmp.2334, _3937
	cmp	w21, w5	// _2, ic
	bne	.L947		//,
.L941:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldr	x27, [sp, 256]	// ivtmp.2315, %sfp
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	mov	w19, 0	// t,
	ldr	x28, [sp, 360]	// ivtmp.2316, %sfp
.L940:
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w24, 3	// _8,
	ble	.L948		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	x2, [sp, 112]	//, %sfp
	mov	x1, x28	//, ivtmp.2316
	mov	x0, x27	//, ivtmp.2315
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w24, w23	// _8, col
	bgt	.L949		//,
.L1349:
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	cbnz	w19, .L950	// t,
	ldr	x0, [sp, 264]	// _2010, %sfp
	add	x28, x28, 96	// ivtmp.2316, ivtmp.2316,
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	mov	w19, 1	// t,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	mov	x1, x28	//, ivtmp.2316
	ldr	x2, [sp, 112]	//, %sfp
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	x27, x27, x0	// ivtmp.2315, ivtmp.2315, _2010
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	mov	x0, x27	//, ivtmp.2315
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w24, w23	// _8, col
	ble	.L1349		//,
.L949:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldrsw	x1, [sp, 280]	// v, %sfp
	and	x0, x19, 1	// tmp3879, t,
	mov	w3, w23	// col, col
	orr	x0, x0, x0, lsl 1	// tmp3325, tmp3879, tmp3879,
	add	x0, x1, x0, lsl 1	// tmp3327, v, tmp3325,
	ldr	x1, [sp, 360]	// tmp3576, %sfp
	lsl	x0, x0, 4	// tmp3328, tmp3327,
	ldr	q31, [x1, x0]	// _697, MEM[(__Float32x4_t[2][6] *)_2907][t_172][v_2357]
.L952:
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w24, #1	// _5624, _8,
	uxtw	x0, w3	// _4794, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// _697, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w2, w3	// _4806, _5624, col
	cmp	w24, w3	// _8, col
	add	x0, x27, x0, lsl 2	//, ivtmp.2315, _4794,
	add	x1, sp, 624	//,,
	mov	x3, 4	// tmp3339,
	add	x2, x2, 1	// _4804, _4806,
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	x28, x28, 96	// ivtmp.2316, ivtmp.2316,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	lsl	x2, x2, 2	// _4803, _4804,
	csel	x2, x2, x3, gt	//, _4803, tmp3339,
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	ldr	x1, [sp, 264]	// _2010, %sfp
	add	w0, w19, 1	// _449, t,
	mov	w19, 1	// t,
	add	x27, x27, x1	// ivtmp.2315, ivtmp.2315, _2010
	cmp	w0, 2	// _449,
	bne	.L940		//,
.L950:
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	x1, [sp, 256]	// ivtmp.2343, %sfp
	add	x25, x25, 8	// ivtmp.2341, ivtmp.2341,
	ldr	x2, [sp, 424]	// _2181, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	w0, [sp, 312]	//, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	x1, x1, x2	// ivtmp.2343, ivtmp.2343, _2181
	ldr	x2, [sp, 432]	// _2196, %sfp
	str	x1, [sp, 256]	// ivtmp.2343, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	w0, w0, 2	// oc0, oc0,
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	x1, [sp, 272]	// ivtmp.2344, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	str	w0, [sp, 312]	// oc0, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	x1, x1, x2	// ivtmp.2344, ivtmp.2344, _2196
	str	x1, [sp, 272]	// ivtmp.2344, %sfp
	ldr	w1, [sp, 216]	//, %sfp
	cmp	w1, w0	// n_full, oc0
	bgt	.L953		//,
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	w0, [sp, 108]	//, %sfp
	mov	w27, w24	// _8, _8
	mov	x24, x26	// _6, _6
	ldp	x26, x2, [sp, 456]	// ivtmp.2350, ivtmp.2357,
	ldr	w1, [sp, 288]	//, %sfp
	ldr	x14, [sp, 496]	// _453, %sfp
	ldr	w25, [sp, 472]	//, %sfp
	cmp	w0, w1	// _1, oc0
	ble	.L1350		//,
	mov	x3, x26	// ivtmp.2350, ivtmp.2350
	mov	x19, x20	// ivtmp.2351, ivtmp.2351
	mov	w26, w27	// _8, _8
	mov	w20, w21	// _2, _2
	mov	w27, w23	// col, col
	mov	w8, w25	// oh, oh
	mov	w23, w22	// _3, _3
	mov	x21, x24	// _6, _6
.L954:
	ldp	x28, x0, [sp, 376]	// ivtmp.2304, _1958,
	mov	w22, w26	// _8, _8
	mov	x26, x2	// ivtmp.2357, ivtmp.2357
	str	x3, [sp, 360]	// ivtmp.2350, %sfp
	str	w8, [sp, 440]	// oh, %sfp
	add	x24, x0, x3	// _1959, _1958, ivtmp.2350
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	ldr	x0, [sp, 400]	// ivtmp.2308, %sfp
	add	x24, x2, x24, lsl 2	// ivtmp.2307, ivtmp.2357, _1959,
	str	x14, [sp, 456]	// _453, %sfp
	str	x0, [sp, 256]	// ivtmp.2308, %sfp
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	ldrsw	x0, [sp, 248]	// _3918, %sfp
	str	x0, [sp, 272]	// _3918, %sfp
	lsl	x0, x14, 2	// _4087, _453,
	str	x0, [sp, 312]	// _4087, %sfp
.L970:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v7.4s}, [x28]	//, MEM[(const float *)_1973]
// src/cpp/cnn_internals.cpp:421:                 conv_row_tile<1, NV>(
	ldr	x7, [sp, 256]	// wp, %sfp
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q7, q7, [sp, 672]	//,,
	stp	q7, q7, [sp, 704]	//,,
	stp	q7, q7, [sp, 736]	//,,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w20, 0	// _2,
	ble	.L960		//,
	cmp	w23, 0	// _3,
	ble	.L960		//,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	ldr	x1, [sp, 272]	// _3918, %sfp
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w6, 0	// ic,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	mov	x0, x19	// ivtmp.2298, ivtmp.2351
	.p2align 5,,15
.L966:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w2, 0	// kh,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x4, x0	// ivtmp.2294, ivtmp.2298
.L965:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w3, [sp, 176]	//, %sfp
	cmp	w3, 0	// _4,
	bgt	.L963		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w2, w2, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x21	// ivtmp.2294, ivtmp.2294, _6
	cmp	w23, w2	// _3, kh
	bne	.L965		//,
	.p2align 5,,15
.L964:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w6, w6, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x0, x0, x1	// ivtmp.2298, ivtmp.2298, _3918
	cmp	w20, w6	// _2, ic
	bne	.L966		//,
.L960:
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w22, 3	// _8,
	ble	.L1351		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	add	x25, sp, 672	// tmp3576,,
	ldr	x2, [sp, 112]	//, %sfp
	mov	x0, x24	//, ivtmp.2307
	mov	x1, x25	//, tmp3576
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	ldr	w0, [sp, 280]	//, %sfp
	mov	w3, w27	// col, col
	cmp	w22, w27	// _8, col
	ble	.L969		//,
.L967:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ubfiz	x0, x0, 4, 31	// tmp3368, v,,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w22, #1	// _4947, _8,
	sub	w2, w2, w3	// _4736, _4947, col
	uxtw	x6, w3	// _4726, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [x25, x0]	// tmp6329, MEM[(__Float32x4_t[1][6] *)_2907][0][v_1564]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	cmp	w22, w3	// _8, col
	add	x2, x2, 1	// _4730, _4736,
	add	x0, x24, x6, lsl 2	//, ivtmp.2307, _4726,
	mov	x3, 4	// tmp3380,
	add	x1, sp, 624	//,,
	lsl	x2, x2, 2	// _4729, _4730,
	csel	x2, x2, x3, gt	//, _4729, tmp3380,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// tmp6329, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	bl	memcpy		//
.L969:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	x0, [sp, 264]	// _2010, %sfp
	add	x28, x28, 4	// ivtmp.2304, ivtmp.2304,
	ldr	x1, [sp, 392]	// _1964, %sfp
	add	x24, x24, x0	// ivtmp.2307, ivtmp.2307, _2010
	ldr	x0, [sp, 256]	// ivtmp.2308, %sfp
	add	x0, x0, x1	// ivtmp.2308, ivtmp.2308, _1964
	str	x0, [sp, 256]	// ivtmp.2308, %sfp
	ldr	x0, [sp, 408]	// _1995, %sfp
	cmp	x28, x0	// ivtmp.2304, _1995
	bne	.L970		//,
	ldr	x3, [sp, 360]	// ivtmp.2350, %sfp
	mov	x2, x26	// ivtmp.2357, ivtmp.2357
	mov	w26, w22	// _8, _8
	ldr	x14, [sp, 456]	// _453, %sfp
	ldr	w8, [sp, 440]	//, %sfp
.L971:
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w8, w8, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	x19, x19, x21	// ivtmp.2351, ivtmp.2351, _6
	ldr	x0, [sp, 184]	// _2604, %sfp
	add	x3, x3, x0	// ivtmp.2350, ivtmp.2350, _2604
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, w8	// _7, oh
	bne	.L938		//,
	ldr	x3, [sp, 304]	// iftmp.18_1843, %sfp
	mov	w19, w23	// _3, _3
	mov	w28, w26	// _8, _8
	ldp	x23, x0, [sp, 344]	// _2230, _2237,
	mov	x12, x2	// ivtmp.2357, ivtmp.2357
	ldr	x7, [sp, 320]	// _1845, %sfp
	ldr	w11, [sp, 216]	//, %sfp
	ldr	w13, [sp, 248]	//, %sfp
	ldr	w8, [sp, 288]	//, %sfp
	ldr	w21, [sp, 328]	//, %sfp
	ldr	w22, [sp, 336]	//, %sfp
.L956:
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	w22, w22, 1	// b, b,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	x12, x12, x0	// ivtmp.2357, ivtmp.2357, _2237
	ldr	x1, [sp, 224]	// ivtmp.2356, %sfp
	add	x1, x1, x23	// ivtmp.2356, ivtmp.2356, _2230
	str	x1, [sp, 224]	// ivtmp.2356, %sfp
	cmp	w21, w22	// _1888, b
	bne	.L973		//,
.L937:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x3, .L623	// iftmp.18_1843,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x1, x7	//, _1845
	mov	x0, x3	//, iftmp.18_1843
	bl	_ZdlPvm		//
// src/cpp/cnn_internals.cpp:479:     };
	b	.L623		//
	.p2align 2,,3
.L1352:
	ldr	x2, [sp, 448]	// _6108, %sfp
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w3, w3, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x26	// ivtmp.2330, ivtmp.2330, _6
	add	x9, x9, x2	// wp, wp, _6108
	cmp	w22, w3	// _3, kh
	beq	.L945		//,
.L944:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldp	q16, q4, [sp, 672]	//,,
	ldp	q22, q30, [sp, 704]	//,,
	ldp	q19, q17, [sp, 736]	//,,
	ldp	q5, q3, [sp, 768]	//,,
	ldp	q31, q21, [sp, 800]	//,,
	ldp	q18, q2, [sp, 832]	//,,
	ldr	x2, [sp, 224]	// ivtmp.2356, %sfp
	ldr	x6, [sp, 416]	// _2095, %sfp
	add	x2, x2, x4, lsl 2	// ivtmp.2324, ivtmp.2356, ivtmp.2330,
	add	x7, x9, x6	// _2099, wp, _2095
	mov	x6, x9	// wp, wp
	.p2align 5,,15
.L943:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldp	q25, q24, [x2]	//,,* ivtmp.2324
	ldp	q27, q26, [x2, 32]	//,,
	ldp	q28, q29, [x2, 64]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x2, x2, 4	// ivtmp.2324, ivtmp.2324,
// src/cpp/cnn_internals.cpp:360:                 for (int t = 0; t < OC_T; ++t) wv[t] = wp[t];
	ldp	s1, s0, [x6], 8	//,, wp,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v16.4s, v25.4s, v1.s[0]	//,,
	fmla	v5.4s, v25.4s, v0.s[0]	//,,
	fmla	v4.4s, v24.4s, v1.s[0]	//,,
	fmla	v3.4s, v24.4s, v0.s[0]	//,,
	fmla	v22.4s, v27.4s, v1.s[0]	//,,
	fmla	v31.4s, v27.4s, v0.s[0]	//,,
	fmla	v30.4s, v26.4s, v1.s[0]	//,,
	fmla	v21.4s, v26.4s, v0.s[0]	//,,
	fmla	v19.4s, v28.4s, v1.s[0]	//,,
	fmla	v18.4s, v28.4s, v0.s[0]	//,,
	fmla	v17.4s, v29.4s, v1.s[0]	//,,
	fmla	v2.4s, v29.4s, v0.s[0]	//,,
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q16, q4, [sp, 672]	//,,
	stp	q22, q30, [sp, 704]	//,,
	stp	q19, q17, [sp, 736]	//,,
	stp	q5, q3, [sp, 768]	//,,
	stp	q31, q21, [sp, 800]	//,,
	stp	q18, q2, [sp, 832]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x6, x7	// wp, _2099
	bne	.L943		//,
	b	.L1352		//
	.p2align 2,,3
.L1353:
	ldr	x3, [sp, 312]	// _4087, %sfp
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w2, w2, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x21	// ivtmp.2294, ivtmp.2294, _6
	add	x7, x7, x3	// wp, wp, _4087
	cmp	w23, w2	// _3, kh
	beq	.L964		//,
.L963:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldp	q6, q23, [sp, 672]	//,,
	ldp	q20, q31, [sp, 704]	//,,
	ldp	q30, q29, [sp, 736]	//,,
	ldr	x3, [sp, 224]	// ivtmp.2356, %sfp
	ldr	x5, [sp, 368]	// _1928, %sfp
	add	x3, x3, x4, lsl 2	// ivtmp.2288, ivtmp.2356, ivtmp.2294,
	add	x8, x7, x5	// _1929, wp, _1928
	mov	x5, x7	// wp, wp
	.p2align 5,,15
.L962:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v28.4s}, [x5], 4	//, MEM[(const float *)wp_5029]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldp	q4, q3, [x3]	//,,* ivtmp.2288
	ldp	q2, q1, [x3, 32]	//,,
	ldp	q0, q27, [x3, 64]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x3, x3, 4	// ivtmp.2288, ivtmp.2288,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v6.4s, v4.4s, v28.4s	//,,
	fmla	v23.4s, v3.4s, v28.4s	//,,
	fmla	v20.4s, v2.4s, v28.4s	//,,
	fmla	v31.4s, v1.4s, v28.4s	//,,
	fmla	v30.4s, v0.4s, v28.4s	//,,
	fmla	v29.4s, v27.4s, v28.4s	//,,
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q6, q23, [sp, 672]	//,,
	stp	q20, q31, [sp, 704]	//,,
	stp	q30, q29, [sp, 736]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x8, x5	// _1929, wp
	bne	.L962		//,
	b	.L1353		//
.L1351:
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w22, 0	// _8,
	ble	.L969		//,
	mov	w0, 0	// v,
	mov	w3, 0	// col,
	add	x25, sp, 672	// tmp3576,,
	b	.L967		//
.L948:
	cmp	w24, 0	// _8,
	ble	.L950		//,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [x28]	// _697, MEM[(__Float32x4_t *)_2046]
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	mov	w3, 0	// col,
	b	.L952		//
.L1350:
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	ldr	x0, [sp, 184]	// _2604, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w25, w25, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	x20, x20, x24	// ivtmp.2351, ivtmp.2351, _6
	add	x26, x26, x0	// ivtmp.2350, ivtmp.2350, _2604
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, w25	// _7, oh
	bne	.L955		//,
	ldr	x3, [sp, 304]	// iftmp.18_1843, %sfp
	mov	w20, w21	// _2, _2
	mov	w19, w22	// _3, _3
	mov	w28, w27	// _8, _8
	mov	x12, x2	// ivtmp.2357, ivtmp.2357
	ldr	x7, [sp, 320]	// _1845, %sfp
	mov	w27, w23	// col, col
	mov	x8, x1	//,
	ldp	x23, x0, [sp, 344]	// _2230, _2237,
	ldr	w11, [sp, 216]	//, %sfp
	ldr	w13, [sp, 248]	//, %sfp
	ldr	w21, [sp, 328]	//, %sfp
	ldr	w22, [sp, 336]	//, %sfp
	b	.L956		//
.L1073:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	w6, 0	// niters_vector_mult_vf.1627,
	sxtw	x2, w9	// _3804, ivtmp.2381
	b	.L925		//
.L1348:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	w0, [sp, 108]	//, %sfp
	cmp	w0, 0	// _1,
	bgt	.L954		//,
	b	.L971		//
.L1082:
	.cfi_restore 23
	.cfi_restore 24
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -840
	.cfi_offset 23, -848
// src/cpp/cnn_internals.cpp:482:     switch (NV) {
	mov	x23, 0	// in_pad$_M_end_of_storage,
	str	xzr, [sp, 192]	//, %sfp
	str	xzr, [sp, 240]	//, %sfp
.L621:
	ldr	w1, [sp, 176]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	mov	x0, 2305843009213693951	// tmp2638,
// src/cpp/cnn_internals.cpp:468:         std::vector<float> packed((std::size_t)OC * IC * KH * KW);
	ldr	w3, [sp, 108]	//, %sfp
	smull	x2, w19, w1	// _5928, _3, _4
	sxtw	x24, w1	// _4, _4
	sbfiz	x1, x1, 2, 32	// _5781, _4,,
	smull	x25, w20, w3	// _868, _2, _1
	str	x1, [sp, 376]	// _5781, %sfp
	str	x2, [sp, 264]	// _5928, %sfp
	mul	x25, x25, x2	// _860, _868, _5928
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	cmp	x25, x0	// _860, tmp2638
	bhi	.L1354		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	cbz	x25, .L1055	// _860,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	lsl	x7, x25, 2	// _916, _860,
	mov	x0, x7	//, _916
	str	x7, [sp, 112]	// _916, %sfp
	bl	_Znwm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	str	wzr, [x0]	//, *_917
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x3, x0	// _917, tmp3871
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	cmp	x25, 1	// _860,
	beq	.L1056		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	ldr	x7, [sp, 112]	// _916, %sfp
	mov	w1, 0	//,
	add	x0, x0, 4	//, _917,
	str	x3, [sp, 112]	// _917, %sfp
	sub	x2, x7, #4	//, _916,
	str	x7, [sp, 256]	// _916, %sfp
	bl	memset		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	ldr	x7, [sp, 256]	// _916, %sfp
	ldr	x3, [sp, 112]	// _917, %sfp
	mov	x25, x7	// _860, _916
	b	.L753		//
.L1037:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:99: 	: _M_start(), _M_finish(), _M_end_of_storage()
	mov	x24, 0	// _167,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:99: 	: _M_start(), _M_finish(), _M_end_of_storage()
	mov	x1, 0	// in_pad$_M_finish,
	b	.L612		//
.L1055:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	mov	x3, 0	// _917,
.L753:
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	ldr	w0, [sp, 176]	//, %sfp
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	ldr	w1, [sp, 108]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x22, [x22]	// _862, MEM[(const struct vector *)weight_24(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	mul	w23, w19, w0	// ksz, _3, _4
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	cmp	w1, 0	// _1,
	add	w0, w1, 3	// tmp2648, _1,
	csel	w0, w0, w1, lt	// _1, tmp2648, _1,
// src/cpp/cnn_internals.cpp:318:     const int wcsz  = in_channels * ksz;              // per-oc stride in OIHW
	mul	w7, w20, w23	// wcsz, _2, ksz
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	asr	w12, w0, 2	// tiles, _1,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w1, 3	// _1,
	ble	.L754		//,
	cmp	w20, 0	// _2,
	ble	.L754		//,
	sub	w0, w23, #1	// _5833, ksz,
	lsr	w2, w23, 2	// _1181, ksz,
	add	x0, x0, 1	// _561, _5833,
	mov	x10, 0	// ivtmp.2065,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w1, 0	// tb,
	lsl	x6, x2, 4	// _1182, _1181,
	lsl	x5, x0, 4	// _5779, _561,
	sxtw	x9, w7	// ivtmp.2066, wcsz
	sbfiz	x13, x7, 2, 32	// _1195, wcsz,,
	sbfiz	x14, x7, 1, 32	// _1202, wcsz,,
	sub	w4, w23, #1	// _1090, ksz,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	x11, x3	// packed, _917
	and	w30, w23, -4	// niters_vector_mult_vf.1534, ksz,
.L755:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	add	x17, x10, x14	// _3605, ivtmp.2065, _1202
	add	x18, x9, x14	// _3624, ivtmp.2066, _1202
	mov	w8, 0	// ivtmp.2061,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w2, 0	// ic,
.L762:
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w23, 0	// ksz,
	bgt	.L1355		//,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w2, w2, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w8, w8, w23	// ivtmp.2061, ivtmp.2061, ksz
	cmp	w20, w2	// _2, ic
	bne	.L762		//,
.L760:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	w1, w1, 1	// tb, tb,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	x10, x10, x13	// ivtmp.2065, ivtmp.2065, _1195
	add	x9, x9, x13	// ivtmp.2066, ivtmp.2066, _1195
	cmp	w12, w1	// tiles, tb
	bgt	.L755		//,
.L754:
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	ldr	w0, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:471:         const int n_full = (OC / OC_TILE) * OC_TILE;
	lsl	w8, w12, 2	// n_full, tiles,
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	cmp	w0, w8	// _1, n_full
	bgt	.L1356		//,
.L763:
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w1, [sp, 224]	//, %sfp
// src/cpp/cnn_internals.cpp:404:     const int tile_wsz    = in_channels * ksz * OC_TILE;   // packed floats/tile
	lsl	w7, w7, 2	// tile_wsz, wcsz,
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w0, [sp, 184]	//, %sfp
// src/cpp/cnn_internals.cpp:478:             input.batches, IC, OC, IH, IW, KH, KW, OH, OW);
	ldr	w23, [x21, 24]	//, input_29(D)->batches
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	mul	w0, w0, w1	// in_size, _5, _6
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	ldr	w1, [sp, 180]	//, %sfp
// src/cpp/cnn_internals.cpp:400:     const int in_ch_size  = in_channels * in_size;
	mul	w10, w20, w0	// in_ch_size, _2, in_size
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	str	w0, [sp, 368]	// in_size, %sfp
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	ldr	w0, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	mul	w1, w1, w28	// out_size, _7, _8
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	mul	w9, w0, w1	// out_ch_size, _1, out_size
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	cmp	w23, 0	// _892,
	ble	.L884		//,
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, 0	// _7,
	ble	.L884		//,
	ldr	x2, [sp, 248]	// _12, %sfp
	cmp	w8, 4	// n_full,
	smull	x12, w12, w7	// _576, tiles, tile_wsz
	mov	w14, 4	// tmp2718,
	csel	w14, w8, w14, ge	// oc0, n_full, tmp2718,
	cmp	w8, 0	// n_full,
	csel	w14, w14, wzr, gt	// oc0, oc0,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	sub	w0, w28, #4	// _5210, _8,
	ldr	x4, [sp, 264]	// _5928, %sfp
	lsr	w13, w0, 2	// _5209, _5210,
	and	w0, w0, -4	// _5888, _5210,
	add	w0, w0, 4	// col, _5888,
	add	w30, w13, 1	// _573, _5209,
	uxtw	x15, w14	// _803, oc0
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	mov	w22, 0	// b,
	str	w0, [sp, 424]	// col, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ubfiz	x0, x30, 4, 31	// _5198, _573,,
	sbfiz	x26, x10, 2, 32	// _1123, in_ch_size,,
	mov	x18, x27	// ivtmp.2031, in_ptr
	sbfiz	x21, x9, 2, 32	// _1128, out_ch_size,,
	mul	x11, x2, x4	// _524, _12, _5928
	ldr	w2, [sp, 176]	//, %sfp
	smull	x4, w14, w1	// _842, oc0, out_size
	str	x0, [sp, 224]	// _5198, %sfp
	str	x4, [sp, 440]	// _842, %sfp
	add	x4, x3, x12, lsl 2	// ivtmp.1985, _917, _576,
	sub	w2, w2, #1	// _5880, _4,
	add	x2, x2, 1	// _599, _5880,
	str	x4, [sp, 456]	// ivtmp.1985, %sfp
	lsl	x0, x2, 4	// _5789, _599,
	ldr	w4, [sp, 108]	//, %sfp
	str	x0, [sp, 504]	// _5789, %sfp
	sbfiz	x0, x1, 4, 32	// _1035, out_size,,
	str	x0, [sp, 496]	// _1035, %sfp
	sbfiz	x0, x1, 2, 32	// _934, out_size,,
	sub	w12, w4, w14	// _922, _1, oc0
	ldr	x4, [sp, 200]	// _87, %sfp
	add	x12, x12, w14, sxtw	// _924, _922, oc0
	add	x5, x4, x15, lsl 2	// ivtmp.1981, _87, _803,
	ldr	x15, [sp, 216]	// ivtmp.2032, %sfp
	add	x4, x4, x12, lsl 2	// _931, _87, _924,
	str	x0, [sp, 216]	// _934, %sfp
	sxtw	x0, w28	// _2604, _8
	str	x0, [sp, 184]	// _2604, %sfp
	str	x4, [sp, 464]	// _931, %sfp
	lsl	x4, x11, 2	// _886, _524,
	str	x5, [sp, 432]	// ivtmp.1981, %sfp
	str	x4, [sp, 448]	// _886, %sfp
	sbfiz	x4, x7, 2, 32	// _1050, tile_wsz,,
	str	x4, [sp, 480]	// _1050, %sfp
	lsl	x4, x24, 4	// _967, _4,
	str	x4, [sp, 472]	// _967, %sfp
.L808:
	str	w23, [sp, 408]	// _892, %sfp
	mov	w0, w19	// _3, _3
// src/cpp/cnn_internals.cpp:410:         float*       __restrict out_b = out_ptr   + (std::size_t)b * out_ch_size;
	mov	x17, 0	// ivtmp.2026,
	ldr	w23, [sp, 424]	//, %sfp
	mov	w19, w20	// _2, _2
	mov	x1, 0	// ivtmp.2025,
	str	x26, [sp, 416]	// _1123, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	mov	w7, 0	// oh,
	mov	w20, w0	// _3, _3
	ldr	x26, [sp, 208]	// _6, %sfp
	stp	x3, x2, [sp, 272]	// _917, _599,
	mov	x3, x15	// ivtmp.2032, ivtmp.2032
	str	x25, [sp, 400]	// _860, %sfp
	mov	x25, x21	// _1128, _1128
	mov	x21, x17	// ivtmp.2026, ivtmp.2026
	str	w30, [sp, 112]	// _573, %sfp
	str	x18, [sp, 248]	// ivtmp.2031, %sfp
	str	w14, [sp, 264]	// oc0, %sfp
	str	w22, [sp, 392]	// b, %sfp
.L773:
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w0, w28, #1	// _5766, _8,
	add	x27, sp, 672	// tmp3576,,
	str	w0, [sp, 384]	// _5766, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	cmp	w8, 0	// n_full,
	ble	.L1357		//,
	mov	x22, x1	// ivtmp.2025, ivtmp.2025
	mov	x1, x27	// tmp3576, tmp3576
	mov	x12, x21	// ivtmp.2026, ivtmp.2026
	mov	x27, x26	// _6, _6
	mov	x24, x3	// ivtmp.2032, ivtmp.2032
	mov	w26, w20	// _3, _3
	mov	x0, x25	// _1128, _1128
	mov	w20, w19	// _2, _2
	mov	w21, w7	// oh, oh
	mov	w19, w23	// col, col
	mov	x23, x1	// tmp3576, tmp3576
.L790:
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	ldr	x25, [sp, 200]	// ivtmp.2018, %sfp
	add	x6, x24, x22, lsl 2	// ivtmp.2020, ivtmp.2032, ivtmp.2025,
	mov	w11, w21	// oh, oh
	mov	x13, x22	// ivtmp.2025, ivtmp.2025
	mov	x7, x27	// _6, _6
// src/cpp/cnn_internals.cpp:413:             int oc0 = 0, tile = 0;
	mov	w9, 0	// oc0,
	ldr	x4, [sp, 272]	// ivtmp.2021, %sfp
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	mov	x10, x24	// ivtmp.2032, ivtmp.2032
	mov	w22, w19	// col, col
	mov	w27, w20	// _2, _2
	mov	x21, x23	// tmp3576, tmp3576
.L788:
// src/cpp/cnn_internals.cpp:347:         const float32x4_t bv = vdupq_n_f32(bias_ptr[oc0 + t]);
	ldr	s1, [x25, 12]	// _4602, MEM[(const float *)_1052 + 12B]
// src/cpp/cnn_internals.cpp:415:                 conv_row_tile<OC_TILE, NV>(
	mov	x15, x4	// wp, ivtmp.2021
// src/cpp/cnn_internals.cpp:347:         const float32x4_t bv = vdupq_n_f32(bias_ptr[oc0 + t]);
	ldp	s30, s31, [x25, 4]	// _4576, _4589,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v29.4s}, [x25]	//, MEM[(const float *)_1052]
	dup	v1.4s, v1.s[0]	//, _4602
	dup	v30.4s, v30.s[0]	//, _4576
	dup	v31.4s, v31.s[0]	//, _4589
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q1, q1, [sp, 832]	//,,
	stp	q30, q30, [sp, 736]	//,,
	stp	q29, q29, [sp, 672]	//,,
	stp	q29, q30, [sp, 704]	//,,
	stp	q31, q31, [sp, 768]	//,,
	stp	q31, q1, [sp, 800]	//,,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w27, 0	// _2,
	ble	.L776		//,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	ldrsw	x16, [sp, 368]	// _5619, %sfp
	cmp	w26, 0	// _3,
	ble	.L776		//,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w14, 0	// ic,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	mov	x2, x12	// ivtmp.2011, ivtmp.2026
	.p2align 5,,15
.L782:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w5, 0	// kh,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x17, x2	// ivtmp.2007, ivtmp.2011
.L781:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w1, [sp, 176]	//, %sfp
	cmp	w1, 0	// _4,
	bgt	.L779		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w5, w5, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x17, x17, x7	// ivtmp.2007, ivtmp.2007, _6
	cmp	w26, w5	// _3, kh
	bne	.L781		//,
	.p2align 5,,15
.L780:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w14, w14, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x2, x2, x16	// ivtmp.2011, ivtmp.2011, _5619
	cmp	w27, w14	// _2, ic
	bne	.L782		//,
.L776:
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	mov	w19, 0	// t,
	str	w9, [sp, 256]	// oc0, %sfp
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	mov	x24, 4	// tmp3641,
	mov	w20, w19	// t, t
	mov	w19, w28	// _8, _8
	str	x25, [sp, 288]	// ivtmp.2018, %sfp
	str	w8, [sp, 304]	// n_full, %sfp
	mov	x25, x21	// ivtmp.1993, ivtmp.1993
	mov	x23, x6	// ivtmp.1992, ivtmp.1992
	stp	x6, x4, [sp, 312]	// ivtmp.2020, ivtmp.2021,
	mov	x28, x7	// _6, _6
	stp	x12, x10, [sp, 328]	// ivtmp.2026, ivtmp.2032,
	str	x0, [sp, 344]	// _1128, %sfp
	str	w11, [sp, 352]	// oh, %sfp
	str	x13, [sp, 360]	// ivtmp.2025, %sfp
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w19, 3	// _8,
	ble	.L783		//,
.L786:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	x2, [sp, 224]	//, %sfp
	mov	x1, x25	//, ivtmp.1993
	mov	x0, x23	//, ivtmp.1992
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w19, w22	// _8, col
	bgt	.L784		//,
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	ldr	x0, [sp, 216]	// _934, %sfp
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	w20, w20, 1	// t, t,
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	x25, x25, 48	// ivtmp.1993, ivtmp.1993,
	add	x23, x23, x0	// ivtmp.1992, ivtmp.1992, _934
	cmp	w20, 4	// t,
	bne	.L786		//,
.L1285:
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	x1, [sp, 496]	// _1035, %sfp
	mov	x7, x28	// _6, _6
	mov	w28, w19	// _8, _8
	ldp	x6, x4, [sp, 312]	// ivtmp.2020, ivtmp.2021,
	ldp	x12, x10, [sp, 328]	// ivtmp.2026, ivtmp.2032,
	add	x6, x6, x1	// ivtmp.2020, ivtmp.2020, _1035
	ldr	w9, [sp, 256]	//, %sfp
	ldr	x25, [sp, 288]	// ivtmp.2018, %sfp
	ldr	x1, [sp, 480]	// _1050, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	w9, w9, 4	// oc0, oc0,
	ldr	w8, [sp, 304]	//, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	x25, x25, 16	// ivtmp.2018, ivtmp.2018,
	ldr	x0, [sp, 344]	// _1128, %sfp
	add	x4, x4, x1	// ivtmp.2021, ivtmp.2021, _1050
	ldr	x13, [sp, 360]	// ivtmp.2025, %sfp
	ldr	w11, [sp, 352]	//, %sfp
	cmp	w8, w9	// n_full, oc0
	bgt	.L788		//,
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	mov	w20, w27	// _2, _2
	mov	w19, w22	// col, col
	ldr	w1, [sp, 108]	//, %sfp
	mov	x24, x10	// ivtmp.2032, ivtmp.2032
	mov	x27, x7	// _6, _6
	mov	x23, x21	// tmp3576, tmp3576
	ldr	w2, [sp, 264]	//, %sfp
	cmp	w1, w2	// _1, oc0
	ble	.L1358		//,
	mov	x25, x0	// _1128, _1128
	mov	x0, x23	// tmp3576, tmp3576
	mov	x3, x10	// ivtmp.2032, ivtmp.2032
	mov	w23, w19	// col, col
	mov	w7, w11	// oh, oh
	mov	w19, w20	// _2, _2
	mov	x21, x12	// ivtmp.2026, ivtmp.2026
	mov	w20, w26	// _3, _3
	mov	x1, x13	// ivtmp.2025, ivtmp.2025
	mov	x26, x27	// _6, _6
	mov	x27, x0	// tmp3576, tmp3576
.L789:
	ldp	x24, x0, [sp, 432]	// ivtmp.1981, _842,
	stp	x3, x25, [sp, 320]	// ivtmp.2032, _1128,
	add	x22, x0, x1	// _843, _842, ivtmp.2025
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	ldr	x0, [sp, 456]	// ivtmp.1985, %sfp
	add	x22, x3, x22, lsl 2	// ivtmp.1984, ivtmp.2032, _843,
	str	w8, [sp, 312]	// n_full, %sfp
	mov	x25, x22	// ivtmp.1984, ivtmp.1984
	mov	w22, w28	// _8, _8
	str	w7, [sp, 336]	// oh, %sfp
	mov	x28, x1	// ivtmp.2025, ivtmp.2025
	str	x0, [sp, 256]	// ivtmp.1985, %sfp
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	ldrsw	x0, [sp, 368]	// _5704, %sfp
	str	x0, [sp, 304]	// _5704, %sfp
	ldr	x0, [sp, 280]	// _599, %sfp
	lsl	x0, x0, 2	// _5756, _599,
	str	x0, [sp, 288]	// _5756, %sfp
.L805:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v6.4s}, [x24]	//, MEM[(const float *)_910]
// src/cpp/cnn_internals.cpp:421:                 conv_row_tile<1, NV>(
	ldr	x8, [sp, 256]	// wp, %sfp
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q6, q6, [x27]	//,,
	str	q6, [x27, 32]	//, MEM[(__Float32x4_t[1][3] *)_2907][0][2]
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w19, 0	// _2,
	ble	.L795		//,
	cmp	w20, 0	// _3,
	ble	.L795		//,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	ldr	x1, [sp, 304]	// _5704, %sfp
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w7, 0	// ic,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	mov	x0, x21	// ivtmp.1975, ivtmp.2026
	.p2align 5,,15
.L801:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w2, 0	// kh,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x4, x0	// ivtmp.1971, ivtmp.1975
.L800:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w3, [sp, 176]	//, %sfp
	cmp	w3, 0	// _4,
	bgt	.L798		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w2, w2, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x26	// ivtmp.1971, ivtmp.1971, _6
	cmp	w20, w2	// _3, kh
	bne	.L800		//,
	.p2align 5,,15
.L799:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w7, w7, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x0, x0, x1	// ivtmp.1975, ivtmp.1975, _5704
	cmp	w19, w7	// _2, ic
	bne	.L801		//,
.L795:
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w22, 3	// _8,
	ble	.L1359		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	x2, [sp, 224]	//, %sfp
	mov	x1, x27	//, tmp3576
	mov	x0, x25	//, ivtmp.1984
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	ldr	w1, [sp, 112]	//, %sfp
	mov	w4, w23	// col, col
	cmp	w22, w23	// _8, col
	ble	.L804		//,
.L802:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ubfiz	x1, x1, 4, 31	// tmp2829, v,,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w22, #1	// _204, _8,
	sub	w2, w2, w4	// _5164, _204, col
	uxtw	x0, w4	// _5157, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [x27, x1]	// tmp6104, MEM[(__Float32x4_t[1][3] *)_2907][0][v_5980]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	cmp	w22, w4	// _8, col
	add	x2, x2, 1	// _5162, _5164,
	add	x0, x25, x0, lsl 2	//, ivtmp.1984, _5157,
	mov	x4, 4	// tmp2841,
	add	x1, sp, 624	//,,
	lsl	x2, x2, 2	// _5161, _5162,
	csel	x2, x2, x4, gt	//, _5161, tmp2841,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// tmp6104, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	bl	memcpy		//
.L804:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	x0, [sp, 216]	// _934, %sfp
	add	x24, x24, 4	// ivtmp.1981, ivtmp.1981,
	ldr	x1, [sp, 448]	// _886, %sfp
	add	x25, x25, x0	// ivtmp.1984, ivtmp.1984, _934
	ldr	x0, [sp, 256]	// ivtmp.1985, %sfp
	add	x0, x0, x1	// ivtmp.1985, ivtmp.1985, _886
	str	x0, [sp, 256]	// ivtmp.1985, %sfp
	ldr	x0, [sp, 464]	// _931, %sfp
	cmp	x24, x0	// ivtmp.1981, _931
	bne	.L805		//,
	ldr	w8, [sp, 312]	//, %sfp
	mov	x1, x28	// ivtmp.2025, ivtmp.2025
	mov	w28, w22	// _8, _8
	ldp	x3, x25, [sp, 320]	// ivtmp.2032, _1128,
	ldr	w7, [sp, 336]	//, %sfp
.L806:
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w7, w7, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	x21, x21, x26	// ivtmp.2026, ivtmp.2026, _6
	ldr	x0, [sp, 184]	// _2604, %sfp
	add	x1, x1, x0	// ivtmp.2025, ivtmp.2025, _2604
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, w7	// _7, oh
	bne	.L773		//,
	ldr	x18, [sp, 248]	// ivtmp.2031, %sfp
	mov	x15, x3	// ivtmp.2032, ivtmp.2032
	mov	x21, x25	// _1128, _1128
	ldp	x3, x2, [sp, 272]	// _917, _599,
	mov	w0, w20	// _3, _3
	mov	w20, w19	// _2, _2
	mov	w19, w0	// _3, _3
	ldr	x25, [sp, 400]	// _860, %sfp
	ldr	x26, [sp, 416]	// _1123, %sfp
	ldr	w30, [sp, 112]	//, %sfp
	ldr	w14, [sp, 264]	//, %sfp
	ldr	w22, [sp, 392]	//, %sfp
	ldr	w23, [sp, 408]	//, %sfp
.L791:
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	w22, w22, 1	// b, b,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	x18, x18, x26	// ivtmp.2031, ivtmp.2031, _1123
	add	x15, x15, x21	// ivtmp.2032, ivtmp.2032, _1128
	cmp	w23, w22	// _892, b
	bne	.L808		//,
.L884:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x3, .L623	// _1581,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x1, x25	//, _1524
	mov	x0, x3	//, _1581
	bl	_ZdlPvm		//
// src/cpp/cnn_internals.cpp:479:     };
	b	.L623		//
	.p2align 2,,3
.L1360:
	ldr	x1, [sp, 504]	// _5789, %sfp
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w5, w5, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x17, x17, x7	// ivtmp.2007, ivtmp.2007, _6
	add	x15, x15, x1	// wp, wp, _5789
	cmp	w26, w5	// _3, kh
	beq	.L780		//,
.L779:
	ldr	x1, [sp, 248]	// ivtmp.2031, %sfp
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldp	q16, q28, [sp, 672]	//,,
	ldp	q24, q3, [sp, 704]	//,,
	add	x3, x1, x17, lsl 2	// ivtmp.2001, ivtmp.2031, ivtmp.2007,
	ldp	q27, q23, [sp, 736]	//,,
	ldp	q2, q26, [sp, 768]	//,,
	ldp	q22, q0, [sp, 800]	//,,
	ldp	q25, q21, [sp, 832]	//,,
	ldr	x1, [sp, 472]	// _967, %sfp
	add	x18, x15, x1	// _968, wp, _967
	mov	x1, x15	// wp, wp
	.p2align 5,,15
.L778:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldp	q5, q4, [x3]	//,,* ivtmp.2001
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x3, x3, 4	// ivtmp.2001, ivtmp.2001,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldr	q7, [x3, 28]	//, MEM <__Float32x4_t> [(float * {ref-all})_963 + 32B]
// src/cpp/cnn_internals.cpp:360:                 for (int t = 0; t < OC_T; ++t) wv[t] = wp[t];
	ldp	s20, s19, [x1]	//,,* wp
	ldp	s18, s17, [x1, 8]	//,,
// src/cpp/cnn_internals.cpp:361:                 wp += OC_T;
	add	x1, x1, 16	// wp, wp,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v16.4s, v5.4s, v20.s[0]	//,,
	fmla	v3.4s, v5.4s, v19.s[0]	//,,
	fmla	v2.4s, v5.4s, v18.s[0]	//,,
	fmla	v0.4s, v5.4s, v17.s[0]	//,,
	fmla	v28.4s, v4.4s, v20.s[0]	//,,
	fmla	v27.4s, v4.4s, v19.s[0]	//,,
	fmla	v26.4s, v4.4s, v18.s[0]	//,,
	fmla	v25.4s, v4.4s, v17.s[0]	//,,
	fmla	v24.4s, v7.4s, v20.s[0]	//,,
	fmla	v23.4s, v7.4s, v19.s[0]	//,,
	fmla	v22.4s, v7.4s, v18.s[0]	//,,
	fmla	v21.4s, v7.4s, v17.s[0]	//,,
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q16, q28, [sp, 672]	//,,
	stp	q24, q3, [sp, 704]	//,,
	stp	q27, q23, [sp, 736]	//,,
	stp	q2, q26, [sp, 768]	//,,
	stp	q22, q0, [sp, 800]	//,,
	stp	q25, q21, [sp, 832]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x18, x1	// _968, wp
	bne	.L778		//,
	b	.L1360		//
	.p2align 2,,3
.L1361:
	ldr	x3, [sp, 288]	// _5756, %sfp
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w2, w2, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x26	// ivtmp.1971, ivtmp.1971, _6
	add	x8, x8, x3	// wp, wp, _5756
	cmp	w20, w2	// _3, kh
	beq	.L799		//,
.L798:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldp	q1, q31, [x27]	//,,
	ldr	x3, [sp, 248]	// ivtmp.2031, %sfp
	ldr	x5, [sp, 376]	// _5781, %sfp
	ldr	q30, [x27, 32]	//, MEM[(__Float32x4_t[1][3] *)_2907][0][2]
	add	x3, x3, x4, lsl 2	// ivtmp.1965, ivtmp.2031, ivtmp.1971,
	add	x6, x8, x5	// _753, wp, _5781
	mov	x5, x8	// wp, wp
	.p2align 5,,15
.L797:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldp	q2, q0, [x3]	//,,* ivtmp.1965
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x3, x3, 4	// ivtmp.1965, ivtmp.1965,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v29.4s}, [x5], 4	//, MEM[(const float *)wp_682]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldr	q28, [x3, 28]	//, MEM <__Float32x4_t> [(float * {ref-all})_719 + 32B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v1.4s, v2.4s, v29.4s	//,,
	fmla	v31.4s, v0.4s, v29.4s	//,,
	fmla	v30.4s, v28.4s, v29.4s	//,,
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q1, q31, [x27]	//,,
	str	q30, [x27, 32]	//, MEM[(__Float32x4_t[1][3] *)_2907][0][2]
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x6, x5	// _753, wp
	bne	.L797		//,
	b	.L1361		//
.L784:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldrsw	x1, [sp, 112]	// v, %sfp
	sbfiz	x0, x20, 1, 32	// tmp2769, t,,
	add	x0, x0, w20, sxtw	// tmp2770, tmp2769, t
	mov	w7, w22	// col, col
	add	x0, x0, x1	// tmp2771, tmp2770, v
	lsl	x0, x0, 4	// tmp2772, tmp2771,
	ldr	q31, [x21, x0]	// _1064, MEM[(__Float32x4_t[4][3] *)_2907][t_2240][v_2996]
.L787:
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	cmp	w19, w7	// _8, col
	add	x1, sp, 624	// tmp6084,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// _1064, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	ldr	w0, [sp, 384]	//, %sfp
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	w20, w20, 1	// t, t,
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	x25, x25, 48	// ivtmp.1993, ivtmp.1993,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w0, w7	// _5233, _5766, col
	uxtw	x0, w7	// _5222, col
	add	x0, x23, x0, lsl 2	//, ivtmp.1992, _5222,
	add	x2, x2, 1	// _5232, _5233,
	lsl	x2, x2, 2	// _5229, _5232,
	csel	x2, x2, x24, gt	//, _5229, tmp3641,
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	ldr	x0, [sp, 216]	// _934, %sfp
	add	x23, x23, x0	// ivtmp.1992, ivtmp.1992, _934
	cmp	w20, 4	// t,
	beq	.L1285		//,
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w19, 3	// _8,
	bgt	.L786		//,
.L783:
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w19, 0	// _8,
	ble	.L1285		//,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [x25]	// _1064, MEM[(__Float32x4_t *)_957]
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	mov	w7, 0	// col,
	b	.L787		//
.L1359:
	mov	w1, 0	// v,
	mov	w4, 0	// col,
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w22, 0	// _8,
	bgt	.L802		//,
	b	.L804		//
.L1358:
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	ldr	x1, [sp, 184]	// _2604, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w21, w11, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	x12, x12, x7	// ivtmp.2026, ivtmp.2026, _6
	add	x22, x13, x1	// ivtmp.2025, ivtmp.2025, _2604
	ldr	w1, [sp, 180]	//, %sfp
	cmp	w1, w21	// _7, oh
	bne	.L790		//,
	ldr	x18, [sp, 248]	// ivtmp.2031, %sfp
	mov	w19, w26	// _3, _3
	mov	x14, x2	//,
	mov	x15, x10	// ivtmp.2032, ivtmp.2032
	ldp	x3, x2, [sp, 272]	// _917, _599,
	mov	x21, x0	// _1128, _1128
	ldr	x25, [sp, 400]	// _860, %sfp
	ldr	x26, [sp, 416]	// _1123, %sfp
	ldr	w30, [sp, 112]	//, %sfp
	ldr	w22, [sp, 392]	//, %sfp
	ldr	w23, [sp, 408]	//, %sfp
	b	.L791		//
.L1355:
	str	w19, [sp, 272]	// _3, %sfp
.L759:
	cmp	w4, 2	// _1090,
	bls	.L1057		//,
	sxtw	x0, w8	// _5304, ivtmp.2061
	mov	x26, x11	// ivtmp.2049, packed
	add	x16, x0, x10	// _1095, _5304, ivtmp.2065
	add	x19, x0, x17	// _525, _5304, _3605
	add	x16, x22, x16, lsl 2	// vectp.1538, _862, _1095,
	add	x15, x0, x18	// _836, _5304, _3624
	str	x0, [sp, 280]	// _5304, %sfp
	add	x19, x22, x19, lsl 2	// vectp.1544, _862, _525,
	str	x16, [sp, 112]	// vectp.1538, %sfp
	add	x16, x0, x9	// _5322, _5304, ivtmp.2066
	add	x16, x22, x16, lsl 2	// vectp.1541, _862, _5322,
	str	x16, [sp, 256]	// vectp.1541, %sfp
	add	x16, x22, x15, lsl 2	// vectp.1547, _862, _836,
	mov	x15, 0	// ivtmp.2046,
.L757:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	ldr	x0, [sp, 112]	// vectp.1538, %sfp
	ldr	q27, [x19, x15]	//, MEM <const vector(4) float> [(const float *)vectp.1544_974 + ivtmp.2046_1174 * 1]
	ldr	q28, [x16, x15]	//, MEM <const vector(4) float> [(const float *)vectp.1547_517 + ivtmp.2046_1174 * 1]
	ldr	q25, [x0, x15]	//, MEM <const vector(4) float> [(const float *)vectp.1538_2923 + ivtmp.2046_1174 * 1]
	ldr	x0, [sp, 256]	// vectp.1541, %sfp
	ldr	q26, [x0, x15]	//, MEM <const vector(4) float> [(const float *)vectp.1541_3554 + ivtmp.2046_1174 * 1]
	add	x15, x15, 16	// ivtmp.2046, ivtmp.2046,
	st4	{v25.4s - v28.4s}, [x26]	//, MEM <float[16]> [(float *)vectp_packed.1549_834]
	add	x26, x26, 64	// ivtmp.2049, ivtmp.2049,
	cmp	x15, x6	// ivtmp.2046, _1182
	bne	.L757		//,
	ldr	x0, [sp, 280]	// _5304, %sfp
	mov	w16, w30	// niters_vector_mult_vf.1534, niters_vector_mult_vf.1534
	cmp	w30, w23	// niters_vector_mult_vf.1534, ksz
	beq	.L758		//,
.L756:
	uxtw	x15, w16	// _3366, niters_vector_mult_vf.1534
	add	x19, x10, x0	// _3365, ivtmp.2065, _5304
	add	x19, x19, x15	// _3367, _3365, _3366
	add	x26, x9, x0	// _853, ivtmp.2066, _5304
	sub	w16, w23, w16	// bnd.1553_3362, ksz, niters_vector_mult_vf.1534
	whilelo	p7.s, wzr, w16	// max_mask_4403,, bnd.1553_3362
	ld1w	z17.s, p7/z, [x22, x19, lsl 2]	//, max_mask_4403,* _862
	add	x19, x17, x0	// _2391, _3605, _5304
	add	x0, x18, x0	// _432, _3624, _5304
	add	x16, x11, x15, lsl 4	// vectp_packed.1567, packed, _3366,
	add	x26, x26, x15	// _497, _853, _3366
	ld1w	z18.s, p7/z, [x22, x26, lsl 2]	//, max_mask_4403,* _862
	add	x19, x19, x15	// _1231, _2391, _3366
	add	x0, x0, x15	// _2218, _432, _3366
	ld1w	z19.s, p7/z, [x22, x19, lsl 2]	//, max_mask_4403,* _862
	ld1w	z20.s, p7/z, [x22, x0, lsl 2]	//, max_mask_4403,* _862
	st4w	{z17.s - z20.s}, p7, [x16]	//, max_mask_4403,* vectp_packed.1567
.L758:
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w2, w2, 1	// ic, ic,
	add	x11, x11, x5	// packed, packed, _5779
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w8, w8, w23	// ivtmp.2061, ivtmp.2061, ksz
	cmp	w20, w2	// _2, ic
	bne	.L759		//,
	ldr	w19, [sp, 272]	//, %sfp
	b	.L760		//
.L1057:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	w16, 0	// niters_vector_mult_vf.1534,
	sxtw	x0, w8	// _5304, ivtmp.2061
	b	.L756		//
.L1045:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	mov	x3, 0	// _257,
.L628:
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	ldr	w0, [sp, 176]	//, %sfp
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	ldr	w1, [sp, 108]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x22, [x22]	// _202, MEM[(const struct vector *)weight_24(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	mul	w23, w19, w0	// ksz, _3, _4
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	cmp	w1, 0	// _1,
	add	w0, w1, 3	// tmp2198, _1,
	csel	w0, w0, w1, lt	// _1, tmp2198, _1,
// src/cpp/cnn_internals.cpp:318:     const int wcsz  = in_channels * ksz;              // per-oc stride in OIHW
	mul	w7, w20, w23	// wcsz, _2, ksz
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	asr	w12, w0, 2	// tiles, _1,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w1, 3	// _1,
	ble	.L629		//,
	cmp	w20, 0	// _2,
	ble	.L629		//,
	sub	w0, w23, #1	// _5983, ksz,
	lsr	w2, w23, 2	// _124, ksz,
	add	x0, x0, 1	// _608, _5983,
	mov	x10, 0	// ivtmp.1824,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w1, 0	// tb,
	lsl	x6, x2, 4	// _125, _124,
	lsl	x5, x0, 4	// _3712, _608,
	sxtw	x9, w7	// ivtmp.1825, wcsz
	sbfiz	x13, x7, 2, 32	// _135, wcsz,,
	sbfiz	x14, x7, 1, 32	// _143, wcsz,,
	sub	w4, w23, #1	// _2129, ksz,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	x11, x3	// packed, _257
	and	w30, w23, -4	// niters_vector_mult_vf.1460, ksz,
.L630:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	add	x17, x10, x14	// _3187, ivtmp.1824, _143
	add	x18, x9, x14	// _3206, ivtmp.1825, _143
	mov	w8, 0	// ivtmp.1820,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w2, 0	// ic,
.L637:
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w23, 0	// ksz,
	bgt	.L1362		//,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w2, w2, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w8, w8, w23	// ivtmp.1820, ivtmp.1820, ksz
	cmp	w20, w2	// _2, ic
	bne	.L637		//,
.L635:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	w1, w1, 1	// tb, tb,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	x10, x10, x13	// ivtmp.1824, ivtmp.1824, _135
	add	x9, x9, x13	// ivtmp.1825, ivtmp.1825, _135
	cmp	w12, w1	// tiles, tb
	bgt	.L630		//,
.L629:
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	ldr	w0, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:471:         const int n_full = (OC / OC_TILE) * OC_TILE;
	lsl	w9, w12, 2	// n_full, tiles,
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	cmp	w0, w9	// _1, n_full
	bgt	.L1363		//,
.L638:
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w0, [sp, 184]	//, %sfp
// src/cpp/cnn_internals.cpp:404:     const int tile_wsz    = in_channels * ksz * OC_TILE;   // packed floats/tile
	lsl	w10, w7, 2	// tile_wsz, wcsz,
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w1, [sp, 224]	//, %sfp
// src/cpp/cnn_internals.cpp:478:             input.batches, IC, OC, IH, IW, KH, KW, OH, OW);
	ldr	w22, [x21, 24]	//, input_29(D)->batches
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	mul	w30, w0, w1	// in_size, _5, _6
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	ldr	w0, [sp, 180]	//, %sfp
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	ldr	w1, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:400:     const int in_ch_size  = in_channels * in_size;
	mul	w13, w20, w30	// in_ch_size, _2, in_size
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	mul	w0, w0, w28	// out_size, _7, _8
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	mul	w14, w1, w0	// out_ch_size, _1, out_size
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	cmp	w22, 0	// _232,
	ble	.L884		//,
	ldr	w1, [sp, 180]	//, %sfp
	cmp	w1, 0	// _7,
	ble	.L884		//,
	ldr	x4, [sp, 248]	// _12, %sfp
	cmp	w9, 4	// n_full,
	smull	x2, w12, w10	// _661, tiles, tile_wsz
	mov	w8, 4	// tmp2265,
	csel	w8, w9, w8, ge	// oc0, n_full, tmp2265,
	cmp	w9, 0	// n_full,
	csel	w8, w8, wzr, gt	// oc0, oc0,,
	sub	w1, w28, #4	//, _8,
	ldr	x5, [sp, 272]	// _5960, %sfp
	and	w7, w1, -4	// _6036, _676,
	sub	w17, w28, #1	// _538, _8,
	add	x2, x3, x2, lsl 2	// ivtmp.1735, _257, _661,
	cmp	w28, 3	// _8,
	add	w7, w7, 4	// col, _6036,
	csel	w7, w7, wzr, gt	// col, col,,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	str	x1, [sp, 472]	// _3560, %sfp
	sub	w11, w17, w7	// _5392, _538, col
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	mov	w21, 0	// b,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	add	x16, x11, 1	// _5391, _5392,
	sbfiz	x23, x14, 2, 32	// _63, out_ch_size,,
	str	x2, [sp, 408]	// ivtmp.1735, %sfp
	uxtw	x2, w8	// _5120, oc0
	sbfiz	x26, x13, 2, 32	// _74, in_ch_size,,
	mul	x12, x4, x5	// _520, _12, _5960
	sbfiz	x18, x0, 2, 32	// _93, out_size,,
	smull	x4, w8, w0	// _2034, oc0, out_size
	str	x4, [sp, 384]	// _2034, %sfp
	ldr	w4, [sp, 176]	//, %sfp
	sub	w15, w4, #1	// _6017, _4,
	ldr	x4, [sp, 200]	// _87, %sfp
	add	x2, x4, x2, lsl 2	// ivtmp.1731, _87, _5120,
	str	x2, [sp, 400]	// ivtmp.1731, %sfp
	ldr	w2, [sp, 108]	//, %sfp
	sub	w11, w2, w8	// _5405, _1, oc0
	add	x11, x11, w8, sxtw	// _5401, _5405, oc0
	add	x4, x4, x11, lsl 2	// _5369, _87, _5401,
	add	x2, x15, 1	// _687, _6017,
	mov	x15, x27	// ivtmp.1788, in_ptr
	ldr	x11, [sp, 216]	// ivtmp.1787, %sfp
	str	x4, [sp, 432]	// _5369, %sfp
	lsl	x4, x12, 2	// _5267, _520,
	str	x4, [sp, 416]	// _5267, %sfp
	sbfiz	x4, x10, 2, 32	// _5684, tile_wsz,,
	str	x4, [sp, 448]	// _5684, %sfp
	lsl	x4, x24, 4	// _5297, _4,
	str	x4, [sp, 424]	// _5297, %sfp
	lsl	x4, x16, 2	// _5388, _5391,
	str	x4, [sp, 224]	// _5388, %sfp
	sbfiz	x4, x7, 2, 32	// _5377, col,,
	str	x4, [sp, 440]	// _5377, %sfp
	lsl	x4, x2, 4	// _5873, _687,
	str	x4, [sp, 464]	// _5873, %sfp
	lsr	w4, w1, 2	// _1812, _676,
	sbfiz	x1, x0, 4, 32	// _2878, out_size,,
	str	x1, [sp, 392]	// _2878, %sfp
	ldr	x1, [sp, 208]	// _6, %sfp
	str	w4, [sp, 376]	// _1812, %sfp
	lsl	x1, x1, 2	// _1702, _6,
	str	x1, [sp, 368]	// _1702, %sfp
	sxtw	x1, w28	// _2604, _8
	str	x1, [sp, 184]	// _2604, %sfp
	add	x1, x11, w0, sxtw 3	// ivtmp.1790, ivtmp.1787, out_size,
.L684:
	add	x0, x1, x18	// _25, ivtmp.1790, _93
// src/cpp/cnn_internals.cpp:410:         float*       __restrict out_b = out_ptr   + (std::size_t)b * out_ch_size;
	mov	x10, 0	// ivtmp.1776,
	str	w9, [sp, 264]	// n_full, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	mov	w14, 0	// oh,
	mov	w27, w17	// _538, _538
	stp	x1, x18, [sp, 312]	// ivtmp.1790, _93,
	mov	x9, x10	// ivtmp.1776, ivtmp.1776
	mov	x1, x11	// ivtmp.1787, ivtmp.1787
	str	w8, [sp, 336]	// oc0, %sfp
	mov	w8, w14	// oh, oh
	stp	x2, x0, [sp, 344]	// _687, _25,
	add	x0, x11, x18	// _55, ivtmp.1787, _93
	str	x26, [sp, 480]	// _74, %sfp
	mov	w26, w28	// _8, _8
	mov	w28, w7	// col, col
// src/cpp/cnn_internals.cpp:410:         float*       __restrict out_b = out_ptr   + (std::size_t)b * out_ch_size;
	str	xzr, [sp, 216]	//, %sfp
	str	x15, [sp, 256]	// ivtmp.1788, %sfp
	str	w30, [sp, 272]	// in_size, %sfp
	str	x3, [sp, 328]	// _257, %sfp
	str	x0, [sp, 360]	// _55, %sfp
	stp	x23, x25, [sp, 496]	// _63, _200,
	str	w22, [sp, 512]	// _232, %sfp
	str	w21, [sp, 516]	// b, %sfp
.L648:
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	w0, [sp, 264]	//, %sfp
	cmp	w0, 0	// n_full,
	ble	.L683		//,
	ldr	x0, [sp, 352]	// _25, %sfp
	add	x22, x1, x9, lsl 2	// ivtmp.1766, ivtmp.1787, ivtmp.1776,
// src/cpp/cnn_internals.cpp:413:             int oc0 = 0, tile = 0;
	str	wzr, [sp, 304]	//, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	x21, [sp, 200]	// ivtmp.1761, %sfp
	str	x1, [sp, 520]	// ivtmp.1787, %sfp
	str	w8, [sp, 528]	// oh, %sfp
	add	x25, x0, x9, lsl 2	// ivtmp.1763, _25, ivtmp.1776,
	ldr	x0, [sp, 328]	// _257, %sfp
	str	w28, [sp, 532]	// col, %sfp
	str	x9, [sp, 536]	// ivtmp.1776, %sfp
	str	x0, [sp, 248]	// _257, %sfp
	ldr	x0, [sp, 312]	// ivtmp.1790, %sfp
	add	x24, x0, x9, lsl 2	// ivtmp.1764, ivtmp.1790, ivtmp.1776,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	ldr	x0, [sp, 472]	// _3560, %sfp
	lsl	x0, x0, 2	// _3034, _3560,
	str	x0, [sp, 456]	// _3034, %sfp
	ldr	x0, [sp, 360]	// _55, %sfp
	add	x23, x0, x9, lsl 2	// ivtmp.1765, _55, ivtmp.1776,
.L667:
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	ld1r	{v31.4s}, [x21]	// tmp5881, MEM[(const float *)_5689]
// src/cpp/cnn_internals.cpp:347:         const float32x4_t bv = vdupq_n_f32(bias_ptr[oc0 + t]);
	ldr	s30, [x21, 12]	// _4310, MEM[(const float *)_5689 + 12B]
// src/cpp/cnn_internals.cpp:415:                 conv_row_tile<OC_TILE, NV>(
	ldr	x7, [sp, 248]	// wp, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	dup	v30.4s, v30.s[0]	// _3927, _4310
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	str	q31, [sp, 112]	// tmp5881, %sfp
// src/cpp/cnn_internals.cpp:347:         const float32x4_t bv = vdupq_n_f32(bias_ptr[oc0 + t]);
	ldp	s29, s31, [x21, 4]	// _4290, _4300,
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	str	q30, [sp, 160]	// _3927, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	str	q30, [sp, 288]	// _3927, %sfp
	dup	v29.4s, v29.s[0]	// _4291, _4290
	dup	v31.4s, v31.s[0]	// _1821, _4300
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q29, q31, [sp, 128]	// _4291, _1821,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w20, 0	// _2,
	ble	.L659		//,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	ldrsw	x6, [sp, 272]	// _5991, %sfp
	cmp	w19, 0	// _3,
	ble	.L659		//,
	ldp	x9, x4, [sp, 208]	// _6, ivtmp.1754,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w5, 0	// ic,
	.p2align 5,,15
.L665:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w1, 0	// kh,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x2, x4	// ivtmp.1750, ivtmp.1754
.L664:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w0, [sp, 176]	//, %sfp
	cmp	w0, 0	// _4,
	bgt	.L662		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w1, w1, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x2, x2, x9	// ivtmp.1750, ivtmp.1750, _6
	cmp	w19, w1	// _3, kh
	bne	.L664		//,
	.p2align 5,,15
.L663:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w5, w5, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x4, x4, x6	// ivtmp.1754, ivtmp.1754, _5991
	cmp	w20, w5	// _2, ic
	bne	.L665		//,
.L659:
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w26, 3	// _8,
	bgt	.L650		//,
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w26, 0	// _8,
	bgt	.L1364		//,
.L666:
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	w0, [sp, 304]	//, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	x21, x21, 16	// ivtmp.1761, ivtmp.1761,
	ldr	x2, [sp, 448]	// _5684, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	w1, w0, 4	// oc0, oc0,
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	x0, [sp, 392]	// _2878, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	str	w1, [sp, 304]	// oc0, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	x25, x25, x0	// ivtmp.1763, ivtmp.1763, _2878
	add	x24, x24, x0	// ivtmp.1764, ivtmp.1764, _2878
	add	x23, x23, x0	// ivtmp.1765, ivtmp.1765, _2878
	add	x22, x22, x0	// ivtmp.1766, ivtmp.1766, _2878
	ldr	x0, [sp, 248]	// ivtmp.1767, %sfp
	add	x0, x0, x2	// ivtmp.1767, ivtmp.1767, _5684
	str	x0, [sp, 248]	// ivtmp.1767, %sfp
	ldr	w0, [sp, 264]	//, %sfp
	cmp	w0, w1	// n_full, oc0
	bgt	.L667		//,
	ldr	x1, [sp, 520]	// ivtmp.1787, %sfp
	ldr	x9, [sp, 536]	// ivtmp.1776, %sfp
	ldr	w8, [sp, 528]	//, %sfp
	ldr	w28, [sp, 532]	//, %sfp
.L683:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	w0, [sp, 108]	//, %sfp
	ldr	w2, [sp, 336]	//, %sfp
	cmp	w0, w2	// _1, oc0
	ble	.L681		//,
	ldr	x0, [sp, 384]	// _2034, %sfp
	ldp	x21, x22, [sp, 400]	// ivtmp.1731, ivtmp.1735,
	str	w27, [sp, 456]	// _538, %sfp
	add	x4, x0, x9	// _1997, _2034, ivtmp.1776
	ldr	w0, [sp, 376]	//, %sfp
	add	x24, x1, x4, lsl 2	// ivtmp.1734, ivtmp.1787, _1997,
	mov	x27, x21	// ivtmp.1731, ivtmp.1731
	mov	w21, w20	// _2, _2
	str	w8, [sp, 288]	// oh, %sfp
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	ldrsw	x23, [sp, 272]	// _6042, %sfp
	mov	w20, w26	// _8, _8
	mov	x26, x24	// ivtmp.1734, ivtmp.1734
	mov	w24, w28	// col, col
	str	x1, [sp, 304]	// ivtmp.1787, %sfp
	mov	x28, x9	// ivtmp.1776, ivtmp.1776
	add	w0, w0, 1	// _690, _1812,
	str	w0, [sp, 248]	// _690, %sfp
	ldr	x0, [sp, 344]	// _687, %sfp
	lsl	x25, x0, 2	// _2025, _687,
.L680:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v3.4s}, [x27]	//, MEM[(const float *)_5215]
// src/cpp/cnn_internals.cpp:421:                 conv_row_tile<1, NV>(
	mov	x5, x22	// wp, ivtmp.1735
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w21, 0	// _2,
	ble	.L670		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	cmp	w19, 0	// _3,
	ble	.L670		//,
	ldr	x3, [sp, 216]	// ivtmp.1724, %sfp
	mov	w4, 0	// ic,
	ldr	x8, [sp, 256]	// ivtmp.1788, %sfp
	ldr	x6, [sp, 280]	// _197, %sfp
	ldr	x7, [sp, 368]	// _1702, %sfp
	.p2align 5,,15
.L676:
	add	x1, x8, x3, lsl 2	// ivtmp.1720, ivtmp.1788, ivtmp.1724,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w2, 0	// kh,
.L675:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w0, [sp, 176]	//, %sfp
	cmp	w0, 0	// _4,
	bgt	.L673		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w2, w2, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x1, x1, x7	// ivtmp.1720, ivtmp.1720, _1702
	cmp	w19, w2	// _3, kh
	bne	.L675		//,
	.p2align 5,,15
.L674:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w4, w4, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x3, x3, x23	// ivtmp.1724, ivtmp.1724, _6042
	cmp	w21, w4	// _2, ic
	bne	.L676		//,
.L670:
	ldr	w0, [sp, 248]	//, %sfp
	add	x2, x26, w0, uxtw 4	// _4799, ivtmp.1734, _690,
	ldr	x0, [sp, 440]	// _5377, %sfp
	add	x0, x26, x0	// _5375, ivtmp.1734, _5377
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w20, 3	// _8,
	ble	.L677		//,
	mov	x1, x26	// ivtmp.1706, ivtmp.1734
	.p2align 5,,15
.L678:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q3, [x1], 16	//, MEM <__Float32x4_t> [(float * {ref-all})_1333]
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	x1, x2	// ivtmp.1706, _4799
	bne	.L678		//,
.L677:
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w20, w24	// _8, col
	ble	.L679		//,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	ldr	x2, [sp, 224]	//, %sfp
	add	x1, sp, 672	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q3, [sp, 672]	//, MEM <__Float32x4_t> [(float * {ref-all})_2907]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	bl	memcpy		//
.L679:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	x0, [sp, 320]	// _93, %sfp
	add	x27, x27, 4	// ivtmp.1731, ivtmp.1731,
	add	x26, x26, x0	// ivtmp.1734, ivtmp.1734, _93
	ldr	x0, [sp, 416]	// _5267, %sfp
	add	x22, x22, x0	// ivtmp.1735, ivtmp.1735, _5267
	ldr	x0, [sp, 432]	// _5369, %sfp
	cmp	x27, x0	// ivtmp.1731, _5369
	bne	.L680		//,
	ldr	x1, [sp, 304]	// ivtmp.1787, %sfp
	mov	w26, w20	// _8, _8
	mov	x9, x28	// ivtmp.1776, ivtmp.1776
	mov	w20, w21	// _2, _2
	mov	w28, w24	// col, col
	ldr	w8, [sp, 288]	//, %sfp
	ldr	w27, [sp, 456]	//, %sfp
.L681:
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w8, w8, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	ldr	x0, [sp, 184]	// _2604, %sfp
	add	x9, x9, x0	// ivtmp.1776, ivtmp.1776, _2604
	ldp	x2, x0, [sp, 208]	// _6, ivtmp.1777,
	add	x0, x0, x2	// ivtmp.1777, ivtmp.1777, _6
	str	x0, [sp, 216]	// ivtmp.1777, %sfp
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, w8	// _7, oh
	bne	.L648		//,
	ldr	w21, [sp, 516]	//, %sfp
	mov	x11, x1	// ivtmp.1787, ivtmp.1787
	mov	w7, w28	// col, col
	ldp	x23, x25, [sp, 496]	// _63, _200,
	mov	w28, w26	// _8, _8
	mov	w17, w27	// _538, _538
	ldp	x1, x18, [sp, 312]	// ivtmp.1790, _93,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	w21, w21, 1	// b, b,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	x11, x11, x23	// ivtmp.1787, ivtmp.1787, _63
	ldr	x15, [sp, 256]	// ivtmp.1788, %sfp
	add	x1, x1, x23	// ivtmp.1790, ivtmp.1790, _63
	ldr	x26, [sp, 480]	// _74, %sfp
	ldr	w22, [sp, 512]	//, %sfp
	ldr	x3, [sp, 328]	// _257, %sfp
	add	x15, x15, x26	// ivtmp.1788, ivtmp.1788, _74
	ldr	x2, [sp, 344]	// _687, %sfp
	ldr	w9, [sp, 264]	//, %sfp
	ldr	w30, [sp, 272]	//, %sfp
	ldr	w8, [sp, 336]	//, %sfp
	cmp	w22, w21	// _232, b
	bne	.L684		//,
	b	.L884		//
.L1357:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	w0, [sp, 108]	//, %sfp
	cmp	w0, 0	// _1,
	bgt	.L789		//,
	b	.L806		//
	.p2align 2,,3
.L1365:
	ldr	x0, [sp, 464]	// _5873, %sfp
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w1, w1, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x2, x2, x9	// ivtmp.1750, ivtmp.1750, _6
	add	x7, x7, x0	// wp, wp, _5873
	cmp	w19, w1	// _3, kh
	beq	.L663		//,
.L662:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	add	x0, sp, 112	// tmp6637,,
	ld1	{v23.4s - v26.4s}, [x0]	// acc, %sfp
	ldr	x0, [sp, 256]	// ivtmp.1788, %sfp
	add	x3, x0, x2, lsl 2	// ivtmp.1744, ivtmp.1788, ivtmp.1750,
	ldr	x0, [sp, 424]	// _5297, %sfp
	mov	v27.16b, v26.16b	// _68, acc
	mov	v28.16b, v24.16b	// _246, acc
	mov	v22.16b, v23.16b	// _1236, acc
	mov	v30.16b, v25.16b	// _2951, acc
	add	x8, x7, x0	// _5293, wp, _5297
	mov	x0, x7	// wp, wp
	.p2align 5,,15
.L661:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldr	q23, [x3], 4	// MEM <__Float32x4_t> [(float * {ref-all})_5315], MEM <__Float32x4_t> [(float * {ref-all})_5315]
// src/cpp/cnn_internals.cpp:360:                 for (int t = 0; t < OC_T; ++t) wv[t] = wp[t];
	ldp	s26, s24, [x0]	// _3095, _3103,* wp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v22.4s, v23.4s, v26.s[0]	// _1236, MEM <__Float32x4_t> [(float * {ref-all})_5315], _3095
// src/cpp/cnn_internals.cpp:360:                 for (int t = 0; t < OC_T; ++t) wv[t] = wp[t];
	ldp	s25, s26, [x0, 8]	// _3111, _3119,
// src/cpp/cnn_internals.cpp:361:                 wp += OC_T;
	add	x0, x0, 16	// wp, wp,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v28.4s, v23.4s, v24.s[0]	// _246, MEM <__Float32x4_t> [(float * {ref-all})_5315], _3103
	fmla	v30.4s, v23.4s, v25.s[0]	// _2951, MEM <__Float32x4_t> [(float * {ref-all})_5315], _3111
	fmla	v27.4s, v23.4s, v26.s[0]	// _68, MEM <__Float32x4_t> [(float * {ref-all})_5315], _3119
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q22, q28, [sp, 112]	// _1236, _246,
	stp	q30, q27, [sp, 144]	// _2951, _68,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x0, x8	// wp, _5293
	bne	.L661		//,
	b	.L1365		//
	.p2align 2,,3
.L1366:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w2, w2, 1	// kh, kh,
	add	x5, x5, x25	// wp, wp, _2025
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x1, x1, x7	// ivtmp.1720, ivtmp.1720, _1702
	cmp	w19, w2	// _3, kh
	beq	.L674		//,
.L673:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	x0, 0	// ivtmp.1711,
	.p2align 5,,15
.L672:
// src/cpp/cnn_internals.cpp:360:                 for (int t = 0; t < OC_T; ++t) wv[t] = wp[t];
	ldr	s2, [x5, x0]	//, MEM[(const float *)wp_2851 + ivtmp.1711_1779 * 1]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldr	q1, [x1, x0]	//, MEM <__Float32x4_t> [(float * {ref-all})_4940 + ivtmp.1711_1779 * 1]
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x0, x0, 4	// ivtmp.1711, ivtmp.1711,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v3.4s, v1.4s, v2.s[0]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x6, x0	// _197, ivtmp.1711
	bne	.L672		//,
	b	.L1366		//
.L650:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	add	x0, sp, 112	// tmp6639,,
	ld1	{v25.4s - v28.4s}, [x0]	// acc, %sfp
	str	q25, [x22]	// tmp5885, MEM <__Float32x4_t> [(float * {ref-all})o_3772]
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w26, 4	// _8,
	bne	.L653		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q26, [x23]	// tmp5886, MEM <__Float32x4_t> [(float * {ref-all})o_5665]
	str	q27, [x24]	// tmp5887, MEM <__Float32x4_t> [(float * {ref-all})o_1917]
	str	q28, [x25]	// tmp5888, MEM <__Float32x4_t> [(float * {ref-all})o_2070]
	b	.L666		//
.L1364:
	ldr	q31, [sp, 112]	// tmp5884, %sfp
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	add	x28, sp, 672	// tmp3576,,
	ubfiz	x2, x26, 2, 32	//, _8,,
	mov	x1, x28	//, tmp3576
	mov	x0, x22	//, ivtmp.1766
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 672]	// tmp5884, MEM <__Float32x4_t> [(float * {ref-all})_2907]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [sp, 128]	// _1821, %sfp
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	mov	w4, 0	// col,
.L652:
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w27, w4	// _2419, _538, col
	uxtw	x0, w4	// _3703, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 672]	// _1821, MEM <__Float32x4_t> [(float * {ref-all})_2907]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	cmp	w26, w4	// _8, col
	add	x0, x23, x0, lsl 2	//, ivtmp.1765, _3703,
	mov	x4, 4	// tmp2316,
	add	w2, w2, 1	// _3711, _2419,
	mov	x1, x28	//, tmp3576
	lsl	x2, x2, 2	// _3705, _3711,
	csel	x2, x2, x4, gt	//, _3705, tmp2316,
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w26, 3	// _8,
	mov	x4, 4	// tmp2316,
	bgt	.L1367		//,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [sp, 144]	// tmp5890, %sfp
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	mov	w2, 0	// col,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	str	q31, [sp, 288]	// tmp5890, %sfp
.L658:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [sp, 288]	// _3927, %sfp
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w4, w27, w2	// _3903, _538, col
	uxtw	x0, w2	// _3332, col
	add	w4, w4, 1	// _3899, _3903,
	cmp	w26, w2	// _8, col
	add	x0, x24, x0, lsl 2	//, ivtmp.1764, _3332,
	lsl	x2, x4, 2	// _3891, _3899,
	mov	x1, x28	//, tmp3576
	mov	x4, 4	// tmp2331,
	csel	x2, x2, x4, gt	//, _3891, tmp2331,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 672]	// _3927, MEM <__Float32x4_t> [(float * {ref-all})_2907]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w26, 3	// _8,
	mov	x4, 4	// tmp2331,
	bgt	.L1368		//,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [sp, 160]	// _2712, %sfp
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	mov	w2, 0	// col,
.L1030:
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w4, w27, w2	// _5443, _538, col
	uxtw	x0, w2	// _5411, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 672]	// _2712, MEM <__Float32x4_t> [(float * {ref-all})_2907]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	add	w4, w4, 1	// _5427, _5443,
	cmp	w26, w2	// _8, col
	add	x0, x25, x0, lsl 2	//, ivtmp.1763, _5411,
	lsl	x2, x4, 2	// _5426, _5427,
	mov	x1, x28	//, tmp3576
	mov	x4, 4	// tmp2362,
	csel	x2, x2, x4, gt	//, _5426, tmp2362,
	bl	memcpy		//
	b	.L666		//
.L653:
	add	x28, sp, 672	// tmp3576,,
	ldr	x2, [sp, 456]	//, %sfp
	add	x0, x22, 16	//, ivtmp.1766,
	mov	x1, x28	//, tmp3576
	str	q31, [sp, 544]	// _1821, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q29, [sp, 672]	// _4291, MEM <__Float32x4_t> [(float * {ref-all})_2907]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [sp, 128]	// tmp6418, %sfp
	mov	w4, 4	// col,
	str	q31, [x23]	// tmp6418, MEM <__Float32x4_t> [(float * {ref-all})o_1161]
	ldr	q31, [sp, 544]	// _1821, %sfp
	b	.L652		//
.L1368:
	add	x1, sp, 112	// tmp6635,,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	add	x0, sp, 560	// tmp3551,,
	mov	w2, w4	// col, tmp2331
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ld1	{v27.4s - v30.4s}, [x1]	// acc, %sfp
	str	q30, [x25]	// tmp6415, MEM <__Float32x4_t> [(float * {ref-all})o_420]
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [sp, 624]	// _2712,
	st1	{v27.4s - v30.4s}, [x0]	// acc,
	b	.L1030		//
.L1367:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [sp, 144]	// tmp6417, %sfp
	mov	w2, w4	// col, tmp2316
	str	q31, [x24]	// tmp6417, MEM <__Float32x4_t> [(float * {ref-all})o_2276]
	b	.L658		//
.L1362:
	str	w19, [sp, 304]	// _3, %sfp
.L634:
	cmp	w4, 2	// _2129,
	bls	.L1047		//,
	sxtw	x0, w8	// _4584, ivtmp.1820
	mov	x26, x11	// ivtmp.1811, packed
	add	x16, x10, x0	// _5827, ivtmp.1824, _4584
	add	x19, x17, x0	// _2487, _3187, _4584
	add	x16, x22, x16, lsl 2	// vectp.1464, _202, _5827,
	add	x15, x18, x0	// _4792, _3206, _4584
	str	x0, [sp, 288]	// _4584, %sfp
	add	x19, x22, x19, lsl 2	// vectp.1470, _202, _2487,
	str	x16, [sp, 264]	// vectp.1464, %sfp
	add	x16, x9, x0	// _5998, ivtmp.1825, _4584
	add	x16, x22, x16, lsl 2	// vectp.1467, _202, _5998,
	str	x16, [sp, 256]	// vectp.1467, %sfp
	add	x16, x22, x15, lsl 2	// vectp.1473, _202, _4792,
	mov	x15, 0	// ivtmp.1805,
.L632:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	ldr	x0, [sp, 264]	// vectp.1464, %sfp
	ldr	q26, [x19, x15]	//, MEM <const vector(4) float> [(const float *)vectp.1470_2485 + ivtmp.1805_112 * 1]
	ldr	q27, [x16, x15]	//, MEM <const vector(4) float> [(const float *)vectp.1473_2492 + ivtmp.1805_112 * 1]
	ldr	q24, [x0, x15]	//, MEM <const vector(4) float> [(const float *)vectp.1464_1242 + ivtmp.1805_112 * 1]
	ldr	x0, [sp, 256]	// vectp.1467, %sfp
	ldr	q25, [x0, x15]	//, MEM <const vector(4) float> [(const float *)vectp.1467_97 + ivtmp.1805_112 * 1]
	add	x15, x15, 16	// ivtmp.1805, ivtmp.1805,
	st4	{v24.4s - v27.4s}, [x26]	//, MEM <float[16]> [(float *)vectp_packed.1475_2499]
	add	x26, x26, 64	// ivtmp.1811, ivtmp.1811,
	cmp	x15, x6	// ivtmp.1805, _125
	bne	.L632		//,
	ldr	x0, [sp, 288]	// _4584, %sfp
	mov	w16, w30	// niters_vector_mult_vf.1460, niters_vector_mult_vf.1460
	cmp	w30, w23	// niters_vector_mult_vf.1460, ksz
	beq	.L633		//,
.L631:
	uxtw	x15, w16	// _2425, niters_vector_mult_vf.1460
	add	x19, x10, x0	// _2424, ivtmp.1824, _4584
	add	x19, x19, x15	// _2426, _2424, _2425
	add	x26, x9, x0	// _4190, ivtmp.1825, _4584
	sub	w16, w23, w16	// bnd.1479_2421, ksz, niters_vector_mult_vf.1460
	whilelo	p7.s, wzr, w16	// max_mask_2086,, bnd.1479_2421
	ld1w	z16.s, p7/z, [x22, x19, lsl 2]	//, max_mask_2086,* _202
	add	x19, x17, x0	// _729, _3187, _4584
	add	x0, x18, x0	// _2155, _3206, _4584
	add	x16, x11, x15, lsl 4	// vectp_packed.1493, packed, _2425,
	add	x26, x26, x15	// _4192, _4190, _2425
	ld1w	z17.s, p7/z, [x22, x26, lsl 2]	//, max_mask_2086,* _202
	add	x19, x19, x15	// _3059, _729, _2425
	add	x0, x0, x15	// _3039, _2155, _2425
	ld1w	z18.s, p7/z, [x22, x19, lsl 2]	//, max_mask_2086,* _202
	ld1w	z19.s, p7/z, [x22, x0, lsl 2]	//, max_mask_2086,* _202
	st4w	{z16.s - z19.s}, p7, [x16]	//, max_mask_2086,* vectp_packed.1493
.L633:
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w2, w2, 1	// ic, ic,
	add	x11, x11, x5	// packed, packed, _3712
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w8, w8, w23	// ivtmp.1820, ivtmp.1820, ksz
	cmp	w20, w2	// _2, ic
	bne	.L634		//,
	ldr	w19, [sp, 304]	//, %sfp
	b	.L635		//
.L1047:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	w16, 0	// niters_vector_mult_vf.1460,
	sxtw	x0, w8	// _4584, ivtmp.1820
	b	.L631		//
.L1066:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	mov	x3, 0	// _1581,
.L869:
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	ldr	w0, [sp, 176]	//, %sfp
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	ldr	w2, [sp, 108]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1268:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x1, [x22]	// _1526, MEM[(const struct vector *)weight_24(D)].D.54862._M_impl.D.54163._M_start
// src/cpp/cnn_internals.cpp:317:     const int ksz  = kernel_h * kernel_w;
	mul	w0, w19, w0	// ksz, _3, _4
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	add	w22, w2, w2, lsr 31	// tmp3076, _1, _1,
// src/cpp/cnn_internals.cpp:318:     const int wcsz  = in_channels * ksz;              // per-oc stride in OIHW
	mul	w23, w20, w0	// wcsz, _2, ksz
// src/cpp/cnn_internals.cpp:319:     const int tiles = out_channels / OC_T;
	asr	w22, w22, 1	// tiles, tmp3076,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w2, 1	// _1,
	ble	.L870		//,
	cmp	w20, 0	// _2,
	ble	.L870		//,
	sub	w2, w0, #1	// _5679, ksz,
	lsr	w7, w0, 2	// _1885, ksz,
	add	x2, x2, 1	// _469, _5679,
	mov	x12, 0	// ivtmp.2280,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w9, 0	// tb,
	lsl	x18, x7, 4	// _1886, _1885,
	lsl	x30, x2, 3	// _1060, _469,
	sxtw	x11, w23	// ivtmp.2281, wcsz
	sbfiz	x13, x23, 1, 32	// _1894, wcsz,,
	sub	w4, w0, #1	// _4104, ksz,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	x10, x3	// packed, _1581
	and	w17, w0, -4	// niters_vector_mult_vf.1602, ksz,
.L871:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	w8, 0	// ivtmp.2276,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w7, 0	// ic,
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w0, 0	// ksz,
	bgt	.L875		//,
.L1273:
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w7, w7, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w8, w8, w0	// ivtmp.2276, ivtmp.2276, ksz
	cmp	w20, w7	// _2, ic
	beq	.L876		//,
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w0, 0	// ksz,
	ble	.L1273		//,
.L875:
	cmp	w4, 2	// _4104,
	bls	.L1068		//,
	sxtw	x2, w8	// _3996, ivtmp.2276
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	x5, 0	// ivtmp.2269,
	add	x14, x12, x2	// _4036, ivtmp.2280, _3996
	add	x6, x11, x2	// _4023, ivtmp.2281, _3996
	add	x14, x1, x14, lsl 2	// vectp.1606, _1526, _4036,
	mov	x15, x10	// ivtmp.2267, packed
	add	x6, x1, x6, lsl 2	// vectp.1609, _1526, _4023,
.L873:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	ldr	q21, [x14, x5]	//, MEM <const vector(4) float> [(const float *)vectp.1606_4065 + ivtmp.2269_1883 * 1]
	ldr	q22, [x6, x5]	//, MEM <const vector(4) float> [(const float *)vectp.1609_4026 + ivtmp.2269_1883 * 1]
	add	x5, x5, 16	// ivtmp.2269, ivtmp.2269,
	st2	{v21.4s - v22.4s}, [x15], 32	//, MEM <float[8]> [(float *)vectp_packed.1611_4016]
	cmp	x5, x18	// ivtmp.2269, _1886
	bne	.L873		//,
	mov	w5, w17	// niters_vector_mult_vf.1602, niters_vector_mult_vf.1602
	cmp	w17, w0	// niters_vector_mult_vf.1602, ksz
	beq	.L874		//,
.L872:
	uxtw	x14, w5	// _3994, niters_vector_mult_vf.1602
	add	x6, x12, x2	// _3995, ivtmp.2280, _3996
	sub	w5, w0, w5	// bnd.1615_4001, ksz, niters_vector_mult_vf.1602
	add	x2, x11, x2	// _3976, ivtmp.2281, _3996
	whilelo	p7.s, wzr, w5	// max_mask_3945,, bnd.1615_4001
	add	x15, x10, x14, lsl 3	// vectp_packed.1623, packed, _3994,
	add	x5, x6, x14	// _3991, _3995, _3994
	add	x2, x2, x14	// _3972, _3976, _3994
	ld1w	z17.s, p7/z, [x1, x5, lsl 2]	//, max_mask_3945,* _1526
	ld1w	z18.s, p7/z, [x1, x2, lsl 2]	//, max_mask_3945,* _1526
	st2w	{z17.s - z18.s}, p7, [x15]	//, max_mask_3945,* vectp_packed.1623
.L874:
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w7, w7, 1	// ic, ic,
	add	x10, x10, x30	// packed, packed, _1060
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w8, w8, w0	// ivtmp.2276, ivtmp.2276, ksz
	cmp	w20, w7	// _2, ic
	bne	.L875		//,
.L876:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	w9, w9, 1	// tb, tb,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	x12, x12, x13	// ivtmp.2280, ivtmp.2280, _1894
	add	x11, x11, x13	// ivtmp.2281, ivtmp.2281, _1894
	cmp	w22, w9	// tiles, tb
	bgt	.L871		//,
.L870:
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	ldr	w2, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:471:         const int n_full = (OC / OC_TILE) * OC_TILE;
	lsl	w8, w22, 1	// n_full, tiles,
// src/cpp/cnn_internals.cpp:472:         if (n_full < OC)
	cmp	w2, w8	// _1, n_full
	bgt	.L1369		//,
.L879:
// src/cpp/cnn_internals.cpp:478:             input.batches, IC, OC, IH, IW, KH, KW, OH, OW);
	ldr	w21, [x21, 24]	//, input_29(D)->batches
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	cmp	w21, 0	// _1556,
	ble	.L884		//,
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w0, [sp, 184]	//, %sfp
// src/cpp/cnn_internals.cpp:404:     const int tile_wsz    = in_channels * ksz * OC_TILE;   // packed floats/tile
	lsl	w7, w23, 1	// tile_wsz, wcsz,
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	ldr	w1, [sp, 224]	//, %sfp
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	ldr	w5, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:399:     const int in_size     = input_h * input_w;
	mul	w9, w0, w1	// in_size, _5, _6
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	ldr	w0, [sp, 180]	//, %sfp
// src/cpp/cnn_internals.cpp:400:     const int in_ch_size  = in_channels * in_size;
	mul	w12, w20, w9	// in_ch_size, _2, in_size
// src/cpp/cnn_internals.cpp:401:     const int out_size    = output_h * output_w;
	mul	w2, w0, w28	// out_size, _7, _8
// src/cpp/cnn_internals.cpp:402:     const int out_ch_size = out_channels * out_size;
	mul	w11, w5, w2	// out_ch_size, _1, out_size
	cmp	w0, 0	// _7,
	ble	.L884		//,
	cmp	w8, 2	// n_full,
	smull	x15, w22, w7	// _482, tiles, tile_wsz
	mov	w13, 2	// tmp3122,
	csel	w13, w8, w13, ge	// oc0, n_full, tmp3122,
	ldr	x1, [sp, 248]	// _12, %sfp
	cmp	w8, 0	// n_full,
	csel	w13, w13, wzr, gt	// oc0, oc0,,
	smull	x6, w13, w2	// _1606, oc0, out_size
	uxtw	x16, w13	// _1598, oc0
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	sub	w0, w28, #4	// _4923, _8,
	ldr	w4, [sp, 176]	//, %sfp
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	mov	w22, 0	// b,
	sbfiz	x23, x12, 2, 32	// _1824, in_ch_size,,
	sbfiz	x11, x11, 2, 32	// _1840, out_ch_size,,
	str	x6, [sp, 424]	// _1606, %sfp
	add	x6, x3, x15, lsl 2	// ivtmp.2203, _1581, _482,
	sub	w15, w5, w13	// _1625, _1, oc0
	add	x15, x15, w13, sxtw	// _1626, _1625, oc0
	ldr	x5, [sp, 200]	// _87, %sfp
	mul	x14, x1, x24	// _1516, _12, _5871
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	lsr	w1, w0, 2	// _4919, _4923,
	str	x6, [sp, 440]	// ivtmp.2203, %sfp
	sub	w10, w4, #1	// _5711, _4,
	add	x10, x10, 1	// _490, _5711,
	add	x6, x5, x16, lsl 2	// ivtmp.2199, _87, _1598,
	mov	x16, x27	// ivtmp.2251, in_ptr
	str	x6, [sp, 416]	// ivtmp.2199, %sfp
	add	w6, w1, 1	// _477, _4919,
	and	w1, w0, -4	// _5727, _4923,
	add	x0, x5, x15, lsl 2	// _1629, _87, _1626,
	add	w27, w1, 4	// col, _5727,
	str	w6, [sp, 360]	// _477, %sfp
	str	x0, [sp, 448]	// _1629, %sfp
	lsl	x0, x14, 2	// _1616, _1516,
	ldr	x14, [sp, 216]	// ivtmp.2252, %sfp
	str	x0, [sp, 432]	// _1616, %sfp
	sbfiz	x0, x7, 2, 32	// _1781, tile_wsz,,
	str	x0, [sp, 472]	// _1781, %sfp
	mov	x0, x6	// tmp6218, _477
	ubfiz	x0, x0, 4, 31	// _4910, tmp6218,,
	str	x0, [sp, 112]	// _4910, %sfp
	lsl	x0, x10, 3	// _5706, _490,
	str	x0, [sp, 496]	// _5706, %sfp
	sbfiz	x0, x2, 3, 32	// _1763, out_size,,
	str	x0, [sp, 464]	// _1763, %sfp
	sbfiz	x0, x2, 2, 32	// _1632, out_size,,
	str	x0, [sp, 248]	// _1632, %sfp
	sbfiz	x0, x4, 3, 32	// _1698, _4,,
	str	x0, [sp, 456]	// _1698, %sfp
	sbfiz	x0, x4, 2, 32	// _1569, _4,,
	str	x0, [sp, 408]	// _1569, %sfp
	sxtw	x0, w28	// _2604, _8
	str	x0, [sp, 184]	// _2604, %sfp
.L920:
	stp	x23, x11, [sp, 384]	// _1824, _1840,
	mov	w23, w20	// _2, _2
	mov	w20, w28	// _8, _8
	mov	x0, x25	// _1524, _1524
// src/cpp/cnn_internals.cpp:410:         float*       __restrict out_b = out_ptr   + (std::size_t)b * out_ch_size;
	mov	x15, 0	// ivtmp.2246,
	mov	x7, 0	// ivtmp.2245,
	mov	w25, w19	// _3, _3
	ldr	x28, [sp, 208]	// _6, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	mov	w2, 0	// oh,
	mov	x26, x15	// ivtmp.2246, ivtmp.2246
	mov	x19, x0	// _1524, _1524
	stp	x3, x10, [sp, 264]	// _1581, _490,
	mov	w3, w8	// n_full, n_full
	str	w21, [sp, 376]	// _1556, %sfp
	mov	x21, x7	// ivtmp.2245, ivtmp.2245
	str	x16, [sp, 216]	// ivtmp.2251, %sfp
	str	w13, [sp, 256]	// oc0, %sfp
	str	w22, [sp, 368]	// b, %sfp
.L885:
	add	x0, sp, 672	// tmp3576,,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	sxtw	x18, w9	// _3470, in_size
	str	x0, [sp, 400]	// tmp3576, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	cmp	w3, 0	// n_full,
	ble	.L1370		//,
	mov	w24, w23	// _2, _2
	mov	x8, x26	// ivtmp.2246, ivtmp.2246
	mov	w23, w20	// _8, _8
	mov	w22, w3	// n_full, n_full
	mov	x20, x19	// _1524, _1524
	mov	w26, w25	// _3, _3
	mov	w19, w2	// oh, oh
.L902:
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	ldr	x25, [sp, 200]	// ivtmp.2236, %sfp
// src/cpp/cnn_internals.cpp:413:             int oc0 = 0, tile = 0;
	mov	w6, 0	// oc0,
	add	x5, x14, x21, lsl 2	// ivtmp.2238, ivtmp.2252, ivtmp.2245,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	mov	w11, w19	// oh, oh
	mov	x12, x20	// _1524, _1524
	mov	x3, x28	// _6, _6
	ldr	x4, [sp, 264]	// ivtmp.2239, %sfp
	mov	w28, w24	// _2, _2
	mov	w19, w6	// oc0, oc0
	mov	x20, x21	// ivtmp.2245, ivtmp.2245
	mov	w7, w22	// n_full, n_full
	mov	w24, w23	// _8, _8
.L900:
// src/cpp/cnn_internals.cpp:347:         const float32x4_t bv = vdupq_n_f32(bias_ptr[oc0 + t]);
	ldr	s16, [x25, 4]	// _4967, MEM[(const float *)_1787 + 4B]
// src/cpp/cnn_internals.cpp:415:                 conv_row_tile<OC_TILE, NV>(
	mov	x16, x4	// wp, ivtmp.2239
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v7.4s}, [x25]	//, MEM[(const float *)_1787]
	dup	v16.4s, v16.s[0]	//, _4967
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q16, q16, [sp, 768]	//,,
	stp	q7, q7, [sp, 672]	//,,
	stp	q7, q7, [sp, 704]	//,,
	stp	q7, q16, [sp, 736]	//,,
	stp	q16, q16, [sp, 800]	//,,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w28, 0	// _2,
	ble	.L888		//,
	cmp	w26, 0	// _3,
	ble	.L888		//,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w15, 0	// ic,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	mov	x0, x8	// ivtmp.2229, ivtmp.2246
	.p2align 5,,15
.L894:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w6, 0	// kh,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x10, x0	// ivtmp.2225, ivtmp.2229
.L893:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w1, [sp, 176]	//, %sfp
	cmp	w1, 0	// _4,
	bgt	.L891		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w6, w6, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x10, x10, x3	// ivtmp.2225, ivtmp.2225, _6
	cmp	w26, w6	// _3, kh
	bne	.L893		//,
	.p2align 5,,15
.L892:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w15, w15, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x0, x0, x18	// ivtmp.2229, ivtmp.2229, _3470
	cmp	w28, w15	// _2, ic
	bne	.L894		//,
.L888:
	ldr	x23, [sp, 400]	// ivtmp.2211, %sfp
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	mov	w22, 0	// t,
	str	x20, [sp, 224]	// ivtmp.2245, %sfp
	mov	x20, x5	// ivtmp.2210, ivtmp.2210
	str	w19, [sp, 280]	// oc0, %sfp
	mov	w19, w24	// _8, _8
	mov	x24, x18	// _3470, _3470
	str	x12, [sp, 288]	// _1524, %sfp
	str	w11, [sp, 304]	// oh, %sfp
	mov	x21, x23	// ivtmp.2211, ivtmp.2211
	mov	x23, x3	// _6, _6
	str	w7, [sp, 312]	// n_full, %sfp
	str	w9, [sp, 320]	// in_size, %sfp
	stp	x5, x4, [sp, 328]	// ivtmp.2238, ivtmp.2239,
	stp	x8, x14, [sp, 344]	// ivtmp.2246, ivtmp.2252,
.L887:
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w19, 3	// _8,
	ble	.L895		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	mov	x1, x21	//, ivtmp.2211
	mov	x0, x20	//, ivtmp.2210
	ldr	x2, [sp, 112]	//, %sfp
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w19, w27	// _8, col
	bgt	.L896		//,
.L1371:
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	cbnz	w22, .L1294	// t,
	ldr	x0, [sp, 248]	// _1632, %sfp
	add	x21, x21, 80	// ivtmp.2211, ivtmp.2211,
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	mov	w22, 1	// t,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	mov	x1, x21	//, ivtmp.2211
	ldr	x2, [sp, 112]	//, %sfp
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	x20, x20, x0	// ivtmp.2210, ivtmp.2210, _1632
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	mov	x0, x20	//, ivtmp.2210
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w19, w27	// _8, col
	ble	.L1371		//,
.L896:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldrsw	x1, [sp, 360]	// v, %sfp
	and	x0, x22, 1	// tmp3878, t,
	mov	w7, w27	// col, col
	orr	x0, x0, x0, lsl 2	// tmp3165, tmp3878, tmp3878,
	add	x0, x0, x1	// tmp3166, tmp3165, v
	ldr	x1, [sp, 400]	// tmp3576, %sfp
	lsl	x0, x0, 4	// tmp3167, tmp3166,
	ldr	q31, [x1, x0]	// _1758, MEM[(__Float32x4_t[2][5] *)_2907][t_5902][v_4017]
.L899:
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w19, #1	// _613, _8,
	uxtw	x0, w7	// _4930, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// _1758, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w2, w7	// _4944, _613, col
	cmp	w19, w7	// _8, col
	add	x0, x20, x0, lsl 2	//, ivtmp.2210, _4930,
	add	x1, sp, 624	//,,
	mov	x7, 4	// tmp3178,
	add	x2, x2, 1	// _4941, _4944,
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	add	x21, x21, 80	// ivtmp.2211, ivtmp.2211,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	lsl	x2, x2, 2	// _4937, _4941,
	csel	x2, x2, x7, gt	//, _4937, tmp3178,
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:374:     for (int t = 0; t < OC_T; ++t) {
	ldr	x1, [sp, 248]	// _1632, %sfp
	add	w0, w22, 1	// _487, t,
	mov	w22, 1	// t,
	add	x20, x20, x1	// ivtmp.2210, ivtmp.2210, _1632
	cmp	w0, 2	// _487,
	bne	.L887		//,
.L1294:
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	ldr	x0, [sp, 464]	// _1763, %sfp
	mov	x18, x24	// _3470, _3470
	mov	w24, w19	// _8, _8
	ldp	x5, x4, [sp, 328]	// ivtmp.2238, ivtmp.2239,
	mov	x3, x23	// _6, _6
	add	x25, x25, 8	// ivtmp.2236, ivtmp.2236,
	ldp	x8, x14, [sp, 344]	// ivtmp.2246, ivtmp.2252,
	add	x5, x5, x0	// ivtmp.2238, ivtmp.2238, _1763
	ldr	w19, [sp, 280]	//, %sfp
	ldr	x0, [sp, 472]	// _1781, %sfp
	ldr	w7, [sp, 312]	//, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	w19, w19, 2	// oc0, oc0,
	ldr	x20, [sp, 224]	// ivtmp.2245, %sfp
// src/cpp/cnn_internals.cpp:414:             for (; oc0 < n_full; oc0 += OC_TILE, ++tile)
	add	x4, x4, x0	// ivtmp.2239, ivtmp.2239, _1781
	ldr	x12, [sp, 288]	// _1524, %sfp
	ldr	w11, [sp, 304]	//, %sfp
	ldr	w9, [sp, 320]	//, %sfp
	cmp	w7, w19	// n_full, oc0
	bgt	.L900		//,
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	mov	w23, w24	// _8, _8
	mov	x21, x20	// ivtmp.2245, ivtmp.2245
	ldr	w0, [sp, 108]	//, %sfp
	mov	w24, w28	// _2, _2
	mov	x20, x12	// _1524, _1524
	mov	w22, w7	// n_full, n_full
	mov	x28, x3	// _6, _6
	ldr	w1, [sp, 256]	//, %sfp
	cmp	w0, w1	// _1, oc0
	ble	.L1372		//,
	mov	w25, w26	// _3, _3
	mov	w20, w23	// _8, _8
	mov	w2, w11	// oh, oh
	mov	x19, x12	// _1524, _1524
	mov	w23, w24	// _2, _2
	mov	w3, w7	// n_full, n_full
	mov	x26, x8	// ivtmp.2246, ivtmp.2246
.L901:
	ldp	x24, x0, [sp, 416]	// ivtmp.2199, _1606,
	str	x19, [sp, 312]	// _1524, %sfp
	mov	w19, w20	// _8, _8
	str	w2, [sp, 288]	// oh, %sfp
	add	x22, x21, x0	// _1607, ivtmp.2245, _1606
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	ldr	x0, [sp, 440]	// ivtmp.2203, %sfp
	add	x22, x14, x22, lsl 2	// ivtmp.2202, ivtmp.2252, _1607,
	str	w3, [sp, 320]	// n_full, %sfp
	str	w9, [sp, 328]	// in_size, %sfp
	mov	x20, x22	// ivtmp.2202, ivtmp.2202
	mov	x22, x14	// ivtmp.2252, ivtmp.2252
	str	x0, [sp, 224]	// ivtmp.2203, %sfp
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	sxtw	x0, w9	// _3335, in_size
	str	x0, [sp, 280]	// _3335, %sfp
	ldr	x0, [sp, 272]	// _490, %sfp
	lsl	x0, x0, 2	// _5692, _490,
	str	x0, [sp, 304]	// _5692, %sfp
.L917:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v20.4s}, [x24]	//, MEM[(const float *)_1620]
// src/cpp/cnn_internals.cpp:421:                 conv_row_tile<1, NV>(
	ldr	x7, [sp, 224]	// wp, %sfp
// src/cpp/cnn_internals.cpp:348:         for (int v = 0; v < NV; ++v) acc[t][v] = bv;
	stp	q20, q20, [sp, 672]	//,,
	stp	q20, q20, [sp, 704]	//,,
	str	q20, [sp, 736]	//, MEM[(__Float32x4_t[1][5] *)_2907][0][4]
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	cmp	w23, 0	// _2,
	ble	.L907		//,
	cmp	w25, 0	// _3,
	ble	.L907		//,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	ldr	x1, [sp, 280]	// _3335, %sfp
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	w6, 0	// ic,
// src/cpp/cnn_internals.cpp:354:         const float* __restrict in_c = in_batch + (std::size_t)ic * in_size;
	mov	x0, x26	// ivtmp.2193, ivtmp.2246
	.p2align 5,,15
.L913:
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	mov	w2, 0	// kh,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	mov	x4, x0	// ivtmp.2189, ivtmp.2193
.L912:
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	ldr	w3, [sp, 176]	//, %sfp
	cmp	w3, 0	// _4,
	bgt	.L910		//,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w2, w2, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x28	// ivtmp.2189, ivtmp.2189, _6
	cmp	w25, w2	// _3, kh
	bne	.L912		//,
	.p2align 5,,15
.L911:
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	w6, w6, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:353:     for (int ic = 0; ic < in_channels; ++ic) {
	add	x0, x0, x1	// ivtmp.2193, ivtmp.2193, _3335
	cmp	w23, w6	// _2, ic
	bne	.L913		//,
.L907:
// src/cpp/cnn_internals.cpp:379:         for (; col + 4 <= output_w; col += 4, ++v) vst1q_f32(o + col, acc[t][v]);
	cmp	w19, 3	// _8,
	ble	.L1373		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	add	x1, sp, 672	// tmp3576,,
	ldr	x2, [sp, 112]	//, %sfp
	mov	x0, x20	//, ivtmp.2202
	str	x1, [sp, 336]	// tmp3576, %sfp
	bl	memcpy		//
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w19, w27	// _8, col
	ldr	x1, [sp, 336]	// tmp3576, %sfp
	mov	w3, w27	// col, col
	ldr	w0, [sp, 360]	//, %sfp
	ble	.L916		//,
.L914:
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ubfiz	x0, x0, 4, 31	// tmp3206, v,,
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	sub	w2, w19, #1	// _5673, _8,
	sub	w2, w2, w3	// _4890, _5673, col
	uxtw	x6, w3	// _4876, col
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	ldr	q31, [x1, x0]	// tmp6252, MEM[(__Float32x4_t[1][5] *)_2907][0][v_2420]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	cmp	w19, w3	// _8, col
	add	x2, x2, 1	// _4887, _4890,
	add	x0, x20, x6, lsl 2	//, ivtmp.2202, _4876,
	mov	x3, 4	// tmp3218,
	add	x1, sp, 624	//,,
	lsl	x2, x2, 2	// _4883, _4887,
	csel	x2, x2, x3, gt	//, _4883, tmp3218,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:22099:   __builtin_aarch64_st1v4sf ((__builtin_aarch64_simd_sf *) __a, __b);
	str	q31, [sp, 624]	// tmp6252, MEM <__Float32x4_t> [(float * {ref-all})_1253]
// src/cpp/cnn_internals.cpp:383:             for (int r = 0; col + r < output_w; ++r) o[col + r] = tmp[r];
	bl	memcpy		//
.L916:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	x0, [sp, 248]	// _1632, %sfp
	add	x24, x24, 4	// ivtmp.2199, ivtmp.2199,
	ldr	x1, [sp, 432]	// _1616, %sfp
	add	x20, x20, x0	// ivtmp.2202, ivtmp.2202, _1632
	ldr	x0, [sp, 224]	// ivtmp.2203, %sfp
	add	x0, x0, x1	// ivtmp.2203, ivtmp.2203, _1616
	str	x0, [sp, 224]	// ivtmp.2203, %sfp
	ldr	x0, [sp, 448]	// _1629, %sfp
	cmp	x24, x0	// ivtmp.2199, _1629
	bne	.L917		//,
	mov	w20, w19	// _8, _8
	ldr	x19, [sp, 312]	// _1524, %sfp
	mov	x14, x22	// ivtmp.2252, ivtmp.2252
	ldr	w2, [sp, 288]	//, %sfp
	ldr	w3, [sp, 320]	//, %sfp
	ldr	w9, [sp, 328]	//, %sfp
.L918:
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w2, w2, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	x26, x26, x28	// ivtmp.2246, ivtmp.2246, _6
	ldr	x0, [sp, 184]	// _2604, %sfp
	add	x21, x21, x0	// ivtmp.2245, ivtmp.2245, _2604
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, w2	// _7, oh
	bne	.L885		//,
	ldr	x16, [sp, 216]	// ivtmp.2251, %sfp
	mov	w28, w20	// _8, _8
	mov	w8, w3	// n_full, n_full
	mov	w20, w23	// _2, _2
	ldp	x3, x10, [sp, 264]	// _1581, _490,
	mov	x0, x19	// _1524, _1524
	mov	w19, w25	// _3, _3
	mov	x25, x0	// _1524, _1524
	ldp	x23, x11, [sp, 384]	// _1824, _1840,
	ldr	w13, [sp, 256]	//, %sfp
	ldr	w22, [sp, 368]	//, %sfp
	ldr	w21, [sp, 376]	//, %sfp
.L903:
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	w22, w22, 1	// b, b,
// src/cpp/cnn_internals.cpp:408:     for (int b = 0; b < batches; ++b) {
	add	x16, x16, x23	// ivtmp.2251, ivtmp.2251, _1824
	add	x14, x14, x11	// ivtmp.2252, ivtmp.2252, _1840
	cmp	w21, w22	// _1556, b
	bne	.L920		//,
	b	.L884		//
	.p2align 2,,3
.L1374:
	ldr	x1, [sp, 496]	// _5706, %sfp
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w6, w6, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x10, x10, x3	// ivtmp.2225, ivtmp.2225, _6
	add	x16, x16, x1	// wp, wp, _5706
	cmp	w26, w6	// _3, kh
	beq	.L892		//,
.L891:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldp	q5, q28, [sp, 672]	//,,
	ldp	q26, q24, [sp, 704]	//,,
	ldp	q6, q29, [sp, 736]	//,,
	ldp	q27, q25, [sp, 768]	//,,
	ldp	q23, q4, [sp, 800]	//,,
	ldr	x1, [sp, 216]	// ivtmp.2251, %sfp
	ldr	x2, [sp, 456]	// _1698, %sfp
	add	x1, x1, x10, lsl 2	// ivtmp.2219, ivtmp.2251, ivtmp.2225,
	add	x13, x16, x2	// _1700, wp, _1698
	mov	x2, x16	// wp, wp
	.p2align 5,,15
.L890:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldp	q19, q22, [x1]	//,,* ivtmp.2219
	ldp	q21, q0, [x1, 32]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x1, x1, 4	// ivtmp.2219, ivtmp.2219,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldr	q1, [x1, 60]	//, MEM <__Float32x4_t> [(float * {ref-all})_1669 + 64B]
// src/cpp/cnn_internals.cpp:360:                 for (int t = 0; t < OC_T; ++t) wv[t] = wp[t];
	ldp	s3, s2, [x2], 8	//,, wp,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v5.4s, v19.4s, v3.s[0]	//,,
	fmla	v29.4s, v19.4s, v2.s[0]	//,,
	fmla	v28.4s, v22.4s, v3.s[0]	//,,
	fmla	v27.4s, v22.4s, v2.s[0]	//,,
	fmla	v26.4s, v21.4s, v3.s[0]	//,,
	fmla	v25.4s, v21.4s, v2.s[0]	//,,
	fmla	v24.4s, v0.4s, v3.s[0]	//,,
	fmla	v23.4s, v0.4s, v2.s[0]	//,,
	fmla	v6.4s, v1.4s, v3.s[0]	//,,
	fmla	v4.4s, v1.4s, v2.s[0]	//,,
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q5, q28, [sp, 672]	//,,
	stp	q26, q24, [sp, 704]	//,,
	stp	q6, q29, [sp, 736]	//,,
	stp	q27, q25, [sp, 768]	//,,
	stp	q23, q4, [sp, 800]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x13, x2	// _1700, wp
	bne	.L890		//,
	b	.L1374		//
	.p2align 2,,3
.L1375:
	ldr	x3, [sp, 304]	// _5692, %sfp
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	w2, w2, 1	// kh, kh,
// src/cpp/cnn_internals.cpp:355:         for (int kh = 0; kh < kernel_h; ++kh) {
	add	x4, x4, x28	// ivtmp.2189, ivtmp.2189, _6
	add	x7, x7, x3	// wp, wp, _5692
	cmp	w25, w2	// _3, kh
	beq	.L911		//,
.L910:
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	ldp	q18, q17, [sp, 672]	//,,
	ldp	q31, q30, [sp, 704]	//,,
	ldr	x3, [sp, 216]	// ivtmp.2251, %sfp
	ldr	x5, [sp, 408]	// _1569, %sfp
	ldr	q16, [sp, 736]	//, MEM[(__Float32x4_t[1][5] *)_2907][0][4]
	add	x3, x3, x4, lsl 2	// ivtmp.2183, ivtmp.2251, ivtmp.2189,
	add	x8, x7, x5	// _1570, wp, _1569
	mov	x5, x7	// wp, wp
	.p2align 5,,15
.L909:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:10805:   return (float32x4_t) {__a, __a, __a, __a};
	ld1r	{v7.4s}, [x5], 4	//, MEM[(const float *)wp_4238]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldp	q3, q2, [x3]	//,,* ivtmp.2183
	ldp	q1, q0, [x3, 32]	//,,
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	add	x3, x3, 4	// ivtmp.2183, ivtmp.2183,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:12531:   return __builtin_aarch64_ld1v4sf ((const __builtin_aarch64_simd_sf *) __a);
	ldr	q29, [x3, 60]	//, MEM <__Float32x4_t> [(float * {ref-all})_1560 + 64B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/include/arm_neon.h:11902:   return __builtin_aarch64_fmav4sf (__b, vdupq_n_f32 (__c), __a);
	fmla	v18.4s, v3.4s, v7.4s	//,,
	fmla	v17.4s, v2.4s, v7.4s	//,,
	fmla	v31.4s, v1.4s, v7.4s	//,,
	fmla	v30.4s, v0.4s, v7.4s	//,,
	fmla	v16.4s, v29.4s, v7.4s	//,,
// src/cpp/cnn_internals.cpp:367:                         acc[t][v] = vfmaq_n_f32(acc[t][v], xv, wv[t]);
	stp	q18, q17, [sp, 672]	//,,
	stp	q31, q30, [sp, 704]	//,,
	str	q16, [sp, 736]	//, MEM[(__Float32x4_t[1][5] *)_2907][0][4]
// src/cpp/cnn_internals.cpp:357:             for (int kw = 0; kw < kernel_w; ++kw) {
	cmp	x8, x5	// _1570, wp
	bne	.L909		//,
	b	.L1375		//
.L1373:
// src/cpp/cnn_internals.cpp:380:         if (col < output_w) {                             // 1..3 leftover lanes
	cmp	w19, 0	// _8,
	ble	.L916		//,
	mov	w0, 0	// v,
	mov	w3, 0	// col,
	add	x1, sp, 672	// tmp3576,,
	b	.L914		//
.L895:
	cmp	w19, 0	// _8,
	ble	.L1294		//,
// src/cpp/cnn_internals.cpp:382:             vst1q_f32(tmp, acc[t][v]);
	ldr	q31, [x21]	// _1758, MEM[(__Float32x4_t *)_1651]
// src/cpp/cnn_internals.cpp:378:         int v = 0, col = 0;
	mov	w7, 0	// col,
	b	.L899		//
.L1370:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	ldr	w0, [sp, 108]	//, %sfp
	cmp	w0, 0	// _1,
	bgt	.L901		//,
	b	.L918		//
.L1068:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	mov	w5, 0	// niters_vector_mult_vf.1602,
	sxtw	x2, w8	// _3996, ivtmp.2276
	b	.L872		//
.L1372:
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	ldr	x0, [sp, 184]	// _2604, %sfp
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	w19, w11, 1	// oh, oh,
// src/cpp/cnn_internals.cpp:412:         for (int oh = 0; oh < output_h; ++oh) {            // oh above oc
	add	x8, x8, x3	// ivtmp.2246, ivtmp.2246, _6
	add	x21, x21, x0	// ivtmp.2245, ivtmp.2245, _2604
	ldr	w0, [sp, 180]	//, %sfp
	cmp	w0, w19	// _7, oh
	bne	.L902		//,
	ldr	x16, [sp, 216]	// ivtmp.2251, %sfp
	mov	w28, w23	// _8, _8
	mov	w19, w26	// _3, _3
	ldp	x3, x10, [sp, 264]	// _1581, _490,
	mov	x13, x1	//,
	mov	x25, x12	// _1524, _1524
	mov	w20, w24	// _2, _2
	mov	w8, w7	// n_full, n_full
	ldp	x23, x11, [sp, 384]	// _1824, _1840,
	ldr	w22, [sp, 368]	//, %sfp
	ldr	w21, [sp, 376]	//, %sfp
	b	.L903		//
.L616:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:413: 	{ *__to = *__from; }
	ldr	s31, [x27]	// tmp5813, MEM[(const float *)_49]
	str	s31, [x0]	// tmp5813, *_167
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:413: 	{ *__to = *__from; }
	b	.L617		//
	.p2align 2,,3
.L1369:
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	ldr	w4, [sp, 176]	//, %sfp
	smull	x7, w20, w19	// _5703, _2, _3
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w2, 0	// _1,
	and	w2, w2, 1	// tmp3104, _1,
	csneg	w2, w2, w2, ge	// _1548, tmp3104, tmp3104,
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	smull	x9, w4, w8	// _5739, _4, n_full
	mul	x7, x7, x9	// _2903, _5703, _5739
	lsl	x10, x7, 2	// _1545, _2903,
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	add	x7, x3, x7, lsl 2	// packed, _1581, _2903,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w2, 1	// _1548,
	bne	.L879		//,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	cmp	w20, 0	// _2,
	ble	.L879		//,
	cmp	w0, 0	// ksz,
	ble	.L879		//,
	add	x1, x1, x10	// ivtmp.2261, _1526, _1545
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w9, 0	// ic,
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	ubfiz	x2, x0, 2, 32	// _4976, ksz,,
	lsl	x10, x24, 2	// _1859, _5871,
.L880:
	mov	x0, x7	//, packed
	str	x1, [sp, 112]	// ivtmp.2261, %sfp
	str	x2, [sp, 256]	// _4976, %sfp
	str	w9, [sp, 264]	// ic, %sfp
	str	w8, [sp, 272]	// n_full, %sfp
	str	x3, [sp, 280]	// _1581, %sfp
	str	x10, [sp, 304]	// _1859, %sfp
	bl	memmove		//
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldr	w9, [sp, 264]	//, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldr	x1, [sp, 112]	// ivtmp.2261, %sfp
	ldr	x2, [sp, 256]	// _4976, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w9, w9, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldr	x10, [sp, 304]	// _1859, %sfp
	cmp	w20, w9	// _2, ic
	ldr	x3, [sp, 280]	// _1581, %sfp
	add	x7, x0, x2	// packed, packed, _4976
	ldr	w8, [sp, 272]	//, %sfp
	add	x1, x1, x10	// ivtmp.2261, ivtmp.2261, _1859
	bne	.L880		//,
	b	.L879		//
.L1067:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	mov	x25, 4	// _1524,
	b	.L869		//
.L1363:
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	smull	x0, w20, w19	// _1725, _2, _3
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	sxtw	x1, w9	// _214, n_full
	ldr	w4, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	mul	x0, x0, x1	// _5896, _1725, _214
	ldr	x1, [sp, 280]	// _197, %sfp
	negs	w2, w4	// tmp2237, _1
	and	w8, w4, 3	// tmp2238, _1,
	and	w2, w2, 3	// tmp2239, tmp2237,
	csneg	w15, w8, w2, mi	// _224, tmp2238, tmp2239,
	mul	x18, x0, x1	// _221, _5896, _197
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	add	x13, x3, x18	// packed, _257, _221
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w15, 0	// _224,
	ble	.L638		//,
	cmp	w20, 0	// _2,
	ble	.L638		//,
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	sub	w2, w23, #1	// _5463, ksz,
	sxtw	x0, w7	// _111, wcsz
	add	x2, x2, 1	// _5462, _5463,
	mov	x26, x0	// _111, _111
	mov	x11, 0	// ivtmp.1801,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w14, 0	// tb,
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	lsl	x2, x2, 2	// _5461, _5462,
.L639:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w10, 0	// ivtmp.1797,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w8, 0	// ic,
.L643:
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w23, 0	// ksz,
	bgt	.L640		//,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w8, w8, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w10, w10, w23	// ivtmp.1797, ivtmp.1797, ksz
	cmp	w20, w8	// _2, ic
	bne	.L643		//,
.L641:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	w14, w14, 1	// tb, tb,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	x11, x11, x26	// ivtmp.1801, ivtmp.1801, _111
	cmp	w15, w14	// _224, tb
	bne	.L639		//,
	b	.L638		//
	.p2align 2,,3
.L640:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	add	x1, x11, w10, sxtw	// _3134, ivtmp.1801, ivtmp.1797
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	add	x1, x18, x1, lsl 2	// _5459, _221, _3134,
	mov	x0, x13	//, packed
	stp	x2, x18, [sp, 256]	// _5461, _221,
	add	x1, x22, x1	//, _202, _5459
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	str	x11, [sp, 288]	// ivtmp.1801, %sfp
	str	w10, [sp, 304]	// ivtmp.1797, %sfp
	str	w9, [sp, 312]	// n_full, %sfp
	str	w15, [sp, 320]	// _224, %sfp
	str	x3, [sp, 328]	// _257, %sfp
	str	w7, [sp, 336]	// wcsz, %sfp
	str	w12, [sp, 344]	// tiles, %sfp
	str	w14, [sp, 352]	// tb, %sfp
	str	w8, [sp, 360]	// ic, %sfp
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	bl	memmove		//
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldr	w8, [sp, 360]	//, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldp	x2, x18, [sp, 256]	// _5461, _221,
	ldr	w10, [sp, 304]	//, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w8, w8, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	cmp	w20, w8	// _2, ic
	add	x13, x0, x2	// packed, packed, _5461
	ldr	x11, [sp, 288]	// ivtmp.1801, %sfp
	ldr	x3, [sp, 328]	// _257, %sfp
	add	w10, w10, w23	// ivtmp.1797, ivtmp.1797, ksz
	ldr	w9, [sp, 312]	//, %sfp
	ldr	w15, [sp, 320]	//, %sfp
	ldr	w7, [sp, 336]	//, %sfp
	ldr	w12, [sp, 344]	//, %sfp
	ldr	w14, [sp, 352]	//, %sfp
	bne	.L640		//,
	b	.L641		//
.L1077:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	mov	x26, 4	// _1847,
	b	.L975		//
.L1337:
// src/cpp/cnn_internals.cpp:420:             for (int r = 0; oc0 < out_channels; ++oc0, ++r)
	mov	w21, 0	// oc0,
	add	x25, sp, 672	// tmp3576,,
	b	.L846		//
.L1318:
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	ldrsw	x1, [sp, 256]	// _1206, %sfp
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	smull	x0, w20, w19	// _6100, _2, _3
	mov	w9, 21846	// tmp2903,
	movk	w9, 0x5555, lsl 16	// tmp2903,,
	ldr	w2, [sp, 108]	//, %sfp
	mul	x0, x0, x1	// _4326, _6100, _1206
	ldr	x1, [sp, 360]	// _5700, %sfp
	smull	x9, w2, w9	// tmp2902, _1, tmp2903
	lsr	x9, x9, 32	// tmp2904, tmp2902,
	mul	x15, x0, x1	// _1213, _4326, _5700
	sub	w9, w9, w16	// _1216, tmp2904, tmp3581
	add	w9, w9, w9, lsl 1	// tmp2908, _1216, _1216,
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	add	x13, x23, x15	// packed, _1249, _1213
	sub	w9, w2, w9	// _1216, _1, tmp2908
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w9, 0	// _1216,
	ble	.L820		//,
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	sub	w2, w25, #1	// _5117, ksz,
	sxtw	x18, w22	// _1527, wcsz
	add	x2, x2, 1	// _5111, _5117,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	x12, 0	// ivtmp.2152,
	mov	w14, 0	// tb,
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	lsl	x2, x2, 2	// _5110, _5111,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	cmp	w20, 0	// _2,
	ble	.L820		//,
.L821:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w11, 0	// ivtmp.2148,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w10, 0	// ic,
.L825:
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w25, 0	// ksz,
	bgt	.L822		//,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w10, w10, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w11, w11, w25	// ivtmp.2148, ivtmp.2148, ksz
	cmp	w20, w10	// _2, ic
	bne	.L825		//,
.L823:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	w14, w14, 1	// tb, tb,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	x12, x12, x18	// ivtmp.2152, ivtmp.2152, _1527
	cmp	w9, w14	// _1216, tb
	bne	.L821		//,
	b	.L820		//
	.p2align 2,,3
.L822:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	add	x1, x12, w11, sxtw	// _3735, ivtmp.2152, ivtmp.2148
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	add	x1, x15, x1, lsl 2	// _5107, _1213, _3735,
	mov	x0, x13	//, packed
	str	x8, [sp, 112]	// _1194, %sfp
	stp	x2, x15, [sp, 264]	// _5110, _1213,
	add	x1, x8, x1	//, _1194, _5107
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	str	w11, [sp, 280]	// ivtmp.2148, %sfp
	str	w10, [sp, 288]	// ic, %sfp
	stp	x12, x7, [sp, 304]	// ivtmp.2152, _1192,
	str	w9, [sp, 320]	// _1216, %sfp
	str	w3, [sp, 328]	// tiles, %sfp
	str	w14, [sp, 336]	// tb, %sfp
	str	x18, [sp, 344]	// _1527, %sfp
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	bl	memmove		//
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldr	w10, [sp, 288]	//, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldp	x2, x15, [sp, 264]	// _5110, _1213,
	ldp	x12, x7, [sp, 304]	// ivtmp.2152, _1192,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w10, w10, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	cmp	w20, w10	// _2, ic
	add	x13, x0, x2	// packed, packed, _5110
	ldr	w11, [sp, 280]	//, %sfp
	ldr	x8, [sp, 112]	// _1194, %sfp
	ldr	x18, [sp, 344]	// _1527, %sfp
	add	w11, w11, w25	// ivtmp.2148, ivtmp.2148, ksz
	ldr	w9, [sp, 320]	//, %sfp
	ldr	w3, [sp, 328]	//, %sfp
	ldr	w14, [sp, 336]	//, %sfp
	bne	.L822		//,
	b	.L823		//
.L1323:
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	smull	x0, w20, w19	// _5778, _2, _3
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	sxtw	x1, w7	// _542, n_full
	ldr	w4, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	mul	x0, x0, x1	// _3984, _5778, _542
	ldr	x1, [sp, 256]	// _563, %sfp
	negs	w2, w4	// tmp2439, _1
	and	w9, w4, 3	// tmp2440, _1,
	and	w2, w2, 3	// tmp2441, tmp2439,
	csneg	w15, w9, w2, mi	// _552, tmp2440, tmp2441,
	mul	x18, x0, x1	// _549, _3984, _563
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	ldr	x0, [sp, 112]	// _585, %sfp
	add	x13, x0, x18	// packed, _585, _549
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w15, 0	// _552,
	ble	.L696		//,
	cmp	w20, 0	// _2,
	ble	.L696		//,
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	sub	w2, w23, #1	// _5357, ksz,
	sxtw	x0, w8	// _635, wcsz
	add	x2, x2, 1	// _5356, _5357,
	mov	x25, x0	// _635, _635
	mov	x11, 0	// ivtmp.1932,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w14, 0	// tb,
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	lsl	x2, x2, 2	// _5355, _5356,
.L697:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w10, 0	// ivtmp.1928,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w9, 0	// ic,
.L701:
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w23, 0	// ksz,
	bgt	.L698		//,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w9, w9, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w10, w10, w23	// ivtmp.1928, ivtmp.1928, ksz
	cmp	w20, w9	// _2, ic
	bne	.L701		//,
.L699:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	w14, w14, 1	// tb, tb,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	x11, x11, x25	// ivtmp.1932, ivtmp.1932, _635
	cmp	w15, w14	// _552, tb
	bne	.L697		//,
	b	.L696		//
	.p2align 2,,3
.L698:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	add	x1, x11, w10, sxtw	// _3361, ivtmp.1932, ivtmp.1928
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	add	x1, x18, x1, lsl 2	// _5348, _549, _3361,
	mov	x0, x13	//, packed
	stp	x2, x18, [sp, 272]	// _5355, _549,
	add	x1, x22, x1	//, _530, _5348
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	str	x11, [sp, 288]	// ivtmp.1932, %sfp
	str	w10, [sp, 304]	// ivtmp.1928, %sfp
	str	x3, [sp, 312]	// _528, %sfp
	str	w7, [sp, 320]	// n_full, %sfp
	str	w15, [sp, 328]	// _552, %sfp
	str	w8, [sp, 336]	// wcsz, %sfp
	str	w12, [sp, 344]	// tiles, %sfp
	str	w14, [sp, 352]	// tb, %sfp
	str	w9, [sp, 360]	// ic, %sfp
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	bl	memmove		//
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldr	w9, [sp, 360]	//, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldp	x2, x18, [sp, 272]	// _5355, _549,
	ldr	w10, [sp, 304]	//, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w9, w9, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	cmp	w20, w9	// _2, ic
	add	x13, x0, x2	// packed, packed, _5355
	ldr	x11, [sp, 288]	// ivtmp.1932, %sfp
	ldr	x3, [sp, 312]	// _528, %sfp
	add	w10, w10, w23	// ivtmp.1928, ivtmp.1928, ksz
	ldr	w7, [sp, 320]	//, %sfp
	ldr	w15, [sp, 328]	//, %sfp
	ldr	w8, [sp, 336]	//, %sfp
	ldr	w12, [sp, 344]	//, %sfp
	ldr	w14, [sp, 352]	//, %sfp
	bne	.L698		//,
	b	.L699		//
.L1049:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	mov	x3, 4	// _528,
	b	.L686		//
.L1061:
	mov	x7, 4	// _1192,
	b	.L810		//
.L1056:
	mov	x25, 4	// _860,
	b	.L753		//
.L1347:
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	ldr	w2, [sp, 176]	//, %sfp
	smull	x9, w20, w19	// _4109, _2, _3
	smull	x8, w2, w11	// _3682, _4, n_full
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	ldr	w2, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	mul	x8, x8, x9	// _3015, _3682, _4109
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w2, 0	// _1,
	and	w2, w2, 1	// tmp3263, _1,
	csneg	w2, w2, w2, ge	// _1880, tmp3263, tmp3263,
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	lsl	x10, x8, 2	// _1877, _3015,
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	add	x8, x3, x8, lsl 2	// packed, iftmp.18_1843, _3015,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w2, 1	// _1880,
	bne	.L932		//,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	cmp	w20, 0	// _2,
	ble	.L932		//,
	cmp	w0, 0	// ksz,
	ble	.L932		//,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w9, 0	// ic,
	add	x1, x1, x10	// ivtmp.2366, _1858, _1877
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	ubfiz	x2, x0, 2, 32	// _4830, ksz,,
	lsl	x24, x25, 2	// _2248, _5895,
.L933:
	mov	x0, x8	//, packed
	str	x1, [sp, 112]	// ivtmp.2366, %sfp
	stp	x2, x3, [sp, 256]	// _4830, iftmp.18_1843,
	str	x7, [sp, 272]	// _1845, %sfp
	str	w11, [sp, 280]	// n_full, %sfp
	str	w9, [sp, 304]	// ic, %sfp
	bl	memmove		//
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldr	w9, [sp, 304]	//, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldp	x2, x3, [sp, 256]	// _4830, iftmp.18_1843,
	ldr	x1, [sp, 112]	// ivtmp.2366, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w9, w9, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	cmp	w20, w9	// _2, ic
	add	x8, x0, x2	// packed, packed, _4830
	ldr	x7, [sp, 272]	// _1845, %sfp
	ldr	w11, [sp, 280]	//, %sfp
	add	x1, x1, x24	// ivtmp.2366, ivtmp.2366, _2248
	bne	.L933		//,
	b	.L932		//
.L1340:
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	ldr	w4, [sp, 176]	//, %sfp
	smull	x7, w20, w19	// _3243, _2, _3
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w2, 0	// _1,
	and	w2, w2, 1	// tmp3425, _1,
	csneg	w2, w2, w2, ge	// _2212, tmp3425, tmp3425,
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	smull	x9, w4, w8	// _4844, _4, n_full
	mul	x7, x7, x9	// _2789, _3243, _4844
	lsl	x10, x7, 2	// _2209, _2789,
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	add	x7, x3, x7, lsl 2	// packed, iftmp.18_1846, _2789,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w2, 1	// _2212,
	bne	.L985		//,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	cmp	w20, 0	// _2,
	ble	.L985		//,
	cmp	w0, 0	// ksz,
	ble	.L985		//,
	add	x1, x1, x10	// ivtmp.2471, _2190, _2209
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w9, 0	// ic,
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	ubfiz	x2, x0, 2, 32	// _4681, ksz,,
	lsl	x10, x24, 2	// _2626, _3070,
.L986:
	mov	x0, x7	//, packed
	str	x1, [sp, 112]	// ivtmp.2471, %sfp
	stp	x2, x3, [sp, 256]	// _4681, iftmp.18_1846,
	str	w8, [sp, 272]	// n_full, %sfp
	str	w9, [sp, 280]	// ic, %sfp
	str	x10, [sp, 304]	// _2626, %sfp
	bl	memmove		//
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldr	w9, [sp, 280]	//, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldp	x2, x3, [sp, 256]	// _4681, iftmp.18_1846,
	ldr	x1, [sp, 112]	// ivtmp.2471, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w9, w9, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	cmp	w20, w9	// _2, ic
	add	x7, x0, x2	// packed, packed, _4681
	ldr	x10, [sp, 304]	// _2626, %sfp
	ldr	w8, [sp, 272]	//, %sfp
	add	x1, x1, x10	// ivtmp.2471, ivtmp.2471, _2626
	bne	.L986		//,
	b	.L985		//
.L1072:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	mov	x7, 4	// _1845,
	b	.L922		//
.L1356:
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	smull	x0, w20, w19	// _328, _2, _3
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	sxtw	x1, w8	// _874, n_full
	ldr	w4, [sp, 108]	//, %sfp
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	mul	x0, x0, x1	// _3724, _328, _874
	ldr	x1, [sp, 376]	// _5781, %sfp
	negs	w2, w4	// tmp2687, _1
	and	w9, w4, 3	// tmp2688, _1,
	and	w2, w2, 3	// tmp2689, tmp2687,
	csneg	w15, w9, w2, mi	// _884, tmp2688, tmp2689,
	mul	x18, x0, x1	// _881, _3724, _5781
// src/cpp/cnn_internals.cpp:473:             pack_weights<1>(weight.data.data() + (std::size_t)n_full * IC * KH * KW,
	add	x13, x3, x18	// packed, _917, _881
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	cmp	w15, 0	// _884,
	ble	.L763		//,
	cmp	w20, 0	// _2,
	ble	.L763		//,
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	sub	w2, w23, #1	// _5261, ksz,
	sxtw	x0, w7	// _1170, wcsz
	add	x2, x2, 1	// _5259, _5261,
	mov	x26, x0	// _1170, _1170
	mov	x11, 0	// ivtmp.2042,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w14, 0	// tb,
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	lsl	x2, x2, 2	// _5258, _5259,
.L764:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	mov	w10, 0	// ivtmp.2038,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	mov	w9, 0	// ic,
.L768:
// src/cpp/cnn_internals.cpp:323:             for (int k = 0; k < ksz; ++k)
	cmp	w23, 0	// ksz,
	bgt	.L765		//,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w9, w9, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w10, w10, w23	// ivtmp.2038, ivtmp.2038, ksz
	cmp	w20, w9	// _2, ic
	bne	.L768		//,
.L766:
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	w14, w14, 1	// tb, tb,
// src/cpp/cnn_internals.cpp:321:     for (int tb = 0; tb < tiles; ++tb) {
	add	x11, x11, x26	// ivtmp.2042, ivtmp.2042, _1170
	cmp	w15, w14	// _884, tb
	bne	.L764		//,
	b	.L763		//
	.p2align 2,,3
.L765:
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	add	x1, x11, w10, sxtw	// _3552, ivtmp.2042, ivtmp.2038
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	add	x1, x18, x1, lsl 2	// _5249, _881, _3552,
	mov	x0, x13	//, packed
	str	x2, [sp, 112]	// _5258, %sfp
	str	x18, [sp, 256]	// _881, %sfp
	add	x1, x22, x1	//, _862, _5249
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	str	w10, [sp, 272]	// ivtmp.2038, %sfp
	str	x11, [sp, 280]	// ivtmp.2042, %sfp
	str	w9, [sp, 288]	// ic, %sfp
	str	w14, [sp, 304]	// tb, %sfp
	str	w8, [sp, 312]	// n_full, %sfp
	str	w15, [sp, 320]	// _884, %sfp
	str	x3, [sp, 328]	// _917, %sfp
	str	w7, [sp, 336]	// wcsz, %sfp
	str	w12, [sp, 344]	// tiles, %sfp
// src/cpp/cnn_internals.cpp:325:                     *packed++ = w[(std::size_t)(tb * OC_T + t) * wcsz + ic * ksz + k];
	bl	memmove		//
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldr	w9, [sp, 288]	//, %sfp
	ldr	x2, [sp, 112]	// _5258, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	ldr	w10, [sp, 272]	//, %sfp
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	add	w9, w9, 1	// ic, ic,
// src/cpp/cnn_internals.cpp:322:         for (int ic = 0; ic < in_channels; ++ic)
	cmp	w20, w9	// _2, ic
	ldr	x18, [sp, 256]	// _881, %sfp
	add	x13, x0, x2	// packed, packed, _5258
	ldr	x11, [sp, 280]	// ivtmp.2042, %sfp
	add	w10, w10, w23	// ivtmp.2038, ivtmp.2038, ksz
	ldr	x3, [sp, 328]	// _917, %sfp
	ldr	w14, [sp, 304]	//, %sfp
	ldr	w8, [sp, 312]	//, %sfp
	ldr	w15, [sp, 320]	//, %sfp
	ldr	w7, [sp, 336]	//, %sfp
	ldr	w12, [sp, 344]	//, %sfp
	bne	.L765		//,
	b	.L766		//
.L1046:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	mov	x25, 4	// _200,
	b	.L628		//
.L1313:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1920: 	  __throw_length_error(
	adrp	x0, .LC0	// tmp3068,
	add	x0, x0, :lo12:.LC0	//, tmp3068,
	bl	_ZSt20__throw_length_errorPKc		//
.LEHE3:
.L1088:
// src/cpp/cnn_internals.cpp:456:         in_pad.resize(numel + slack, 0.0f);
	mov	x19, x0	// tmp3549, tmp3867
.L615:
	mov	x0, x19	//, tmp3549
.LEHB4:
	bl	_Unwind_Resume		//
.LEHE4:
.L1314:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1920: 	  __throw_length_error(
	adrp	x0, .LC0	// tmp2178,
	add	x0, x0, :lo12:.LC0	//, tmp2178,
.LEHB5:
	bl	_ZSt20__throw_length_errorPKc		//
.LEHE5:
.L1338:
	adrp	x0, .LC0	// tmp3389,
	add	x0, x0, :lo12:.LC0	//, tmp3389,
.LEHB6:
	bl	_ZSt20__throw_length_errorPKc		//
.L1316:
	adrp	x0, .LC0	// tmp2392,
	add	x0, x0, :lo12:.LC0	//, tmp2392,
	bl	_ZSt20__throw_length_errorPKc		//
.L1089:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	mov	x19, x0	// tmp3549, tmp3877
.L1028:
	ldr	x0, [sp, 192]	// in_pad$_M_start, %sfp
	sub	x1, x23, x0	// _94, in_pad$_M_end_of_storage, in_pad$_M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L615	// in_pad$_M_start,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/alloc_traits.h:513:       { __a.deallocate(__p, __n); }
	b	.L615		//
.L1315:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1920: 	  __throw_length_error(
	adrp	x0, .LC0	// tmp2851,
	add	x0, x0, :lo12:.LC0	//, tmp2851,
	bl	_ZSt20__throw_length_errorPKc		//
.L1339:
	adrp	x0, .LC0	// tmp3227,
	add	x0, x0, :lo12:.LC0	//, tmp3227,
	bl	_ZSt20__throw_length_errorPKc		//
.L1087:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1037: 	  _M_fill_insert(end(), __new_size - size(), __x);
	ldr	x1, [sp, 640]	// in_pad$_M_start, MEM <float *> [(struct vector *)&in_pad]
	mov	x19, x0	// tmp3549, tmp3876
	ldr	x23, [sp, 656]	// in_pad$_M_end_of_storage, MEM <float *> [(struct vector *)&in_pad + 16B]
	str	x1, [sp, 192]	// in_pad$_M_start, %sfp
	b	.L1028		//
.L1354:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1920: 	  __throw_length_error(
	adrp	x0, .LC0	// tmp2640,
	add	x0, x0, :lo12:.LC0	//, tmp2640,
	bl	_ZSt20__throw_length_errorPKc		//
.L1312:
	adrp	x0, .LC0	// tmp2190,
	add	x0, x0, :lo12:.LC0	//, tmp2190,
	bl	_ZSt20__throw_length_errorPKc		//
.LEHE6:
	.cfi_endproc
.LFE5452:
	.section	.gcc_except_table,"a",@progbits
.LLSDA5452:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE5452-.LLSDACSB5452
.LLSDACSB5452:
	.uleb128 .LEHB0-.LFB5452
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L1089-.LFB5452
	.uleb128 0
	.uleb128 .LEHB1-.LFB5452
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L1088-.LFB5452
	.uleb128 0
	.uleb128 .LEHB2-.LFB5452
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L1087-.LFB5452
	.uleb128 0
	.uleb128 .LEHB3-.LFB5452
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L1089-.LFB5452
	.uleb128 0
	.uleb128 .LEHB4-.LFB5452
	.uleb128 .LEHE4-.LEHB4
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB5-.LFB5452
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L1088-.LFB5452
	.uleb128 0
	.uleb128 .LEHB6-.LFB5452
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L1089-.LFB5452
	.uleb128 0
.LLSDACSE5452:
	.text
	.size	_Z34conv2d_forward_specialized_blockedRK6TensorS1_S1_RS_, .-_Z34conv2d_forward_specialized_blockedRK6TensorS1_S1_RS_
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align	3
	.type	DW.ref.__gxx_personality_v0, %object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.xword	__gxx_personality_v0
	.global	__gxx_personality_v0
	.ident	"GCC: (Spack GCC) 14.2.0"
	.section	.note.GNU-stack,"",@progbits
