	.arch armv9-a+crc+i8mm+bf16+sve2-aes+sve2-bitperm+sve2-sha3+sve2-sm4+memtag+profile+nossbs
	.file	"benchmark.cpp"
// GNU C++17 (Spack GCC) version 14.2.0 (aarch64-unknown-linux-gnu)
//	compiled by GNU C version 14.2.0, GMP version 6.3.0, MPFR version 4.2.1-p1, MPC version 1.3.1, isl version none
// GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
// options passed: -mlittle-endian -mabi=lp64 -mcpu=neoverse-v2+crc+sve2-aes+sve2-sha3+sve2-sm4+norng+nossbs -O3 -std=c++17 -ffast-math -ftree-vectorize -fopt-info-vec-optimized
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.align	2
	.p2align 5,,15
	.type	_ZL11_cmp_doublePKvS0_, %function
_ZL11_cmp_doublePKvS0_:
.LFB81:
	.cfi_startproc
// src/cpp/include/timing.h:96:     double da = *(const double *)a, db = *(const double *)b;
	ldr	d31, [x0]	//, MEM[(const double *)a_6(D)]
// src/cpp/include/timing.h:96:     double da = *(const double *)a, db = *(const double *)b;
	ldr	d30, [x1]	//, MEM[(const double *)b_8(D)]
// src/cpp/include/timing.h:97:     return (da > db) - (da < db);
	fcmpe	d31, d30	//,
	cset	w1, gt	// _1,
// src/cpp/include/timing.h:97:     return (da > db) - (da < db);
	cset	w0, mi	// _3,
// src/cpp/include/timing.h:98: }
	sub	w0, w1, w0	//, _1, _3
	ret	
	.cfi_endproc
.LFE81:
	.size	_ZL11_cmp_doublePKvS0_, .-_ZL11_cmp_doublePKvS0_
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC9:
	.string	"cannot create std::vector larger than max_size()"
	.text
	.align	2
	.p2align 5,,15
	.type	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0, %function
_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0:
.LFB8387:
	.cfi_startproc
	stp	x29, x30, [sp, -64]!	//,,,
	.cfi_def_cfa_offset 64
	.cfi_offset 29, -64
	.cfi_offset 30, -56
	mov	x29, sp	//,
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -48
	.cfi_offset 20, -40
	mov	x20, x1	// __n, tmp125
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	mov	x1, 1152921504606846975	// tmp113,
	cmp	x20, x1	// __n, tmp113
	bhi	.L12		//,
	mov	x19, x0	// this, tmp124
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:99: 	: _M_start(), _M_finish(), _M_end_of_storage()
	stp	xzr, xzr, [x0]	//,,* this
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:99: 	: _M_start(), _M_finish(), _M_end_of_storage()
	str	xzr, [x0, 16]	//, MEM[(struct _Vector_impl_data *)this_1(D)]._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	cbz	x20, .L13	// __n,
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 22, -24
	.cfi_offset 21, -32
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	lsl	x22, x20, 3	// _26, __n,
	mov	x0, x22	//, _26
	str	x23, [sp, 48]	//,
	.cfi_offset 23, -16
	bl	_Znwm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	add	x23, x0, x22	// _15, tmp116, _26
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	mov	x3, x0	// __first, tmp116
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:398: 	this->_M_impl._M_start = this->_M_allocate(__n);
	str	x0, [x19]	// tmp116, MEM[(struct _Vector_base *)this_1(D)]._M_impl.D.100808._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x21, x0	// tmp116, tmp126
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	str	xzr, [x3], 8	//, *_27
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	str	x23, [x19, 16]	// _15, MEM[(struct _Vector_base *)this_1(D)]._M_impl.D.100808._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	cmp	x20, 1	// __n,
	beq	.L11		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	sub	x2, x22, #8	//, _26,
	mov	x0, x3	//, __first
	mov	w1, 0	//,
	bl	memset		//
	add	x3, x0, x23	// _24, __first, _15
	ldr	x23, [sp, 48]	//,
	.cfi_restore 23
	sub	x3, x3, x21	// _44, _24, tmp116
	ldp	x21, x22, [sp, 32]	//,,
	.cfi_restore 22
	.cfi_restore 21
	sub	x3, x3, #8	// __first, _44,
.L6:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1719: 	this->_M_impl._M_finish =
	str	x3, [x19, 8]	// __first, *this_1(D).D.101473._M_impl.D.100808._M_finish
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:558:       { _M_default_initialize(__n); }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x29, x30, [sp], 64	//,,,
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
.L13:
	.cfi_restore_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	mov	x3, 0	// __first,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:398: 	this->_M_impl._M_start = this->_M_allocate(__n);
	str	xzr, [x0]	//, MEM[(struct _Vector_base *)this_1(D)]._M_impl.D.100808._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	str	xzr, [x0, 16]	//, MEM[(struct _Vector_base *)this_1(D)]._M_impl.D.100808._M_end_of_storage
	b	.L6		//
.L11:
	.cfi_offset 21, -32
	.cfi_offset 22, -24
	.cfi_offset 23, -16
	ldr	x23, [sp, 48]	//,
	.cfi_restore 23
	ldp	x21, x22, [sp, 32]	//,,
	.cfi_restore 22
	.cfi_restore 21
	b	.L6		//
.L12:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1920: 	  __throw_length_error(
	adrp	x0, .LC9	// tmp115,
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 22, -24
	.cfi_offset 21, -32
	add	x0, x0, :lo12:.LC9	//, tmp115,
	str	x23, [sp, 48]	//,
	.cfi_offset 23, -16
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1920: 	  __throw_length_error(
	bl	_ZSt20__throw_length_errorPKc		//
	.cfi_endproc
.LFE8387:
	.size	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0, .-_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0
	.align	2
	.p2align 5,,15
	.type	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0, %function
_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0:
.LFB8390:
	.cfi_startproc
	stp	x29, x30, [sp, -64]!	//,,,
	.cfi_def_cfa_offset 64
	.cfi_offset 29, -64
	.cfi_offset 30, -56
	mov	x29, sp	//,
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -48
	.cfi_offset 20, -40
	mov	x20, x1	// __n, tmp125
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	mov	x1, 2305843009213693951	// tmp113,
	cmp	x20, x1	// __n, tmp113
	bhi	.L23		//,
	mov	x19, x0	// this, tmp124
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:99: 	: _M_start(), _M_finish(), _M_end_of_storage()
	stp	xzr, xzr, [x0]	//,,* this
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:99: 	: _M_start(), _M_finish(), _M_end_of_storage()
	str	xzr, [x0, 16]	//, MEM[(struct _Vector_impl_data *)this_1(D)]._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	cbz	x20, .L24	// __n,
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 22, -24
	.cfi_offset 21, -32
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	lsl	x22, x20, 2	// _26, __n,
	mov	x0, x22	//, _26
	str	x23, [sp, 48]	//,
	.cfi_offset 23, -16
	bl	_Znwm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	add	x23, x0, x22	// _15, tmp116, _26
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	mov	x3, x0	// __first, tmp116
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:398: 	this->_M_impl._M_start = this->_M_allocate(__n);
	str	x0, [x19]	// tmp116, MEM[(struct _Vector_base *)this_1(D)]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x21, x0	// tmp116, tmp126
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	str	wzr, [x3], 4	//, *_27
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	str	x23, [x19, 16]	// _15, MEM[(struct _Vector_base *)this_1(D)]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	cmp	x20, 1	// __n,
	beq	.L22		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	sub	x2, x22, #4	//, _26,
	mov	x0, x3	//, __first
	mov	w1, 0	//,
	bl	memset		//
	add	x3, x0, x23	// _24, __first, _15
	ldr	x23, [sp, 48]	//,
	.cfi_restore 23
	sub	x3, x3, x21	// _44, _24, tmp116
	ldp	x21, x22, [sp, 32]	//,,
	.cfi_restore 22
	.cfi_restore 21
	sub	x3, x3, #4	// __first, _44,
.L17:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1719: 	this->_M_impl._M_finish =
	str	x3, [x19, 8]	// __first, *this_1(D).D.57594._M_impl.D.56895._M_finish
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:558:       { _M_default_initialize(__n); }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x29, x30, [sp], 64	//,,,
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
.L24:
	.cfi_restore_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	mov	x3, 0	// __first,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:398: 	this->_M_impl._M_start = this->_M_allocate(__n);
	str	xzr, [x0]	//, MEM[(struct _Vector_base *)this_1(D)]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	str	xzr, [x0, 16]	//, MEM[(struct _Vector_base *)this_1(D)]._M_impl.D.56895._M_end_of_storage
	b	.L17		//
.L22:
	.cfi_offset 21, -32
	.cfi_offset 22, -24
	.cfi_offset 23, -16
	ldr	x23, [sp, 48]	//,
	.cfi_restore 23
	ldp	x21, x22, [sp, 32]	//,,
	.cfi_restore 22
	.cfi_restore 21
	b	.L17		//
.L23:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1920: 	  __throw_length_error(
	adrp	x0, .LC9	// tmp115,
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 22, -24
	.cfi_offset 21, -32
	add	x0, x0, :lo12:.LC9	//, tmp115,
	str	x23, [sp, 48]	//,
	.cfi_offset 23, -16
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1920: 	  __throw_length_error(
	bl	_ZSt20__throw_length_errorPKc		//
	.cfi_endproc
.LFE8390:
	.size	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0, .-_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0
	.section	.rodata.str1.8
	.align	3
.LC10:
	.string	"basic_string::append"
	.text
	.align	2
	.p2align 5,,15
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKcm.isra.0, %function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKcm.isra.0:
.LFB8393:
	.cfi_startproc
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:404: 	if (this->max_size() - (this->size() - __n1) < __n2)
	ldr	x4, [x0, 8]	// MEM[(const struct basic_string *)this_1(D)]._M_string_length, MEM[(const struct basic_string *)this_1(D)]._M_string_length
	mov	x3, 4611686018427387903	// tmp106,
	sub	x3, x3, x4	// _3, tmp106, MEM[(const struct basic_string *)this_1(D)]._M_string_length
	cmp	x3, x2	// _3, tmp111
	bcc	.L30		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:1470: 	return _M_append(__s, __n);
	b	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_appendEPKcm		//
.L30:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:405: 	  __throw_length_error(__N(__s));
	adrp	x0, .LC10	// tmp109,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:1466:       append(const _CharT* __s, size_type __n)
	stp	x29, x30, [sp, -16]!	//,,,
	.cfi_def_cfa_offset 16
	.cfi_offset 29, -16
	.cfi_offset 30, -8
	mov	x29, sp	//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:405: 	  __throw_length_error(__N(__s));
	add	x0, x0, :lo12:.LC10	//, tmp109,
	bl	_ZSt20__throw_length_errorPKc		//
	.cfi_endproc
.LFE8393:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKcm.isra.0, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKcm.isra.0
	.align	2
	.p2align 5,,15
	.type	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_.isra.0, %function
_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_.isra.0:
.LFB8394:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA8394
	stp	x29, x30, [sp, -64]!	//,,,
	.cfi_def_cfa_offset 64
	.cfi_offset 29, -64
	.cfi_offset 30, -56
	mov	x29, sp	//,
	str	x23, [sp, 48]	//,
	.cfi_offset 23, -16
	mov	x23, x0	// tmp114, ISRA.477
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x2	//, __rhs
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3652:     operator+(const basic_string<_CharT, _Traits, _Alloc>& __lhs,
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -48
	.cfi_offset 20, -40
	mov	x19, x8	// <retval>, tmp113
	mov	x20, x2	// __rhs, tmp116
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 21, -32
	.cfi_offset 22, -24
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3652:     operator+(const basic_string<_CharT, _Traits, _Alloc>& __lhs,
	mov	x22, x1	// ISRA.478, tmp115
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:235: 	return std::pointer_traits<pointer>::pointer_to(*_M_local_buf);
	add	x1, x19, 16	// _12, <retval>,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x21, x0	// _5, tmp117
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x19, 16]	//, MEM[(char_type &)_1(D) + 16]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3582:       __str.reserve(__lhs_len + __rhs_len);
	mov	x0, x19	//, <retval>
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	stp	x1, xzr, [x19]	// _12,,* <retval>
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3582:       __str.reserve(__lhs_len + __rhs_len);
	add	x1, x21, x22	//, _5, ISRA.478
.LEHB0:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3583:       __str.append(__lhs, __lhs_len);
	mov	x2, x22	//, ISRA.478
	mov	x1, x23	//, ISRA.477
	mov	x0, x19	//, <retval>
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKcm.isra.0		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3584:       __str.append(__rhs, __rhs_len);
	mov	x2, x21	//, _5
	mov	x1, x20	//, __rhs
	mov	x0, x19	//, <retval>
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKcm.isra.0		//
.LEHE0:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3660:     }
	ldr	x23, [sp, 48]	//,
	mov	x0, x19	//, <retval>
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x29, x30, [sp], 64	//,,,
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 23
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
.L34:
	.cfi_restore_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x1, x0	// tmp118,
	mov	x0, x19	//, <retval>
	mov	x19, x1	// tmp111, tmp118
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	mov	x0, x19	//, tmp111
.LEHB1:
	bl	_Unwind_Resume		//
.LEHE1:
	.cfi_endproc
.LFE8394:
	.section	.gcc_except_table,"a",@progbits
.LLSDA8394:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE8394-.LLSDACSB8394
.LLSDACSB8394:
	.uleb128 .LEHB0-.LFB8394
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L34-.LFB8394
	.uleb128 0
	.uleb128 .LEHB1-.LFB8394
	.uleb128 .LEHE1-.LEHB1
	.uleb128 0
	.uleb128 0
.LLSDACSE8394:
	.text
	.size	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_.isra.0, .-_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_.isra.0
	.section	.rodata.str1.8
	.align	3
.LC11:
	.string	"basic_string: construction from null is not valid"
	.text
	.align	2
	.p2align 5,,15
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_.constprop.0, %function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_.constprop.0:
.LFB8396:
	.cfi_startproc
	stp	x29, x30, [sp, -64]!	//,,,
	.cfi_def_cfa_offset 64
	.cfi_offset 29, -64
	.cfi_offset 30, -56
	mov	x29, sp	//,
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 21, -32
	.cfi_offset 22, -24
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:235: 	return std::pointer_traits<pointer>::pointer_to(*_M_local_buf);
	add	x22, x0, 16	// _2, this,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:646:       basic_string(const _CharT* __s, const _Alloc& __a = _Alloc())
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -48
	.cfi_offset 20, -40
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	str	x22, [x0]	// _2, MEM[(struct _Alloc_hider *)this_1(D)]._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:650: 	if (__s == 0)
	cbz	x1, .L47	// __s,
	mov	x19, x0	// this, tmp120
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x1	//, __s
	mov	x21, x1	// __s, tmp121
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:225: 	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
	str	x0, [sp, 56]	// _4, __dnew
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x20, x0	// _4, tmp122
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:227: 	if (__dnew > size_type(_S_local_capacity))
	cmp	x0, 15	// _4,
	bhi	.L48		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:432: 	if (__n == 1)
	cmp	x0, 1	// _4,
	bne	.L41		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldrb	w0, [x21]	//, MEM[(const char_type &)__s_3(D)]
	strb	w0, [x19, 16]	// MEM[(const char_type &)__s_3(D)], MEM[(char_type &)this_1(D) + 16]
.L42:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:251: 	_M_set_length(__dnew);
	ldr	x0, [sp, 56]	// __dnew.121_18, __dnew
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldr	x1, [x19]	// MEM[(const struct basic_string *)this_1(D)]._M_dataplus._M_p, MEM[(const struct basic_string *)this_1(D)]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x0, [x19, 8]	// __dnew.121_18, *this_1(D)._M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x1, x0]	//, MEM[(char_type &)_20]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:655:       }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x29, x30, [sp], 64	//,,,
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
.L41:
	.cfi_restore_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	cbz	x0, .L42	// _4,
	b	.L40		//
.L48:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	add	x1, sp, 56	//,,
	mov	x2, 0	//,
	mov	x0, x19	//, this
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	ldr	x1, [sp, 56]	// __dnew, __dnew
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	mov	x22, x0	// _2, tmp123
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:218:       { _M_dataplus._M_p = __p; }
	str	x0, [x19]	// _2, *this_1(D)._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	str	x1, [x19, 16]	// __dnew, *this_1(D).D.64159._M_allocated_capacity
.L40:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	mov	x2, x20	//, _4
	mov	x1, x21	//, __s
	mov	x0, x22	//, _2
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	b	.L42		//
.L47:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:651: 	  std::__throw_logic_error(__N("basic_string: "
	adrp	x0, .LC11	// tmp110,
	add	x0, x0, :lo12:.LC11	//, tmp110,
	bl	_ZSt19__throw_logic_errorPKc		//
	.cfi_endproc
.LFE8396:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_.constprop.0, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_.constprop.0
	.section	.rodata.str1.8
	.align	3
.LC12:
	.string	"CNN_BENCH_FILTER_EXACT"
	.align	3
.LC13:
	.string	" \t"
	.align	3
.LC14:
	.string	"basic_string::substr"
	.align	3
.LC15:
	.string	"%s: __pos (which is %zu) > this->size() (which is %zu)"
	.text
	.align	2
	.p2align 5,,15
	.type	_ZL29implementation_matches_filterRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKc, %function
_ZL29implementation_matches_filterRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKc:
.LFB6766:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA6766
	stp	x29, x30, [sp, -208]!	//,,,
	.cfi_def_cfa_offset 208
	.cfi_offset 29, -208
	.cfi_offset 30, -200
	mov	x29, sp	//,
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -192
	.cfi_offset 20, -184
// src/cpp/include/utils.h:142:     if (!filter || !*filter) return true;
	cbz	x1, .L109	// filter,
	mov	x19, x1	// filter, tmp352
// src/cpp/include/utils.h:142:     if (!filter || !*filter) return true;
	mov	w20, 1	// _7,
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 22, -168
	.cfi_offset 21, -176
	mov	x22, x0	// name, tmp351
// src/cpp/include/utils.h:142:     if (!filter || !*filter) return true;
	ldrb	w0, [x1]	// *filter_17(D), *filter_17(D)
	cbnz	w0, .L160	// *filter_17(D),
	ldp	x21, x22, [sp, 32]	//,,
	.cfi_restore 22
	.cfi_restore 21
.L126:
// src/cpp/include/utils.h:160: }
	mov	w0, w20	//, _7
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x29, x30, [sp], 208	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L160:
	.cfi_def_cfa_offset 208
	.cfi_offset 19, -192
	.cfi_offset 20, -184
	.cfi_offset 21, -176
	.cfi_offset 22, -168
	.cfi_offset 29, -208
	.cfi_offset 30, -200
// src/cpp/include/utils.h:143:     bool exact = std::getenv("CNN_BENCH_FILTER_EXACT") != nullptr;
	adrp	x0, .LC12	// tmp203,
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 24, -152
	.cfi_offset 23, -160
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	add	x23, sp, 128	// tmp334,,
// src/cpp/include/utils.h:143:     bool exact = std::getenv("CNN_BENCH_FILTER_EXACT") != nullptr;
	add	x0, x0, :lo12:.LC12	//, tmp203,
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 26, -136
	.cfi_offset 25, -144
	str	x27, [sp, 80]	//,
	.cfi_offset 27, -128
// src/cpp/include/utils.h:143:     bool exact = std::getenv("CNN_BENCH_FILTER_EXACT") != nullptr;
	bl	getenv		//
	mov	x24, x0	// tmp353,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x19	//, filter
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	str	x23, [sp, 112]	// tmp334, MEM[(struct _Alloc_hider *)&f]._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:225: 	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
	str	x0, [sp, 176]	// prephitmp_65, MEM[(long unsigned int *)_289]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x21, x0	// prephitmp_65, tmp354
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:227: 	if (__dnew > size_type(_S_local_capacity))
	cmp	x0, 15	// prephitmp_65,
	bhi	.L161		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:432: 	if (__n == 1)
	cmp	x0, 1	// prephitmp_65,
	bne	.L53		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldrb	w0, [x19]	//, MEM[(const char_type &)filter_17(D)]
	strb	w0, [sp, 128]	// MEM[(const char_type &)filter_17(D)], MEM[(char_type &)&f + 16]
.L55:
	mov	x0, x23	// pretmp_5, tmp334
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x21, [sp, 120]	// prephitmp_65, f._M_string_length
// src/cpp/include/utils.h:146:     size_t pos = 0;
	mov	x20, 0	// pos,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x0, x21]	//, MEM[(char_type &)_76]
	cbz	x24, .L151	// _2,
.L169:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3089: 	return this->find_first_not_of(__s, __pos, traits_type::length(__s));
	adrp	x24, .LC13	// tmp349,
	add	x24, x24, :lo12:.LC13	// tmp350, tmp349,
	.p2align 5,,15
.L56:
// src/cpp/include/utils.h:148:         size_t comma = f.find(',', pos);
	mov	x2, x20	//, pos
	mov	w1, 44	//,
	add	x0, sp, 112	// tmp378,,
	bl	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4findEcm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:1077:       { return _M_string_length; }
	ldr	x19, [sp, 120]	// _78, f._M_string_length
// src/cpp/include/utils.h:149:         std::string token = f.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
	cmn	x0, #1	// comma,
// src/cpp/include/utils.h:148:         size_t comma = f.find(',', pos);
	mov	x26, x0	// comma, tmp362
// src/cpp/include/utils.h:149:         std::string token = f.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
	sub	x0, x0, x20	// tmp341, comma, pos
	csinv	x0, x0, xzr, ne	// iftmp.124_9, tmp341,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:393: 	if (__pos > this->size())
	cmp	x20, x19	// pos, _78
	bhi	.L58		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:590: 	const _CharT* __start = __str._M_data()
	ldr	x1, [sp, 112]	// MEM[(const struct basic_string *)&f]._M_dataplus._M_p, MEM[(const struct basic_string *)&f]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:198: 	: allocator_type(std::move(__a)), _M_p(__dat) { }
	add	x21, sp, 160	// tmp336,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:414: 	const bool __testoff =  __off < this->size() - __pos;
	sub	x19, x19, x20	// _83, _78, pos
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:415: 	return __testoff ? __off : this->size() - __pos;
	cmp	x19, x0	// _83, iftmp.124_9
	csel	x19, x19, x0, ls	// prephitmp_63, _83, iftmp.124_9,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:225: 	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
	str	x19, [sp, 176]	// prephitmp_63, MEM[(long unsigned int *)_289]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:198: 	: allocator_type(std::move(__a)), _M_p(__dat) { }
	add	x25, sp, 144	// tmp379,,
	str	x21, [sp, 144]	// tmp336, MEM[(struct _Alloc_hider *)&token]._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:590: 	const _CharT* __start = __str._M_data()
	add	x20, x1, x20	// __start, MEM[(const struct basic_string *)&f]._M_dataplus._M_p, pos
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:227: 	if (__dnew > size_type(_S_local_capacity))
	cmp	x19, 15	// prephitmp_63,
	bhi	.L162		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:432: 	if (__n == 1)
	cmp	x19, 1	// prephitmp_63,
	beq	.L163		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	cbnz	x19, .L164	// prephitmp_63,
.L90:
	mov	x4, x21	// prephitmp_57, tmp336
.L89:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x19, [sp, 152]	// prephitmp_63, token._M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3089: 	return this->find_first_not_of(__s, __pos, traits_type::length(__s));
	mov	x3, 2	//,
	mov	x2, 0	//,
	mov	x1, x24	//, tmp350
	add	x0, sp, 144	// tmp381,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x4, x19]	//, MEM[(char_type &)_94]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3089: 	return this->find_first_not_of(__s, __pos, traits_type::length(__s));
	bl	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE17find_first_not_ofEPKcmm		//
	mov	x20, x0	// tmp364,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3176: 	return this->find_last_not_of(__s, __pos, traits_type::length(__s));
	mov	x1, x24	//, tmp350
	add	x0, sp, 144	// tmp382,,
	mov	x3, 2	//,
	mov	x2, -1	//,
	bl	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16find_last_not_ofEPKcmm		//
// src/cpp/include/utils.h:152:         if (b != std::string::npos) {
	cmn	x20, #1	// _23,
	beq	.L91		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:1077:       { return _M_string_length; }
	ldr	x19, [sp, 152]	// _96, token._M_string_length
// src/cpp/include/utils.h:153:             std::string trimmed = token.substr(b, e - b + 1);
	add	x0, x0, 1	// _161, _25,
	sub	x0, x0, x20	// _4, _161, _23
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:393: 	if (__pos > this->size())
	cmp	x20, x19	// _23, _96
	bhi	.L66		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:590: 	const _CharT* __start = __str._M_data()
	ldr	x1, [sp, 144]	// MEM[(const struct basic_string *)&token]._M_dataplus._M_p, MEM[(const struct basic_string *)&token]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:198: 	: allocator_type(std::move(__a)), _M_p(__dat) { }
	add	x27, sp, 192	// tmp337,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:414: 	const bool __testoff =  __off < this->size() - __pos;
	sub	x19, x19, x20	// _101, _96, _23
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:415: 	return __testoff ? __off : this->size() - __pos;
	cmp	x19, x0	// _101, _4
	csel	x19, x19, x0, ls	// prephitmp_49, _101, _4,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:225: 	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
	str	x19, [sp, 104]	// prephitmp_49, __dnew
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:198: 	: allocator_type(std::move(__a)), _M_p(__dat) { }
	str	x27, [sp, 176]	// tmp337, MEM[(struct _Alloc_hider *)_289]._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:590: 	const _CharT* __start = __str._M_data()
	add	x20, x1, x20	// __start, MEM[(const struct basic_string *)&token]._M_dataplus._M_p, _23
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:227: 	if (__dnew > size_type(_S_local_capacity))
	cmp	x19, 15	// prephitmp_49,
	bhi	.L165		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:432: 	if (__n == 1)
	cmp	x19, 1	// prephitmp_49,
	beq	.L166		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	cbnz	x19, .L167	// prephitmp_49,
.L97:
	mov	x0, x27	// prephitmp_30, tmp337
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x19, [sp, 184]	// prephitmp_49, MEM[(struct basic_string *)_289]._M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x0, x19]	//, MEM[(char_type &)_112]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:1077:       { return _M_string_length; }
	ldr	x2, [x22, 8]	// _113, MEM[(const struct basic_string *)name_31(D)]._M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3760: 	       && !_Traits::compare(__lhs.data(), __rhs.data(), __lhs.size());
	ldp	x19, x0, [sp, 176]	// _257, MEM[(struct basic_string *)_289]._M_string_length,
	cmp	x2, x0	// _113, MEM[(struct basic_string *)_289]._M_string_length
	beq	.L168		//,
.L98:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x19, x27	// _257, tmp337
	beq	.L91		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 192]	// MEM[(struct basic_string *)_289].D.64159._M_allocated_capacity, MEM[(struct basic_string *)_289].D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x0, x19	//, _257
	add	x1, x1, 1	//, MEM[(struct basic_string *)_289].D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L91:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x0, [sp, 144]	// pretmp_130, token._M_dataplus._M_p
// src/cpp/include/utils.h:156:         if (comma == std::string::npos) break;
	cmn	x26, #1	// comma,
	beq	.L80		//,
// src/cpp/include/utils.h:157:         pos = comma + 1;
	add	x20, x26, 1	// pos, comma,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x0, x21	// pretmp_130, tmp336
	beq	.L103		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 160]	// token.D.64159._M_allocated_capacity, token.D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	add	x1, x1, 1	//, token.D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L103:
// src/cpp/include/utils.h:147:     while (pos <= f.size()) {
	ldr	x0, [sp, 120]	// f._M_string_length, f._M_string_length
	cmp	x20, x0	// pos, f._M_string_length
	bls	.L56		//,
.L104:
// src/cpp/include/utils.h:159:     return false;
	mov	w20, 0	// _7,
.L101:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x0, [sp, 112]	// _149, f._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x0, x23	// _149, tmp334
	beq	.L158		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 128]	// f.D.64159._M_allocated_capacity, f.D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	add	x1, x1, 1	//, f.D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	ldr	x27, [sp, 80]	//,
	.cfi_remember_state
	.cfi_restore 27
// src/cpp/include/utils.h:160: }
	mov	w0, w20	//, _7
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	ldp	x21, x22, [sp, 32]	//,,
	.cfi_restore 22
	.cfi_restore 21
	ldp	x23, x24, [sp, 48]	//,,
	.cfi_restore 24
	.cfi_restore 23
	ldp	x25, x26, [sp, 64]	//,,
	.cfi_restore 26
	.cfi_restore 25
// src/cpp/include/utils.h:160: }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x29, x30, [sp], 208	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L164:
	.cfi_restore_state
	mov	x0, x21	// _68, tmp336
.L87:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	mov	x2, x19	//, prephitmp_63
	mov	x1, x20	//, __start
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x4, [sp, 144]	// prephitmp_57, token._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:251: 	_M_set_length(__dnew);
	ldr	x19, [sp, 176]	// prephitmp_63, MEM[(long unsigned int *)_289]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	b	.L89		//
	.p2align 2,,3
.L167:
	mov	x0, x27	// _86, tmp337
.L94:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	mov	x2, x19	//, prephitmp_49
	mov	x1, x20	//, __start
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:251: 	_M_set_length(__dnew);
	ldr	x19, [sp, 104]	// prephitmp_49, __dnew
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x0, [sp, 176]	// prephitmp_30, MEM[(struct basic_string *)_289]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x19, [sp, 184]	// prephitmp_49, MEM[(struct basic_string *)_289]._M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x0, x19]	//, MEM[(char_type &)_112]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:1077:       { return _M_string_length; }
	ldr	x2, [x22, 8]	// _113, MEM[(const struct basic_string *)name_31(D)]._M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3760: 	       && !_Traits::compare(__lhs.data(), __rhs.data(), __lhs.size());
	ldp	x19, x0, [sp, 176]	// _257, MEM[(struct basic_string *)_289]._M_string_length,
	cmp	x2, x0	// _113, MEM[(struct basic_string *)_289]._M_string_length
	bne	.L98		//,
.L168:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:368: 	if (__n == 0)
	cbz	x2, .L74	// _113,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:381: 	return __builtin_memcmp(__s1, __s2, __n);
	ldr	x0, [x22]	//, MEM[(const struct basic_string *)name_31(D)]._M_dataplus._M_p
	mov	x1, x19	//, _257
	bl	memcmp		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3760: 	       && !_Traits::compare(__lhs.data(), __rhs.data(), __lhs.size());
	cbnz	w0, .L98	// tmp367,
.L74:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x19, x27	// _257, tmp337
	beq	.L99		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 192]	// MEM[(struct basic_string *)_289].D.64159._M_allocated_capacity, MEM[(struct basic_string *)_289].D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x0, x19	//, _257
	add	x1, x1, 1	//, MEM[(struct basic_string *)_289].D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L99:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x0, [sp, 144]	// _119, token._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x0, x21	// _119, tmp336
	beq	.L100		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 160]	// token.D.64159._M_allocated_capacity, token.D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	add	x1, x1, 1	//, token.D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L100:
// src/cpp/include/utils.h:154:             if (exact ? (name == trimmed) : contains_substr(name, trimmed)) return true;
	mov	w20, 1	// _7,
	b	.L101		//
	.p2align 2,,3
.L53:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	cbz	x0, .L55	// prephitmp_65,
	mov	x0, x23	// _70, tmp334
	b	.L52		//
	.p2align 2,,3
.L163:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldrb	w0, [x20]	// _91, MEM[(const char_type &)__start_82]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	w0, [sp, 160]	// _91, MEM[(char_type &)&token + 16]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:351:       }
	b	.L90		//
	.p2align 2,,3
.L162:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	mov	x2, 0	//,
	add	x1, sp, 176	//,,
	mov	x0, x25	// tmp380, tmp379
.LEHB2:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm		//
.LEHE2:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	ldr	x1, [sp, 176]	// MEM[(long unsigned int *)_289], MEM[(long unsigned int *)_289]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:218:       { _M_dataplus._M_p = __p; }
	str	x0, [sp, 144]	// _68, token._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	str	x1, [sp, 160]	// MEM[(long unsigned int *)_289], token.D.64159._M_allocated_capacity
	b	.L87		//
	.p2align 2,,3
.L166:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldrb	w0, [x20]	// _109, MEM[(const char_type &)__start_100]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	w0, [sp, 192]	// _109, MEM[(char_type &)_289 + 16]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:351:       }
	b	.L97		//
	.p2align 2,,3
.L165:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	add	x0, sp, 176	// tmp383,,
	mov	x2, 0	//,
	add	x1, sp, 104	//,,
.LEHB3:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm		//
.LEHE3:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	ldr	x1, [sp, 104]	// __dnew, __dnew
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:218:       { _M_dataplus._M_p = __p; }
	str	x0, [sp, 176]	// _86, MEM[(struct basic_string *)_289]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	str	x1, [sp, 192]	// __dnew, MEM[(struct basic_string *)_289].D.64159._M_allocated_capacity
	b	.L94		//
	.p2align 2,,3
.L109:
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 27
// src/cpp/include/utils.h:142:     if (!filter || !*filter) return true;
	mov	w20, 1	// _7,
// src/cpp/include/utils.h:160: }
	mov	w0, w20	//, _7
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x29, x30, [sp], 208	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L80:
	.cfi_def_cfa_offset 208
	.cfi_offset 19, -192
	.cfi_offset 20, -184
	.cfi_offset 21, -176
	.cfi_offset 22, -168
	.cfi_offset 23, -160
	.cfi_offset 24, -152
	.cfi_offset 25, -144
	.cfi_offset 26, -136
	.cfi_offset 27, -128
	.cfi_offset 29, -208
	.cfi_offset 30, -200
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x0, x21	// pretmp_130, tmp336
	beq	.L104		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 160]	// token.D.64159._M_allocated_capacity, token.D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	add	x1, x1, 1	//, token.D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	b	.L104		//
	.p2align 2,,3
.L161:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	add	x1, sp, 176	//,,
	mov	x2, 0	//,
	add	x0, sp, 112	// tmp371,,
.LEHB4:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm		//
.LEHE4:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	ldr	x1, [sp, 176]	// MEM[(long unsigned int *)_289], MEM[(long unsigned int *)_289]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:218:       { _M_dataplus._M_p = __p; }
	str	x0, [sp, 112]	// _70, f._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	str	x1, [sp, 128]	// MEM[(long unsigned int *)_289], f.D.64159._M_allocated_capacity
.L52:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	mov	x2, x21	//, prephitmp_65
	mov	x1, x19	//, filter
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:251: 	_M_set_length(__dnew);
	ldr	x21, [sp, 176]	// prephitmp_65, MEM[(long unsigned int *)_289]
// src/cpp/include/utils.h:146:     size_t pos = 0;
	mov	x20, 0	// pos,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x0, [sp, 112]	// pretmp_5, f._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x21, [sp, 120]	// prephitmp_65, f._M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x0, x21]	//, MEM[(char_type &)_76]
	cbnz	x24, .L169	// _2,
.L151:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3089: 	return this->find_first_not_of(__s, __pos, traits_type::length(__s));
	adrp	x24, .LC13	// tmp345,
	add	x24, x24, :lo12:.LC13	// tmp346, tmp345,
	.p2align 5,,15
.L82:
// src/cpp/include/utils.h:148:         size_t comma = f.find(',', pos);
	mov	x2, x20	//, pos
	mov	w1, 44	//,
	add	x0, sp, 112	// tmp372,,
	bl	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4findEcm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:1077:       { return _M_string_length; }
	ldr	x19, [sp, 120]	// _78, f._M_string_length
// src/cpp/include/utils.h:149:         std::string token = f.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
	cmn	x0, #1	// comma,
// src/cpp/include/utils.h:148:         size_t comma = f.find(',', pos);
	mov	x26, x0	// comma, tmp356
// src/cpp/include/utils.h:149:         std::string token = f.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
	sub	x0, x0, x20	// tmp339, comma, pos
	csinv	x0, x0, xzr, ne	// iftmp.124_87, tmp339,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:393: 	if (__pos > this->size())
	cmp	x19, x20	// _78, pos
	bcc	.L58		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:590: 	const _CharT* __start = __str._M_data()
	ldr	x1, [sp, 112]	// MEM[(const struct basic_string *)&f]._M_dataplus._M_p, MEM[(const struct basic_string *)&f]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:198: 	: allocator_type(std::move(__a)), _M_p(__dat) { }
	add	x21, sp, 160	// tmp336,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:414: 	const bool __testoff =  __off < this->size() - __pos;
	sub	x19, x19, x20	// _13, _78, pos
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:415: 	return __testoff ? __off : this->size() - __pos;
	cmp	x19, x0	// _13, iftmp.124_87
	csel	x19, x19, x0, ls	// _11, _13, iftmp.124_87,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:225: 	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
	str	x19, [sp, 176]	// _11, MEM[(long unsigned int *)_289]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:198: 	: allocator_type(std::move(__a)), _M_p(__dat) { }
	add	x25, sp, 144	// tmp373,,
	str	x21, [sp, 144]	// tmp336, MEM[(struct _Alloc_hider *)&token]._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:590: 	const _CharT* __start = __str._M_data()
	add	x20, x1, x20	// __start, MEM[(const struct basic_string *)&f]._M_dataplus._M_p, pos
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:227: 	if (__dnew > size_type(_S_local_capacity))
	cmp	x19, 15	// _11,
	bhi	.L170		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:432: 	if (__n == 1)
	cmp	x19, 1	// _11,
	beq	.L62		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	mov	x0, x21	// _6, tmp336
	cbnz	x19, .L64	// _11,
.L63:
	mov	x4, x21	// pretmp_92, tmp336
.L61:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x19, [sp, 152]	// _11, token._M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3089: 	return this->find_first_not_of(__s, __pos, traits_type::length(__s));
	mov	x3, 2	//,
	mov	x2, 0	//,
	mov	x1, x24	//, tmp346
	add	x0, sp, 144	// tmp375,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x4, x19]	//, MEM[(char_type &)_22]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3089: 	return this->find_first_not_of(__s, __pos, traits_type::length(__s));
	bl	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE17find_first_not_ofEPKcmm		//
	mov	x20, x0	// tmp358,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3176: 	return this->find_last_not_of(__s, __pos, traits_type::length(__s));
	mov	x1, x24	//, tmp346
	add	x0, sp, 144	// tmp376,,
	mov	x3, 2	//,
	mov	x2, -1	//,
	bl	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16find_last_not_ofEPKcmm		//
// src/cpp/include/utils.h:152:         if (b != std::string::npos) {
	cmn	x20, #1	// _23,
	beq	.L65		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:1077:       { return _M_string_length; }
	ldr	x19, [sp, 152]	// _96, token._M_string_length
// src/cpp/include/utils.h:153:             std::string trimmed = token.substr(b, e - b + 1);
	add	x0, x0, 1	// _233, _232,
	sub	x0, x0, x20	// _234, _233, _23
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:393: 	if (__pos > this->size())
	cmp	x20, x19	// _23, _96
	bhi	.L66		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:590: 	const _CharT* __start = __str._M_data()
	ldr	x1, [sp, 144]	// MEM[(const struct basic_string *)&token]._M_dataplus._M_p, MEM[(const struct basic_string *)&token]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:198: 	: allocator_type(std::move(__a)), _M_p(__dat) { }
	add	x27, sp, 192	// tmp337,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:414: 	const bool __testoff =  __off < this->size() - __pos;
	sub	x19, x19, x20	// _241, _96, _23
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:415: 	return __testoff ? __off : this->size() - __pos;
	cmp	x19, x0	// _241, _234
	csel	x19, x19, x0, ls	// _242, _241, _234,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:225: 	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
	str	x19, [sp, 104]	// _242, __dnew
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:198: 	: allocator_type(std::move(__a)), _M_p(__dat) { }
	str	x27, [sp, 176]	// tmp337, MEM[(struct _Alloc_hider *)_289]._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:590: 	const _CharT* __start = __str._M_data()
	add	x20, x1, x20	// __start, MEM[(const struct basic_string *)&token]._M_dataplus._M_p, _23
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:227: 	if (__dnew > size_type(_S_local_capacity))
	cmp	x19, 15	// _242,
	bhi	.L171		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:432: 	if (__n == 1)
	cmp	x19, 1	// _242,
	beq	.L77		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	mov	x0, x27	// _271, tmp337
	cbnz	x19, .L79	// _242,
.L78:
	mov	x0, x27	// prephitmp_246, tmp337
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x19, [sp, 184]	// _242, MEM[(struct basic_string *)_289]._M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x0, x19]	//, MEM[(char_type &)_248]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:1077:       { return _M_string_length; }
	ldp	x19, x3, [sp, 176]	// _257, pretmp_251,
// src/cpp/include/utils.h:138:     return needle.empty() ? false : haystack.find(needle) != std::string::npos;
	cbnz	x3, .L172	// pretmp_251,
.L69:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x19, x27	// _257, tmp337
	beq	.L65		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x0, x19	//, _257
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 192]	// MEM[(struct basic_string *)_289].D.64159._M_allocated_capacity, MEM[(struct basic_string *)_289].D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	add	x1, x1, 1	//, MEM[(struct basic_string *)_289].D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L65:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x0, [sp, 144]	// pretmp_130, token._M_dataplus._M_p
// src/cpp/include/utils.h:156:         if (comma == std::string::npos) break;
	cmn	x26, #1	// comma,
	beq	.L80		//,
// src/cpp/include/utils.h:157:         pos = comma + 1;
	add	x20, x26, 1	// pos, comma,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x0, x21	// pretmp_130, tmp336
	beq	.L81		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 160]	// token.D.64159._M_allocated_capacity, token.D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	add	x1, x1, 1	//, token.D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L81:
// src/cpp/include/utils.h:147:     while (pos <= f.size()) {
	ldr	x0, [sp, 120]	// f._M_string_length, f._M_string_length
	cmp	x20, x0	// pos, f._M_string_length
	bls	.L82		//,
	b	.L104		//
	.p2align 2,,3
.L170:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	mov	x2, 0	//,
	add	x1, sp, 176	//,,
	add	x0, sp, 144	// tmp374,,
.LEHB5:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm		//
.LEHE5:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	ldr	x1, [sp, 176]	// MEM[(long unsigned int *)_289], MEM[(long unsigned int *)_289]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:218:       { _M_dataplus._M_p = __p; }
	str	x0, [sp, 144]	// _6, token._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	str	x1, [sp, 160]	// MEM[(long unsigned int *)_289], token.D.64159._M_allocated_capacity
.L64:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	mov	x2, x19	//, _11
	mov	x1, x20	//, __start
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x4, [sp, 144]	// pretmp_92, token._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:251: 	_M_set_length(__dnew);
	ldr	x19, [sp, 176]	// _11, MEM[(long unsigned int *)_289]
	b	.L61		//
	.p2align 2,,3
.L171:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	add	x0, sp, 176	// tmp377,,
	mov	x2, 0	//,
	add	x1, sp, 104	//,,
.LEHB6:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	ldr	x1, [sp, 104]	// __dnew, __dnew
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:218:       { _M_dataplus._M_p = __p; }
	str	x0, [sp, 176]	// _271, MEM[(struct basic_string *)_289]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	str	x1, [sp, 192]	// __dnew, MEM[(struct basic_string *)_289].D.64159._M_allocated_capacity
.L79:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	mov	x2, x19	//, _242
	mov	x1, x20	//, __start
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:251: 	_M_set_length(__dnew);
	ldr	x19, [sp, 104]	// _242, __dnew
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x0, [sp, 176]	// prephitmp_246, MEM[(struct basic_string *)_289]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x19, [sp, 184]	// _242, MEM[(struct basic_string *)_289]._M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x0, x19]	//, MEM[(char_type &)_248]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:1077:       { return _M_string_length; }
	ldp	x19, x3, [sp, 176]	// _257, pretmp_251,
// src/cpp/include/utils.h:138:     return needle.empty() ? false : haystack.find(needle) != std::string::npos;
	cbz	x3, .L69	// pretmp_251,
.L172:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:2708:       { return this->find(__str.data(), __pos, __str.size()); }
	mov	x1, x19	//, _257
	mov	x2, 0	//,
	mov	x0, x22	//, name
	bl	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4findEPKcmm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x19, [sp, 176]	// _257, MEM[(struct basic_string *)_289]._M_dataplus._M_p
// src/cpp/include/utils.h:154:             if (exact ? (name == trimmed) : contains_substr(name, trimmed)) return true;
	cmn	x0, #1	// tmp360,
	beq	.L69		//,
	b	.L74		//
	.p2align 2,,3
.L62:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldrb	w0, [x20]	// _93, MEM[(const char_type &)__start_20]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	w0, [sp, 160]	// _93, MEM[(char_type &)&token + 16]
	b	.L63		//
	.p2align 2,,3
.L77:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldrb	w0, [x20]	// _276, MEM[(const char_type &)__start_240]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	w0, [sp, 192]	// _276, MEM[(char_type &)_289 + 16]
	b	.L78		//
	.p2align 2,,3
.L158:
	ldr	x27, [sp, 80]	//,
	.cfi_remember_state
	.cfi_restore 27
	ldp	x21, x22, [sp, 32]	//,,
	.cfi_restore 22
	.cfi_restore 21
	ldp	x23, x24, [sp, 48]	//,,
	.cfi_restore 24
	.cfi_restore 23
	ldp	x25, x26, [sp, 64]	//,,
	.cfi_restore 26
	.cfi_restore 25
	b	.L126		//
.L66:
	.cfi_restore_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:394: 	  __throw_out_of_range_fmt(__N("%s: __pos (which is %zu) > "
	adrp	x1, .LC14	// tmp283,
	adrp	x0, .LC15	// tmp285,
	mov	x3, x19	//, _96
	mov	x2, x20	//, _23
	add	x1, x1, :lo12:.LC14	//, tmp283,
	add	x0, x0, :lo12:.LC15	//, tmp285,
	bl	_ZSt24__throw_out_of_range_fmtPKcz		//
.LEHE6:
.L58:
	adrp	x1, .LC14	// tmp260,
	adrp	x0, .LC15	// tmp262,
	mov	x3, x19	//, _78
	mov	x2, x20	//, pos
	add	x1, x1, :lo12:.LC14	//, tmp260,
	add	x0, x0, :lo12:.LC15	//, tmp262,
.LEHB7:
	bl	_ZSt24__throw_out_of_range_fmtPKcz		//
.LEHE7:
.L115:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x19, x0	// tmp368,
	mov	x0, x25	//, tmp335
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
.L108:
	add	x0, sp, 112	// tmp384,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	mov	x0, x19	//, tmp329
.LEHB8:
	bl	_Unwind_Resume		//
.LEHE8:
.L114:
	mov	x19, x0	// tmp329, tmp369
	b	.L108		//
	.cfi_endproc
.LFE6766:
	.section	.gcc_except_table
.LLSDA6766:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE6766-.LLSDACSB6766
.LLSDACSB6766:
	.uleb128 .LEHB2-.LFB6766
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L114-.LFB6766
	.uleb128 0
	.uleb128 .LEHB3-.LFB6766
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L115-.LFB6766
	.uleb128 0
	.uleb128 .LEHB4-.LFB6766
	.uleb128 .LEHE4-.LEHB4
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB5-.LFB6766
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L114-.LFB6766
	.uleb128 0
	.uleb128 .LEHB6-.LFB6766
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L115-.LFB6766
	.uleb128 0
	.uleb128 .LEHB7-.LFB6766
	.uleb128 .LEHE7-.LEHB7
	.uleb128 .L114-.LFB6766
	.uleb128 0
	.uleb128 .LEHB8-.LFB6766
	.uleb128 .LEHE8-.LEHB8
	.uleb128 0
	.uleb128 0
.LLSDACSE6766:
	.text
	.size	_ZL29implementation_matches_filterRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKc, .-_ZL29implementation_matches_filterRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKc
	.section	.text.unlikely,"ax",@progbits
	.align	2
	.type	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_.isra.0, %function
_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_.isra.0:
.LFB8400:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA8400
	stp	x29, x30, [sp, -64]!	//,,,
	.cfi_def_cfa_offset 64
	.cfi_offset 29, -64
	.cfi_offset 30, -56
	mov	x29, sp	//,
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -48
	.cfi_offset 20, -40
	mov	x19, x8	// <retval>, tmp113
	mov	x20, x2	// ISRA.494, tmp116
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 21, -32
	.cfi_offset 22, -24
	mov	x22, x1	// ISRA.493, tmp115
	str	x23, [sp, 48]	//,
	.cfi_offset 23, -16
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3616:     operator+(const _CharT* __lhs,
	mov	x23, x0	// __lhs, tmp114
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:235: 	return std::pointer_traits<pointer>::pointer_to(*_M_local_buf);
	add	x1, x19, 16	// _12, <retval>,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x21, x0	// _3, tmp117
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x19, 16]	//, MEM[(char_type &)_1(D) + 16]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3582:       __str.reserve(__lhs_len + __rhs_len);
	mov	x0, x19	//, <retval>
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	stp	x1, xzr, [x19]	// _12,,* <retval>
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3582:       __str.reserve(__lhs_len + __rhs_len);
	add	x1, x21, x20	//, _3, ISRA.494
.LEHB9:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3583:       __str.append(__lhs, __lhs_len);
	mov	x2, x21	//, _3
	mov	x1, x23	//, __lhs
	mov	x0, x19	//, <retval>
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKcm.isra.0		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3584:       __str.append(__rhs, __rhs_len);
	mov	x2, x20	//, ISRA.494
	mov	x1, x22	//, ISRA.493
	mov	x0, x19	//, <retval>
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKcm.isra.0		//
.LEHE9:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3624:     }
	ldr	x23, [sp, 48]	//,
	mov	x0, x19	//, <retval>
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x29, x30, [sp], 64	//,,,
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 23
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
.L176:
	.cfi_restore_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x1, x0	// tmp118,
	mov	x0, x19	//, <retval>
	mov	x19, x1	// tmp111, tmp118
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	mov	x0, x19	//, tmp111
.LEHB10:
	bl	_Unwind_Resume		//
.LEHE10:
	.cfi_endproc
.LFE8400:
	.section	.gcc_except_table
.LLSDA8400:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE8400-.LLSDACSB8400
.LLSDACSB8400:
	.uleb128 .LEHB9-.LFB8400
	.uleb128 .LEHE9-.LEHB9
	.uleb128 .L176-.LFB8400
	.uleb128 0
	.uleb128 .LEHB10-.LFB8400
	.uleb128 .LEHE10-.LEHB10
	.uleb128 0
	.uleb128 0
.LLSDACSE8400:
	.section	.text.unlikely
	.size	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_.isra.0, .-_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_.isra.0
	.section	.text._ZSteqIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_,"axG",@progbits,_ZSteqIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_,comdat
	.align	2
	.p2align 5,,15
	.weak	_ZSteqIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_
	.type	_ZSteqIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_, %function
_ZSteqIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_:
.LFB7793:
	.cfi_startproc
	stp	x29, x30, [sp, -48]!	//,,,
	.cfi_def_cfa_offset 48
	.cfi_offset 29, -48
	.cfi_offset 30, -40
	mov	x29, sp	//,
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -32
	.cfi_offset 20, -24
	mov	x20, x0	// __lhs, tmp115
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x1	//, __rhs
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3772:     operator==(const basic_string<_CharT, _Traits, _Alloc>& __lhs,
	mov	x19, x1	// __rhs, tmp116
	str	x21, [sp, 32]	//,
	.cfi_offset 21, -16
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:1077:       { return _M_string_length; }
	ldr	x21, [x20, 8]	// _6, __lhs_3(D)->_M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3776: 	       && !_Traits::compare(__lhs.data(), __rhs, __lhs.size());
	cmp	x21, x0	// _6, tmp117
	beq	.L185		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3776: 	       && !_Traits::compare(__lhs.data(), __rhs, __lhs.size());
	mov	w0, 0	// iftmp.65_1,
.L180:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3777:     }
	ldr	x21, [sp, 32]	//,
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x29, x30, [sp], 48	//,,,
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 21
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L185:
	.cfi_restore_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3776: 	       && !_Traits::compare(__lhs.data(), __rhs, __lhs.size());
	mov	w0, 1	// iftmp.65_1,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:368: 	if (__n == 0)
	cbz	x21, .L180	// _6,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:381: 	return __builtin_memcmp(__s1, __s2, __n);
	ldr	x0, [x20]	//, __lhs_3(D)->_M_dataplus._M_p
	mov	x2, x21	//, _6
	mov	x1, x19	//, __rhs
	bl	memcmp		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3777:     }
	ldr	x21, [sp, 32]	//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3776: 	       && !_Traits::compare(__lhs.data(), __rhs, __lhs.size());
	cmp	w0, 0	// tmp118,
	cset	w0, eq	// iftmp.65_1,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3777:     }
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x29, x30, [sp], 48	//,,,
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 21
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret	
	.cfi_endproc
.LFE7793:
	.size	_ZSteqIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_, .-_ZSteqIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_
	.section	.rodata._Z17load_mnist_imagesRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEER6Tensori.str1.8,"aMS",@progbits,1
	.align	3
.LC16:
	.string	"The MNIST image file could not be opened: "
	.section	.text._Z17load_mnist_imagesRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEER6Tensori,"axG",@progbits,_Z17load_mnist_imagesRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEER6Tensori,comdat
	.align	2
	.p2align 5,,15
	.weak	_Z17load_mnist_imagesRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEER6Tensori
	.type	_Z17load_mnist_imagesRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEER6Tensori, %function
_Z17load_mnist_imagesRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEER6Tensori:
.LFB6753:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA6753
	sub	sp, sp, #720	//,,
	.cfi_def_cfa_offset 720
	stp	x29, x30, [sp]	//,,
	.cfi_offset 29, -720
	.cfi_offset 30, -712
	mov	x29, sp	//,
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 21, -688
	.cfi_offset 22, -680
	mov	x21, x0	// tmp335, filepath
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	add	x0, sp, 456	// tmp350,,
// src/cpp/include/utils.h:78: inline void load_mnist_images(const std::string& filepath, Tensor& images_tensor, int num_images_to_load) {
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -704
	.cfi_offset 20, -696
	mov	x20, x1	// images_tensor, tmp336
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	add	x19, sp, 192	// tmp327,,
// src/cpp/include/utils.h:78: inline void load_mnist_images(const std::string& filepath, Tensor& images_tensor, int num_images_to_load) {
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 23, -672
	.cfi_offset 24, -664
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	adrp	x23, _ZTVSt9basic_iosIcSt11char_traitsIcEE+16	// tmp326,
	add	x23, x23, :lo12:_ZTVSt9basic_iosIcSt11char_traitsIcEE+16	// tmp321, tmp326,
// src/cpp/include/utils.h:78: inline void load_mnist_images(const std::string& filepath, Tensor& images_tensor, int num_images_to_load) {
	stp	x25, x26, [sp, 64]	//,,
	stp	x27, x28, [sp, 80]	//,,
	.cfi_offset 25, -656
	.cfi_offset 26, -648
	.cfi_offset 27, -640
	.cfi_offset 28, -632
	mov	w27, w2	// num_images_to_load, tmp337
	str	d15, [sp, 96]	//,
	.cfi_offset 79, -624
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	bl	_ZNSt8ios_baseC2Ev		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:697:       : _M_gcount(streamsize(0))
	adrp	x2, _ZTTSt14basic_ifstreamIcSt11char_traitsIcEE	// tmp187,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:465:       : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), 
	strh	wzr, [sp, 680]	//, MEM <unsigned short> [(void *)&file + 488B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:698:       { this->init(0); }
	mov	x1, 0	//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:697:       : _M_gcount(streamsize(0))
	add	x2, x2, :lo12:_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE	// tmp186, tmp187,
	ldp	x22, x26, [x2, 8]	// _154, _158,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	add	x2, sp, 688	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:697:       : _M_gcount(streamsize(0))
	ldr	x0, [x22, -24]	// MEM[(long int *)_154 + -24B], MEM[(long int *)_154 + -24B]
	str	x22, [sp, 192]	// _154, MEM[(struct basic_istream *)&file]._vptr.basic_istream
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	str	x23, [sp, 456]	// tmp321, MEM[(struct basic_ios *)&file + 264B].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:465:       : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), 
	str	xzr, [sp, 672]	//, MEM[(struct basic_ios *)&file + 264B]._M_tie
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	stp	xzr, xzr, [x2]	//,,
	stp	xzr, xzr, [x2, 16]	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:697:       : _M_gcount(streamsize(0))
	str	x26, [x19, x0]	// _158, MEM[(struct basic_ios *)_157].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:697:       : _M_gcount(streamsize(0))
	str	xzr, [sp, 200]	//, MEM[(struct basic_istream *)&file]._M_gcount
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:698:       { this->init(0); }
	ldr	x0, [x22, -24]	// MEM[(long int *)_154 + -24B], MEM[(long int *)_154 + -24B]
	add	x0, x19, x0	//, tmp327, MEM[(long int *)_154 + -24B]
.LEHB11:
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E		//
.LEHE11:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:604:       : __istream_type(), _M_filebuf()
	adrp	x25, _ZTVSt14basic_ifstreamIcSt11char_traitsIcEE+24	// tmp323,
	adrp	x24, _ZTVSt14basic_ifstreamIcSt11char_traitsIcEE+64	// tmp328,
	add	x25, x25, :lo12:_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE+24	// tmp332, tmp323,
	add	x24, x24, :lo12:_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE+64	// tmp325, tmp328,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:604:       : __istream_type(), _M_filebuf()
	add	x0, sp, 208	// tmp351,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:604:       : __istream_type(), _M_filebuf()
	str	x25, [sp, 192]	// tmp332, MEM[(struct basic_ifstream *)&file].D.88669._vptr.basic_istream
	str	x24, [sp, 456]	// tmp325, MEM[(struct basic_ios *)&file + 264B].D.81517._vptr.ios_base
.LEHB12:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:604:       : __istream_type(), _M_filebuf()
	bl	_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev		//
.LEHE12:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:606: 	this->init(&_M_filebuf);
	add	x1, sp, 208	// tmp352,,
	add	x0, sp, 456	// tmp353,,
.LEHB13:
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:342:       { return open(__s.c_str(), __mode); }
	ldr	x1, [x21]	//, MEM[(char * *)filepath_24(D)]
	mov	w2, 12	//,
	add	x0, sp, 208	// tmp354,,
	bl	_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode		//
	mov	x2, x0	// tmp338,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:740: 	  this->setstate(ios_base::failbit);
	ldr	x0, [sp, 192]	// file.D.88669._vptr.basic_istream, file.D.88669._vptr.basic_istream
	ldr	x1, [x0, -24]	// MEM[(long int *)pretmp_301 + -24B], MEM[(long int *)pretmp_301 + -24B]
	add	x0, x19, x1	// _304, tmp327, MEM[(long int *)pretmp_301 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:739: 	if (!_M_filebuf.open(__s, __mode | ios_base::in))
	cbz	x2, .L222	// tmp338,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:744: 	  this->clear();
	mov	w1, 0	//,
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
.LEHE13:
.L188:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:276:       { return _M_file.is_open(); }
	add	x0, sp, 320	// tmp355,,
	bl	_ZNKSt12__basic_fileIcE7is_openEv		//
// src/cpp/include/utils.h:80:     if (!file.is_open())
	tbz	x0, 0, .L223	// tmp339,,
// src/cpp/include/utils.h:84:     file.read((char*)&magic_number,      sizeof(magic_number));      magic_number      = reverse_int(magic_number);
	mov	x2, 4	//,
	add	x1, sp, 148	//,,
// src/cpp/include/utils.h:83:     int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
	stp	wzr, wzr, [sp, 148]	//,,
// src/cpp/include/utils.h:84:     file.read((char*)&magic_number,      sizeof(magic_number));      magic_number      = reverse_int(magic_number);
	mov	x0, x19	//, tmp327
// src/cpp/include/utils.h:83:     int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
	stp	wzr, wzr, [sp, 156]	//,,
.LEHB14:
// src/cpp/include/utils.h:84:     file.read((char*)&magic_number,      sizeof(magic_number));      magic_number      = reverse_int(magic_number);
	bl	_ZNSi4readEPcl		//
	ldr	w3, [sp, 148]	//, MEM[(int *)&magic_number]
// src/cpp/include/utils.h:85:     file.read((char*)&number_of_images,  sizeof(number_of_images));  number_of_images  = reverse_int(number_of_images);
	mov	x2, 4	//,
	add	x1, sp, 152	//,,
	mov	x0, x19	//, tmp327
	rev	w3, w3	// bswapdst_44, MEM[(int *)&magic_number]
// src/cpp/include/utils.h:84:     file.read((char*)&magic_number,      sizeof(magic_number));      magic_number      = reverse_int(magic_number);
	str	w3, [sp, 148]	// bswapdst_44, magic_number
// src/cpp/include/utils.h:85:     file.read((char*)&number_of_images,  sizeof(number_of_images));  number_of_images  = reverse_int(number_of_images);
	bl	_ZNSi4readEPcl		//
	ldr	w3, [sp, 152]	//, MEM[(int *)&number_of_images]
// src/cpp/include/utils.h:86:     file.read((char*)&n_rows,            sizeof(n_rows));             n_rows            = reverse_int(n_rows);
	mov	x2, 4	//,
	add	x1, sp, 156	//,,
	mov	x0, x19	//, tmp327
	rev	w3, w3	// bswapdst_19, MEM[(int *)&number_of_images]
// src/cpp/include/utils.h:85:     file.read((char*)&number_of_images,  sizeof(number_of_images));  number_of_images  = reverse_int(number_of_images);
	str	w3, [sp, 152]	// bswapdst_19, number_of_images
// src/cpp/include/utils.h:86:     file.read((char*)&n_rows,            sizeof(n_rows));             n_rows            = reverse_int(n_rows);
	bl	_ZNSi4readEPcl		//
	ldr	w3, [sp, 156]	//, MEM[(int *)&n_rows]
// src/cpp/include/utils.h:87:     file.read((char*)&n_cols,            sizeof(n_cols));             n_cols            = reverse_int(n_cols);
	mov	x2, 4	//,
	add	x1, sp, 160	//,,
	mov	x0, x19	//, tmp327
	rev	w3, w3	// bswapdst_50, MEM[(int *)&n_rows]
// src/cpp/include/utils.h:86:     file.read((char*)&n_rows,            sizeof(n_rows));             n_rows            = reverse_int(n_rows);
	str	w3, [sp, 156]	// bswapdst_50, n_rows
// src/cpp/include/utils.h:87:     file.read((char*)&n_cols,            sizeof(n_cols));             n_cols            = reverse_int(n_cols);
	bl	_ZNSi4readEPcl		//
// src/cpp/include/utils.h:90:     images_tensor = {std::vector<float>(images_to_read * n_rows * n_cols),
	ldp	w21, w5, [sp, 152]	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	mov	x0, 2305843009213693951	// tmp263,
	ldr	w28, [sp, 160]	//, MEM[(int *)_69]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:238:       if (__b < __a)
	cmp	w21, w27	// number_of_images, num_images_to_load
	csel	w21, w21, w27, le	// _27, number_of_images, num_images_to_load,
// src/cpp/include/utils.h:90:     images_tensor = {std::vector<float>(images_to_read * n_rows * n_cols),
	mul	w27, w5, w21	// _6, n_rows.74_5, _27
	rev	w28, w28	// _125, MEM[(int *)_69]
// src/cpp/include/utils.h:90:     images_tensor = {std::vector<float>(images_to_read * n_rows * n_cols),
	mul	w27, w27, w28	// _7, _6, _125
// src/cpp/include/utils.h:87:     file.read((char*)&n_cols,            sizeof(n_cols));             n_cols            = reverse_int(n_cols);
	str	w28, [sp, 160]	// _125, MEM[(int *)_69]
// src/cpp/include/utils.h:98: }
	sxtw	x27, w27	// _8, _7
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1919: 	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	cmp	x27, x0	// _8, tmp263
	bhi	.L224		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	cbz	x27, .L204	// _8,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	lsl	x2, x27, 2	// _195, _8,
	mov	x0, x2	//, _195
	str	x2, [sp, 120]	// _195, %sfp
	bl	_Znwm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	ldr	x2, [sp, 120]	// _195, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	mov	x3, x0	// __first, iftmp.90_133
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x4, x0	// iftmp.90_133, tmp344
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	str	wzr, [x3], 4	//, *_196
// src/cpp/include/utils.h:98: }
	ldp	w5, w28, [sp, 156]	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	add	x6, x0, x2	// _131, iftmp.90_133, _195
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:1146:       if (__n <= 0)
	cmp	x27, 1	// _8,
	beq	.L195		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:952: 	*__first = __tmp;
	mov	x0, x3	//, __first
	sub	x2, x2, #4	//, _195,
	str	w5, [sp, 120]	// n_rows.74_5, %sfp
	mov	w1, 0	//,
	stp	x6, x4, [sp, 128]	// _131, iftmp.90_133,
	bl	memset		//
	ldp	x6, x4, [sp, 128]	// _131, iftmp.90_133,
	mov	x3, x0	// __first,
	mov	x0, -4	// tmp273,
	ldr	w5, [sp, 120]	//, %sfp
	add	x3, x6, x3	// _206, _131, __first
	sub	x0, x0, x4	// _14, tmp273, iftmp.90_133
	add	x3, x0, x3	// __first, _14, _206
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	mov	x6, x3	// _131, __first
.L195:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:114: 	  _M_start = __x._M_start;
	ldr	x0, [x20]	// _204, MEM[(const struct _Vector_impl_data &)images_tensor_51(D)]._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:114: 	  _M_start = __x._M_start;
	stp	x4, x3, [x20]	// iftmp.90_133, __first,* images_tensor
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:116: 	  _M_end_of_storage = __x._M_end_of_storage;
	ldr	x1, [x20, 16]	// _205, MEM[(const struct _Vector_impl_data &)images_tensor_51(D)]._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:116: 	  _M_end_of_storage = __x._M_end_of_storage;
	str	x6, [x20, 16]	// _131, MEM[(struct _Vector_impl_data *)images_tensor_51(D)]._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L205	// _204,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, _205, _204
	str	w5, [sp, 120]	// n_rows.74_5, %sfp
	bl	_ZdlPvm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/alloc_traits.h:513:       { __a.deallocate(__p, __n); }
	ldr	w5, [sp, 120]	//, %sfp
// src/cpp/include/utils.h:93:     for (int i = 0; i < images_to_read * n_rows * n_cols; ++i) {
	ldp	w1, w3, [sp, 156]	//,,
.L196:
// src/cpp/include/utils.h:93:     for (int i = 0; i < images_to_read * n_rows * n_cols; ++i) {
	mul	w1, w21, w1	// _264, _27, pretmp_55
// src/cpp/include/cnn_internals.h:15: struct Tensor {
	mov	w2, 1	// tmp276,
// src/cpp/include/utils.h:96:         images_tensor.data[i] = static_cast<float>(pixel) / 255.0f;
	mov	w0, 32897	// tmp348,
// src/cpp/include/cnn_internals.h:15: struct Tensor {
	stp	w5, w28, [x20, 32]	// n_rows.74_5, _125,
// src/cpp/include/utils.h:96:         images_tensor.data[i] = static_cast<float>(pixel) / 255.0f;
	movk	w0, 0x3b80, lsl 16	// tmp348,,
// src/cpp/include/utils.h:93:     for (int i = 0; i < images_to_read * n_rows * n_cols; ++i) {
	mov	x27, 0	// ivtmp.508,
// src/cpp/include/cnn_internals.h:15: struct Tensor {
	stp	w21, w2, [x20, 24]	// _27, tmp276,
// src/cpp/include/utils.h:93:     for (int i = 0; i < images_to_read * n_rows * n_cols; ++i) {
	mul	w1, w1, w3	// _138, _264, prephitmp_53
// src/cpp/include/utils.h:96:         images_tensor.data[i] = static_cast<float>(pixel) / 255.0f;
	fmov	s15, w0	// tmp334, tmp348
// src/cpp/include/utils.h:93:     for (int i = 0; i < images_to_read * n_rows * n_cols; ++i) {
	cmp	w1, 0	// _138,
	ble	.L199		//,
	.p2align 5,,15
.L197:
// src/cpp/include/utils.h:95:         file.read((char*)&pixel, sizeof(pixel));
	mov	x2, 1	//,
	add	x1, sp, 147	// tmp362,,
// src/cpp/include/utils.h:94:         unsigned char pixel = 0;
	strb	wzr, [sp, 147]	//, pixel
// src/cpp/include/utils.h:95:         file.read((char*)&pixel, sizeof(pixel));
	mov	x0, x19	//, tmp327
	bl	_ZNSi4readEPcl		//
.LEHE14:
// src/cpp/include/utils.h:96:         images_tensor.data[i] = static_cast<float>(pixel) / 255.0f;
	ldr	b31, [sp, 147]	// pixel, pixel
// src/cpp/include/utils.h:93:     for (int i = 0; i < images_to_read * n_rows * n_cols; ++i) {
	ldp	w0, w2, [sp, 156]	//,,
// src/cpp/include/utils.h:96:         images_tensor.data[i] = static_cast<float>(pixel) / 255.0f;
	ldr	x1, [x20]	// MEM[(struct vector *)images_tensor_51(D)].D.57594._M_impl.D.56895._M_start, MEM[(struct vector *)images_tensor_51(D)].D.57594._M_impl.D.56895._M_start
// src/cpp/include/utils.h:96:         images_tensor.data[i] = static_cast<float>(pixel) / 255.0f;
	ucvtf	s31, s31	// _12, pixel
// src/cpp/include/utils.h:93:     for (int i = 0; i < images_to_read * n_rows * n_cols; ++i) {
	mul	w0, w21, w0	// _16, _27, n_rows
// src/cpp/include/utils.h:93:     for (int i = 0; i < images_to_read * n_rows * n_cols; ++i) {
	mul	w0, w0, w2	// _18, _16, MEM[(int *)_69]
// src/cpp/include/utils.h:96:         images_tensor.data[i] = static_cast<float>(pixel) / 255.0f;
	fmul	s31, s31, s15	// _13, _12, tmp334
// src/cpp/include/utils.h:96:         images_tensor.data[i] = static_cast<float>(pixel) / 255.0f;
	str	s31, [x1, w27, uxtw 2]	// _13, *_137
// src/cpp/include/utils.h:93:     for (int i = 0; i < images_to_read * n_rows * n_cols; ++i) {
	add	x27, x27, 1	// ivtmp.508, ivtmp.508,
	cmp	w0, w27	// _18, ivtmp.508
	bgt	.L197		//,
.L199:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:262:       }
	adrp	x1, _ZTVSt13basic_filebufIcSt11char_traitsIcEE+16	// tmp284,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:259: 	  { this->close(); }
	add	x0, sp, 208	// tmp361,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:640:       { }
	str	x25, [sp, 192]	// tmp332, file.D.88669._vptr.basic_istream
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:262:       }
	add	x1, x1, :lo12:_ZTVSt13basic_filebufIcSt11char_traitsIcEE+16	// tmp283, tmp284,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:640:       { }
	str	x24, [sp, 456]	// tmp325, MEM[(struct basic_ios *)&file + 264B].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:262:       }
	str	x1, [sp, 208]	// tmp283, MEM[(struct basic_filebuf *)&file + 16B].D.88477._vptr.basic_streambuf
.LEHB15:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:259: 	  { this->close(); }
	bl	_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv		//
.LEHE15:
.L198:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:262:       }
	add	x0, sp, 320	// tmp363,,
	bl	_ZNSt12__basic_fileIcED1Ev		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/streambuf:205:       { }
	adrp	x1, _ZTVSt15basic_streambufIcSt11char_traitsIcEE+16	// tmp303,
	add	x0, sp, 264	//,,
	add	x1, x1, :lo12:_ZTVSt15basic_streambufIcSt11char_traitsIcEE+16	// tmp302, tmp303,
	str	x1, [sp, 208]	// tmp302, MEM[(struct basic_streambuf *)&file + 16B]._vptr.basic_streambuf
	bl	_ZNSt6localeD1Ev		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:106:       { _M_gcount = streamsize(0); }
	ldr	x1, [x22, -24]	// MEM[(long int *)_154 + -24B], MEM[(long int *)_154 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:286:       ~basic_ios() { }
	add	x0, sp, 456	// tmp364,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:106:       { _M_gcount = streamsize(0); }
	str	x22, [sp, 192]	// _154, MEM[(struct basic_istream *)&file]._vptr.basic_istream
	str	x26, [x19, x1]	// _158, MEM[(struct basic_ios *)_216].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:106:       { _M_gcount = streamsize(0); }
	str	xzr, [sp, 200]	//, MEM[(struct basic_istream *)&file]._M_gcount
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:286:       ~basic_ios() { }
	str	x23, [sp, 456]	// tmp321, MEM[(struct basic_ios *)&file + 264B].D.81517._vptr.ios_base
	bl	_ZNSt8ios_baseD2Ev		//
// src/cpp/include/utils.h:98: }
	ldr	d15, [sp, 96]	//,
	ldp	x29, x30, [sp]	//,,
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x25, x26, [sp, 64]	//,,
	ldp	x27, x28, [sp, 80]	//,,
	add	sp, sp, 720	//,,
	.cfi_remember_state
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
	.cfi_restore 29
	.cfi_restore 30
	.cfi_restore 79
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L205:
	.cfi_restore_state
	mov	w1, w5	// pretmp_55, n_rows.74_5
	mov	w3, w28	// prephitmp_53, _125
	b	.L196		//
	.p2align 2,,3
.L204:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:400: 	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
	mov	x6, 0	// _131,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:380: 	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
	mov	x4, 0	// iftmp.90_133,
	mov	x3, 0	// __first,
	b	.L195		//
	.p2align 2,,3
.L222:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:187:   { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
	ldr	w1, [x0, 32]	//, MEM[(const struct basic_ios *)_304].D.81517._M_streambuf_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	orr	w1, w1, 4	//, MEM[(const struct basic_ios *)_304].D.81517._M_streambuf_state,
.LEHB16:
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
.LEHE16:
	b	.L188		//
.L212:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:608:       }
	mov	x20, x0	// tmp340,
	add	x0, sp, 208	// tmp356,,
	bl	_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev		//
.L192:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:106:       { _M_gcount = streamsize(0); }
	ldr	x0, [x22, -24]	// MEM[(long int *)_154 + -24B], MEM[(long int *)_154 + -24B]
	str	x22, [sp, 192]	// _154, MEM[(struct basic_istream *)&file]._vptr.basic_istream
	str	x26, [x19, x0]	// _158, MEM[(struct basic_ios *)_167].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:106:       { _M_gcount = streamsize(0); }
	str	xzr, [sp, 200]	//, MEM[(struct basic_istream *)&file]._M_gcount
.L193:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:286:       ~basic_ios() { }
	add	x0, sp, 456	// tmp357,,
	str	x23, [sp, 456]	// tmp321, MEM[(struct basic_ios *)&file + 264B].D.81517._vptr.ios_base
	bl	_ZNSt8ios_baseD2Ev		//
	mov	x0, x20	//, tmp225
.LEHB17:
	bl	_Unwind_Resume		//
.L209:
// src/cpp/include/utils.h:98: }
	mov	x20, x0	// tmp318, tmp347
.L203:
	mov	x0, x19	//, tmp327
	bl	_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev		//
	mov	x0, x20	//, tmp318
	bl	_Unwind_Resume		//
.LEHE17:
.L223:
// src/cpp/include/utils.h:81:         throw std::runtime_error("The MNIST image file could not be opened: " + filepath);
	mov	x0, 16	//,
	bl	__cxa_allocate_exception		//
	mov	x3, x0	// tmp343,
// src/cpp/include/utils.h:81:         throw std::runtime_error("The MNIST image file could not be opened: " + filepath);
	adrp	x0, .LC16	// tmp233,
	ldp	x1, x2, [x21]	//,,* filepath
	add	x8, sp, 160	// tmp358,,
	add	x0, x0, :lo12:.LC16	//, tmp233,
// src/cpp/include/utils.h:81:         throw std::runtime_error("The MNIST image file could not be opened: " + filepath);
	mov	x21, x3	// _71, tmp343
.LEHB18:
// src/cpp/include/utils.h:81:         throw std::runtime_error("The MNIST image file could not be opened: " + filepath);
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_.isra.0		//
.LEHE18:
// src/cpp/include/utils.h:81:         throw std::runtime_error("The MNIST image file could not be opened: " + filepath);
	add	x1, sp, 160	// tmp359,,
	mov	x0, x21	//, _71
.LEHB19:
	bl	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE		//
.LEHE19:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	add	x0, sp, 160	// tmp360,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/include/utils.h:81:         throw std::runtime_error("The MNIST image file could not be opened: " + filepath);
	adrp	x2, _ZNSt13runtime_errorD1Ev	// tmp238,
	adrp	x1, _ZTISt13runtime_error	// tmp240,
	mov	x0, x21	//, _71
	add	x2, x2, :lo12:_ZNSt13runtime_errorD1Ev	//, tmp238,
	add	x1, x1, :lo12:_ZTISt13runtime_error	//, tmp240,
.LEHB20:
	bl	__cxa_throw		//
.LEHE20:
.L208:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x20, x0	// tmp345,
	add	x0, sp, 160	// tmp365,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
.L202:
// src/cpp/include/utils.h:81:         throw std::runtime_error("The MNIST image file could not be opened: " + filepath);
	mov	x0, x21	//, _71
	bl	__cxa_free_exception		//
	b	.L203		//
.L207:
	mov	x20, x0	// tmp314, tmp346
	b	.L202		//
.L213:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:260: 	__catch(...)
	bl	__cxa_begin_catch		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:260: 	__catch(...)
	bl	__cxa_end_catch		//
	b	.L198		//
.L224:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1920: 	  __throw_length_error(
	adrp	x0, .LC9	// tmp265,
	add	x0, x0, :lo12:.LC9	//, tmp265,
.LEHB21:
	bl	_ZSt20__throw_length_errorPKc		//
.LEHE21:
.L211:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:106:       { _M_gcount = streamsize(0); }
	mov	x20, x0	// tmp219, tmp341
	b	.L192		//
.L210:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:286:       ~basic_ios() { }
	mov	x20, x0	// tmp225, tmp342
	b	.L193		//
	.cfi_endproc
.LFE6753:
	.section	.gcc_except_table
	.align	2
.LLSDA6753:
	.byte	0xff
	.byte	0x9b
	.uleb128 .LLSDATT6753-.LLSDATTD6753
.LLSDATTD6753:
	.byte	0x1
	.uleb128 .LLSDACSE6753-.LLSDACSB6753
.LLSDACSB6753:
	.uleb128 .LEHB11-.LFB6753
	.uleb128 .LEHE11-.LEHB11
	.uleb128 .L210-.LFB6753
	.uleb128 0
	.uleb128 .LEHB12-.LFB6753
	.uleb128 .LEHE12-.LEHB12
	.uleb128 .L211-.LFB6753
	.uleb128 0
	.uleb128 .LEHB13-.LFB6753
	.uleb128 .LEHE13-.LEHB13
	.uleb128 .L212-.LFB6753
	.uleb128 0
	.uleb128 .LEHB14-.LFB6753
	.uleb128 .LEHE14-.LEHB14
	.uleb128 .L209-.LFB6753
	.uleb128 0
	.uleb128 .LEHB15-.LFB6753
	.uleb128 .LEHE15-.LEHB15
	.uleb128 .L213-.LFB6753
	.uleb128 0x1
	.uleb128 .LEHB16-.LFB6753
	.uleb128 .LEHE16-.LEHB16
	.uleb128 .L212-.LFB6753
	.uleb128 0
	.uleb128 .LEHB17-.LFB6753
	.uleb128 .LEHE17-.LEHB17
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB18-.LFB6753
	.uleb128 .LEHE18-.LEHB18
	.uleb128 .L207-.LFB6753
	.uleb128 0
	.uleb128 .LEHB19-.LFB6753
	.uleb128 .LEHE19-.LEHB19
	.uleb128 .L208-.LFB6753
	.uleb128 0
	.uleb128 .LEHB20-.LFB6753
	.uleb128 .LEHE20-.LEHB20
	.uleb128 .L209-.LFB6753
	.uleb128 0
	.uleb128 .LEHB21-.LFB6753
	.uleb128 .LEHE21-.LEHB21
	.uleb128 .L209-.LFB6753
	.uleb128 0
.LLSDACSE6753:
	.byte	0x1
	.byte	0
	.align	2
	.4byte	0

.LLSDATT6753:
	.section	.text._Z17load_mnist_imagesRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEER6Tensori,"axG",@progbits,_Z17load_mnist_imagesRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEER6Tensori,comdat
	.size	_Z17load_mnist_imagesRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEER6Tensori, .-_Z17load_mnist_imagesRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEER6Tensori
	.section	.rodata._Z11save_binaryRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIfSaIfEE.str1.8,"aMS",@progbits,1
	.align	3
.LC17:
	.string	"The file could not be opened for writing: "
	.section	.text._Z11save_binaryRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIfSaIfEE,"axG",@progbits,_Z11save_binaryRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIfSaIfEE,comdat
	.align	2
	.p2align 5,,15
	.weak	_Z11save_binaryRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIfSaIfEE
	.type	_Z11save_binaryRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIfSaIfEE, %function
_Z11save_binaryRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIfSaIfEE:
.LFB6762:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA6762
	sub	sp, sp, #640	//,,
	.cfi_def_cfa_offset 640
	stp	x29, x30, [sp]	//,,
	.cfi_offset 29, -640
	.cfi_offset 30, -632
	mov	x29, sp	//,
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 25, -576
	.cfi_offset 26, -568
	mov	x25, x0	// tmp238, filepath
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	add	x0, sp, 368	// tmp250,,
// src/cpp/include/utils.h:101: inline void save_binary(const std::string& filepath, const std::vector<float>& v) {
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -624
	.cfi_offset 20, -616
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	add	x19, sp, 112	// tmp233,,
// src/cpp/include/utils.h:101: inline void save_binary(const std::string& filepath, const std::vector<float>& v) {
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 21, -608
	.cfi_offset 22, -600
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	adrp	x21, _ZTVSt9basic_iosIcSt11char_traitsIcEE+16	// tmp231,
	add	x21, x21, :lo12:_ZTVSt9basic_iosIcSt11char_traitsIcEE+16	// tmp229, tmp231,
// src/cpp/include/utils.h:101: inline void save_binary(const std::string& filepath, const std::vector<float>& v) {
	stp	x23, x24, [sp, 48]	//,,
	.cfi_offset 23, -592
	.cfi_offset 24, -584
// src/cpp/include/utils.h:101: inline void save_binary(const std::string& filepath, const std::vector<float>& v) {
	mov	x24, x1	// v, tmp239
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	bl	_ZNSt8ios_baseC2Ev		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:438:       { this->init(0); }
	adrp	x2, _ZTTSt14basic_ofstreamIcSt11char_traitsIcEE	// tmp140,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:465:       : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), 
	strh	wzr, [sp, 592]	//, MEM <unsigned short> [(void *)&file + 480B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	add	x3, sp, 600	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:438:       { this->init(0); }
	add	x2, x2, :lo12:_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE	// tmp139, tmp140,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:438:       { this->init(0); }
	mov	x1, 0	//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:438:       { this->init(0); }
	ldp	x20, x26, [x2, 8]	// _31, _35,
	ldr	x2, [x20, -24]	// MEM[(long int *)_31 + -24B], MEM[(long int *)_31 + -24B]
	str	x20, [sp, 112]	// _31, MEM[(struct basic_ostream *)&file]._vptr.basic_ostream
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	str	x21, [sp, 368]	// tmp229, MEM[(struct basic_ios *)&file + 256B].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:465:       : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), 
	str	xzr, [sp, 584]	//, MEM[(struct basic_ios *)&file + 256B]._M_tie
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	stp	xzr, xzr, [x3]	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:438:       { this->init(0); }
	add	x0, x19, x2	//, tmp233, MEM[(long int *)_31 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	stp	xzr, xzr, [x3, 16]	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:438:       { this->init(0); }
	str	x26, [x19, x2]	// _35, MEM[(struct basic_ios *)_34].D.81517._vptr.ios_base
.LEHB22:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:438:       { this->init(0); }
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E		//
.LEHE22:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:876:       : __ostream_type(), _M_filebuf()
	adrp	x23, _ZTVSt14basic_ofstreamIcSt11char_traitsIcEE+24	// tmp232,
	adrp	x22, _ZTVSt14basic_ofstreamIcSt11char_traitsIcEE+64	// tmp236,
	add	x23, x23, :lo12:_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE+24	// tmp230, tmp232,
	add	x22, x22, :lo12:_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE+64	// tmp235, tmp236,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:876:       : __ostream_type(), _M_filebuf()
	add	x0, sp, 120	// tmp251,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:876:       : __ostream_type(), _M_filebuf()
	str	x23, [sp, 112]	// tmp230, MEM[(struct basic_ofstream *)&file].D.88839._vptr.basic_ostream
	str	x22, [sp, 368]	// tmp235, MEM[(struct basic_ios *)&file + 256B].D.81517._vptr.ios_base
.LEHB23:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:876:       : __ostream_type(), _M_filebuf()
	bl	_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev		//
.LEHE23:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:878: 	this->init(&_M_filebuf);
	add	x1, sp, 120	// tmp252,,
	add	x0, sp, 368	// tmp253,,
.LEHB24:
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:342:       { return open(__s.c_str(), __mode); }
	ldr	x1, [x25]	//, MEM[(char * *)filepath_4(D)]
	mov	w2, 20	//,
	add	x0, sp, 120	// tmp254,,
	bl	_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:1011: 	if (!_M_filebuf.open(__s, __mode | ios_base::out))
	cbz	x0, .L248	// tmp240,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:1016: 	  this->clear();
	ldr	x0, [sp, 112]	// file.D.88839._vptr.basic_ostream,
	mov	w1, 0	//,
	ldr	x0, [x0, -24]	// MEM[(long int *)_47 + -24B], MEM[(long int *)_47 + -24B]
	add	x0, x19, x0	//, tmp233, MEM[(long int *)_47 + -24B]
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
.LEHE24:
.L229:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:276:       { return _M_file.is_open(); }
	add	x0, sp, 232	// tmp255,,
	bl	_ZNKSt12__basic_fileIcE7is_openEv		//
// src/cpp/include/utils.h:103:     if (!file.is_open())
	tbz	x0, 0, .L249	// tmp241,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:993:       { return size_type(this->_M_impl._M_finish - this->_M_impl._M_start); }
	ldp	x1, x2, [x24]	// _22, v_7(D)->D.57594._M_impl.D.56895._M_finish,* v
// src/cpp/include/utils.h:105:     file.write(reinterpret_cast<const char*>(v.data()),
	mov	x0, x19	//, tmp233
	sub	x2, x2, x1	//, v_7(D)->D.57594._M_impl.D.56895._M_finish, _22
.LEHB25:
	bl	_ZNSo5writeEPKcl		//
.LEHE25:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:262:       }
	adrp	x1, _ZTVSt13basic_filebufIcSt11char_traitsIcEE+16	// tmp203,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:259: 	  { this->close(); }
	add	x0, sp, 120	// tmp261,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:912:       { }
	str	x22, [sp, 368]	// tmp235, MEM[(struct basic_ios *)&file + 256B].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:262:       }
	add	x1, x1, :lo12:_ZTVSt13basic_filebufIcSt11char_traitsIcEE+16	// tmp202, tmp203,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:912:       { }
	stp	x23, x1, [sp, 112]	// tmp230, tmp202,
.LEHB26:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:259: 	  { this->close(); }
	bl	_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv		//
.LEHE26:
.L234:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:262:       }
	add	x0, sp, 232	// tmp262,,
	bl	_ZNSt12__basic_fileIcED1Ev		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/streambuf:205:       { }
	adrp	x1, _ZTVSt15basic_streambufIcSt11char_traitsIcEE+16	// tmp209,
	add	x0, sp, 176	//,,
	add	x1, x1, :lo12:_ZTVSt15basic_streambufIcSt11char_traitsIcEE+16	// tmp208, tmp209,
	str	x1, [sp, 120]	// tmp208, MEM[(struct basic_streambuf *)&file + 8B]._vptr.basic_streambuf
	bl	_ZNSt6localeD1Ev		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:101:       ~basic_ostream() { }
	ldr	x1, [x20, -24]	// MEM[(long int *)_31 + -24B], MEM[(long int *)_31 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:286:       ~basic_ios() { }
	add	x0, sp, 368	// tmp263,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:101:       ~basic_ostream() { }
	str	x20, [sp, 112]	// _31, MEM[(struct basic_ostream *)&file]._vptr.basic_ostream
	str	x26, [x19, x1]	// _35, MEM[(struct basic_ios *)_62].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:286:       ~basic_ios() { }
	str	x21, [sp, 368]	// tmp229, MEM[(struct basic_ios *)&file + 256B].D.81517._vptr.ios_base
	bl	_ZNSt8ios_baseD2Ev		//
// src/cpp/include/utils.h:107: }
	ldp	x29, x30, [sp]	//,,
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x25, x26, [sp, 64]	//,,
	add	sp, sp, 640	//,,
	.cfi_remember_state
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 29
	.cfi_restore 30
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L248:
	.cfi_restore_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:740: 	  this->setstate(ios_base::failbit);
	ldr	x0, [sp, 112]	// MEM[(struct basic_ifstream *)&file].D.88669._vptr.basic_istream,
	ldr	x0, [x0, -24]	// MEM[(long int *)_51 + -24B], MEM[(long int *)_51 + -24B]
	add	x0, x19, x0	// _54, tmp233, MEM[(long int *)_51 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:187:   { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
	ldr	w1, [x0, 32]	//, MEM[(const struct basic_ios *)_54].D.81517._M_streambuf_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	orr	w1, w1, 4	//, MEM[(const struct basic_ios *)_54].D.81517._M_streambuf_state,
.LEHB27:
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
.LEHE27:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	b	.L229		//
.L243:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:880:       }
	mov	x22, x0	// tmp242,
	add	x0, sp, 120	// tmp256,,
	bl	_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev		//
.L231:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:101:       ~basic_ostream() { }
	ldr	x0, [x20, -24]	// MEM[(long int *)_31 + -24B], MEM[(long int *)_31 + -24B]
	str	x20, [sp, 112]	// _31, MEM[(struct basic_ostream *)&file]._vptr.basic_ostream
	str	x26, [x19, x0]	// _35, MEM[(struct basic_ios *)_41].D.81517._vptr.ios_base
.L232:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:286:       ~basic_ios() { }
	add	x0, sp, 368	// tmp257,,
	str	x21, [sp, 368]	// tmp229, MEM[(struct basic_ios *)&file + 256B].D.81517._vptr.ios_base
	bl	_ZNSt8ios_baseD2Ev		//
	mov	x0, x22	//, tmp179
.LEHB28:
	bl	_Unwind_Resume		//
.LEHE28:
.L242:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:101:       ~basic_ostream() { }
	mov	x22, x0	// tmp173, tmp243
	b	.L231		//
.L241:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:286:       ~basic_ios() { }
	mov	x22, x0	// tmp179, tmp244
	b	.L232		//
.L244:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:260: 	__catch(...)
	bl	__cxa_begin_catch		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:260: 	__catch(...)
	bl	__cxa_end_catch		//
	b	.L234		//
.L240:
// src/cpp/include/utils.h:107: }
	mov	x20, x0	// tmp224, tmp248
.L237:
	mov	x0, x19	//, tmp233
	bl	_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev		//
	mov	x0, x20	//, tmp224
.LEHB29:
	bl	_Unwind_Resume		//
.LEHE29:
.L249:
// src/cpp/include/utils.h:104:         throw std::runtime_error("The file could not be opened for writing: " + filepath);
	mov	x0, 16	//,
	bl	__cxa_allocate_exception		//
	mov	x21, x0	// tmp245,
// src/cpp/include/utils.h:104:         throw std::runtime_error("The file could not be opened for writing: " + filepath);
	adrp	x0, .LC17	// tmp187,
	ldp	x1, x2, [x25]	//,,* filepath
	add	x8, sp, 80	// tmp258,,
	add	x0, x0, :lo12:.LC17	//, tmp187,
.LEHB30:
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_.isra.0		//
.LEHE30:
// src/cpp/include/utils.h:104:         throw std::runtime_error("The file could not be opened for writing: " + filepath);
	add	x1, sp, 80	// tmp259,,
	mov	x0, x21	//, _13
.LEHB31:
	bl	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE		//
.LEHE31:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	add	x0, sp, 80	// tmp260,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/include/utils.h:104:         throw std::runtime_error("The file could not be opened for writing: " + filepath);
	adrp	x2, _ZNSt13runtime_errorD1Ev	// tmp192,
	adrp	x1, _ZTISt13runtime_error	// tmp194,
	mov	x0, x21	//, _13
	add	x2, x2, :lo12:_ZNSt13runtime_errorD1Ev	//, tmp192,
	add	x1, x1, :lo12:_ZTISt13runtime_error	//, tmp194,
.LEHB32:
	bl	__cxa_throw		//
.LEHE32:
.L239:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x20, x0	// tmp246,
	add	x0, sp, 80	// tmp264,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
.L236:
// src/cpp/include/utils.h:104:         throw std::runtime_error("The file could not be opened for writing: " + filepath);
	mov	x0, x21	//, _13
	bl	__cxa_free_exception		//
	b	.L237		//
.L238:
	mov	x20, x0	// tmp220, tmp247
	b	.L236		//
	.cfi_endproc
.LFE6762:
	.section	.gcc_except_table
	.align	2
.LLSDA6762:
	.byte	0xff
	.byte	0x9b
	.uleb128 .LLSDATT6762-.LLSDATTD6762
.LLSDATTD6762:
	.byte	0x1
	.uleb128 .LLSDACSE6762-.LLSDACSB6762
.LLSDACSB6762:
	.uleb128 .LEHB22-.LFB6762
	.uleb128 .LEHE22-.LEHB22
	.uleb128 .L241-.LFB6762
	.uleb128 0
	.uleb128 .LEHB23-.LFB6762
	.uleb128 .LEHE23-.LEHB23
	.uleb128 .L242-.LFB6762
	.uleb128 0
	.uleb128 .LEHB24-.LFB6762
	.uleb128 .LEHE24-.LEHB24
	.uleb128 .L243-.LFB6762
	.uleb128 0
	.uleb128 .LEHB25-.LFB6762
	.uleb128 .LEHE25-.LEHB25
	.uleb128 .L240-.LFB6762
	.uleb128 0
	.uleb128 .LEHB26-.LFB6762
	.uleb128 .LEHE26-.LEHB26
	.uleb128 .L244-.LFB6762
	.uleb128 0x1
	.uleb128 .LEHB27-.LFB6762
	.uleb128 .LEHE27-.LEHB27
	.uleb128 .L243-.LFB6762
	.uleb128 0
	.uleb128 .LEHB28-.LFB6762
	.uleb128 .LEHE28-.LEHB28
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB29-.LFB6762
	.uleb128 .LEHE29-.LEHB29
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB30-.LFB6762
	.uleb128 .LEHE30-.LEHB30
	.uleb128 .L238-.LFB6762
	.uleb128 0
	.uleb128 .LEHB31-.LFB6762
	.uleb128 .LEHE31-.LEHB31
	.uleb128 .L239-.LFB6762
	.uleb128 0
	.uleb128 .LEHB32-.LFB6762
	.uleb128 .LEHE32-.LEHB32
	.uleb128 .L240-.LFB6762
	.uleb128 0
.LLSDACSE6762:
	.byte	0x1
	.byte	0
	.align	2
	.4byte	0

.LLSDATT6762:
	.section	.text._Z11save_binaryRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIfSaIfEE,"axG",@progbits,_Z11save_binaryRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIfSaIfEE,comdat
	.size	_Z11save_binaryRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIfSaIfEE, .-_Z11save_binaryRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIfSaIfEE
	.section	.rodata._Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE.str1.8,"aMS",@progbits,1
	.align	3
.LC18:
	.string	"The file could not be opened: "
	.align	3
.LC19:
	.string	"File size mismatch for: "
	.align	3
.LC20:
	.string	"Successfully loaded: "
	.align	3
.LC21:
	.string	"\n"
	.align	3
.LC22:
	.string	"Read error: "
	.section	.text._Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE,"axG",@progbits,_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE,comdat
	.align	2
	.p2align 5,,15
	.weak	_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE
	.type	_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE, %function
_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE:
.LFB6750:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA6750
	sub	sp, sp, #656	//,,
	.cfi_def_cfa_offset 656
	stp	x29, x30, [sp]	//,,
	.cfi_offset 29, -656
	.cfi_offset 30, -648
	mov	x29, sp	//,
	stp	x19, x20, [sp, 16]	//,,
	.cfi_offset 19, -640
	.cfi_offset 20, -632
	mov	x20, x0	// tmp348, filepath
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	add	x0, sp, 392	// tmp371,,
	add	x19, sp, 128	// tmp340,,
// src/cpp/include/utils.h:39: inline void load_binary_weights(const std::string& filepath, std::vector<float>& target_vector) {
	stp	x21, x22, [sp, 32]	//,,
	.cfi_offset 21, -624
	.cfi_offset 22, -616
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	adrp	x22, _ZTVSt9basic_iosIcSt11char_traitsIcEE+16	// tmp336,
	add	x22, x22, :lo12:_ZTVSt9basic_iosIcSt11char_traitsIcEE+16	// tmp347, tmp336,
// src/cpp/include/utils.h:39: inline void load_binary_weights(const std::string& filepath, std::vector<float>& target_vector) {
	stp	x23, x24, [sp, 48]	//,,
	stp	x25, x26, [sp, 64]	//,,
	.cfi_offset 23, -608
	.cfi_offset 24, -600
	.cfi_offset 25, -592
	.cfi_offset 26, -584
	mov	x26, x1	// target_vector, tmp349
	str	x27, [sp, 80]	//,
	.cfi_offset 27, -576
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	bl	_ZNSt8ios_baseC2Ev		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:697:       : _M_gcount(streamsize(0))
	adrp	x0, _ZTTSt14basic_ifstreamIcSt11char_traitsIcEE	// tmp177,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:465:       : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), 
	strh	wzr, [sp, 616]	//, MEM <unsigned short> [(void *)&file + 488B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	add	x2, sp, 624	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:697:       : _M_gcount(streamsize(0))
	add	x0, x0, :lo12:_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE	// tmp176, tmp177,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:698:       { this->init(0); }
	mov	x1, 0	//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:697:       : _M_gcount(streamsize(0))
	ldp	x21, x25, [x0, 8]	// _87, _91,
	ldr	x0, [x21, -24]	// MEM[(long int *)_87 + -24B], MEM[(long int *)_87 + -24B]
	str	x21, [sp, 128]	// _87, MEM[(struct basic_istream *)&file]._vptr.basic_istream
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	str	x22, [sp, 392]	// tmp347, MEM[(struct basic_ios *)&file + 264B].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:465:       : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), 
	str	xzr, [sp, 608]	//, MEM[(struct basic_ios *)&file + 264B]._M_tie
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:466: 	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
	stp	xzr, xzr, [x2]	//,,
	stp	xzr, xzr, [x2, 16]	//,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:697:       : _M_gcount(streamsize(0))
	str	x25, [x19, x0]	// _91, MEM[(struct basic_ios *)_90].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:697:       : _M_gcount(streamsize(0))
	str	xzr, [sp, 136]	//, MEM[(struct basic_istream *)&file]._M_gcount
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:698:       { this->init(0); }
	ldr	x0, [x21, -24]	// MEM[(long int *)_87 + -24B], MEM[(long int *)_87 + -24B]
	add	x0, x19, x0	//, tmp340, MEM[(long int *)_87 + -24B]
.LEHB33:
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E		//
.LEHE33:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:604:       : __istream_type(), _M_filebuf()
	adrp	x24, _ZTVSt14basic_ifstreamIcSt11char_traitsIcEE+24	// tmp346,
	adrp	x23, _ZTVSt14basic_ifstreamIcSt11char_traitsIcEE+64	// tmp338,
	add	x24, x24, :lo12:_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE+24	// tmp345, tmp346,
	add	x23, x23, :lo12:_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE+64	// tmp337, tmp338,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:604:       : __istream_type(), _M_filebuf()
	add	x0, sp, 144	// tmp372,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:604:       : __istream_type(), _M_filebuf()
	str	x24, [sp, 128]	// tmp345, MEM[(struct basic_ifstream *)&file].D.88669._vptr.basic_istream
	str	x23, [sp, 392]	// tmp337, MEM[(struct basic_ios *)&file + 264B].D.81517._vptr.ios_base
.LEHB34:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:604:       : __istream_type(), _M_filebuf()
	bl	_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev		//
.LEHE34:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:606: 	this->init(&_M_filebuf);
	add	x1, sp, 144	// tmp373,,
	add	x0, sp, 392	// tmp374,,
.LEHB35:
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:342:       { return open(__s.c_str(), __mode); }
	ldr	x1, [x20]	//, MEM[(char * *)filepath_12(D)]
	mov	w2, 14	//,
	add	x0, sp, 144	// tmp375,,
	bl	_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode		//
	mov	x2, x0	// tmp350,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:740: 	  this->setstate(ios_base::failbit);
	ldr	x0, [sp, 128]	// file.D.88669._vptr.basic_istream, file.D.88669._vptr.basic_istream
	ldr	x1, [x0, -24]	// MEM[(long int *)pretmp_179 + -24B], MEM[(long int *)pretmp_179 + -24B]
	add	x0, x19, x1	// _135, tmp340, MEM[(long int *)pretmp_179 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:739: 	if (!_M_filebuf.open(__s, __mode | ios_base::in))
	cbz	x2, .L291	// tmp350,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:744: 	  this->clear();
	mov	w1, 0	//,
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
.LEHE35:
.L252:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:276:       { return _M_file.is_open(); }
	add	x0, sp, 256	// tmp376,,
	bl	_ZNKSt12__basic_fileIcE7is_openEv		//
// src/cpp/include/utils.h:41:     if (!file.is_open())
	tbz	x0, 0, .L292	// tmp351,,
// src/cpp/include/utils.h:44:     std::streamsize size = file.tellg();
	mov	x0, x19	//, tmp340
.LEHB36:
	bl	_ZNSi5tellgEv		//
	mov	x27, x0	// _57, tmp356
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:993:       { return size_type(this->_M_impl._M_finish - this->_M_impl._M_start); }
	ldp	x1, x0, [x26]	// MEM[(const struct vector *)target_vector_18(D)].D.57594._M_impl.D.56895._M_start, MEM[(const struct vector *)target_vector_18(D)].D.57594._M_impl.D.56895._M_finish,* target_vector
	sub	x0, x0, x1	// _14, MEM[(const struct vector *)target_vector_18(D)].D.57594._M_impl.D.56895._M_finish, MEM[(const struct vector *)target_vector_18(D)].D.57594._M_impl.D.56895._M_start
// src/cpp/include/utils.h:45:     if (size != static_cast<std::streamsize>(target_vector.size() * sizeof(float)))
	cmp	x0, x27	// _14, _57
	bne	.L293		//,
// src/cpp/include/utils.h:48:     file.seekg(0, std::ios::beg);
	mov	w2, 0	//,
	mov	x1, 0	//,
	mov	x0, x19	//, tmp340
	bl	_ZNSi5seekgElSt12_Ios_Seekdir		//
// src/cpp/include/utils.h:49:     if (file.read(reinterpret_cast<char*>(target_vector.data()), size))
	ldr	x1, [x26]	//, target_vector_18(D)->D.57594._M_impl.D.56895._M_start
	mov	x2, x27	//, _57
	mov	x0, x19	//, tmp340
	bl	_ZNSi4readEPcl		//
	ldr	x1, [x0]	// _23->_vptr.basic_istream, _23->_vptr.basic_istream
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:182:   { return _Ios_Iostate(static_cast<int>(__a) & static_cast<int>(__b)); }
	mov	w2, 5	// tmp253,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:142:       { return _M_streambuf_state; }
	ldr	x1, [x1, -24]	// MEM[(long int *)_1 + -24B], MEM[(long int *)_1 + -24B]
	add	x1, x0, x1	// tmp250, _23, MEM[(long int *)_1 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:182:   { return _Ios_Iostate(static_cast<int>(__a) & static_cast<int>(__b)); }
	ldr	w0, [x1, 32]	//, MEM[(const struct basic_ios *)_4].D.81517._M_streambuf_state
// src/cpp/include/utils.h:49:     if (file.read(reinterpret_cast<char*>(target_vector.data()), size))
	tst	w0, w2	// MEM[(const struct basic_ios *)_4].D.81517._M_streambuf_state, tmp253
	bne	.L259		//,
// src/cpp/include/utils.h:50:         std::cout << Color::DIM_GREEN << "Successfully loaded: " << filepath << Color::RESET << "\n";
	adrp	x0, _ZN5Color9DIM_GREENE	// tmp255,
	ldr	x27, [x0, #:lo12:_ZN5Color9DIM_GREENE]	// DIM_GREEN.87_5, DIM_GREEN
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:670:       if (!__s)
	cbz	x27, .L294	// DIM_GREEN.87_5,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x27	//, DIM_GREEN.87_5
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x26, _ZSt4cout	// tmp344,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	add	x26, x26, :lo12:_ZSt4cout	// tmp343, tmp344,
	mov	x2, x0	//, tmp359
	mov	x1, x27	//, DIM_GREEN.87_5
	mov	x0, x26	//, tmp343
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
.L261:
	adrp	x1, .LC20	// tmp268,
	mov	x2, 21	//,
	add	x1, x1, :lo12:.LC20	//, tmp268,
	mov	x0, x26	//, tmp343
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:4082:       return __ostream_insert(__os, __str.data(), __str.size());
	ldp	x1, x2, [x20]	//,,* filepath
	mov	x0, x26	//, tmp343
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/include/utils.h:50:         std::cout << Color::DIM_GREEN << "Successfully loaded: " << filepath << Color::RESET << "\n";
	adrp	x1, _ZN5Color5RESETE	// tmp276,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:4082:       return __ostream_insert(__os, __str.data(), __str.size());
	mov	x20, x0	// _64, tmp360
// src/cpp/include/utils.h:50:         std::cout << Color::DIM_GREEN << "Successfully loaded: " << filepath << Color::RESET << "\n";
	ldr	x26, [x1, #:lo12:_ZN5Color5RESETE]	// RESET.88_6, RESET
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:670:       if (!__s)
	cbz	x26, .L295	// RESET.88_6,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x26	//, RESET.88_6
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, x0	//, tmp361
	mov	x1, x26	//, RESET.88_6
	mov	x0, x20	//, _64
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
.L263:
	adrp	x1, .LC21	// tmp283,
	mov	x0, x20	//, _64
	add	x1, x1, :lo12:.LC21	//, tmp283,
	mov	x2, 1	//,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
.LEHE36:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:262:       }
	adrp	x1, _ZTVSt13basic_filebufIcSt11char_traitsIcEE+16	// tmp299,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:259: 	  { this->close(); }
	add	x0, sp, 144	// tmp379,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:640:       { }
	str	x24, [sp, 128]	// tmp345, file.D.88669._vptr.basic_istream
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:262:       }
	add	x1, x1, :lo12:_ZTVSt13basic_filebufIcSt11char_traitsIcEE+16	// tmp298, tmp299,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:640:       { }
	str	x23, [sp, 392]	// tmp337, MEM[(struct basic_ios *)&file + 264B].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:262:       }
	str	x1, [sp, 144]	// tmp298, MEM[(struct basic_filebuf *)&file + 16B].D.88477._vptr.basic_streambuf
.LEHB37:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:259: 	  { this->close(); }
	bl	_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv		//
.LEHE37:
.L266:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:262:       }
	add	x0, sp, 256	// tmp380,,
	bl	_ZNSt12__basic_fileIcED1Ev		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/streambuf:205:       { }
	adrp	x1, _ZTVSt15basic_streambufIcSt11char_traitsIcEE+16	// tmp305,
	add	x0, sp, 200	//,,
	add	x1, x1, :lo12:_ZTVSt15basic_streambufIcSt11char_traitsIcEE+16	// tmp304, tmp305,
	str	x1, [sp, 144]	// tmp304, MEM[(struct basic_streambuf *)&file + 16B]._vptr.basic_streambuf
	bl	_ZNSt6localeD1Ev		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:106:       { _M_gcount = streamsize(0); }
	ldr	x1, [x21, -24]	// MEM[(long int *)_87 + -24B], MEM[(long int *)_87 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:286:       ~basic_ios() { }
	add	x0, sp, 392	// tmp381,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:106:       { _M_gcount = streamsize(0); }
	str	x21, [sp, 128]	// _87, MEM[(struct basic_istream *)&file]._vptr.basic_istream
	str	x25, [x19, x1]	// _91, MEM[(struct basic_ios *)_139].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:106:       { _M_gcount = streamsize(0); }
	str	xzr, [sp, 136]	//, MEM[(struct basic_istream *)&file]._M_gcount
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:286:       ~basic_ios() { }
	str	x22, [sp, 392]	// tmp347, MEM[(struct basic_ios *)&file + 264B].D.81517._vptr.ios_base
	bl	_ZNSt8ios_baseD2Ev		//
// src/cpp/include/utils.h:53: }
	ldr	x27, [sp, 80]	//,
	ldp	x29, x30, [sp]	//,,
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x25, x26, [sp, 64]	//,,
	add	sp, sp, 656	//,,
	.cfi_remember_state
	.cfi_restore 27
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 29
	.cfi_restore 30
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L291:
	.cfi_restore_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:187:   { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
	ldr	w1, [x0, 32]	//, MEM[(const struct basic_ios *)_135].D.81517._M_streambuf_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	orr	w1, w1, 4	//, MEM[(const struct basic_ios *)_135].D.81517._M_streambuf_state,
.LEHB38:
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
.LEHE38:
	b	.L252		//
	.p2align 2,,3
.L295:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:671: 	__out.setstate(ios_base::badbit);
	ldr	x0, [x0]	// _64->_vptr.basic_ostream, _64->_vptr.basic_ostream
	ldr	x0, [x0, -24]	// MEM[(long int *)_126 + -24B], MEM[(long int *)_126 + -24B]
	add	x0, x20, x0	// _129, _64, MEM[(long int *)_126 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:187:   { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
	ldr	w1, [x0, 32]	//, MEM[(const struct basic_ios *)_129].D.81517._M_streambuf_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	orr	w1, w1, 1	//, MEM[(const struct basic_ios *)_129].D.81517._M_streambuf_state,
.LEHB39:
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
	b	.L263		//
	.p2align 2,,3
.L294:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:671: 	__out.setstate(ios_base::badbit);
	adrp	x26, _ZSt4cout	// tmp344,
	ldr	x0, [x26, #:lo12:_ZSt4cout]	// cout._vptr.basic_ostream, cout._vptr.basic_ostream
	add	x26, x26, :lo12:_ZSt4cout	// tmp343, tmp344,
	ldr	x0, [x0, -24]	// MEM[(long int *)_118 + -24B], MEM[(long int *)_118 + -24B]
	add	x0, x26, x0	// _121, tmp343, MEM[(long int *)_118 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:187:   { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
	ldr	w1, [x0, 32]	//, MEM[(const struct basic_ios *)_121].D.81517._M_streambuf_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	orr	w1, w1, 1	//, MEM[(const struct basic_ios *)_121].D.81517._M_streambuf_state,
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
.LEHE39:
	b	.L261		//
.L283:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:608:       }
	mov	x20, x0	// tmp352,
	add	x0, sp, 144	// tmp377,,
	bl	_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev		//
.L256:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:106:       { _M_gcount = streamsize(0); }
	ldr	x0, [x21, -24]	// MEM[(long int *)_87 + -24B], MEM[(long int *)_87 + -24B]
	str	x21, [sp, 128]	// _87, MEM[(struct basic_istream *)&file]._vptr.basic_istream
	str	x25, [x19, x0]	// _91, MEM[(struct basic_ios *)_100].D.81517._vptr.ios_base
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:106:       { _M_gcount = streamsize(0); }
	str	xzr, [sp, 136]	//, MEM[(struct basic_istream *)&file]._M_gcount
.L257:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:286:       ~basic_ios() { }
	add	x0, sp, 392	// tmp378,,
	str	x22, [sp, 392]	// tmp347, MEM[(struct basic_ios *)&file + 264B].D.81517._vptr.ios_base
	bl	_ZNSt8ios_baseD2Ev		//
	mov	x0, x20	//, tmp215
.LEHB40:
	bl	_Unwind_Resume		//
.LEHE40:
.L282:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/istream:106:       { _M_gcount = streamsize(0); }
	mov	x20, x0	// tmp209, tmp353
	b	.L256		//
.L281:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:286:       ~basic_ios() { }
	mov	x20, x0	// tmp215, tmp354
	b	.L257		//
.L284:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:260: 	__catch(...)
	bl	__cxa_begin_catch		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:260: 	__catch(...)
	bl	__cxa_end_catch		//
	b	.L266		//
.L259:
// src/cpp/include/utils.h:52:         throw std::runtime_error("Read error: " + filepath);
	mov	x0, 16	//,
// src/cpp/include/utils.h:52:         throw std::runtime_error("Read error: " + filepath);
	add	x21, sp, 96	// tmp335,,
// src/cpp/include/utils.h:52:         throw std::runtime_error("Read error: " + filepath);
	bl	__cxa_allocate_exception		//
	mov	x22, x0	// tmp362,
// src/cpp/include/utils.h:52:         throw std::runtime_error("Read error: " + filepath);
	adrp	x0, .LC22	// tmp286,
	ldp	x1, x2, [x20]	//,,* filepath
	mov	x8, x21	//, tmp335
	add	x0, x0, :lo12:.LC22	//, tmp286,
.LEHB41:
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_.isra.0		//
.LEHE41:
// src/cpp/include/utils.h:52:         throw std::runtime_error("Read error: " + filepath);
	mov	x1, x21	//, tmp335
	mov	x0, x22	//, _25
.LEHB42:
	bl	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE		//
.LEHE42:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x21	//, tmp335
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/include/utils.h:52:         throw std::runtime_error("Read error: " + filepath);
	adrp	x2, _ZNSt13runtime_errorD1Ev	// tmp291,
	adrp	x1, _ZTISt13runtime_error	// tmp293,
	mov	x0, x22	//, _25
	add	x2, x2, :lo12:_ZNSt13runtime_errorD1Ev	//, tmp291,
	add	x1, x1, :lo12:_ZTISt13runtime_error	//, tmp293,
.LEHB43:
	bl	__cxa_throw		//
.LEHE43:
.L293:
// src/cpp/include/utils.h:46:         throw std::runtime_error("File size mismatch for: " + filepath);
	mov	x0, 16	//,
// src/cpp/include/utils.h:46:         throw std::runtime_error("File size mismatch for: " + filepath);
	add	x21, sp, 96	// tmp335,,
// src/cpp/include/utils.h:46:         throw std::runtime_error("File size mismatch for: " + filepath);
	bl	__cxa_allocate_exception		//
	mov	x22, x0	// tmp357,
// src/cpp/include/utils.h:46:         throw std::runtime_error("File size mismatch for: " + filepath);
	adrp	x0, .LC19	// tmp237,
	ldp	x1, x2, [x20]	//,,* filepath
	mov	x8, x21	//, tmp335
	add	x0, x0, :lo12:.LC19	//, tmp237,
.LEHB44:
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_.isra.0		//
.LEHE44:
// src/cpp/include/utils.h:46:         throw std::runtime_error("File size mismatch for: " + filepath);
	mov	x1, x21	//, tmp335
	mov	x0, x22	//, _40
.LEHB45:
	bl	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE		//
.LEHE45:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x21	//, tmp335
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/include/utils.h:46:         throw std::runtime_error("File size mismatch for: " + filepath);
	adrp	x2, _ZNSt13runtime_errorD1Ev	// tmp242,
	adrp	x1, _ZTISt13runtime_error	// tmp244,
	mov	x0, x22	//, _40
	add	x2, x2, :lo12:_ZNSt13runtime_errorD1Ev	//, tmp242,
	add	x1, x1, :lo12:_ZTISt13runtime_error	//, tmp244,
.LEHB46:
	bl	__cxa_throw		//
.LEHE46:
.L280:
.L290:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x20, x0	// tmp367,
	mov	x0, x21	//, tmp335
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
.L273:
// src/cpp/include/utils.h:52:         throw std::runtime_error("Read error: " + filepath);
	mov	x0, x22	//, _25
	bl	__cxa_free_exception		//
.L269:
// src/cpp/include/utils.h:53: }
	mov	x0, x19	//, tmp340
	bl	_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev		//
	mov	x0, x20	//, tmp320
.LEHB47:
	bl	_Unwind_Resume		//
.LEHE47:
.L279:
// src/cpp/include/utils.h:52:         throw std::runtime_error("Read error: " + filepath);
	mov	x20, x0	// tmp328, tmp368
	b	.L273		//
.L276:
// src/cpp/include/utils.h:44:     std::streamsize size = file.tellg();
	mov	x20, x0	// tmp320, tmp369
	b	.L269		//
.L292:
// src/cpp/include/utils.h:42:         throw std::runtime_error("The file could not be opened: " + filepath);
	mov	x0, 16	//,
// src/cpp/include/utils.h:42:         throw std::runtime_error("The file could not be opened: " + filepath);
	add	x21, sp, 96	// tmp335,,
// src/cpp/include/utils.h:42:         throw std::runtime_error("The file could not be opened: " + filepath);
	bl	__cxa_allocate_exception		//
	mov	x22, x0	// tmp355,
// src/cpp/include/utils.h:42:         throw std::runtime_error("The file could not be opened: " + filepath);
	adrp	x0, .LC18	// tmp223,
	ldp	x1, x2, [x20]	//,,* filepath
	mov	x8, x21	//, tmp335
	add	x0, x0, :lo12:.LC18	//, tmp223,
.LEHB48:
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_.isra.0		//
.LEHE48:
// src/cpp/include/utils.h:42:         throw std::runtime_error("The file could not be opened: " + filepath);
	mov	x1, x21	//, tmp335
	mov	x0, x22	//, _49
.LEHB49:
	bl	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE		//
.LEHE49:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x21	//, tmp335
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/include/utils.h:42:         throw std::runtime_error("The file could not be opened: " + filepath);
	adrp	x2, _ZNSt13runtime_errorD1Ev	// tmp228,
	adrp	x1, _ZTISt13runtime_error	// tmp230,
	mov	x0, x22	//, _49
	add	x2, x2, :lo12:_ZNSt13runtime_errorD1Ev	//, tmp228,
	add	x1, x1, :lo12:_ZTISt13runtime_error	//, tmp230,
.LEHB50:
	bl	__cxa_throw		//
.LEHE50:
.L278:
	b	.L290		//
.L277:
// src/cpp/include/utils.h:52:         throw std::runtime_error("Read error: " + filepath);
	mov	x20, x0	// tmp328, tmp368
	b	.L273		//
.L275:
	b	.L290		//
.L274:
	mov	x20, x0	// tmp328, tmp368
	b	.L273		//
	.cfi_endproc
.LFE6750:
	.section	.gcc_except_table
	.align	2
.LLSDA6750:
	.byte	0xff
	.byte	0x9b
	.uleb128 .LLSDATT6750-.LLSDATTD6750
.LLSDATTD6750:
	.byte	0x1
	.uleb128 .LLSDACSE6750-.LLSDACSB6750
.LLSDACSB6750:
	.uleb128 .LEHB33-.LFB6750
	.uleb128 .LEHE33-.LEHB33
	.uleb128 .L281-.LFB6750
	.uleb128 0
	.uleb128 .LEHB34-.LFB6750
	.uleb128 .LEHE34-.LEHB34
	.uleb128 .L282-.LFB6750
	.uleb128 0
	.uleb128 .LEHB35-.LFB6750
	.uleb128 .LEHE35-.LEHB35
	.uleb128 .L283-.LFB6750
	.uleb128 0
	.uleb128 .LEHB36-.LFB6750
	.uleb128 .LEHE36-.LEHB36
	.uleb128 .L276-.LFB6750
	.uleb128 0
	.uleb128 .LEHB37-.LFB6750
	.uleb128 .LEHE37-.LEHB37
	.uleb128 .L284-.LFB6750
	.uleb128 0x1
	.uleb128 .LEHB38-.LFB6750
	.uleb128 .LEHE38-.LEHB38
	.uleb128 .L283-.LFB6750
	.uleb128 0
	.uleb128 .LEHB39-.LFB6750
	.uleb128 .LEHE39-.LEHB39
	.uleb128 .L276-.LFB6750
	.uleb128 0
	.uleb128 .LEHB40-.LFB6750
	.uleb128 .LEHE40-.LEHB40
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB41-.LFB6750
	.uleb128 .LEHE41-.LEHB41
	.uleb128 .L279-.LFB6750
	.uleb128 0
	.uleb128 .LEHB42-.LFB6750
	.uleb128 .LEHE42-.LEHB42
	.uleb128 .L280-.LFB6750
	.uleb128 0
	.uleb128 .LEHB43-.LFB6750
	.uleb128 .LEHE43-.LEHB43
	.uleb128 .L276-.LFB6750
	.uleb128 0
	.uleb128 .LEHB44-.LFB6750
	.uleb128 .LEHE44-.LEHB44
	.uleb128 .L277-.LFB6750
	.uleb128 0
	.uleb128 .LEHB45-.LFB6750
	.uleb128 .LEHE45-.LEHB45
	.uleb128 .L278-.LFB6750
	.uleb128 0
	.uleb128 .LEHB46-.LFB6750
	.uleb128 .LEHE46-.LEHB46
	.uleb128 .L276-.LFB6750
	.uleb128 0
	.uleb128 .LEHB47-.LFB6750
	.uleb128 .LEHE47-.LEHB47
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB48-.LFB6750
	.uleb128 .LEHE48-.LEHB48
	.uleb128 .L274-.LFB6750
	.uleb128 0
	.uleb128 .LEHB49-.LFB6750
	.uleb128 .LEHE49-.LEHB49
	.uleb128 .L275-.LFB6750
	.uleb128 0
	.uleb128 .LEHB50-.LFB6750
	.uleb128 .LEHE50-.LEHB50
	.uleb128 .L276-.LFB6750
	.uleb128 0
.LLSDACSE6750:
	.byte	0x1
	.byte	0
	.align	2
	.4byte	0

.LLSDATT6750:
	.section	.text._Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE,"axG",@progbits,_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE,comdat
	.size	_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE, .-_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE
	.section	.rodata.str1.8
	.align	3
.LC23:
	.string	""
	.align	3
.LC24:
	.string	"verify"
	.align	3
.LC25:
	.string	"profile"
	.align	3
.LC26:
	.string	"--verbose"
	.align	3
.LC27:
	.string	"verbose"
	.align	3
.LC28:
	.string	"perf_event_open failed: %s\n"
	.align	3
.LC29:
	.string	"PMU init failed \342\200\224 cycle counts will be invalid.\n"
	.align	3
.LC30:
	.string	"Memory allocation is starting..."
	.align	3
.LC31:
	.string	"../../data/MNIST/raw/t10k-images-idx3-ubyte"
	.align	3
.LC32:
	.string	"MNIST dataset successfully loaded."
	.align	3
.LC33:
	.string	"\nLoading real weights..."
	.align	3
.LC34:
	.string	"../python/weights_cpp/"
	.align	3
.LC35:
	.string	"model_0_weight.bin"
	.align	3
.LC36:
	.string	"model_0_bias.bin"
	.align	3
.LC37:
	.string	"model_3_weight.bin"
	.align	3
.LC38:
	.string	"model_3_bias.bin"
	.align	3
.LC39:
	.string	"model_6_weight.bin"
	.align	3
.LC40:
	.string	"model_6_bias.bin"
	.align	3
.LC41:
	.string	"model_10_weight.bin"
	.align	3
.LC42:
	.string	"model_10_bias.bin"
	.align	3
.LC44:
	.string	"  VERIFYING IMPLEMENTATION: "
	.align	3
.LC6:
	.string	"Specialized MaxPool2D"
	.align	3
.LC45:
	.string	"  "
	.align	3
.LC46:
	.string	"\n\n"
	.align	3
.LC47:
	.string	"../python/weights_cpp/cpp_logits.bin"
	.align	3
.LC48:
	.string	"Wrote logits for "
	.align	3
.LC49:
	.string	" images (implementation: "
	.align	3
.LC50:
	.string	") to "
	.align	3
.LC51:
	.string	"\n--- Per-Layer Profiling ("
	.align	3
.LC52:
	.string	" Runs) ---"
	.align	3
.LC53:
	.string	"conv1"
	.align	3
.LC54:
	.string	"relu1"
	.align	3
.LC55:
	.string	"pool1"
	.align	3
.LC56:
	.string	"conv2"
	.align	3
.LC57:
	.string	"relu2"
	.align	3
.LC58:
	.string	"pool2"
	.align	3
.LC59:
	.string	"conv3"
	.align	3
.LC60:
	.string	"relu3"
	.align	3
.LC61:
	.string	"avgpool"
	.align	3
.LC62:
	.string	"fc"
	.align	3
.LC63:
	.string	"=== Per-Layer Breakdown (median cycles) ==="
	.align	3
.LC64:
	.string	" cycles  ("
	.align	3
.LC65:
	.string	"%)\n"
	.align	3
.LC66:
	.string	"  total: "
	.align	3
.LC67:
	.string	" cycles\n\n"
	.align	3
.LC68:
	.string	"CNN_BENCH_FILTER"
	.align	3
.LC69:
	.string	"Error: CNN_BENCH_FILTER matched no implementations: "
	.align	3
.LC70:
	.string	"Benchmark filter: "
	.align	3
.LC71:
	.string	" ("
	.align	3
.LC72:
	.string	" of "
	.align	3
.LC73:
	.string	" implementations)"
	.align	3
.LC74:
	.string	"Runs: "
	.align	3
.LC75:
	.string	" measured, "
	.align	3
.LC76:
	.string	" warmup\n\n"
	.align	3
.LC77:
	.string	"==================================================\n"
	.align	3
.LC78:
	.string	" -> Benchmarking "
	.align	3
.LC79:
	.string	" (WARM CACHE)\n"
	.align	3
.LC80:
	.string	"=================================================="
	.align	3
.LC81:
	.string	"--- Performance Benchmarking ("
	.align	3
.LC82:
	.string	"Running "
	.align	3
.LC83:
	.string	" warmup run"
	.align	3
.LC84:
	.string	"s"
	.align	3
.LC85:
	.string	"...\n"
	.align	3
.LC86:
	.string	"Warmup complete. Starting "
	.align	3
.LC87:
	.string	" benchmark run"
	.align	3
.LC88:
	.string	"=== Results ==="
	.align	3
.LC89:
	.string	"  Median cycles : "
	.align	3
.LC90:
	.string	"  Median time   : "
	.align	3
.LC91:
	.string	" s ("
	.align	3
.LC92:
	.string	" ms)\n\n"
	.align	3
.LC93:
	.string	"The network has successfully predicted the digit: "
	.align	3
.LC94:
	.string	"Raw Logits (Computational Verification):"
	.align	3
.LC95:
	.string	"Class "
	.align	3
.LC96:
	.string	": "
	.align	3
.LC97:
	.string	"=== Final Performance Summary ==="
	.align	3
.LC98:
	.string	"Seconds"
	.align	3
.LC99:
	.string	"Cycles"
	.align	3
.LC100:
	.string	"vs Base"
	.align	3
.LC101:
	.string	"Implementation"
	.align	3
.LC102:
	.string	"%-35s | %-12s | %-20s | %-14s\n"
	.align	3
.LC103:
	.string	"---------------------------------------------------------------------------------------"
	.align	3
.LC0:
	.string	"Baseline Nested-Loop"
	.align	3
.LC104:
	.string	"%-35s | %-12s | %-20s | %-14s"
	.align	3
.LC105:
	.string	"%.2f"
	.align	3
.LC106:
	.string	"%.6f"
	.align	3
.LC107:
	.string	"%.2fx"
	.align	3
.LC108:
	.string	"\nBenchmark sink: "
	.align	3
.LC109:
	.string	"../python/weights_cpp/cpp_timing.json"
	.align	3
.LC110:
	.string	"{\n"
	.align	3
.LC111:
	.string	"  \"implementation\": \""
	.align	3
.LC112:
	.string	"\",\n"
	.align	3
.LC113:
	.string	"  \"median_time_sec\": "
	.align	3
.LC114:
	.string	",\n"
	.align	3
.LC115:
	.string	"  \"median_cycles\": "
	.align	3
.LC116:
	.string	"}\n"
	.section	.text.startup,"ax",@progbits
	.align	2
	.p2align 5,,15
	.global	main
	.type	main, %function
main:
.LFB7375:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA7375
	sub	sp, sp, #1824	//,,
	.cfi_def_cfa_offset 1824
	stp	x29, x30, [sp]	//,,
	.cfi_offset 29, -1824
	.cfi_offset 30, -1816
	mov	x29, sp	//,
	stp	x19, x20, [sp, 16]	//,,
	stp	x21, x22, [sp, 32]	//,,
	stp	x23, x24, [sp, 48]	//,,
	stp	x25, x26, [sp, 64]	//,,
	stp	x27, x28, [sp, 80]	//,,
	stp	d12, d13, [sp, 96]	//,,
	stp	d14, d15, [sp, 112]	//,,
	.cfi_offset 19, -1808
	.cfi_offset 20, -1800
	.cfi_offset 21, -1792
	.cfi_offset 22, -1784
	.cfi_offset 23, -1776
	.cfi_offset 24, -1768
	.cfi_offset 25, -1760
	.cfi_offset 26, -1752
	.cfi_offset 27, -1744
	.cfi_offset 28, -1736
	.cfi_offset 76, -1728
	.cfi_offset 77, -1720
	.cfi_offset 78, -1712
	.cfi_offset 79, -1704
// src/cpp/benchmark.cpp:85:     for (int i = 1; i < argc; ++i) {
	cmp	w0, 1	// argc,
	ble	.L557		//,
// src/cpp/benchmark.cpp:91:         else if (arg == "-v" || arg == "--verbose" || arg == "verbose")
	adrp	x24, .LC26	// tmp2438,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:381: 	return __builtin_memcmp(__s1, __s2, __n);
	mov	w25, 29296	// tmp2440,
// src/cpp/benchmark.cpp:82:     bool verify_mode = false;
	str	wzr, [sp, 128]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:381: 	return __builtin_memcmp(__s1, __s2, __n);
	mov	w23, 25974	// tmp2441,
// src/cpp/benchmark.cpp:91:         else if (arg == "-v" || arg == "--verbose" || arg == "verbose")
	add	x24, x24, :lo12:.LC26	// tmp2439, tmp2438,
// src/cpp/benchmark.cpp:83:     bool profile_mode = false;
	str	wzr, [sp, 136]	//, %sfp
	add	x19, sp, 1304	// tmp2346,,
	add	x20, sp, 1320	// tmp2347,,
// src/cpp/benchmark.cpp:84:     bool verbose = false;
	str	wzr, [sp, 208]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:381: 	return __builtin_memcmp(__s1, __s2, __n);
	movk	w25, 0x666f, lsl 16	// tmp2440,,
	movk	w23, 0x6972, lsl 16	// tmp2441,,
	add	x21, x1, 8	// ivtmp.809, argv,
	sub	w0, w0, #2	// _5324, argc,
	add	x1, x1, 16	// _1823, argv,
	add	x22, x1, w0, uxtw 3	// _4985, _1823, _5324,
	.p2align 5,,15
.L314:
// src/cpp/benchmark.cpp:86:         std::string arg = argv[i];
	ldr	x27, [x21]	// _4, MEM[(char * *)_5364]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	str	x20, [sp, 1304]	// tmp2347, MEM[(struct _Alloc_hider *)_1195]._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:650: 	if (__s == 0)
	cbz	x27, .L863	// _4,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x27	//, _4
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:225: 	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
	str	x0, [sp, 1168]	// _1306, MEM[(long unsigned int *)_2139]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x26, x0	// _1306, tmp2444
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:227: 	if (__dnew > size_type(_S_local_capacity))
	cmp	x0, 15	// _1306,
	bhi	.L864		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:432: 	if (__n == 1)
	cmp	x0, 1	// _1306,
	bne	.L301		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldrb	w0, [x27]	// _1313, MEM[(const char_type &)_4]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	w0, [sp, 1320]	// _1313, MEM[(char_type &)_1195 + 16]
.L303:
	mov	x0, x20	// pretmp_5516, tmp2347
.L302:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x26, [sp, 1312]	// _1306, MEM[(struct basic_string *)_1195]._M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x0, x26]	//, MEM[(char_type &)_1316]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:1077:       { return _M_string_length; }
	ldr	x0, [sp, 1312]	// _1300, MEM[(struct basic_string *)_1195]._M_string_length
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x26, [sp, 1304]	// pretmp_5671, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3776: 	       && !_Traits::compare(__lhs.data(), __rhs, __lhs.size());
	cmp	x0, 6	// _1300,
	beq	.L865		//,
	cmp	x0, 7	// _1300,
	beq	.L866		//,
	cmp	x0, 2	// _1300,
	bne	.L307		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3776: 	       && !_Traits::compare(__lhs.data(), __rhs, __lhs.size());
	ldrh	w1, [x26]	// MEM <unsigned short> [(char * {ref-all})pretmp_5671], MEM <unsigned short> [(char * {ref-all})pretmp_5671]
	mov	w0, 30253	// tmp981,
	cmp	w1, w0	// MEM <unsigned short> [(char * {ref-all})pretmp_5671], tmp981
	beq	.L867		//,
	.p2align 5,,15
.L307:
// src/cpp/benchmark.cpp:91:         else if (arg == "-v" || arg == "--verbose" || arg == "verbose")
	mov	x1, x24	//, tmp2439
	mov	x0, x19	//, tmp2346
	bl	_ZSteqIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_		//
	and	w1, w0, 255	// _818, tmp2446
// src/cpp/benchmark.cpp:91:         else if (arg == "-v" || arg == "--verbose" || arg == "verbose")
	tbnz	x0, 0, .L561	// tmp2446,,
// src/cpp/benchmark.cpp:91:         else if (arg == "-v" || arg == "--verbose" || arg == "verbose")
	adrp	x1, .LC27	// tmp990,
	mov	x0, x19	//, tmp2346
	add	x1, x1, :lo12:.LC27	//, tmp990,
	bl	_ZSteqIcSt11char_traitsIcESaIcEEbRKNSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_		//
// src/cpp/benchmark.cpp:91:         else if (arg == "-v" || arg == "--verbose" || arg == "verbose")
	ldr	w1, [sp, 208]	//, %sfp
// src/cpp/benchmark.cpp:91:         else if (arg == "-v" || arg == "--verbose" || arg == "verbose")
	and	w0, w0, 255	// _820, tmp2447
// src/cpp/benchmark.cpp:91:         else if (arg == "-v" || arg == "--verbose" || arg == "verbose")
	cmp	w1, 0	// verbose,
	ccmp	w0, 0, 0, eq	// _820,,,
	cset	w0, ne	// verbose,
	str	w0, [sp, 208]	// verbose, %sfp
.L313:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x26, x20	// pretmp_5671, tmp2347
	beq	.L308		//,
.L309:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 1320]	// MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity, MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x0, x26	//, pretmp_5671
	add	x1, x1, 1	//, MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L308:
// src/cpp/benchmark.cpp:85:     for (int i = 1; i < argc; ++i) {
	add	x21, x21, 8	// ivtmp.809, ivtmp.809,
	cmp	x21, x22	// ivtmp.809, _4985
	bne	.L314		//,
// src/cpp/benchmark.cpp:95:     int batch_size  = verify_mode ? 10000 : 1;
	ldr	w0, [sp, 128]	//, %sfp
	cbz	w0, .L297	// verify_mode,
	mov	x0, 34464	// prephitmp_5533,
	mov	x20, 34816	// prephitmp_5531,
	movk	x0, 0x1, lsl 16	// prephitmp_5533,,
	movk	x20, 0x13, lsl 16	// prephitmp_5531,,
	mov	x27, 41984	// prephitmp_5525,
	mov	x23, 12800	// prephitmp_5523,
	str	x0, [sp, 176]	// prephitmp_5533, %sfp
	add	x0, x20, 10240000	// prephitmp_5529, prephitmp_5531,
	movk	x27, 0x49d, lsl 16	// prephitmp_5525,,
	movk	x23, 0x339, lsl 16	// prephitmp_5523,,
// src/cpp/benchmark.cpp:95:     int batch_size  = verify_mode ? 10000 : 1;
	mov	w25, 10000	// iftmp.4_168,
	str	x0, [sp, 160]	// prephitmp_5529, %sfp
	mov	x0, 9216	// prephitmp_5527,
	movk	x0, 0xf4, lsl 16	// prephitmp_5527,,
	str	x0, [sp, 152]	// prephitmp_5527, %sfp
	mov	x0, 51200	// prephitmp_5521,
	movk	x0, 0xce4, lsl 16	// prephitmp_5521,,
	str	x0, [sp, 144]	// prephitmp_5521, %sfp
	b	.L315		//
.L557:
	add	x19, sp, 1304	// tmp2346,,
// src/cpp/benchmark.cpp:83:     bool profile_mode = false;
	str	wzr, [sp, 136]	//, %sfp
// src/cpp/benchmark.cpp:84:     bool verbose = false;
	str	wzr, [sp, 208]	//, %sfp
.L297:
// src/cpp/benchmark.cpp:95:     int batch_size  = verify_mode ? 10000 : 1;
	mov	x0, 10	// prephitmp_5533,
	mov	x20, 128	// prephitmp_5531,
	str	wzr, [sp, 128]	//, %sfp
	mov	x27, 7744	// prephitmp_5525,
	mov	x23, 5408	// prephitmp_5523,
	str	x0, [sp, 176]	// prephitmp_5533, %sfp
	mov	x0, 1152	// prephitmp_5529,
// src/cpp/benchmark.cpp:95:     int batch_size  = verify_mode ? 10000 : 1;
	mov	w25, 1	// iftmp.4_168,
// src/cpp/benchmark.cpp:95:     int batch_size  = verify_mode ? 10000 : 1;
	str	x0, [sp, 160]	// prephitmp_5529, %sfp
	mov	x0, 1600	// prephitmp_5527,
	str	x0, [sp, 152]	// prephitmp_5527, %sfp
	mov	x0, 21632	// prephitmp_5521,
	str	x0, [sp, 144]	// prephitmp_5521, %sfp
.L315:
// src/cpp/include/timing.h:122:     memset(&pe, 0, sizeof(pe));
	movi	v31.4s, 0	// tmp1002
	add	x6, sp, 1136	// tmp2683,,
// src/cpp/include/timing.h:126:     pe.disabled       = 1;
	mov	w7, 97	// tmp1010,
// src/cpp/include/timing.h:124:     pe.size           = sizeof(pe);
	mov	w8, 136	// tmp1004,
// src/cpp/include/timing.h:117:     return syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
	mov	w4, -1	//,
	mov	x5, 0	//,
	mov	w3, w4	//,
	mov	w2, 0	//,
	mov	x1, x19	//, tmp2346
// src/cpp/include/timing.h:122:     memset(&pe, 0, sizeof(pe));
	str	q31, [x6, 168]	// tmp1002, MEM <char[1:136]> [(void *)_1195]
// src/cpp/include/timing.h:117:     return syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
	mov	x0, 241	//,
// src/cpp/include/timing.h:122:     memset(&pe, 0, sizeof(pe));
	stp	q31, q31, [x19, 16]	// tmp1002, tmp1002,
	stp	q31, q31, [x19, 48]	// tmp1002, tmp1002,
	stp	q31, q31, [x19, 80]	// tmp1002, tmp1002,
	str	q31, [x19, 112]	// tmp1002, MEM <char[1:136]> [(void *)_1195]
	str	xzr, [x19, 128]	//, MEM <char[1:136]> [(void *)_1195]
// src/cpp/include/timing.h:126:     pe.disabled       = 1;
	ldrb	w6, [sp, 1344]	//, MEM <unsigned char> [(struct perf_event_attr *)_1195 + 40B]
// src/cpp/include/timing.h:124:     pe.size           = sizeof(pe);
	str	w8, [sp, 1308]	// tmp1004, MEM[(struct perf_event_attr *)_1195].size
// src/cpp/include/timing.h:126:     pe.disabled       = 1;
	orr	w6, w6, w7	// tmp1009, MEM <unsigned char> [(struct perf_event_attr *)_1195 + 40B], tmp1010
	strb	w6, [sp, 1344]	// tmp1009, MEM <unsigned char> [(struct perf_event_attr *)_1195 + 40B]
// src/cpp/include/timing.h:117:     return syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
	bl	syscall		//
	str	x0, [sp, 224]	// _1333, %sfp
// src/cpp/include/timing.h:130:     ctx->fd = _perf_event_open(&pe, 0 /* self */, -1 /* any cpu */, -1, 0);
	mov	w21, w0	// _1334, _1333
// src/cpp/include/timing.h:131:     if (ctx->fd == -1) {
	cmn	w0, #1	// tmp2684,
	beq	.L868		//,
.L317:
// src/cpp/benchmark.cpp:106:     std::cout << Color::CYAN << "Memory allocation is starting..." << Color::RESET << "\n";
	adrp	x0, _ZSt4cout	// tmp2362,
	add	x28, x0, :lo12:_ZSt4cout	// tmp2355, tmp2362,
	adrp	x0, _ZN5Color4CYANE	// tmp2686,
	ldr	x1, [x0, #:lo12:_ZN5Color4CYANE]	//, CYAN
	mov	x0, x28	//, tmp2355
.LEHB51:
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, 32	//,
// src/cpp/benchmark.cpp:106:     std::cout << Color::CYAN << "Memory allocation is starting..." << Color::RESET << "\n";
	mov	x22, x0	// _236, tmp2452
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC30	// tmp1031,
	add	x1, x1, :lo12:.LC30	//, tmp1031,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:106:     std::cout << Color::CYAN << "Memory allocation is starting..." << Color::RESET << "\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2687,
	mov	x0, x22	//, _236
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:106:     std::cout << Color::CYAN << "Memory allocation is starting..." << Color::RESET << "\n";
	adrp	x1, .LC21	// tmp2332,
	add	x24, x1, :lo12:.LC21	// tmp2327, tmp2332,
	mov	x1, x24	//, tmp2327
	str	x24, [sp, 168]	// tmp2327, %sfp
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.LEHE51:
// src/cpp/benchmark.cpp:111:         load_mnist_images("../../data/MNIST/raw/t10k-images-idx3-ubyte", input_batch, batch_size);
	adrp	x1, .LC31	// tmp1038,
	mov	x0, x19	//, tmp2346
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:99: 	: _M_start(), _M_finish(), _M_end_of_storage()
	stp	xzr, xzr, [sp, 368]	//,,
// src/cpp/benchmark.cpp:111:         load_mnist_images("../../data/MNIST/raw/t10k-images-idx3-ubyte", input_batch, batch_size);
	add	x1, x1, :lo12:.LC31	//, tmp1038,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:99: 	: _M_start(), _M_finish(), _M_end_of_storage()
	str	xzr, [sp, 384]	//, MEM[(struct _Vector_impl_data *)&input_batch]._M_end_of_storage
.LEHB52:
// src/cpp/benchmark.cpp:111:         load_mnist_images("../../data/MNIST/raw/t10k-images-idx3-ubyte", input_batch, batch_size);
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_.constprop.0		//
.LEHE52:
// src/cpp/benchmark.cpp:111:         load_mnist_images("../../data/MNIST/raw/t10k-images-idx3-ubyte", input_batch, batch_size);
	mov	w2, w25	//, iftmp.4_168
	add	x1, sp, 368	// tmp2689,,
	mov	x0, x19	//, tmp2346
.LEHB53:
	bl	_Z17load_mnist_imagesRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEER6Tensori		//
.LEHE53:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/benchmark.cpp:112:         std::cout << Color::GREEN << "MNIST dataset successfully loaded." << Color::RESET << "\n";
	adrp	x1, _ZN5Color5GREENE	// tmp2690,
	mov	x0, x28	//, tmp2355
	ldr	x1, [x1, #:lo12:_ZN5Color5GREENE]	//, GREEN
.LEHB54:
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC32	// tmp1049,
	mov	x2, 34	//,
	add	x1, x1, :lo12:.LC32	//, tmp1049,
// src/cpp/benchmark.cpp:112:         std::cout << Color::GREEN << "MNIST dataset successfully loaded." << Color::RESET << "\n";
	mov	x22, x0	// _247, tmp2453
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:112:         std::cout << Color::GREEN << "MNIST dataset successfully loaded." << Color::RESET << "\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2691,
	mov	x0, x22	//, _247
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:112:         std::cout << Color::GREEN << "MNIST dataset successfully loaded." << Color::RESET << "\n";
	mov	x1, x24	//, tmp2327
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.LEHE54:
// src/cpp/benchmark.cpp:123:     std::cout << Color::BOLD_CYAN << "\nLoading real weights..." << Color::RESET << "\n";
	adrp	x1, _ZN5Color9BOLD_CYANE	// tmp2692,
	mov	x0, x28	//, tmp2355
	ldr	x1, [x1, #:lo12:_ZN5Color9BOLD_CYANE]	//, BOLD_CYAN
.LEHB55:
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC33	// tmp1061,
	mov	x2, 24	//,
	add	x1, x1, :lo12:.LC33	//, tmp1061,
// src/cpp/benchmark.cpp:123:     std::cout << Color::BOLD_CYAN << "\nLoading real weights..." << Color::RESET << "\n";
	mov	x22, x0	// _252, tmp2454
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:123:     std::cout << Color::BOLD_CYAN << "\nLoading real weights..." << Color::RESET << "\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2693,
	mov	x0, x22	//, _252
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:123:     std::cout << Color::BOLD_CYAN << "\nLoading real weights..." << Color::RESET << "\n";
	mov	x1, x24	//, tmp2327
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:124:     const std::string w_path = "../python/weights_cpp/";
	adrp	x1, .LC34	// tmp1068,
	add	x0, sp, 272	// tmp2694,,
	add	x1, x1, :lo12:.LC34	//, tmp1068,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_.constprop.0		//
.LEHE55:
// src/cpp/benchmark.cpp:126:     Tensor conv1_weight = {std::vector<float>(32 * 1 * 3 * 3), 32,  1,  3,   3};
	mov	x3, 32	// tmp1070,
	mov	x2, 12884901891	// tmp1071,
	stp	xzr, xzr, [sp, 416]	// MEM <char[24]> [(struct Tensor *)&conv1_weight]
	movk	x3, 0x1, lsl 32	// tmp1070,,
// src/cpp/benchmark.cpp:126:     Tensor conv1_weight = {std::vector<float>(32 * 1 * 3 * 3), 32,  1,  3,   3};
	mov	x1, 288	//,
	add	x0, sp, 416	// tmp2695,,
// src/cpp/benchmark.cpp:126:     Tensor conv1_weight = {std::vector<float>(32 * 1 * 3 * 3), 32,  1,  3,   3};
	stp	xzr, x3, [sp, 432]	//, tmp1070,
	str	x2, [sp, 448]	// tmp1071, MEM <unsigned long> [(int *)&conv1_weight + 32B]
.LEHB56:
// src/cpp/benchmark.cpp:126:     Tensor conv1_weight = {std::vector<float>(32 * 1 * 3 * 3), 32,  1,  3,   3};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE56:
// src/cpp/benchmark.cpp:127:     Tensor conv1_bias   = {std::vector<float>(32), 32,  1,  1,   1};
	mov	x3, 32	// tmp1073,
	mov	x2, 4294967297	// tmp1074,
	stp	xzr, xzr, [sp, 464]	// MEM <char[24]> [(struct Tensor *)&conv1_bias]
	movk	x3, 0x1, lsl 32	// tmp1073,,
// src/cpp/benchmark.cpp:127:     Tensor conv1_bias   = {std::vector<float>(32), 32,  1,  1,   1};
	mov	x1, 32	//,
	add	x0, sp, 464	// tmp2696,,
// src/cpp/benchmark.cpp:127:     Tensor conv1_bias   = {std::vector<float>(32), 32,  1,  1,   1};
	stp	xzr, x3, [sp, 480]	//, tmp1073,
	str	x2, [sp, 496]	// tmp1074, MEM <unsigned long> [(int *)&conv1_bias + 32B]
.LEHB57:
// src/cpp/benchmark.cpp:127:     Tensor conv1_bias   = {std::vector<float>(32), 32,  1,  1,   1};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE57:
// src/cpp/benchmark.cpp:128:     load_binary_weights(w_path + "model_0_weight.bin", conv1_weight.data);
	adrp	x2, .LC35	// tmp1077,
	ldp	x0, x1, [sp, 272]	//,,
	mov	x8, x19	//, tmp2346
	add	x2, x2, :lo12:.LC35	//, tmp1077,
.LEHB58:
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_.isra.0		//
.LEHE58:
// src/cpp/benchmark.cpp:128:     load_binary_weights(w_path + "model_0_weight.bin", conv1_weight.data);
	add	x1, sp, 416	// tmp2697,,
	mov	x0, x19	//, tmp2346
.LEHB59:
	bl	_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE		//
.LEHE59:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/benchmark.cpp:129:     load_binary_weights(w_path + "model_0_bias.bin", conv1_bias.data);
	adrp	x2, .LC36	// tmp1085,
	ldp	x0, x1, [sp, 272]	//,,
	mov	x8, x19	//, tmp2346
	add	x2, x2, :lo12:.LC36	//, tmp1085,
.LEHB60:
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_.isra.0		//
.LEHE60:
// src/cpp/benchmark.cpp:129:     load_binary_weights(w_path + "model_0_bias.bin", conv1_bias.data);
	add	x1, sp, 464	// tmp2698,,
	mov	x0, x19	//, tmp2346
.LEHB61:
	bl	_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE		//
.LEHE61:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/benchmark.cpp:131:     Tensor conv2_weight = {std::vector<float>(64 * 32 * 3 * 3), 64, 32,  3,   3};
	mov	x3, 64	// tmp1092,
	add	x0, sp, 512	// tmp2998,,
	movk	x3, 0x20, lsl 32	// tmp1092,,
	mov	x2, 12884901891	// tmp1093,
// src/cpp/benchmark.cpp:131:     Tensor conv2_weight = {std::vector<float>(64 * 32 * 3 * 3), 64, 32,  3,   3};
	mov	x1, 18432	//,
// src/cpp/benchmark.cpp:131:     Tensor conv2_weight = {std::vector<float>(64 * 32 * 3 * 3), 64, 32,  3,   3};
	stp	xzr, xzr, [x0]	// MEM <char[24]> [(struct Tensor *)&conv2_weight]
	str	xzr, [sp, 528]	//, MEM <char[24]> [(struct Tensor *)&conv2_weight]
	str	x3, [sp, 536]	// tmp1092, MEM <unsigned long> [(int *)&conv2_weight + 24B]
	str	x2, [sp, 544]	// tmp1093, MEM <unsigned long> [(int *)&conv2_weight + 32B]
.LEHB62:
// src/cpp/benchmark.cpp:131:     Tensor conv2_weight = {std::vector<float>(64 * 32 * 3 * 3), 64, 32,  3,   3};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE62:
// src/cpp/benchmark.cpp:132:     Tensor conv2_bias   = {std::vector<float>(64), 64,  1,  1,   1};
	add	x0, sp, 512	// tmp2996,,
	mov	x3, 64	// tmp1095,
	movk	x3, 0x1, lsl 32	// tmp1095,,
	mov	x2, 4294967297	// tmp1096,
// src/cpp/benchmark.cpp:132:     Tensor conv2_bias   = {std::vector<float>(64), 64,  1,  1,   1};
	mov	x1, 64	//,
// src/cpp/benchmark.cpp:132:     Tensor conv2_bias   = {std::vector<float>(64), 64,  1,  1,   1};
	stp	xzr, xzr, [x0, 48]	// MEM <char[24]> [(struct Tensor *)&conv2_bias]
// src/cpp/benchmark.cpp:132:     Tensor conv2_bias   = {std::vector<float>(64), 64,  1,  1,   1};
	add	x0, sp, 560	// tmp2700,,
// src/cpp/benchmark.cpp:132:     Tensor conv2_bias   = {std::vector<float>(64), 64,  1,  1,   1};
	str	xzr, [sp, 576]	//, MEM <char[24]> [(struct Tensor *)&conv2_bias]
	str	x3, [sp, 584]	// tmp1095, MEM <unsigned long> [(int *)&conv2_bias + 24B]
	str	x2, [sp, 592]	// tmp1096, MEM <unsigned long> [(int *)&conv2_bias + 32B]
.LEHB63:
// src/cpp/benchmark.cpp:132:     Tensor conv2_bias   = {std::vector<float>(64), 64,  1,  1,   1};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE63:
// src/cpp/benchmark.cpp:133:     load_binary_weights(w_path + "model_3_weight.bin", conv2_weight.data);
	adrp	x2, .LC37	// tmp1099,
	ldp	x0, x1, [sp, 272]	//,,
	mov	x8, x19	//, tmp2346
	add	x2, x2, :lo12:.LC37	//, tmp1099,
.LEHB64:
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_.isra.0		//
.LEHE64:
// src/cpp/benchmark.cpp:133:     load_binary_weights(w_path + "model_3_weight.bin", conv2_weight.data);
	add	x1, sp, 512	// tmp2701,,
	mov	x0, x19	//, tmp2346
.LEHB65:
	bl	_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE		//
.LEHE65:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/benchmark.cpp:134:     load_binary_weights(w_path + "model_3_bias.bin", conv2_bias.data);
	adrp	x2, .LC38	// tmp1107,
	ldp	x0, x1, [sp, 272]	//,,
	mov	x8, x19	//, tmp2346
	add	x2, x2, :lo12:.LC38	//, tmp1107,
.LEHB66:
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_.isra.0		//
.LEHE66:
// src/cpp/benchmark.cpp:134:     load_binary_weights(w_path + "model_3_bias.bin", conv2_bias.data);
	add	x1, sp, 560	// tmp2702,,
	mov	x0, x19	//, tmp2346
.LEHB67:
	bl	_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE		//
.LEHE67:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/benchmark.cpp:136:     Tensor conv3_weight = {std::vector<float>(128 * 64 * 3 * 3), 128, 64,  3,   3};
	add	x0, sp, 512	// tmp2994,,
	mov	x3, 128	// tmp1114,
	movk	x3, 0x40, lsl 32	// tmp1114,,
	mov	x2, 12884901891	// tmp1115,
// src/cpp/benchmark.cpp:136:     Tensor conv3_weight = {std::vector<float>(128 * 64 * 3 * 3), 128, 64,  3,   3};
	mov	x1, 8192	//,
// src/cpp/benchmark.cpp:136:     Tensor conv3_weight = {std::vector<float>(128 * 64 * 3 * 3), 128, 64,  3,   3};
	stp	xzr, xzr, [x0, 96]	// MEM <char[24]> [(struct Tensor *)&conv3_weight]
// src/cpp/benchmark.cpp:136:     Tensor conv3_weight = {std::vector<float>(128 * 64 * 3 * 3), 128, 64,  3,   3};
	add	x0, sp, 608	// tmp2703,,
	movk	x1, 0x1, lsl 16	//,,
// src/cpp/benchmark.cpp:136:     Tensor conv3_weight = {std::vector<float>(128 * 64 * 3 * 3), 128, 64,  3,   3};
	str	xzr, [sp, 624]	//, MEM <char[24]> [(struct Tensor *)&conv3_weight]
	str	x3, [sp, 632]	// tmp1114, MEM <unsigned long> [(int *)&conv3_weight + 24B]
	str	x2, [sp, 640]	// tmp1115, MEM <unsigned long> [(int *)&conv3_weight + 32B]
.LEHB68:
// src/cpp/benchmark.cpp:136:     Tensor conv3_weight = {std::vector<float>(128 * 64 * 3 * 3), 128, 64,  3,   3};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE68:
// src/cpp/benchmark.cpp:137:     Tensor conv3_bias   = {std::vector<float>(128), 128,  1,  1,   1};
	add	x0, sp, 624	// tmp2705,,
	mov	x3, 128	// tmp1118,
	movk	x3, 0x1, lsl 32	// tmp1118,,
	mov	x2, 4294967297	// tmp1119,
// src/cpp/benchmark.cpp:137:     Tensor conv3_bias   = {std::vector<float>(128), 128,  1,  1,   1};
	add	x26, sp, 656	// tmp2323,,
	mov	x1, 128	//,
// src/cpp/benchmark.cpp:137:     Tensor conv3_bias   = {std::vector<float>(128), 128,  1,  1,   1};
	stp	xzr, xzr, [x0, 32]	// MEM <char[24]> [(struct Tensor *)&conv3_bias]
// src/cpp/benchmark.cpp:137:     Tensor conv3_bias   = {std::vector<float>(128), 128,  1,  1,   1};
	mov	x0, x26	//, tmp2323
// src/cpp/benchmark.cpp:137:     Tensor conv3_bias   = {std::vector<float>(128), 128,  1,  1,   1};
	str	xzr, [sp, 672]	//, MEM <char[24]> [(struct Tensor *)&conv3_bias]
	str	x3, [sp, 680]	// tmp1118, MEM <unsigned long> [(int *)&conv3_bias + 24B]
	str	x2, [sp, 688]	// tmp1119, MEM <unsigned long> [(int *)&conv3_bias + 32B]
.LEHB69:
// src/cpp/benchmark.cpp:137:     Tensor conv3_bias   = {std::vector<float>(128), 128,  1,  1,   1};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE69:
// src/cpp/benchmark.cpp:138:     load_binary_weights(w_path + "model_6_weight.bin", conv3_weight.data);
	adrp	x2, .LC39	// tmp1122,
	ldp	x0, x1, [sp, 272]	//,,
	mov	x8, x19	//, tmp2346
	add	x2, x2, :lo12:.LC39	//, tmp1122,
.LEHB70:
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_.isra.0		//
.LEHE70:
// src/cpp/benchmark.cpp:138:     load_binary_weights(w_path + "model_6_weight.bin", conv3_weight.data);
	add	x1, sp, 608	// tmp2706,,
	mov	x0, x19	//, tmp2346
.LEHB71:
	bl	_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE		//
.LEHE71:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/benchmark.cpp:139:     load_binary_weights(w_path + "model_6_bias.bin", conv3_bias.data);
	adrp	x2, .LC40	// tmp1130,
	ldp	x0, x1, [sp, 272]	//,,
	mov	x8, x19	//, tmp2346
	add	x2, x2, :lo12:.LC40	//, tmp1130,
.LEHB72:
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_.isra.0		//
.LEHE72:
// src/cpp/benchmark.cpp:139:     load_binary_weights(w_path + "model_6_bias.bin", conv3_bias.data);
	mov	x1, x26	//, tmp2323
	mov	x0, x19	//, tmp2346
.LEHB73:
	bl	_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE		//
.LEHE73:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x19	//, tmp2346
// src/cpp/benchmark.cpp:141:     Tensor fc_weight    = {std::vector<float>(num_classes * 128),        1,  1, num_classes, 128};
	add	x24, sp, 696	// tmp2325,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/benchmark.cpp:141:     Tensor fc_weight    = {std::vector<float>(num_classes * 128),        1,  1, num_classes, 128};
	add	x0, sp, 624	// tmp2708,,
	add	x4, sp, 624	// tmp2710,,
	add	x14, sp, 728	//,,
	mov	w3, 128	// tmp1139,
	mov	w2, 1	// tmp1137,
	stp	xzr, xzr, [x0, 88]	// MEM <char[36]> [(struct Tensor *)&fc_weight]
// src/cpp/benchmark.cpp:141:     Tensor fc_weight    = {std::vector<float>(num_classes * 128),        1,  1, num_classes, 128};
	mov	x1, 1280	//,
	mov	x0, x24	//, tmp2325
// src/cpp/benchmark.cpp:141:     Tensor fc_weight    = {std::vector<float>(num_classes * 128),        1,  1, num_classes, 128};
	stp	xzr, xzr, [x4, 72]	// MEM <char[36]> [(struct Tensor *)&fc_weight]
	stp	w2, w2, [x14, -8]	// tmp1137, tmp1137,
	stp	wzr, w3, [x14]	//, tmp1139,
.LEHB74:
// src/cpp/benchmark.cpp:141:     Tensor fc_weight    = {std::vector<float>(num_classes * 128),        1,  1, num_classes, 128};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE74:
// src/cpp/benchmark.cpp:141:     Tensor fc_weight    = {std::vector<float>(num_classes * 128),        1,  1, num_classes, 128};
	mov	w1, 10	// tmp1141,
// src/cpp/benchmark.cpp:142:     Tensor fc_bias      = {std::vector<float>(num_classes), num_classes, 1,  1,   1};
	add	x0, sp, 624	// tmp2712,,
	mov	w3, 1	// tmp1142,
	mov	x2, 4294967297	// tmp1143,
// src/cpp/benchmark.cpp:142:     Tensor fc_bias      = {std::vector<float>(num_classes), num_classes, 1,  1,   1};
	add	x22, sp, 736	// tmp2326,,
// src/cpp/benchmark.cpp:142:     Tensor fc_bias      = {std::vector<float>(num_classes), num_classes, 1,  1,   1};
	stp	xzr, xzr, [x0, 112]	// MEM <char[28]> [(struct Tensor *)&fc_bias]
// src/cpp/benchmark.cpp:142:     Tensor fc_bias      = {std::vector<float>(num_classes), num_classes, 1,  1,   1};
	mov	x0, x22	//, tmp2326
// src/cpp/benchmark.cpp:141:     Tensor fc_weight    = {std::vector<float>(num_classes * 128),        1,  1, num_classes, 128};
	str	w1, [sp, 728]	// tmp1141, fc_weight.height
// src/cpp/benchmark.cpp:142:     Tensor fc_bias      = {std::vector<float>(num_classes), num_classes, 1,  1,   1};
	mov	x1, 10	//,
// src/cpp/benchmark.cpp:142:     Tensor fc_bias      = {std::vector<float>(num_classes), num_classes, 1,  1,   1};
	str	xzr, [sp, 752]	//, MEM <char[28]> [(struct Tensor *)&fc_bias]
	str	wzr, [sp, 760]	//, MEM <char[28]> [(struct Tensor *)&fc_bias]
	str	w3, [sp, 764]	// tmp1142, fc_bias.channels
	str	x2, [sp, 768]	// tmp1143, MEM <unsigned long> [(int *)&fc_bias + 32B]
.LEHB75:
// src/cpp/benchmark.cpp:142:     Tensor fc_bias      = {std::vector<float>(num_classes), num_classes, 1,  1,   1};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE75:
// src/cpp/benchmark.cpp:142:     Tensor fc_bias      = {std::vector<float>(num_classes), num_classes, 1,  1,   1};
	mov	w3, 10	// tmp1145,
// src/cpp/benchmark.cpp:143:     load_binary_weights(w_path + "model_10_weight.bin", fc_weight.data);
	adrp	x2, .LC41	// tmp1147,
	ldp	x0, x1, [sp, 272]	//,,
	add	x2, x2, :lo12:.LC41	//, tmp1147,
	mov	x8, x19	//, tmp2346
// src/cpp/benchmark.cpp:142:     Tensor fc_bias      = {std::vector<float>(num_classes), num_classes, 1,  1,   1};
	str	w3, [sp, 760]	// tmp1145, fc_bias.batches
.LEHB76:
// src/cpp/benchmark.cpp:143:     load_binary_weights(w_path + "model_10_weight.bin", fc_weight.data);
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_.isra.0		//
.LEHE76:
// src/cpp/benchmark.cpp:143:     load_binary_weights(w_path + "model_10_weight.bin", fc_weight.data);
	mov	x1, x24	//, tmp2325
	mov	x0, x19	//, tmp2346
.LEHB77:
	bl	_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE		//
.LEHE77:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/benchmark.cpp:144:     load_binary_weights(w_path + "model_10_bias.bin", fc_bias.data);
	adrp	x2, .LC42	// tmp1155,
	ldp	x0, x1, [sp, 272]	//,,
	mov	x8, x19	//, tmp2346
	add	x2, x2, :lo12:.LC42	//, tmp1155,
.LEHB78:
	bl	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_.isra.0		//
.LEHE78:
// src/cpp/benchmark.cpp:144:     load_binary_weights(w_path + "model_10_bias.bin", fc_bias.data);
	mov	x1, x22	//, tmp2326
	mov	x0, x19	//, tmp2346
.LEHB79:
	bl	_Z19load_binary_weightsRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIfSaIfEE		//
.LEHE79:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// src/cpp/benchmark.cpp:147:     Tensor conv1_out   = {std::vector<float>(batch_size * 32  * 26 * 26), batch_size,  32, 26, 26};
	add	x0, sp, 624	// tmp2714,,
	add	x13, sp, 808	//,,
// src/cpp/benchmark.cpp:147:     Tensor conv1_out   = {std::vector<float>(batch_size * 32  * 26 * 26), batch_size,  32, 26, 26};
	ldr	x1, [sp, 144]	//, %sfp
// src/cpp/benchmark.cpp:147:     Tensor conv1_out   = {std::vector<float>(batch_size * 32  * 26 * 26), batch_size,  32, 26, 26};
	mov	w3, 32	// tmp1162,
	mov	w2, 26	// tmp1163,
	stp	xzr, xzr, [x0, 152]	// MEM <char[28]> [(struct Tensor *)&conv1_out]
// src/cpp/benchmark.cpp:147:     Tensor conv1_out   = {std::vector<float>(batch_size * 32  * 26 * 26), batch_size,  32, 26, 26};
	add	x0, sp, 776	// tmp2715,,
// src/cpp/benchmark.cpp:147:     Tensor conv1_out   = {std::vector<float>(batch_size * 32  * 26 * 26), batch_size,  32, 26, 26};
	str	xzr, [sp, 792]	//, MEM <char[28]> [(struct Tensor *)&conv1_out]
	stp	wzr, w3, [x13, -8]	//, tmp1162,
	stp	w2, w2, [x13]	// tmp1163, tmp1163,
.LEHB80:
// src/cpp/benchmark.cpp:147:     Tensor conv1_out   = {std::vector<float>(batch_size * 32  * 26 * 26), batch_size,  32, 26, 26};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE80:
// src/cpp/benchmark.cpp:148:     Tensor pool1_out   = {std::vector<float>(batch_size * 32  * 13 * 13), batch_size,  32, 13, 13};
	add	x0, sp, 624	// tmp2717,,
	add	x12, sp, 848	//,,
// src/cpp/benchmark.cpp:147:     Tensor conv1_out   = {std::vector<float>(batch_size * 32  * 26 * 26), batch_size,  32, 26, 26};
	str	w25, [sp, 800]	// iftmp.4_168, conv1_out.batches
// src/cpp/benchmark.cpp:148:     Tensor pool1_out   = {std::vector<float>(batch_size * 32  * 13 * 13), batch_size,  32, 13, 13};
	mov	w3, 32	// tmp1165,
	mov	w2, 13	// tmp1166,
	stp	xzr, xzr, [x0, 192]	// MEM <char[28]> [(struct Tensor *)&pool1_out]
// src/cpp/benchmark.cpp:148:     Tensor pool1_out   = {std::vector<float>(batch_size * 32  * 13 * 13), batch_size,  32, 13, 13};
	mov	x1, x23	//, prephitmp_5523
	add	x0, sp, 816	// tmp2718,,
// src/cpp/benchmark.cpp:148:     Tensor pool1_out   = {std::vector<float>(batch_size * 32  * 13 * 13), batch_size,  32, 13, 13};
	str	xzr, [sp, 832]	//, MEM <char[28]> [(struct Tensor *)&pool1_out]
	stp	wzr, w3, [x12, -8]	//, tmp1165,
	stp	w2, w2, [x12]	// tmp1166, tmp1166,
.LEHB81:
// src/cpp/benchmark.cpp:148:     Tensor pool1_out   = {std::vector<float>(batch_size * 32  * 13 * 13), batch_size,  32, 13, 13};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE81:
// src/cpp/benchmark.cpp:149:     Tensor conv2_out   = {std::vector<float>(batch_size * 64  * 11 * 11), batch_size,  64, 11, 11};
	add	x0, sp, 624	// tmp2720,,
	add	x11, sp, 888	//,,
// src/cpp/benchmark.cpp:148:     Tensor pool1_out   = {std::vector<float>(batch_size * 32  * 13 * 13), batch_size,  32, 13, 13};
	str	w25, [sp, 840]	// iftmp.4_168, pool1_out.batches
// src/cpp/benchmark.cpp:149:     Tensor conv2_out   = {std::vector<float>(batch_size * 64  * 11 * 11), batch_size,  64, 11, 11};
	mov	w3, 64	// tmp1168,
	mov	w2, 11	// tmp1169,
	stp	xzr, xzr, [x0, 232]	// MEM <char[28]> [(struct Tensor *)&conv2_out]
// src/cpp/benchmark.cpp:149:     Tensor conv2_out   = {std::vector<float>(batch_size * 64  * 11 * 11), batch_size,  64, 11, 11};
	mov	x1, x27	//, prephitmp_5525
	add	x0, sp, 856	// tmp2721,,
// src/cpp/benchmark.cpp:149:     Tensor conv2_out   = {std::vector<float>(batch_size * 64  * 11 * 11), batch_size,  64, 11, 11};
	str	xzr, [sp, 872]	//, MEM <char[28]> [(struct Tensor *)&conv2_out]
	stp	wzr, w3, [x11, -8]	//, tmp1168,
	stp	w2, w2, [x11]	// tmp1169, tmp1169,
.LEHB82:
// src/cpp/benchmark.cpp:149:     Tensor conv2_out   = {std::vector<float>(batch_size * 64  * 11 * 11), batch_size,  64, 11, 11};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE82:
// src/cpp/benchmark.cpp:150:     Tensor pool2_out   = {std::vector<float>(batch_size * 64  *  5 *  5), batch_size,  64,  5,  5};
	add	x0, sp, 1136	// tmp2723,,
	add	x10, sp, 928	//,,
// src/cpp/benchmark.cpp:150:     Tensor pool2_out   = {std::vector<float>(batch_size * 64  *  5 *  5), batch_size,  64,  5,  5};
	ldr	x1, [sp, 152]	//, %sfp
// src/cpp/benchmark.cpp:150:     Tensor pool2_out   = {std::vector<float>(batch_size * 64  *  5 *  5), batch_size,  64,  5,  5};
	mov	w3, 64	// tmp1171,
	mov	w2, 5	// tmp1172,
// src/cpp/benchmark.cpp:149:     Tensor conv2_out   = {std::vector<float>(batch_size * 64  * 11 * 11), batch_size,  64, 11, 11};
	str	w25, [sp, 880]	// iftmp.4_168, conv2_out.batches
// src/cpp/benchmark.cpp:150:     Tensor pool2_out   = {std::vector<float>(batch_size * 64  *  5 *  5), batch_size,  64,  5,  5};
	add	x27, sp, 896	// tmp2333,,
// src/cpp/benchmark.cpp:150:     Tensor pool2_out   = {std::vector<float>(batch_size * 64  *  5 *  5), batch_size,  64,  5,  5};
	stp	xzr, xzr, [x0, -240]	// MEM <char[28]> [(struct Tensor *)&pool2_out]
// src/cpp/benchmark.cpp:150:     Tensor pool2_out   = {std::vector<float>(batch_size * 64  *  5 *  5), batch_size,  64,  5,  5};
	mov	x0, x27	//, tmp2333
// src/cpp/benchmark.cpp:150:     Tensor pool2_out   = {std::vector<float>(batch_size * 64  *  5 *  5), batch_size,  64,  5,  5};
	str	xzr, [sp, 912]	//, MEM <char[28]> [(struct Tensor *)&pool2_out]
	stp	wzr, w3, [x10, -8]	//, tmp1171,
	stp	w2, w2, [x10]	// tmp1172, tmp1172,
.LEHB83:
// src/cpp/benchmark.cpp:150:     Tensor pool2_out   = {std::vector<float>(batch_size * 64  *  5 *  5), batch_size,  64,  5,  5};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE83:
// src/cpp/benchmark.cpp:151:     Tensor conv3_out   = {std::vector<float>(batch_size * 128 *  3 *  3), batch_size, 128,  3,  3};
	add	x0, sp, 1136	// tmp2725,,
	mov	w3, 128	// tmp1174,
// src/cpp/benchmark.cpp:151:     Tensor conv3_out   = {std::vector<float>(batch_size * 128 *  3 *  3), batch_size, 128,  3,  3};
	ldr	x1, [sp, 160]	//, %sfp
// src/cpp/benchmark.cpp:151:     Tensor conv3_out   = {std::vector<float>(batch_size * 128 *  3 *  3), batch_size, 128,  3,  3};
	mov	x2, 12884901891	// tmp1175,
// src/cpp/benchmark.cpp:151:     Tensor conv3_out   = {std::vector<float>(batch_size * 128 *  3 *  3), batch_size, 128,  3,  3};
	add	x23, sp, 936	// tmp2334,,
// src/cpp/benchmark.cpp:150:     Tensor pool2_out   = {std::vector<float>(batch_size * 64  *  5 *  5), batch_size,  64,  5,  5};
	str	w25, [sp, 920]	// iftmp.4_168, pool2_out.batches
// src/cpp/benchmark.cpp:151:     Tensor conv3_out   = {std::vector<float>(batch_size * 128 *  3 *  3), batch_size, 128,  3,  3};
	stp	xzr, xzr, [x0, -200]	// MEM <char[28]> [(struct Tensor *)&conv3_out]
// src/cpp/benchmark.cpp:151:     Tensor conv3_out   = {std::vector<float>(batch_size * 128 *  3 *  3), batch_size, 128,  3,  3};
	mov	x0, x23	//, tmp2334
// src/cpp/benchmark.cpp:151:     Tensor conv3_out   = {std::vector<float>(batch_size * 128 *  3 *  3), batch_size, 128,  3,  3};
	str	xzr, [sp, 952]	//, MEM <char[28]> [(struct Tensor *)&conv3_out]
	str	wzr, [sp, 960]	//, MEM <char[28]> [(struct Tensor *)&conv3_out]
	str	w3, [sp, 964]	// tmp1174, conv3_out.channels
	str	x2, [sp, 968]	// tmp1175, MEM <unsigned long> [(int *)&conv3_out + 32B]
.LEHB84:
// src/cpp/benchmark.cpp:151:     Tensor conv3_out   = {std::vector<float>(batch_size * 128 *  3 *  3), batch_size, 128,  3,  3};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE84:
// src/cpp/benchmark.cpp:152:     Tensor avgpool_out = {std::vector<float>(batch_size * 128),           batch_size, 128,  1,  1};
	add	x0, sp, 1136	// tmp2727,,
	mov	w3, 128	// tmp1177,
// src/cpp/benchmark.cpp:151:     Tensor conv3_out   = {std::vector<float>(batch_size * 128 *  3 *  3), batch_size, 128,  3,  3};
	str	w25, [sp, 960]	// iftmp.4_168, conv3_out.batches
// src/cpp/benchmark.cpp:152:     Tensor avgpool_out = {std::vector<float>(batch_size * 128),           batch_size, 128,  1,  1};
	mov	x2, 4294967297	// tmp1178,
// src/cpp/benchmark.cpp:152:     Tensor avgpool_out = {std::vector<float>(batch_size * 128),           batch_size, 128,  1,  1};
	mov	x1, x20	//, prephitmp_5531
// src/cpp/benchmark.cpp:152:     Tensor avgpool_out = {std::vector<float>(batch_size * 128),           batch_size, 128,  1,  1};
	stp	xzr, xzr, [x0, -160]	// MEM <char[28]> [(struct Tensor *)&avgpool_out]
// src/cpp/benchmark.cpp:152:     Tensor avgpool_out = {std::vector<float>(batch_size * 128),           batch_size, 128,  1,  1};
	add	x0, sp, 976	// tmp2728,,
// src/cpp/benchmark.cpp:152:     Tensor avgpool_out = {std::vector<float>(batch_size * 128),           batch_size, 128,  1,  1};
	str	xzr, [sp, 992]	//, MEM <char[28]> [(struct Tensor *)&avgpool_out]
	str	wzr, [sp, 1000]	//, MEM <char[28]> [(struct Tensor *)&avgpool_out]
	str	w3, [sp, 1004]	// tmp1177, avgpool_out.channels
	str	x2, [sp, 1008]	// tmp1178, MEM <unsigned long> [(int *)&avgpool_out + 32B]
.LEHB85:
// src/cpp/benchmark.cpp:152:     Tensor avgpool_out = {std::vector<float>(batch_size * 128),           batch_size, 128,  1,  1};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE85:
// src/cpp/benchmark.cpp:153:     Tensor final_logits= {std::vector<float>(batch_size * num_classes),   batch_size,   1,  1, num_classes};
	add	x0, sp, 1136	// tmp2730,,
	add	x3, sp, 1136	// tmp2732,,
// src/cpp/benchmark.cpp:153:     Tensor final_logits= {std::vector<float>(batch_size * num_classes),   batch_size,   1,  1, num_classes};
	ldr	x1, [sp, 176]	//, %sfp
// src/cpp/benchmark.cpp:153:     Tensor final_logits= {std::vector<float>(batch_size * num_classes),   batch_size,   1,  1, num_classes};
	mov	x2, 4294967297	// tmp1180,
// src/cpp/benchmark.cpp:153:     Tensor final_logits= {std::vector<float>(batch_size * num_classes),   batch_size,   1,  1, num_classes};
	add	x20, sp, 1016	// tmp2339,,
// src/cpp/benchmark.cpp:153:     Tensor final_logits= {std::vector<float>(batch_size * num_classes),   batch_size,   1,  1, num_classes};
	str	xzr, [sp, 1048]	//, final_logits
	stp	xzr, xzr, [x0, -104]	// final_logits
// src/cpp/benchmark.cpp:153:     Tensor final_logits= {std::vector<float>(batch_size * num_classes),   batch_size,   1,  1, num_classes};
	mov	x0, x20	//, tmp2339
// src/cpp/benchmark.cpp:153:     Tensor final_logits= {std::vector<float>(batch_size * num_classes),   batch_size,   1,  1, num_classes};
	stp	xzr, xzr, [x3, -120]	// final_logits
	str	x2, [x3, -92]	// tmp1180, MEM <unsigned long> [(int *)&final_logits + 28B]
// src/cpp/benchmark.cpp:152:     Tensor avgpool_out = {std::vector<float>(batch_size * 128),           batch_size, 128,  1,  1};
	str	w25, [sp, 1000]	// iftmp.4_168, avgpool_out.batches
.LEHB86:
// src/cpp/benchmark.cpp:153:     Tensor final_logits= {std::vector<float>(batch_size * num_classes),   batch_size,   1,  1, num_classes};
	bl	_ZNSt6vectorIfSaIfEEC2EmRKS0_.constprop.0		//
.LEHE86:
// src/cpp/benchmark.cpp:168:     const benchmark_impl_t implementations[] = {
	adrp	x0, .LANCHOR0	// tmp2351,
// src/cpp/benchmark.cpp:161:     };
	add	x2, sp, 416	// tmp2737,,
// src/cpp/benchmark.cpp:153:     Tensor final_logits= {std::vector<float>(batch_size * num_classes),   batch_size,   1,  1, num_classes};
	str	w25, [sp, 1040]	// iftmp.4_168, final_logits.batches
// src/cpp/benchmark.cpp:168:     const benchmark_impl_t implementations[] = {
	add	x1, x0, :lo12:.LANCHOR0	// tmp2348, tmp2351,
// src/cpp/benchmark.cpp:161:     };
	add	x0, sp, 368	// tmp2736,,
	str	x27, [sp, 1232]	// tmp2333, MEM[(struct CNNContext *)_2139].pool2_out
	add	x9, sp, 1248	//,,
	str	x2, [sp, 1176]	// tmp2737, MEM[(struct CNNContext *)_2139].conv1_weight
// src/cpp/benchmark.cpp:168:     const benchmark_impl_t implementations[] = {
	str	x1, [sp, 216]	// tmp2348, %sfp
// src/cpp/benchmark.cpp:161:     };
	str	x0, [sp, 1168]	// tmp2736, MEM[(struct CNNContext *)_2139].input_batch
// src/cpp/benchmark.cpp:153:     Tensor final_logits= {std::vector<float>(batch_size * num_classes),   batch_size,   1,  1, num_classes};
	mov	w0, 10	// tmp1182,
// src/cpp/benchmark.cpp:161:     };
	stp	x26, x23, [x9]	// tmp2323, tmp2334,
	stp	x24, x22, [x9, 24]	// tmp2325, tmp2326,
// src/cpp/benchmark.cpp:168:     const benchmark_impl_t implementations[] = {
	ldp	q31, q27, [x1]	// tmp1203, tmp1204,
// src/cpp/benchmark.cpp:153:     Tensor final_logits= {std::vector<float>(batch_size * num_classes),   batch_size,   1,  1, num_classes};
	str	w0, [sp, 1052]	// tmp1182, final_logits.width
// src/cpp/benchmark.cpp:168:     const benchmark_impl_t implementations[] = {
	mov	x0, x1	// tmp2348, tmp2348
// src/cpp/benchmark.cpp:161:     };
	add	x1, sp, 464	// tmp2740,,
	str	x20, [sp, 1288]	// tmp2339, MEM[(struct CNNContext *)_2139].final_logits
	str	x1, [sp, 1184]	// tmp2740, MEM[(struct CNNContext *)_2139].conv1_bias
	add	x1, sp, 776	// tmp2741,,
// src/cpp/benchmark.cpp:168:     const benchmark_impl_t implementations[] = {
	ldr	q30, [x0, 96]	// tmp1209,
// src/cpp/benchmark.cpp:161:     };
	str	x1, [sp, 1192]	// tmp2741, MEM[(struct CNNContext *)_2139].conv1_out
// src/cpp/benchmark.cpp:168:     const benchmark_impl_t implementations[] = {
	add	x1, x0, 32	// tmp2744, tmp2348,
	ldp	q26, q29, [x1]	// tmp1205, tmp1206,
// src/cpp/benchmark.cpp:161:     };
	add	x1, sp, 816	// tmp2746,,
	str	x1, [sp, 1200]	// tmp2746, MEM[(struct CNNContext *)_2139].pool1_out
// src/cpp/benchmark.cpp:168:     const benchmark_impl_t implementations[] = {
	add	x1, x0, 64	// tmp2749, tmp2348,
// src/cpp/benchmark.cpp:161:     };
	add	x0, sp, 560	// tmp2753,,
// src/cpp/benchmark.cpp:168:     const benchmark_impl_t implementations[] = {
	str	q31, [sp, 1056]	// tmp1203, implementations
// src/cpp/benchmark.cpp:161:     };
	str	x0, [sp, 1216]	// tmp2753, MEM[(struct CNNContext *)_2139].conv2_bias
	add	x0, sp, 856	// tmp2754,,
// src/cpp/benchmark.cpp:168:     const benchmark_impl_t implementations[] = {
	ldp	q28, q31, [x1]	// tmp1207, tmp1208,
// src/cpp/benchmark.cpp:161:     };
	add	x1, sp, 512	// tmp2751,,
	str	x0, [sp, 1224]	// tmp2754, MEM[(struct CNNContext *)_2139].conv2_out
	add	x0, sp, 608	// tmp2755,,
	str	x1, [sp, 1208]	// tmp2751, MEM[(struct CNNContext *)_2139].conv2_weight
	str	x0, [sp, 1240]	// tmp2755, MEM[(struct CNNContext *)_2139].conv3_weight
	add	x0, sp, 976	// tmp2756,,
	str	x0, [sp, 1264]	// tmp2756, MEM[(struct CNNContext *)_2139].avgpool_out
// src/cpp/benchmark.cpp:168:     const benchmark_impl_t implementations[] = {
	add	x0, sp, 1072	// tmp2991,,
	stp	q27, q26, [x0]	// tmp1204, tmp1205,
	add	x0, sp, 1104	// tmp2986,,
	stp	q29, q28, [x0]	// tmp1206, tmp1207,
	add	x0, sp, 1136	// tmp2758,,
	stp	q31, q30, [x0]	// tmp1208, tmp1209,
// src/cpp/benchmark.cpp:175:     if (verify_mode) {
	ldr	w0, [sp, 128]	//, %sfp
	cbnz	w0, .L869	// verify_mode,
// src/cpp/benchmark.cpp:192:     if (profile_mode) {
	ldr	w0, [sp, 136]	//, %sfp
	cbz	w0, .L320	// profile_mode,
// src/cpp/benchmark.cpp:193:         std::cout << Color::BOLD_CYAN << "\n--- Per-Layer Profiling (" << num_runs << " Runs) ---"
	adrp	x0, _ZN5Color9BOLD_CYANE	// tmp2769,
	ldr	x1, [x0, #:lo12:_ZN5Color9BOLD_CYANE]	//, BOLD_CYAN
	mov	x0, x28	//, tmp2355
.LEHB87:
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC51	// tmp1274,
	mov	x2, 26	//,
	add	x1, x1, :lo12:.LC51	//, tmp1274,
// src/cpp/benchmark.cpp:193:         std::cout << Color::BOLD_CYAN << "\n--- Per-Layer Profiling (" << num_runs << " Runs) ---"
	mov	x20, x0	// _560, tmp2462
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:194:                   << Color::RESET << "\n";
	mov	x0, x20	//, _560
	mov	w1, 10	//,
	bl	_ZNSolsEi		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC52	// tmp1276,
	mov	x2, 10	//,
	add	x1, x1, :lo12:.LC52	//, tmp1276,
// src/cpp/benchmark.cpp:194:                   << Color::RESET << "\n";
	mov	x20, x0	// _562, tmp2463
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:194:                   << Color::RESET << "\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2770,
	mov	x0, x20	//, _562
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:194:                   << Color::RESET << "\n";
	ldr	x1, [sp, 168]	//, %sfp
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.LEHE87:
// src/cpp/benchmark.cpp:208:         };
	adrp	x2, .LC53	// tmp1282,
	mov	x1, 10	//,
	add	x2, x2, :lo12:.LC53	// tmp1281, tmp1282,
	add	x0, sp, 1312	//,,
	str	x2, [sp, 1304]	// tmp1281, MEM[(struct LayerStat *)_1195].name
.LEHB88:
	bl	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0		//
.LEHE88:
// src/cpp/benchmark.cpp:208:         };
	adrp	x2, .LC54	// tmp1286,
	mov	x1, 10	//,
	add	x2, x2, :lo12:.LC54	// tmp1285, tmp1286,
	add	x0, sp, 1344	//,,
	str	x2, [sp, 1336]	// tmp1285, MEM[(struct LayerStat *)_1195 + 32B].name
.LEHB89:
	bl	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0		//
.LEHE89:
// src/cpp/benchmark.cpp:208:         };
	adrp	x2, .LC55	// tmp1290,
	mov	x1, 10	//,
	add	x2, x2, :lo12:.LC55	// tmp1289, tmp1290,
	add	x0, sp, 1376	//,,
	str	x2, [sp, 1368]	// tmp1289, MEM[(struct LayerStat *)_1195 + 64B].name
.LEHB90:
	bl	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0		//
.LEHE90:
// src/cpp/benchmark.cpp:208:         };
	adrp	x2, .LC56	// tmp1294,
	mov	x1, 10	//,
	add	x2, x2, :lo12:.LC56	// tmp1293, tmp1294,
	add	x0, sp, 1408	//,,
	str	x2, [sp, 1400]	// tmp1293, MEM[(struct LayerStat *)_1195 + 96B].name
.LEHB91:
	bl	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0		//
.LEHE91:
// src/cpp/benchmark.cpp:208:         };
	adrp	x2, .LC57	// tmp1298,
	mov	x1, 10	//,
	add	x2, x2, :lo12:.LC57	// tmp1297, tmp1298,
	add	x0, sp, 1440	//,,
	str	x2, [sp, 1432]	// tmp1297, MEM[(struct LayerStat *)_1195 + 128B].name
.LEHB92:
	bl	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0		//
.LEHE92:
// src/cpp/benchmark.cpp:208:         };
	adrp	x2, .LC58	// tmp1302,
	mov	x1, 10	//,
	add	x2, x2, :lo12:.LC58	// tmp1301, tmp1302,
	add	x0, sp, 1472	//,,
	str	x2, [sp, 1464]	// tmp1301, MEM[(struct LayerStat *)_1195 + 160B].name
.LEHB93:
	bl	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0		//
.LEHE93:
// src/cpp/benchmark.cpp:208:         };
	adrp	x2, .LC59	// tmp1306,
	mov	x1, 10	//,
	add	x2, x2, :lo12:.LC59	// tmp1305, tmp1306,
	add	x0, sp, 1504	//,,
	str	x2, [sp, 1496]	// tmp1305, MEM[(struct LayerStat *)_1195 + 192B].name
.LEHB94:
	bl	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0		//
.LEHE94:
// src/cpp/benchmark.cpp:208:         };
	adrp	x2, .LC60	// tmp1310,
	mov	x1, 10	//,
	add	x2, x2, :lo12:.LC60	// tmp1309, tmp1310,
	add	x0, sp, 1536	//,,
	str	x2, [sp, 1528]	// tmp1309, MEM[(struct LayerStat *)_1195 + 224B].name
.LEHB95:
	bl	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0		//
.LEHE95:
// src/cpp/benchmark.cpp:208:         };
	adrp	x2, .LC61	// tmp1314,
	mov	x1, 10	//,
	add	x2, x2, :lo12:.LC61	// tmp1313, tmp1314,
	add	x0, sp, 1568	//,,
	str	x2, [sp, 1560]	// tmp1313, MEM[(struct LayerStat *)_1195 + 256B].name
.LEHB96:
	bl	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0		//
.LEHE96:
// src/cpp/benchmark.cpp:208:         };
	adrp	x2, .LC62	// tmp1318,
	mov	x1, 10	//,
	add	x2, x2, :lo12:.LC62	// tmp1317, tmp1318,
	add	x0, sp, 1600	//,,
	str	x2, [sp, 1592]	// tmp1317, MEM[(struct LayerStat *)_1195 + 288B].name
.LEHB97:
	bl	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0		//
.LEHE97:
// src/cpp/benchmark.cpp:211:         for (int w = 0; w < num_warmup; ++w) cnn_baseline(ctx);
	add	x24, sp, 1168	// tmp2344,,
	mov	x0, x24	//, tmp2344
.LEHB98:
	bl	_Z12cnn_baselineR10CNNContext		//
	mov	x0, x24	//, tmp2344
	bl	_Z12cnn_baselineR10CNNContext		//
	mov	x20, 0	// ivtmp.703,
	.p2align 5,,15
.L357:
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	mov	w2, 0	//,
	mov	x1, 9219	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:140:     ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9216	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/benchmark.cpp:217:             pmu_cycles_start(&pmu); conv2d_forward(ctx.input_batch, ctx.conv1_weight, ctx.conv1_bias, ctx.conv1_out, 1, 0);
	add	x8, sp, 1168	//,,
	mov	w5, 0	//,
	ldp	x0, x1, [x8]	//,,
	mov	w4, 1	//,
	ldp	x2, x3, [x8, 16]	//,,
	bl	_Z14conv2d_forwardRK6TensorS1_S1_RS_ii		//
// src/cpp/include/timing.h:144:     ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9217	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	add	x1, sp, 336	// tmp2771,,
	mov	w0, w21	//, _1334
	mov	x2, 8	//,
	bl	read		//
	movi	d29, #0	//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	cmn	x0, #1	// tmp2473,
	beq	.L340		//,
// src/cpp/benchmark.cpp:218:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	d29, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
	ucvtf	d29, d29	//, MEM[(long unsigned int *)_2140]
.L340:
// src/cpp/benchmark.cpp:218:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	x3, [sp, 1312]	// MEM[(struct vector *)_1195 + 8B].D.101473._M_impl.D.100808._M_start, MEM[(struct vector *)_1195 + 8B].D.101473._M_impl.D.100808._M_start
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	mov	w2, 0	//,
	mov	x1, 9219	//,
	mov	w0, w21	//, _1334
// src/cpp/benchmark.cpp:218:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	str	d29, [x3, x20]	//, *_528
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	bl	ioctl		//
// src/cpp/include/timing.h:140:     ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0);
	mov	w0, w21	//, _1334
	mov	w2, 0	//,
	mov	x1, 9216	//,
	bl	ioctl		//
// src/cpp/benchmark.cpp:220:             pmu_cycles_start(&pmu); relu_forward(ctx.conv1_out);
	ldr	x0, [sp, 1192]	//, MEM[(struct CNNContext *)_2139].conv1_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/include/timing.h:144:     ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9217	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	add	x1, sp, 336	// tmp2772,,
	mov	w0, w21	//, _1334
	mov	x2, 8	//,
	bl	read		//
	movi	d28, #0	//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	cmn	x0, #1	// tmp2474,
	beq	.L341		//,
// src/cpp/benchmark.cpp:221:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	d28, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
	ucvtf	d28, d28	//, MEM[(long unsigned int *)_2140]
.L341:
// src/cpp/benchmark.cpp:221:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	x3, [sp, 1344]	// MEM[(struct vector *)_1195 + 40B].D.101473._M_impl.D.100808._M_start, MEM[(struct vector *)_1195 + 40B].D.101473._M_impl.D.100808._M_start
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	mov	w2, 0	//,
	mov	x1, 9219	//,
	mov	w0, w21	//, _1334
// src/cpp/benchmark.cpp:221:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	str	d28, [x3, x20]	//, *_548
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	bl	ioctl		//
// src/cpp/include/timing.h:140:     ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9216	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/benchmark.cpp:223:             pmu_cycles_start(&pmu); maxpool2d_forward(ctx.conv1_out, ctx.pool1_out, 2, 2);
	mov	w3, 2	//,
	ldr	x0, [sp, 1192]	//, MEM[(struct CNNContext *)_2139].conv1_out
	mov	w2, w3	//,
	ldr	x1, [sp, 1200]	//, MEM[(struct CNNContext *)_2139].pool1_out
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/include/timing.h:144:     ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9217	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	add	x1, sp, 336	// tmp2773,,
	mov	w0, w21	//, _1334
	mov	x2, 8	//,
	bl	read		//
	movi	d27, #0	//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	cmn	x0, #1	// tmp2475,
	beq	.L342		//,
// src/cpp/benchmark.cpp:224:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	d27, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
	ucvtf	d27, d27	//, MEM[(long unsigned int *)_2140]
.L342:
// src/cpp/benchmark.cpp:224:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	x3, [sp, 1376]	// MEM[(struct vector *)_1195 + 72B].D.101473._M_impl.D.100808._M_start, MEM[(struct vector *)_1195 + 72B].D.101473._M_impl.D.100808._M_start
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	mov	w2, 0	//,
	mov	x1, 9219	//,
	mov	w0, w21	//, _1334
// src/cpp/benchmark.cpp:224:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	str	d27, [x3, x20]	//, *_620
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	bl	ioctl		//
// src/cpp/include/timing.h:140:     ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9216	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/benchmark.cpp:226:             pmu_cycles_start(&pmu); conv2d_forward(ctx.pool1_out, ctx.conv2_weight, ctx.conv2_bias, ctx.conv2_out, 1, 0);
	add	x7, sp, 1200	//,,
	mov	w5, 0	//,
	ldp	x0, x1, [x7]	//,,
	mov	w4, 1	//,
	ldp	x2, x3, [x7, 16]	//,,
	bl	_Z14conv2d_forwardRK6TensorS1_S1_RS_ii		//
// src/cpp/include/timing.h:144:     ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9217	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	add	x1, sp, 336	// tmp2774,,
	mov	w0, w21	//, _1334
	mov	x2, 8	//,
	bl	read		//
	movi	d26, #0	//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	cmn	x0, #1	// tmp2476,
	beq	.L343		//,
// src/cpp/benchmark.cpp:227:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	d26, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
	ucvtf	d26, d26	//, MEM[(long unsigned int *)_2140]
.L343:
// src/cpp/benchmark.cpp:227:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	x3, [sp, 1408]	// MEM[(struct vector *)_1195 + 104B].D.101473._M_impl.D.100808._M_start, MEM[(struct vector *)_1195 + 104B].D.101473._M_impl.D.100808._M_start
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	mov	w2, 0	//,
	mov	x1, 9219	//,
	mov	w0, w21	//, _1334
// src/cpp/benchmark.cpp:227:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	str	d26, [x3, x20]	//, *_617
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	bl	ioctl		//
// src/cpp/include/timing.h:140:     ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0);
	mov	w0, w21	//, _1334
	mov	w2, 0	//,
	mov	x1, 9216	//,
	bl	ioctl		//
// src/cpp/benchmark.cpp:229:             pmu_cycles_start(&pmu); relu_forward(ctx.conv2_out);
	ldr	x0, [sp, 1224]	//, MEM[(struct CNNContext *)_2139].conv2_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/include/timing.h:144:     ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9217	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	add	x1, sp, 336	// tmp2775,,
	mov	w0, w21	//, _1334
	mov	x2, 8	//,
	bl	read		//
	movi	d25, #0	//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	cmn	x0, #1	// tmp2477,
	beq	.L344		//,
// src/cpp/benchmark.cpp:230:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	d25, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
	ucvtf	d25, d25	//, MEM[(long unsigned int *)_2140]
.L344:
// src/cpp/benchmark.cpp:230:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	x3, [sp, 1440]	// MEM[(struct vector *)_1195 + 136B].D.101473._M_impl.D.100808._M_start, MEM[(struct vector *)_1195 + 136B].D.101473._M_impl.D.100808._M_start
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	mov	w2, 0	//,
	mov	x1, 9219	//,
	mov	w0, w21	//, _1334
// src/cpp/benchmark.cpp:230:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	str	d25, [x3, x20]	//, *_624
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	bl	ioctl		//
// src/cpp/include/timing.h:140:     ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9216	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/benchmark.cpp:232:             pmu_cycles_start(&pmu); maxpool2d_forward(ctx.conv2_out, ctx.pool2_out, 2, 2);
	mov	w3, 2	//,
	ldr	x0, [sp, 1224]	//, MEM[(struct CNNContext *)_2139].conv2_out
	mov	w2, w3	//,
	ldr	x1, [sp, 1232]	//, MEM[(struct CNNContext *)_2139].pool2_out
	bl	_Z17maxpool2d_forwardRK6TensorRS_ii		//
// src/cpp/include/timing.h:144:     ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9217	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	add	x1, sp, 336	// tmp2776,,
	mov	w0, w21	//, _1334
	mov	x2, 8	//,
	bl	read		//
	movi	d24, #0	//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	cmn	x0, #1	// tmp2478,
	beq	.L345		//,
// src/cpp/benchmark.cpp:233:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	d24, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
	ucvtf	d24, d24	//, MEM[(long unsigned int *)_2140]
.L345:
// src/cpp/benchmark.cpp:233:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	x3, [sp, 1472]	// MEM[(struct vector *)_1195 + 168B].D.101473._M_impl.D.100808._M_start, MEM[(struct vector *)_1195 + 168B].D.101473._M_impl.D.100808._M_start
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	mov	w2, 0	//,
	mov	x1, 9219	//,
	mov	w0, w21	//, _1334
// src/cpp/benchmark.cpp:233:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	str	d24, [x3, x20]	//, *_689
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	bl	ioctl		//
// src/cpp/include/timing.h:140:     ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9216	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/benchmark.cpp:235:             pmu_cycles_start(&pmu); conv2d_forward(ctx.pool2_out, ctx.conv3_weight, ctx.conv3_bias, ctx.conv3_out, 1, 0);
	add	x6, sp, 1232	//,,
	mov	w5, 0	//,
	ldp	x0, x1, [x6]	//,,
	mov	w4, 1	//,
	ldp	x2, x3, [x6, 16]	//,,
	bl	_Z14conv2d_forwardRK6TensorS1_S1_RS_ii		//
// src/cpp/include/timing.h:144:     ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9217	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	add	x1, sp, 336	// tmp2777,,
	mov	w0, w21	//, _1334
	mov	x2, 8	//,
	bl	read		//
	movi	d23, #0	//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	cmn	x0, #1	// tmp2479,
	beq	.L346		//,
// src/cpp/benchmark.cpp:236:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	d23, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
	ucvtf	d23, d23	//, MEM[(long unsigned int *)_2140]
.L346:
// src/cpp/benchmark.cpp:236:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	x3, [sp, 1504]	// MEM[(struct vector *)_1195 + 200B].D.101473._M_impl.D.100808._M_start, MEM[(struct vector *)_1195 + 200B].D.101473._M_impl.D.100808._M_start
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	mov	w2, 0	//,
	mov	x1, 9219	//,
	mov	w0, w21	//, _1334
// src/cpp/benchmark.cpp:236:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	str	d23, [x3, x20]	//, *_676
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	bl	ioctl		//
// src/cpp/include/timing.h:140:     ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0);
	mov	w0, w21	//, _1334
	mov	w2, 0	//,
	mov	x1, 9216	//,
	bl	ioctl		//
// src/cpp/benchmark.cpp:238:             pmu_cycles_start(&pmu); relu_forward(ctx.conv3_out);
	ldr	x0, [sp, 1256]	//, MEM[(struct CNNContext *)_2139].conv3_out
	bl	_Z12relu_forwardR6Tensor		//
// src/cpp/include/timing.h:144:     ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9217	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	add	x1, sp, 336	// tmp2778,,
	mov	w0, w21	//, _1334
	mov	x2, 8	//,
	bl	read		//
	movi	d22, #0	//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	cmn	x0, #1	// tmp2480,
	beq	.L347		//,
// src/cpp/benchmark.cpp:239:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	d22, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
	ucvtf	d22, d22	//, MEM[(long unsigned int *)_2140]
.L347:
// src/cpp/benchmark.cpp:239:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	x3, [sp, 1536]	// MEM[(struct vector *)_1195 + 232B].D.101473._M_impl.D.100808._M_start, MEM[(struct vector *)_1195 + 232B].D.101473._M_impl.D.100808._M_start
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	mov	w2, 0	//,
	mov	x1, 9219	//,
	mov	w0, w21	//, _1334
// src/cpp/benchmark.cpp:239:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	str	d22, [x3, x20]	//, *_659
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	bl	ioctl		//
// src/cpp/include/timing.h:140:     ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0);
	mov	x1, 9216	//,
	mov	w0, w21	//, _1334
	mov	w2, 0	//,
	bl	ioctl		//
// src/cpp/benchmark.cpp:241:             pmu_cycles_start(&pmu); adaptive_avgpool2d_forward(ctx.conv3_out, ctx.avgpool_out);
	ldr	x0, [sp, 1256]	//, MEM[(struct CNNContext *)_2139].conv3_out
	ldr	x1, [sp, 1264]	//, MEM[(struct CNNContext *)_2139].avgpool_out
	bl	_Z26adaptive_avgpool2d_forwardRK6TensorRS_		//
// src/cpp/include/timing.h:144:     ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9217	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	add	x1, sp, 336	// tmp2779,,
	mov	w0, w21	//, _1334
	mov	x2, 8	//,
	bl	read		//
	movi	d21, #0	//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	cmn	x0, #1	// tmp2481,
	beq	.L348		//,
// src/cpp/benchmark.cpp:242:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	d21, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
	ucvtf	d21, d21	//, MEM[(long unsigned int *)_2140]
.L348:
// src/cpp/benchmark.cpp:242:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	x3, [sp, 1568]	// MEM[(struct vector *)_1195 + 264B].D.101473._M_impl.D.100808._M_start, MEM[(struct vector *)_1195 + 264B].D.101473._M_impl.D.100808._M_start
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	mov	w2, 0	//,
	mov	x1, 9219	//,
	mov	w0, w21	//, _1334
// src/cpp/benchmark.cpp:242:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	str	d21, [x3, x20]	//, *_642
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	bl	ioctl		//
// src/cpp/include/timing.h:140:     ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9216	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/benchmark.cpp:244:             pmu_cycles_start(&pmu); linear_forward(ctx.avgpool_out, ctx.fc_weight, ctx.fc_bias, ctx.final_logits);
	add	x4, sp, 1264	//,,
	ldp	x0, x1, [x4]	//,,
	ldp	x2, x3, [x4, 16]	//,,
	bl	_Z14linear_forwardRK6TensorS1_S1_RS_		//
// src/cpp/include/timing.h:144:     ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9217	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	add	x1, sp, 336	// tmp2780,,
	mov	w0, w21	//, _1334
	mov	x2, 8	//,
	bl	read		//
	movi	d20, #0	//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	cmn	x0, #1	// tmp2482,
	beq	.L349		//,
// src/cpp/benchmark.cpp:245:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	d20, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
	ucvtf	d20, d20	//, MEM[(long unsigned int *)_2140]
.L349:
// src/cpp/benchmark.cpp:247:             benchmark_global_sink += checksum_tensor(ctx.final_logits);
	ldr	x0, [sp, 1288]	// _79, MEM[(struct CNNContext *)_2139].final_logits
// src/cpp/benchmark.cpp:245:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	ldr	x1, [sp, 1600]	// MEM[(struct vector *)_1195 + 296B].D.101473._M_impl.D.100808._M_start, MEM[(struct vector *)_1195 + 296B].D.101473._M_impl.D.100808._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_iterator.h:1068:       : _M_current(__i) { }
	ldp	x3, x0, [x0]	// _636, _597,* _79
// src/cpp/benchmark.cpp:245:             c = pmu_cycles_stop(&pmu); layers[li++].cycles[i] = (double)c;
	str	d20, [x1, x20]	//, *_402
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	cmp	x0, x3	// _597, _636
	beq	.L573		//,
	sub	x2, x0, #4	// _147, _597,
	mov	x0, x3	// ivtmp.696, _636
	sub	x2, x2, x3	// _5320, _147, _636
	lsr	x1, x2, 2	// _5319, _5320,
	add	x1, x1, 1	// niters.556, _5319,
	cmp	x2, 56	// _5320,
	bls	.L574		//,
	movi	v18.2d, 0	//
	lsr	x2, x1, 4	// bnd.557_5304, niters.556,
	add	x2, x3, x2, lsl 6	// _1690, _636, bnd.557_5304,
	mov	v17.16b, v18.16b	//,
	mov	v16.16b, v18.16b	//,
	mov	v7.16b, v18.16b	//,
	mov	v6.16b, v18.16b	//,
	mov	v27.16b, v18.16b	//,
	mov	v4.16b, v18.16b	//,
	mov	v28.16b, v18.16b	//,
	.p2align 5,,15
.L352:
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	ldp	q23, q24, [x0]	//,,* ivtmp.696
	ldp	q25, q26, [x0, 32]	//,,
	add	x0, x0, 64	// ivtmp.696, ivtmp.696,
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fcvtl	v0.2d, v23.2s	//,
	fcvtl	v2.2d, v24.2s	//,
	fcvtl2	v23.2d, v23.4s	//,
	fcvtl2	v24.2d, v24.4s	//,
	fcvtl	v1.2d, v25.2s	//,
	fcvtl	v31.2d, v26.2s	//,
	fcvtl2	v25.2d, v25.4s	//,
	fcvtl2	v26.2d, v26.4s	//,
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fadd	v28.2d, v28.2d, v0.2d	//,,
	fadd	v4.2d, v4.2d, v23.2d	//,,
	fadd	v27.2d, v27.2d, v2.2d	//,,
	fadd	v6.2d, v6.2d, v24.2d	//,,
	fadd	v7.2d, v7.2d, v1.2d	//,,
	fadd	v16.2d, v16.2d, v25.2d	//,,
	fadd	v17.2d, v17.2d, v31.2d	//,,
	fadd	v18.2d, v18.2d, v26.2d	//,,
	cmp	x2, x0	// _1690, ivtmp.696
	bne	.L352		//,
	fadd	v27.2d, v6.2d, v27.2d	//,,
	and	x4, x1, -16	// niters_vector_mult_vf.558, niters.556,
	fadd	v28.2d, v4.2d, v28.2d	//,,
	fadd	v27.2d, v27.2d, v16.2d	//,,
	fadd	v28.2d, v28.2d, v7.2d	//,,
	fadd	v27.2d, v27.2d, v18.2d	//,,
	fadd	v28.2d, v28.2d, v17.2d	//,,
	fadd	v28.2d, v27.2d, v28.2d	//,,
	faddp	d19, v28.2d	//,
	tst	x1, 15	// niters.556,
	beq	.L350		//,
.L351:
	add	x3, x3, x4, lsl 2	// vectp.574, _636, niters_vector_mult_vf.558,
// src/cpp/include/utils.h:129:     double s = 0.0;
	mov	x0, 0	// ivtmp_5215,
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	cntw	x2		// tmp1414
// src/cpp/include/utils.h:129:     double s = 0.0;
	mov	z29.d, #0	//
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	ptrue	p4.b, all	// tmp1407
	sub	x1, x1, x4	// bnd.571, niters.556, niters_vector_mult_vf.558
	whilelo	p7.s, xzr, x1	// next_mask_5213,, bnd.571
.L355:
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	ld1w	z5.s, p7/z, [x3, x0, lsl 2]	//, next_mask_5213,* vectp.574
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	punpklo	p5.h, p7.b	// loop_mask_5223, next_mask_5213
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	zip1	z3.s, z5.s, z5.s	//,,
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	punpkhi	p6.h, p7.b	// loop_mask_5222, next_mask_5213
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	add	x0, x0, x2	// ivtmp_5215, ivtmp_5215, tmp1414
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fcvt	z3.d, p4/m, z3.s	//, tmp1407,
	zip2	z5.s, z5.s, z5.s	//,,
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fadd	z29.d, p5/m, z29.d, z3.d	//, loop_mask_5223,
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fcvt	z5.d, p4/m, z5.s	//, tmp1407,
	whilelo	p7.s, x0, x1	// next_mask_5213, ivtmp_5215, bnd.571
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fadd	z29.d, p6/m, z29.d, z5.d	//, loop_mask_5222,
	b.any	.L355	//,
	faddv	d29, p4, z29.d	//, tmp1407,
	fadd	d19, d19, d29	//,,
.L350:
// src/cpp/benchmark.cpp:247:             benchmark_global_sink += checksum_tensor(ctx.final_logits);
	adrp	x0, .LANCHOR1	// tmp1417,
// src/cpp/benchmark.cpp:213:         for (int i = 0; i < num_runs; ++i) {
	add	x20, x20, 8	// ivtmp.703, ivtmp.703,
// src/cpp/benchmark.cpp:247:             benchmark_global_sink += checksum_tensor(ctx.final_logits);
	ldr	d30, [x0, #:lo12:.LANCHOR1]	//, benchmark_global_sink
	fadd	d30, d30, d19	//,,
	str	d30, [x0, #:lo12:.LANCHOR1]	//, benchmark_global_sink
// src/cpp/benchmark.cpp:213:         for (int i = 0; i < num_runs; ++i) {
	cmp	x20, 80	// ivtmp.703,
	bne	.L357		//,
// src/cpp/benchmark.cpp:250:         std::vector<double> medians(num_layers);
	mov	x1, 10	//,
	add	x0, sp, 304	//,,
	bl	_ZNSt6vectorIdSaIdEEC2EmRKS0_.constprop.0		//
.LEHE98:
	ldr	x25, [sp, 304]	// medians$D101473$_M_impl$D100808$_M_start, MEM <double *> [(struct vector *)_807]
// src/cpp/benchmark.cpp:251:         double total = 0.0;
	movi	d15, #0	// total
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	fmov	d14, 5.0e-1	// tmp2398,
	adrp	x22, _ZL11_cmp_doublePKvS0_	// tmp2340,
// src/cpp/benchmark.cpp:250:         std::vector<double> medians(num_layers);
	mov	x20, 0	// ivtmp.674,
	add	x22, x22, :lo12:_ZL11_cmp_doublePKvS0_	// tmp2331, tmp2340,
	ldr	x0, [sp, 320]	// medians$_M_impl$D100808$_M_end_of_storage, MEM <double *> [(struct vector *)_807 + 16B]
	str	x0, [sp, 128]	// medians$_M_impl$D100808$_M_end_of_storage, %sfp
	.p2align 5,,15
.L358:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1263:       { return _M_data_ptr(this->_M_impl._M_start); }
	ubfiz	x0, x20, 5, 32	// _5310, ivtmp.674,,
// src/cpp/include/timing.h:101:     qsort(arr, n, sizeof(double), _cmp_double);
	mov	x3, x22	//, tmp2331
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1263:       { return _M_data_ptr(this->_M_impl._M_start); }
	add	x0, x19, x0	// tmp1424, tmp2346, _5310
// src/cpp/include/timing.h:101:     qsort(arr, n, sizeof(double), _cmp_double);
	mov	x2, 8	//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1263:       { return _M_data_ptr(this->_M_impl._M_start); }
	ldr	x21, [x0, 8]	// _398, MEM[(double * *)_1195 + 8B + _5310 * 1]
// src/cpp/include/timing.h:101:     qsort(arr, n, sizeof(double), _cmp_double);
	mov	x1, 10	//,
	mov	x0, x21	//, _398
.LEHB99:
	bl	qsort		//
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	ldp	d31, d30, [x21, 32]	// MEM[(double *)_398 + 32B], MEM[(double *)_398 + 40B],
	fadd	d31, d31, d30	// _1427, MEM[(double *)_398 + 32B], MEM[(double *)_398 + 40B]
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	fmul	d31, d31, d14	// _1428, _1427, tmp2398
// src/cpp/benchmark.cpp:253:             medians[k] = compute_median(layers[k].cycles.data(), num_runs);
	str	d31, [x25, x20, lsl 3]	// _1428, MEM[(value_type &)medians$D101473$_M_impl$D100808$_M_start_160 + ivtmp.674_5308 * 8]
// src/cpp/benchmark.cpp:252:         for (int k = 0; k < num_layers; ++k) {
	add	x20, x20, 1	// ivtmp.674, ivtmp.674,
// src/cpp/benchmark.cpp:254:             total += medians[k];
	fadd	d15, d15, d31	// total, total, _1428
// src/cpp/benchmark.cpp:252:         for (int k = 0; k < num_layers; ++k) {
	cmp	x20, 10	// ivtmp.674,
	bne	.L358		//,
// src/cpp/benchmark.cpp:257:         std::cout << Color::GREEN << "=== Per-Layer Breakdown (median cycles) ===" << Color::RESET << "\n";
	adrp	x0, _ZN5Color5GREENE	// tmp2782,
	ldr	x1, [x0, #:lo12:_ZN5Color5GREENE]	//, GREEN
	mov	x0, x28	//, tmp2355
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC63	// tmp1436,
	mov	x2, 43	//,
	add	x1, x1, :lo12:.LC63	//, tmp1436,
// src/cpp/benchmark.cpp:257:         std::cout << Color::GREEN << "=== Per-Layer Breakdown (median cycles) ===" << Color::RESET << "\n";
	mov	x20, x0	// _603, tmp2483
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:257:         std::cout << Color::GREEN << "=== Per-Layer Breakdown (median cycles) ===" << Color::RESET << "\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2783,
	mov	x0, x20	//, _603
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:257:         std::cout << Color::GREEN << "=== Per-Layer Breakdown (median cycles) ===" << Color::RESET << "\n";
	ldr	x1, [sp, 168]	//, %sfp
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.LEHE99:
// src/cpp/benchmark.cpp:261:                       << medians[k] << " cycles  (" << (100.0 * medians[k] / total) << "%)\n";
	mov	x0, 4636737291354636288	// tmp2591,
	adrp	x26, .LC45	// tmp2342,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x27, .LC64	// tmp2392,
	add	x26, x26, :lo12:.LC45	// tmp2337, tmp2342,
// src/cpp/benchmark.cpp:261:                       << medians[k] << " cycles  (" << (100.0 * medians[k] / total) << "%)\n";
	fmov	d14, x0	// tmp2394, tmp2591
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	add	x27, x27, :lo12:.LC64	// tmp2393, tmp2392,
	mov	x22, x25	// ivtmp.671, medians$D101473$_M_impl$D100808$_M_start
	mov	x21, x19	// ivtmp.672, tmp2346
	add	x20, sp, 1624	// _694,,
// src/cpp/benchmark.cpp:261:                       << medians[k] << " cycles  (" << (100.0 * medians[k] / total) << "%)\n";
	fdiv	d14, d14, d15	// _5150, tmp2394, total
	b	.L366		//
	.p2align 2,,3
.L359:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x23	//, BOLD.36_85
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, x0	//, tmp2484
	mov	x1, x23	//, BOLD.36_85
	mov	x0, x28	//, tmp2355
.LEHB100:
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:260:                       << std::string(9 - strlen(layers[k].name), ' ')
	ldr	x23, [x21]	// _86, MEM[(const char * *)_5307]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:670:       if (!__s)
	cbz	x23, .L870	// _86,
.L361:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x23	//, _86
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, x0	//, tmp2485
	mov	x1, x23	//, _86
	mov	x0, x28	//, tmp2355
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:260:                       << std::string(9 - strlen(layers[k].name), ' ')
	adrp	x0, _ZN5Color5RESETE	// tmp2784,
	ldr	x24, [x0, #:lo12:_ZN5Color5RESETE]	// RESET.37_87, RESET
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:670:       if (!__s)
	cbz	x24, .L871	// RESET.37_87,
.L363:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x24	//, RESET.37_87
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, x0	//, tmp2486
	mov	x1, x24	//, RESET.37_87
	mov	x0, x28	//, tmp2355
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
.L364:
// src/cpp/benchmark.cpp:260:                       << std::string(9 - strlen(layers[k].name), ' ')
	mov	x0, x23	//, _86
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	add	x24, sp, 352	// tmp2367,,
// src/cpp/benchmark.cpp:260:                       << std::string(9 - strlen(layers[k].name), ' ')
	bl	strlen		//
// src/cpp/benchmark.cpp:260:                       << std::string(9 - strlen(layers[k].name), ' ')
	mov	x1, 9	// tmp1486,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:671:       { _M_construct(__n, __c); }
	mov	w2, 32	//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	str	x24, [sp, 336]	// tmp2367, MEM[(struct _Alloc_hider *)_2140]._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:671:       { _M_construct(__n, __c); }
	sub	x1, x1, x0	//, tmp1486, tmp2487
	add	x0, sp, 336	// tmp2785,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc		//
.LEHE100:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:4082:       return __ostream_insert(__os, __str.data(), __str.size());
	ldp	x1, x2, [sp, 336]	//,,
	mov	x0, x28	//, tmp2355
.LEHB101:
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	ldr	d0, [x22]	//, MEM[(value_type &)_5299]
	bl	_ZNSo9_M_insertIdEERSoT_		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, 10	//,
	mov	x1, x27	//, tmp2393
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	mov	x23, x0	// _329, tmp2488
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:261:                       << medians[k] << " cycles  (" << (100.0 * medians[k] / total) << "%)\n";
	ldr	d0, [x22]	// MEM[(value_type &)_5299], MEM[(value_type &)_5299]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	mov	x0, x23	//, _329
	fmul	d0, d14, d0	//, _5150, MEM[(value_type &)_5299]
	bl	_ZNSo9_M_insertIdEERSoT_		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC65	// tmp1500,
	mov	x2, 3	//,
	add	x1, x1, :lo12:.LC65	//, tmp1500,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
.LEHE101:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x0, [sp, 336]	// _1456, MEM[(struct basic_string *)_2140]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x0, x24	// _1456, tmp2367
	beq	.L365		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 352]	// MEM[(struct basic_string *)_2140].D.64159._M_allocated_capacity, MEM[(struct basic_string *)_2140].D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	add	x1, x1, 1	//, MEM[(struct basic_string *)_2140].D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L365:
// src/cpp/benchmark.cpp:258:         for (int k = 0; k < num_layers; ++k) {
	add	x21, x21, 32	// ivtmp.672, ivtmp.672,
	add	x22, x22, 8	// ivtmp.671, ivtmp.671,
	cmp	x21, x20	// ivtmp.672, _694
	beq	.L872		//,
.L366:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, 2	//,
	mov	x1, x26	//, tmp2337
	mov	x0, x28	//, tmp2355
.LEHB102:
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:260:                       << std::string(9 - strlen(layers[k].name), ' ')
	adrp	x0, _ZN5Color4BOLDE	// tmp1446,
	ldr	x23, [x0, #:lo12:_ZN5Color4BOLDE]	// BOLD.36_85, BOLD
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:670:       if (!__s)
	cbnz	x23, .L359	// BOLD.36_85,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:671: 	__out.setstate(ios_base::badbit);
	ldr	x0, [x28]	// cout._vptr.basic_ostream, cout._vptr.basic_ostream
	ldr	x0, [x0, -24]	// MEM[(long int *)_1432 + -24B], MEM[(long int *)_1432 + -24B]
	add	x0, x28, x0	// _1435, tmp2355, MEM[(long int *)_1432 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:187:   { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
	ldr	w1, [x0, 32]	//, MEM[(const struct basic_ios *)_1435].D.81517._M_streambuf_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	orr	w1, w1, 1	//, MEM[(const struct basic_ios *)_1435].D.81517._M_streambuf_state,
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
// src/cpp/benchmark.cpp:260:                       << std::string(9 - strlen(layers[k].name), ' ')
	ldr	x23, [x21]	// _86, MEM[(const char * *)_5307]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:670:       if (!__s)
	cbnz	x23, .L361	// _86,
.L870:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:671: 	__out.setstate(ios_base::badbit);
	ldr	x0, [x28]	// cout._vptr.basic_ostream, cout._vptr.basic_ostream
	ldr	x0, [x0, -24]	// MEM[(long int *)_1440 + -24B], MEM[(long int *)_1440 + -24B]
	add	x0, x28, x0	// _1443, tmp2355, MEM[(long int *)_1440 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:187:   { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
	ldr	w1, [x0, 32]	//, MEM[(const struct basic_ios *)_1443].D.81517._M_streambuf_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	orr	w1, w1, 1	//, MEM[(const struct basic_ios *)_1443].D.81517._M_streambuf_state,
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
// src/cpp/benchmark.cpp:260:                       << std::string(9 - strlen(layers[k].name), ' ')
	adrp	x0, _ZN5Color5RESETE	// tmp2784,
	ldr	x24, [x0, #:lo12:_ZN5Color5RESETE]	// RESET.37_87, RESET
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:670:       if (!__s)
	cbnz	x24, .L363	// RESET.37_87,
.L871:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:671: 	__out.setstate(ios_base::badbit);
	ldr	x0, [x28]	// cout._vptr.basic_ostream, cout._vptr.basic_ostream
	ldr	x0, [x0, -24]	// MEM[(long int *)_1448 + -24B], MEM[(long int *)_1448 + -24B]
	add	x0, x28, x0	// _1451, tmp2355, MEM[(long int *)_1448 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:187:   { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
	ldr	w1, [x0, 32]	//, MEM[(const struct basic_ios *)_1451].D.81517._M_streambuf_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	orr	w1, w1, 1	//, MEM[(const struct basic_ios *)_1451].D.81517._M_streambuf_state,
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
.LEHE102:
	b	.L364		//
	.p2align 2,,3
.L865:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:381: 	return __builtin_memcmp(__s1, __s2, __n);
	ldr	w0, [x26]	//, MEM <char[1:6]> [(void *)pretmp_5671]
	cmp	w0, w23	// MEM <char[1:6]> [(void *)pretmp_5671], tmp2441
	bne	.L307		//,
	ldrh	w1, [x26, 4]	// MEM <char[1:6]> [(void *)pretmp_5671], MEM <char[1:6]> [(void *)pretmp_5671]
	mov	w0, 31078	// tmp968,
	cmp	w1, w0	// MEM <char[1:6]> [(void *)pretmp_5671], tmp968
	bne	.L307		//,
// src/cpp/benchmark.cpp:88:             verify_mode = true;
	mov	w0, 1	// verify_mode,
	str	w0, [sp, 128]	// verify_mode, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x26, x20	// pretmp_5671, tmp2347
	bne	.L309		//,
	b	.L308		//
	.p2align 2,,3
.L301:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	cbz	x0, .L303	// _1306,
	mov	x0, x20	// _707, tmp2347
	b	.L300		//
	.p2align 2,,3
.L864:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	add	x1, sp, 1168	// tmp2664,,
	mov	x2, 0	//,
	mov	x0, x19	//, tmp2346
.LEHB103:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm		//
.LEHE103:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	ldr	x1, [sp, 1168]	// MEM[(long unsigned int *)_2139], MEM[(long unsigned int *)_2139]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:218:       { _M_dataplus._M_p = __p; }
	str	x0, [sp, 1304]	// _707, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	str	x1, [sp, 1320]	// MEM[(long unsigned int *)_2139], MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity
.L300:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	mov	x2, x26	//, _1306
	mov	x1, x27	//, _4
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:251: 	_M_set_length(__dnew);
	ldr	x26, [sp, 1168]	// _1306, MEM[(long unsigned int *)_2139]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x0, [sp, 1304]	// pretmp_5516, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	b	.L302		//
	.p2align 2,,3
.L866:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:381: 	return __builtin_memcmp(__s1, __s2, __n);
	ldr	w0, [x26]	//, MEM <char[1:7]> [(void *)pretmp_5671]
	cmp	w0, w25	// MEM <char[1:7]> [(void *)pretmp_5671], tmp2440
	bne	.L307		//,
	ldrh	w1, [x26, 4]	// MEM <char[1:7]> [(void *)pretmp_5671], MEM <char[1:7]> [(void *)pretmp_5671]
	mov	w0, 27753	// tmp976,
	cmp	w1, w0	// MEM <char[1:7]> [(void *)pretmp_5671], tmp976
	bne	.L307		//,
	ldrb	w0, [x26, 6]	// MEM <char[1:7]> [(void *)pretmp_5671], MEM <char[1:7]> [(void *)pretmp_5671]
	cmp	w0, 101	// MEM <char[1:7]> [(void *)pretmp_5671],
	bne	.L307		//,
// src/cpp/benchmark.cpp:90:             profile_mode = true;
	mov	w0, 1	// profile_mode,
	str	w0, [sp, 136]	// profile_mode, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x26, x20	// pretmp_5671, tmp2347
	bne	.L309		//,
	b	.L308		//
.L561:
// src/cpp/benchmark.cpp:92:             verbose = true;
	str	w1, [sp, 208]	// _818, %sfp
	b	.L313		//
.L320:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x0, 168	//,
.LEHB104:
	bl	_Znwm		//
.LEHE104:
	mov	x7, x0	// _1475, tmp2489
// src/cpp/benchmark.cpp:271:     const char *bench_filter = std::getenv("CNN_BENCH_FILTER");
	adrp	x1, .LC68	// tmp1556,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:940: 	*__first = __value;
	mov	x3, x7	// tmp1526, _1475
	mov	x2, x7	// tmp1538, _1475
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	str	x7, [sp, 232]	// _1475, %sfp
// src/cpp/benchmark.cpp:271:     const char *bench_filter = std::getenv("CNN_BENCH_FILTER");
	add	x0, x1, :lo12:.LC68	//, tmp1556,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:940: 	*__first = __value;
	mov	x1, x7	// tmp1550, _1475
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	stp	xzr, xzr, [x7]	//,,* _1475
// src/cpp/benchmark.cpp:272:     int num_selected = 0;
	mov	w23, 0	// num_selected,
	add	x24, sp, 1168	// tmp2344,,
	add	x20, sp, 1320	// tmp2347,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:940: 	*__first = __value;
	ldp	x4, x5, [x7]	// MEM[(const struct bench_result_t &)_1475], MEM[(const struct bench_result_t &)_1475]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	strb	wzr, [x7, 16]	//, _1475->has_data
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_algobase.h:940: 	*__first = __value;
	ldr	x6, [x7, 16]	// MEM[(const struct bench_result_t &)_1475], MEM[(const struct bench_result_t &)_1475]
	stp	x4, x5, [x7, 24]	// MEM[(const struct bench_result_t &)_1475], MEM[(struct bench_result_t *)_1475 + 24B]
	mov	x4, x6	// MEM[(const struct bench_result_t &)_1475], MEM[(const struct bench_result_t &)_1475]
	str	x6, [x7, 40]	// MEM[(const struct bench_result_t &)_1475], MEM[(struct bench_result_t *)_1475 + 24B]
	ldp	x6, x7, [x3], 48	// MEM[(const struct bench_result_t &)_1475], MEM[(const struct bench_result_t &)_1475]
	stp	x6, x7, [x2, 48]	// MEM[(const struct bench_result_t &)_1475], MEM[(struct bench_result_t *)_1475 + 48B]
	str	x4, [x3, 16]	// MEM[(const struct bench_result_t &)_1475], MEM[(struct bench_result_t *)_1475 + 48B]
	ldp	x6, x7, [x2]	// MEM[(const struct bench_result_t &)_1475], MEM[(const struct bench_result_t &)_1475]
	ldr	x4, [x2, 16]	// MEM[(const struct bench_result_t &)_1475], MEM[(const struct bench_result_t &)_1475]
	stp	x6, x7, [x2, 72]	// MEM[(const struct bench_result_t &)_1475], MEM[(struct bench_result_t *)_1475 + 72B]
	mov	x6, x2	// _1475, _1475
	mov	x3, x4	// MEM[(const struct bench_result_t &)_1475], MEM[(const struct bench_result_t &)_1475]
	str	x4, [x2, 88]	// MEM[(const struct bench_result_t &)_1475], MEM[(struct bench_result_t *)_1475 + 72B]
	ldp	x4, x5, [x2], 96	// MEM[(const struct bench_result_t &)_1475], MEM[(const struct bench_result_t &)_1475]
	stp	x4, x5, [x1, 96]	// MEM[(const struct bench_result_t &)_1475], MEM[(struct bench_result_t *)_1475 + 96B]
	str	x3, [x2, 16]	// MEM[(const struct bench_result_t &)_1475], MEM[(struct bench_result_t *)_1475 + 96B]
	ldp	x4, x5, [x1]	// MEM[(const struct bench_result_t &)_1475], MEM[(const struct bench_result_t &)_1475]
	ldr	x3, [x1, 16]	// MEM[(const struct bench_result_t &)_1475], MEM[(const struct bench_result_t &)_1475]
	stp	x4, x5, [x1, 120]	// MEM[(const struct bench_result_t &)_1475], MEM[(struct bench_result_t *)_1475 + 120B]
	str	x3, [x1, 136]	// MEM[(const struct bench_result_t &)_1475], MEM[(struct bench_result_t *)_1475 + 120B]
	ldp	x4, x5, [x1], 144	// MEM[(const struct bench_result_t &)_1475], MEM[(const struct bench_result_t &)_1475]
	stp	x4, x5, [x6, 144]	// MEM[(const struct bench_result_t &)_1475], MEM[(struct bench_result_t *)_1475 + 144B]
	str	x3, [x1, 16]	// MEM[(const struct bench_result_t &)_1475], MEM[(struct bench_result_t *)_1475 + 144B]
// src/cpp/benchmark.cpp:271:     const char *bench_filter = std::getenv("CNN_BENCH_FILTER");
	bl	getenv		//
	add	x1, sp, 1056	// tmp2808,,
	str	x0, [sp, 160]	// tmp2490, %sfp
	mov	x22, x1	// tmp2809, tmp2808
	str	x1, [sp, 192]	// tmp2808, %sfp
	.p2align 5,,15
.L379:
// src/cpp/benchmark.cpp:274:         if (implementation_matches_filter(implementations[i].name, bench_filter)) num_selected++;
	ldr	x26, [x22]	// _95, MEM[(const char * *)_1399]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	str	x20, [sp, 1304]	// tmp2347, MEM[(struct _Alloc_hider *)_1195]._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:650: 	if (__s == 0)
	cbz	x26, .L873	// _95,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x26	//, _95
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:225: 	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
	str	x0, [sp, 336]	// _1486, MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x25, x0	// _1486, tmp2491
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:227: 	if (__dnew > size_type(_S_local_capacity))
	cmp	x0, 15	// _1486,
	bhi	.L874		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:432: 	if (__n == 1)
	cmp	x0, 1	// _1486,
	beq	.L875		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	cbnz	x0, .L876	// _1486,
.L377:
	mov	x2, x20	// pretmp_5537, tmp2347
.L376:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x25, [sp, 1312]	// _1486, MEM[(struct basic_string *)_1195]._M_string_length
// src/cpp/benchmark.cpp:274:         if (implementation_matches_filter(implementations[i].name, bench_filter)) num_selected++;
	mov	x0, x19	//, tmp2346
	ldr	x1, [sp, 160]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x2, x25]	//, MEM[(char_type &)_1496]
.LEHB105:
// src/cpp/benchmark.cpp:274:         if (implementation_matches_filter(implementations[i].name, bench_filter)) num_selected++;
	bl	_ZL29implementation_matches_filterRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKc		//
.LEHE105:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x2, [sp, 1304]	// _1497, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// src/cpp/benchmark.cpp:274:         if (implementation_matches_filter(implementations[i].name, bench_filter)) num_selected++;
	and	w25, w0, 255	// _551, tmp2493
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x2, x20	// _1497, tmp2347
	beq	.L378		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 1320]	// MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity, MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x0, x2	//, _1497
	add	x1, x1, 1	//, MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L378:
// src/cpp/benchmark.cpp:273:     for (int i = 0; i < num_impls; ++i)
	add	x22, x22, 16	// ivtmp.804, ivtmp.804,
// src/cpp/benchmark.cpp:274:         if (implementation_matches_filter(implementations[i].name, bench_filter)) num_selected++;
	add	w23, w23, w25	// num_selected, num_selected, _551
// src/cpp/benchmark.cpp:273:     for (int i = 0; i < num_impls; ++i)
	cmp	x24, x22	// tmp2344, ivtmp.804
	bne	.L379		//,
// src/cpp/benchmark.cpp:276:     if (num_selected == 0) {
	cbz	w23, .L877	// num_selected,
// src/cpp/benchmark.cpp:282:     if (bench_filter && *bench_filter) {
	ldr	x0, [sp, 160]	// bench_filter, %sfp
	cbz	x0, .L383	// bench_filter,
// src/cpp/benchmark.cpp:282:     if (bench_filter && *bench_filter) {
	ldrb	w0, [x0]	// *bench_filter_425, *bench_filter_425
	cbnz	w0, .L878	// *bench_filter_425,
.L383:
// src/cpp/benchmark.cpp:288:     std::cout << Color::BOLD_CYAN << "Runs: " << num_runs << " measured, "
	adrp	x0, _ZN5Color9BOLD_CYANE	// tmp2817,
	ldr	x1, [x0, #:lo12:_ZN5Color9BOLD_CYANE]	//, BOLD_CYAN
	mov	x0, x28	//, tmp2355
.LEHB106:
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC74	// tmp1606,
	mov	x2, 6	//,
	add	x1, x1, :lo12:.LC74	//, tmp1606,
// src/cpp/benchmark.cpp:288:     std::cout << Color::BOLD_CYAN << "Runs: " << num_runs << " measured, "
	mov	x22, x0	// _438, tmp2498
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:288:     std::cout << Color::BOLD_CYAN << "Runs: " << num_runs << " measured, "
	mov	x0, x22	//, _438
	mov	w1, 10	//,
	bl	_ZNSolsEi		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC75	// tmp1608,
	mov	x2, 11	//,
	add	x1, x1, :lo12:.LC75	//, tmp1608,
// src/cpp/benchmark.cpp:288:     std::cout << Color::BOLD_CYAN << "Runs: " << num_runs << " measured, "
	mov	x22, x0	// _440, tmp2499
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:289:               << num_warmup << " warmup\n\n" << Color::RESET;
	mov	x0, x22	//, _440
	mov	w1, 2	//,
	bl	_ZNSolsEi		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC76	// tmp1610,
	mov	x2, 9	//,
	add	x1, x1, :lo12:.LC76	//, tmp1610,
// src/cpp/benchmark.cpp:289:               << num_warmup << " warmup\n\n" << Color::RESET;
	mov	x22, x0	// _442, tmp2500
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:289:               << num_warmup << " warmup\n\n" << Color::RESET;
	adrp	x1, _ZN5Color5RESETE	// tmp2818,
	mov	x0, x22	//, _442
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.LEHE106:
// src/cpp/benchmark.cpp:76:               << " s (" << result.seconds_median * 1000.0 << " ms)\n\n";
	mov	x0, 70368744177664	// tmp2590,
// src/cpp/include/timing.h:87:     return (double)(ctx->os_stop_nsec - ctx->os_start_nsec) / 1e9;
	adrp	x1, .LC117	// tmp2819,
// src/cpp/benchmark.cpp:76:               << " s (" << result.seconds_median * 1000.0 << " ms)\n\n";
	movk	x0, 0x408f, lsl 48	// tmp2590,,
// src/cpp/include/timing.h:62:     return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
	mov	x26, 51712	// tmp2431,
// src/cpp/include/timing.h:87:     return (double)(ctx->os_stop_nsec - ctx->os_start_nsec) / 1e9;
	ldr	d15, [x1, #:lo12:.LC117]	// tmp2432,
// src/cpp/include/timing.h:62:     return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
	movk	x26, 0x3b9a, lsl 16	// tmp2431,,
// src/cpp/benchmark.cpp:76:               << " s (" << result.seconds_median * 1000.0 << " ms)\n\n";
	fmov	d14, x0	// tmp2435, tmp2590
	ldr	x1, [sp, 232]	// _1475, %sfp
	str	x1, [sp, 184]	// _1475, %sfp
// src/cpp/benchmark.cpp:289:               << num_warmup << " warmup\n\n" << Color::RESET;
	str	x1, [sp, 200]	// _1475, %sfp
	add	x1, sp, 1056	// tmp2822,,
	str	x1, [sp, 176]	// tmp2822, %sfp
	.p2align 5,,15
.L420:
// src/cpp/benchmark.cpp:295:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	ldr	x0, [sp, 176]	// ivtmp.794, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	str	x20, [sp, 1304]	// tmp2347, MEM[(struct _Alloc_hider *)_1195]._M_p
// src/cpp/benchmark.cpp:295:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	ldr	x22, [x0]	// _101, MEM[(const char * *)_1565]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:650: 	if (__s == 0)
	cbz	x22, .L879	// _101,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x22	//, _101
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:225: 	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
	str	x0, [sp, 336]	// _1504, MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x23, x0	// _1504, tmp2501
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:227: 	if (__dnew > size_type(_S_local_capacity))
	cmp	x0, 15	// _1504,
	bhi	.L880		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:432: 	if (__n == 1)
	cmp	x0, 1	// _1504,
	beq	.L881		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	cbnz	x0, .L882	// _1504,
.L389:
	mov	x2, x20	// pretmp_5543, tmp2347
.L388:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x23, [sp, 1312]	// _1504, MEM[(struct basic_string *)_1195]._M_string_length
// src/cpp/benchmark.cpp:295:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	mov	x0, x19	//, tmp2346
	ldr	x1, [sp, 160]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x2, x23]	//, MEM[(char_type &)_1514]
.LEHB107:
// src/cpp/benchmark.cpp:295:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	bl	_ZL29implementation_matches_filterRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKc		//
.LEHE107:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x2, [sp, 1304]	// _1515, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// src/cpp/benchmark.cpp:295:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	and	w23, w0, 255	// _527, tmp2503
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x2, x20	// _1515, tmp2347
	beq	.L390		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 1320]	// MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity, MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x0, x2	//, _1515
	add	x1, x1, 1	//, MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L390:
// src/cpp/benchmark.cpp:295:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	tbz	x23, 0, .L391	// _527,,
// src/cpp/benchmark.cpp:299:                   << "==================================================" << Color::RESET << "\n";
	adrp	x0, _ZN5Color4CYANE	// tmp2824,
	ldr	x1, [x0, #:lo12:_ZN5Color4CYANE]	//, CYAN
	mov	x0, x28	//, tmp2355
.LEHB108:
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC77	// tmp1639,
	mov	x2, 51	//,
	add	x1, x1, :lo12:.LC77	//, tmp1639,
// src/cpp/benchmark.cpp:299:                   << "==================================================" << Color::RESET << "\n";
	mov	x23, x0	// _532, tmp2504
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	adrp	x1, .LC78	// tmp1641,
	mov	x2, 17	//,
	add	x1, x1, :lo12:.LC78	//, tmp1641,
	mov	x0, x23	//, _532
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:299:                   << "==================================================" << Color::RESET << "\n";
	mov	x1, x22	//, _101
	mov	x0, x23	//, _532
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC79	// tmp1643,
	mov	x2, 14	//,
	add	x1, x1, :lo12:.LC79	//, tmp1643,
// src/cpp/benchmark.cpp:299:                   << "==================================================" << Color::RESET << "\n";
	mov	x22, x0	// _534, tmp2505
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	adrp	x1, .LC80	// tmp1645,
	mov	x2, 50	//,
	add	x1, x1, :lo12:.LC80	//, tmp1645,
	mov	x0, x22	//, _534
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:299:                   << "==================================================" << Color::RESET << "\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2825,
	mov	x0, x22	//, _534
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:299:                   << "==================================================" << Color::RESET << "\n";
	ldr	x1, [sp, 168]	//, %sfp
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:35:     bench_result_t result = {NAN, NAN, false};
	ldr	x0, [sp, 216]	// tmp2348, %sfp
// src/cpp/benchmark.cpp:37:     std::cout << Color::GREEN << "--- Performance Benchmarking (" << num_runs << " Runs) ---"
	adrp	x1, _ZN5Color5GREENE	// tmp2828,
	ldr	x22, [x1, #:lo12:_ZN5Color5GREENE]	// GREEN.108_1521, GREEN
// src/cpp/benchmark.cpp:35:     bench_result_t result = {NAN, NAN, false};
	ldp	x2, x3, [x0, 112]	// tmp1657,
	ldr	x0, [x0, 128]	// tmp1658,
	stp	x2, x3, [sp, 240]	// tmp1657, D.111033
	str	x0, [sp, 256]	// tmp1658, D.111033
// src/cpp/benchmark.cpp:300:         results[i] = benchmark_cnn(implementations[i].function, ctx, pmu, num_runs, num_warmup);
	ldr	x0, [sp, 176]	// ivtmp.794, %sfp
	ldr	x25, [x0, 8]	// _104, MEM[(void (*<T7f54>) (struct CNNContext &) *)_1565 + 8B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:670:       if (!__s)
	cbz	x22, .L883	// GREEN.108_1521,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x22	//, GREEN.108_1521
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, x0	//, tmp2506
	mov	x1, x22	//, GREEN.108_1521
	mov	x0, x28	//, tmp2355
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
.L393:
	adrp	x1, .LC81	// tmp1673,
	mov	x2, 30	//,
	add	x1, x1, :lo12:.LC81	//, tmp1673,
	mov	x0, x28	//, tmp2355
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:38:                << Color::RESET << "\n";
	mov	w1, 10	//,
	mov	x0, x28	//, tmp2355
	bl	_ZNSolsEi		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC52	// tmp1679,
	mov	x2, 10	//,
	add	x1, x1, :lo12:.LC52	//, tmp1679,
// src/cpp/benchmark.cpp:38:                << Color::RESET << "\n";
	mov	x22, x0	// _1523, tmp2507
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:38:                << Color::RESET << "\n";
	adrp	x0, _ZN5Color5RESETE	// tmp2830,
	ldr	x23, [x0, #:lo12:_ZN5Color5RESETE]	// RESET.109_1524, RESET
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:670:       if (!__s)
	cbz	x23, .L884	// RESET.109_1524,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x23	//, RESET.109_1524
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, x0	//, tmp2508
	mov	x1, x23	//, RESET.109_1524
	mov	x0, x22	//, _1523
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
.L395:
	ldr	x1, [sp, 168]	//, %sfp
	mov	x0, x22	//, _1523
	mov	x2, 1	//,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x0, 80	//,
	bl	_Znwm		//
.LEHE108:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	movi	v31.2d, 0	// tmp1690
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x1, x0	// tmp2509,
	mov	x0, 80	//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	add	x2, x1, 32	// tmp2835, _1629,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	str	x1, [sp, 144]	// _1629, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	stp	q31, q31, [x1]	// tmp1690, tmp1690,* _1629
	stp	q31, q31, [x2]	// tmp1690, tmp1690,
	str	q31, [x1, 64]	// tmp1690, MEM <vector(2) double> [(double *)_1629 + 64B]
.LEHB109:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	bl	_Znwm		//
.LEHE109:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	movi	v31.2d, 0	// tmp1696
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	mov	x3, x0	// _1650, tmp2510
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC82	// tmp1702,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	add	x4, x3, 32	// tmp2842, _1650,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	add	x1, x1, :lo12:.LC82	//, tmp1702,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:151: 	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
	str	x3, [sp, 152]	// _1650, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, 8	//,
	mov	x0, x28	//, tmp2355
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_construct.h:119:       ::new((void*)__p) _Tp(std::forward<_Args>(__args)...);
	stp	q31, q31, [x3]	// tmp1696, tmp1696,* _1650
	stp	q31, q31, [x4]	// tmp1696, tmp1696,
	str	q31, [x3, 64]	// tmp1696, MEM <vector(2) double> [(double *)_1650 + 64B]
.LEHB110:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:43:     std::cout << "Running " << num_warmup << " warmup run" << (num_warmup == 1 ? "" : "s") << "...\n";
	mov	w1, 2	//,
	mov	x0, x28	//, tmp2355
	bl	_ZNSolsEi		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC83	// tmp1708,
	mov	x2, 11	//,
	add	x1, x1, :lo12:.LC83	//, tmp1708,
// src/cpp/benchmark.cpp:43:     std::cout << "Running " << num_warmup << " warmup run" << (num_warmup == 1 ? "" : "s") << "...\n";
	mov	x22, x0	// _1527, tmp2511
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	adrp	x0, .LC84	// tmp2360,
	mov	x2, 1	//,
	add	x0, x0, :lo12:.LC84	// tmp2353, tmp2360,
	mov	x1, x0	// tmp2353, tmp2353
	mov	x0, x22	//, _1527
	str	x1, [sp, 128]	// tmp2353, %sfp
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	adrp	x1, .LC85	// tmp2361,
	mov	x0, x22	//, _1527
	add	x1, x1, :lo12:.LC85	// tmp2354, tmp2361,
	mov	x2, 4	//,
	str	x1, [sp, 136]	// tmp2354, %sfp
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	mov	w27, 2	// ivtmp_5344,
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	cntw	x22		// tmp2428
	.p2align 5,,15
.L403:
// src/cpp/benchmark.cpp:45:         f(ctx);
	mov	x0, x24	//, tmp2344
	blr	x25		// _104
// src/cpp/benchmark.cpp:46:         benchmark_global_sink += checksum_tensor(ctx.final_logits);
	ldr	x0, [sp, 1288]	// _1530, MEM[(struct CNNContext *)_2139].final_logits
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	adrp	x4, .LANCHOR1	// tmp2430,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_iterator.h:1068:       : _M_current(__i) { }
	ldp	x3, x0, [x0]	// _1531, _1532,* _1530
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	cmp	x3, x0	// _1531, _1532
	beq	.L577		//,
	sub	x2, x0, #4	// _5149, _1532,
	mov	x0, x3	// ivtmp.782, _1531
	sub	x2, x2, x3	// _5094, _5149, _1531
	lsr	x1, x2, 2	// _5093, _5094,
	add	x1, x1, 1	// niters.602, _5093,
	cmp	x2, 56	// _5094,
	bls	.L578		//,
	movi	v26.2d, 0	// vect_s_1383.606
	lsr	x2, x1, 4	// bnd.603_5078, niters.602,
	add	x2, x3, x2, lsl 6	// _5202, _1531, bnd.603_5078,
	mov	v24.16b, v26.16b	// vect_s_1537.614, vect_s_1383.606
	mov	v23.16b, v26.16b	// vect_s_1537.614, vect_s_1383.606
	mov	v22.16b, v26.16b	// vect_s_1537.614, vect_s_1383.606
	mov	v25.16b, v26.16b	// vect_s_1537.614, vect_s_1383.606
	mov	v21.16b, v26.16b	// vect_s_1537.614, vect_s_1383.606
	mov	v27.16b, v26.16b	// vect_s_1537.614, vect_s_1383.606
	mov	v20.16b, v26.16b	// vect_s_1537.614, vect_s_1383.606
	.p2align 5,,15
.L398:
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	ldp	q28, q29, [x0]	// MEM <const vector(4) float> [(const float &)_5128], MEM <const vector(4) float> [(const float &)_5128 + 16],* ivtmp.782
	ldp	q30, q31, [x0, 32]	// MEM <const vector(4) float> [(const float &)_5128 + 32], MEM <const vector(4) float> [(const float &)_5128 + 48],
	add	x0, x0, 64	// ivtmp.782, ivtmp.782,
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fcvtl	v19.2d, v28.2s	// vect__1535.613_5055, MEM <const vector(4) float> [(const float &)_5128]
	fcvtl	v17.2d, v29.2s	// vect__1535.613_5053, MEM <const vector(4) float> [(const float &)_5128 + 16]
	fcvtl2	v28.2d, v28.4s	// vect__1535.613_5054, MEM <const vector(4) float> [(const float &)_5128]
	fcvtl2	v29.2d, v29.4s	// vect__1535.613_5052, MEM <const vector(4) float> [(const float &)_5128 + 16]
	fcvtl	v18.2d, v30.2s	// vect__1535.613_5051, MEM <const vector(4) float> [(const float &)_5128 + 32]
	fcvtl2	v30.2d, v30.4s	// vect__1535.613_5050, MEM <const vector(4) float> [(const float &)_5128 + 32]
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fadd	v20.2d, v20.2d, v19.2d	// vect_s_1537.614, vect_s_1537.614, vect__1535.613_5055
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fcvtl	v19.2d, v31.2s	// vect__1535.613_5049, MEM <const vector(4) float> [(const float &)_5128 + 48]
	fcvtl2	v31.2d, v31.4s	// vect__1535.613_5048, MEM <const vector(4) float> [(const float &)_5128 + 48]
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fadd	v27.2d, v27.2d, v28.2d	// vect_s_1537.614, vect_s_1537.614, vect__1535.613_5054
	fadd	v21.2d, v21.2d, v17.2d	// vect_s_1537.614, vect_s_1537.614, vect__1535.613_5053
	fadd	v25.2d, v25.2d, v29.2d	// vect_s_1537.614, vect_s_1537.614, vect__1535.613_5052
	fadd	v22.2d, v22.2d, v18.2d	// vect_s_1537.614, vect_s_1537.614, vect__1535.613_5051
	fadd	v23.2d, v23.2d, v30.2d	// vect_s_1537.614, vect_s_1537.614, vect__1535.613_5050
	fadd	v24.2d, v24.2d, v19.2d	// vect_s_1537.614, vect_s_1537.614, vect__1535.613_5049
	fadd	v26.2d, v26.2d, v31.2d	// vect_s_1383.606, vect_s_1383.606, vect__1535.613_5048
	cmp	x0, x2	// ivtmp.782, _5202
	bne	.L398		//,
	fadd	v30.2d, v27.2d, v20.2d	// _4993, vect_s_1537.614, vect_s_1537.614
	and	x2, x1, -16	// niters_vector_mult_vf.604, niters.602,
	fadd	v25.2d, v25.2d, v21.2d	// _5440, vect_s_1537.614, vect_s_1537.614
	fadd	v30.2d, v30.2d, v22.2d	// _5144, _4993, vect_s_1537.614
	fadd	v25.2d, v25.2d, v23.2d	// _5145, _5440, vect_s_1537.614
	fadd	v30.2d, v30.2d, v24.2d	// _5146, _5144, vect_s_1537.614
	fadd	v25.2d, v25.2d, v26.2d	// _5147, _5145, vect_s_1383.606
	fadd	v30.2d, v30.2d, v25.2d	// _5025, _5146, _5147
	faddp	d30, v30.2d	// s, _5025
	tst	x1, 15	// niters.602,
	beq	.L396		//,
.L397:
	add	x3, x3, x2, lsl 2	// vectp.620, _1531, niters_vector_mult_vf.604,
// src/cpp/include/utils.h:129:     double s = 0.0;
	mov	x0, 0	// ivtmp_4989,
	sub	x1, x1, x2	// bnd.617, niters.602, niters_vector_mult_vf.604
	ptrue	p5.b, all	// tmp2427
	mov	z29.d, #0	// vect_s_5091.618
	whilelo	p7.s, xzr, x1	// next_mask_4987,, bnd.617
.L401:
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	punpklo	p6.h, p7.b	// loop_mask_4997, next_mask_4987
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	ld1w	z31.s, p7/z, [x3, x0, lsl 2]	// vect_v_5090.621, next_mask_4987,* vectp.620
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	add	x0, x0, x22	// ivtmp_4989, ivtmp_4989, tmp2428
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	zip1	z28.s, z31.s, z31.s	// tmp1742, vect_v_5090.621, vect_v_5090.621
	zip2	z31.s, z31.s, z31.s	// tmp1746, vect_v_5090.621, vect_v_5090.621
	fcvt	z28.d, p5/m, z28.s	// vect__5089.622_4999, tmp2427, tmp1742
	fcvt	z31.d, p5/m, z31.s	// vect__5089.622_4998, tmp2427, tmp1746
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fadd	z29.d, p6/m, z29.d, z28.d	// vect_s_5088.623, loop_mask_4997, vect__5089.622_4999
	punpkhi	p6.h, p7.b	// loop_mask_4996, next_mask_4987
	whilelo	p7.s, x0, x1	// next_mask_4987, ivtmp_4989, bnd.617
	fadd	z29.d, p6/m, z29.d, z31.d	// vect_s_5091.618, loop_mask_4996, vect__5089.622_4998
	b.any	.L401	//,
	faddv	d29, p5, z29.d	// _4992, tmp2427, vect_s_5091.618
	fadd	d30, d30, d29	// s, s, _4992
.L396:
// src/cpp/benchmark.cpp:46:         benchmark_global_sink += checksum_tensor(ctx.final_logits);
	ldr	d31, [x4, #:lo12:.LANCHOR1]	// benchmark_global_sink.112_1539, benchmark_global_sink
	add	x23, x4, :lo12:.LANCHOR1	// tmp2349, tmp2430,
	fadd	d30, d30, d31	// _1540, s, benchmark_global_sink.112_1539
	str	d30, [x4, #:lo12:.LANCHOR1]	// _1540, benchmark_global_sink
// src/cpp/benchmark.cpp:44:     for (int w = 0; w < num_warmup; ++w) {
	cmp	w27, 1	// ivtmp_5344,
	beq	.L885		//,
	mov	w27, 1	// ivtmp_5344,
	b	.L403		//
.L867:
// src/cpp/benchmark.cpp:92:             verbose = true;
	mov	w0, 1	// verbose,
	str	w0, [sp, 208]	// verbose, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x26, x20	// pretmp_5671, tmp2347
	bne	.L309		//,
	b	.L308		//
.L876:
	mov	x0, x20	// _1490, tmp2347
.L374:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	mov	x2, x25	//, _1486
	mov	x1, x26	//, _95
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:251: 	_M_set_length(__dnew);
	ldr	x25, [sp, 336]	// _1486, MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x2, [sp, 1304]	// pretmp_5537, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	b	.L376		//
.L882:
	mov	x0, x20	// _1508, tmp2347
.L386:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	mov	x2, x23	//, _1504
	mov	x1, x22	//, _101
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:251: 	_M_set_length(__dnew);
	ldr	x23, [sp, 336]	// _1504, MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x2, [sp, 1304]	// pretmp_5543, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	b	.L388		//
	.p2align 2,,3
.L885:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC86	// tmp1757,
	mov	x2, 26	//,
	add	x1, x1, :lo12:.LC86	//, tmp1757,
	mov	x0, x28	//, tmp2355
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:50:               << (num_runs == 1 ? "" : "s") << "...\n";
	mov	w1, 10	//,
	mov	x0, x28	//, tmp2355
	bl	_ZNSolsEi		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC87	// tmp1763,
	mov	x2, 14	//,
	add	x1, x1, :lo12:.LC87	//, tmp1763,
// src/cpp/benchmark.cpp:50:               << (num_runs == 1 ? "" : "s") << "...\n";
	mov	x22, x0	// _1543, tmp2512
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	ldr	x1, [sp, 128]	//, %sfp
	mov	x2, 1	//,
	mov	x0, x22	//, _1543
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	ldr	x1, [sp, 136]	//, %sfp
	mov	x0, x22	//, _1543
	mov	x2, 4	//,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	mov	x22, 0	// ivtmp.768,
	.p2align 5,,15
.L412:
// src/cpp/include/timing.h:139:     ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
	mov	w2, 0	//,
	mov	x1, 9219	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:140:     ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0);
	mov	w0, w21	//, _1334
	mov	w2, 0	//,
	mov	x1, 9216	//,
	bl	ioctl		//
// src/cpp/include/timing.h:40:     __asm__ volatile("" ::: "memory");
// src/cpp/include/timing.h:58:     clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	mov	w0, 4	//,
	mov	x1, x19	//, tmp2346
	bl	clock_gettime		//
// src/cpp/include/timing.h:62:     return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
	ldr	x0, [sp, 1304]	// MEM[(struct timespec *)_1195].tv_sec, MEM[(struct timespec *)_1195].tv_sec
// src/cpp/include/timing.h:62:     return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
	ldr	x27, [sp, 1312]	// _1670, MEM[(struct timespec *)_1195].tv_nsec
// src/cpp/include/timing.h:62:     return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
	mul	x0, x0, x26	// _1668, MEM[(struct timespec *)_1195].tv_sec, tmp2431
	str	x0, [sp, 128]	// _1668, %sfp
// src/cpp/include/timing.h:40:     __asm__ volatile("" ::: "memory");
// src/cpp/benchmark.cpp:58:         f(ctx);
	mov	x0, x24	//, tmp2344
	blr	x25		// _104
// src/cpp/include/timing.h:40:     __asm__ volatile("" ::: "memory");
// src/cpp/include/timing.h:58:     clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	mov	w0, 4	//,
	mov	x1, x19	//, tmp2346
	bl	clock_gettime		//
// src/cpp/include/timing.h:62:     return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
	ldr	x0, [sp, 1304]	// MEM[(struct timespec *)_1195].tv_sec, MEM[(struct timespec *)_1195].tv_sec
// src/cpp/include/timing.h:62:     return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
	ldr	x3, [sp, 1312]	// _1683, MEM[(struct timespec *)_1195].tv_nsec
// src/cpp/include/timing.h:62:     return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
	mul	x0, x0, x26	// _1681, MEM[(struct timespec *)_1195].tv_sec, tmp2431
// src/cpp/include/timing.h:62:     return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
	str	x3, [sp, 136]	// _1683, %sfp
	sub	x27, x0, x27	// _5038, _1681, _1670
// src/cpp/include/timing.h:40:     __asm__ volatile("" ::: "memory");
// src/cpp/include/timing.h:144:     ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0);
	mov	w2, 0	//,
	mov	x1, 9217	//,
	mov	w0, w21	//, _1334
	bl	ioctl		//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	mov	x1, x19	//, tmp2346
	mov	w0, w21	//, _1334
	mov	x2, 8	//,
	bl	read		//
// src/cpp/include/timing.h:146:     if (read(ctx->fd, &val, sizeof(val)) == -1) {
	cmn	x0, #1	// tmp2513,
	ldr	x3, [sp, 136]	// _1683, %sfp
	movi	d31, #0	// _5548
	beq	.L404		//,
// src/cpp/benchmark.cpp:64:         cycles_arr[i] = (double)cycles;
	ldr	d31, [sp, 1304]	// MEM[(long unsigned int *)_1195], MEM[(long unsigned int *)_1195]
	ucvtf	d31, d31	// _5548, MEM[(long unsigned int *)_1195]
.L404:
// src/cpp/include/timing.h:87:     return (double)(ctx->os_stop_nsec - ctx->os_start_nsec) / 1e9;
	ldr	x0, [sp, 128]	// _1668, %sfp
// src/cpp/benchmark.cpp:64:         cycles_arr[i] = (double)cycles;
	ldr	x1, [sp, 144]	// _1629, %sfp
// src/cpp/include/timing.h:87:     return (double)(ctx->os_stop_nsec - ctx->os_start_nsec) / 1e9;
	sub	x3, x3, x0	// _5039, _1683, _1668
// src/cpp/benchmark.cpp:66:         benchmark_global_sink += checksum_tensor(ctx.final_logits);
	ldr	x0, [sp, 1288]	// _1560, MEM[(struct CNNContext *)_2139].final_logits
// src/cpp/include/timing.h:87:     return (double)(ctx->os_stop_nsec - ctx->os_start_nsec) / 1e9;
	add	x3, x3, x27	// _1549, _5039, _5038
// src/cpp/benchmark.cpp:64:         cycles_arr[i] = (double)cycles;
	str	d31, [x1, x22]	// _5548, MEM[(value_type &)_1629 + ivtmp.768_5301 * 1]
// src/cpp/include/timing.h:87:     return (double)(ctx->os_stop_nsec - ctx->os_start_nsec) / 1e9;
	ucvtf	d31, x3	// _1550, _1549
// src/cpp/benchmark.cpp:63:         sec_arr[i] = get_elapsed_os_sec(&timer_ctx);
	ldr	x1, [sp, 152]	// _1650, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_iterator.h:1068:       : _M_current(__i) { }
	ldp	x3, x0, [x0]	// _1561, _1562,* _1560
// src/cpp/include/timing.h:87:     return (double)(ctx->os_stop_nsec - ctx->os_start_nsec) / 1e9;
	fmul	d31, d31, d15	// _1551, _1550, tmp2432
// src/cpp/benchmark.cpp:63:         sec_arr[i] = get_elapsed_os_sec(&timer_ctx);
	str	d31, [x1, x22]	// _1551, MEM[(value_type &)_1650 + ivtmp.768_5301 * 1]
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	cmp	x3, x0	// _1561, _1562
	beq	.L581		//,
	sub	x2, x0, #4	// _5037, _1562,
	mov	x0, x3	// ivtmp.759, _1561
	sub	x2, x2, x3	// _5206, _5037, _1561
	lsr	x1, x2, 2	// _5205, _5206,
	add	x1, x1, 1	// niters.579, _5205,
	cmp	x2, 56	// _5206,
	bls	.L582		//,
	movi	v26.2d, 0	// vect_s_1361.583
	lsr	x2, x1, 4	// bnd.580_5190, niters.579,
	add	x2, x3, x2, lsl 6	// _5312, _1561, bnd.580_5190,
	mov	v24.16b, v26.16b	// vect_s_1567.591, vect_s_1361.583
	mov	v23.16b, v26.16b	// vect_s_1567.591, vect_s_1361.583
	mov	v22.16b, v26.16b	// vect_s_1567.591, vect_s_1361.583
	mov	v25.16b, v26.16b	// vect_s_1567.591, vect_s_1361.583
	mov	v21.16b, v26.16b	// vect_s_1567.591, vect_s_1361.583
	mov	v27.16b, v26.16b	// vect_s_1567.591, vect_s_1361.583
	mov	v20.16b, v26.16b	// vect_s_1567.591, vect_s_1361.583
	.p2align 5,,15
.L407:
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	ldp	q28, q29, [x0]	// MEM <const vector(4) float> [(const float &)_716], MEM <const vector(4) float> [(const float &)_716 + 16],* ivtmp.759
	ldp	q30, q31, [x0, 32]	// MEM <const vector(4) float> [(const float &)_716 + 32], MEM <const vector(4) float> [(const float &)_716 + 48],
	add	x0, x0, 64	// ivtmp.759, ivtmp.759,
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fcvtl	v19.2d, v28.2s	// vect__1565.590_5167, MEM <const vector(4) float> [(const float &)_716]
	fcvtl	v17.2d, v29.2s	// vect__1565.590_5165, MEM <const vector(4) float> [(const float &)_716 + 16]
	fcvtl2	v28.2d, v28.4s	// vect__1565.590_5166, MEM <const vector(4) float> [(const float &)_716]
	fcvtl2	v29.2d, v29.4s	// vect__1565.590_5164, MEM <const vector(4) float> [(const float &)_716 + 16]
	fcvtl	v18.2d, v30.2s	// vect__1565.590_5163, MEM <const vector(4) float> [(const float &)_716 + 32]
	fcvtl2	v30.2d, v30.4s	// vect__1565.590_5162, MEM <const vector(4) float> [(const float &)_716 + 32]
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fadd	v20.2d, v20.2d, v19.2d	// vect_s_1567.591, vect_s_1567.591, vect__1565.590_5167
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fcvtl	v19.2d, v31.2s	// vect__1565.590_5161, MEM <const vector(4) float> [(const float &)_716 + 48]
	fcvtl2	v31.2d, v31.4s	// vect__1565.590_5160, MEM <const vector(4) float> [(const float &)_716 + 48]
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fadd	v27.2d, v27.2d, v28.2d	// vect_s_1567.591, vect_s_1567.591, vect__1565.590_5166
	fadd	v21.2d, v21.2d, v17.2d	// vect_s_1567.591, vect_s_1567.591, vect__1565.590_5165
	fadd	v25.2d, v25.2d, v29.2d	// vect_s_1567.591, vect_s_1567.591, vect__1565.590_5164
	fadd	v22.2d, v22.2d, v18.2d	// vect_s_1567.591, vect_s_1567.591, vect__1565.590_5163
	fadd	v23.2d, v23.2d, v30.2d	// vect_s_1567.591, vect_s_1567.591, vect__1565.590_5162
	fadd	v24.2d, v24.2d, v19.2d	// vect_s_1567.591, vect_s_1567.591, vect__1565.590_5161
	fadd	v26.2d, v26.2d, v31.2d	// vect_s_1361.583, vect_s_1361.583, vect__1565.590_5160
	cmp	x0, x2	// ivtmp.759, _5312
	bne	.L407		//,
	fadd	v30.2d, v27.2d, v20.2d	// _1388, vect_s_1567.591, vect_s_1567.591
	and	x5, x1, -16	// niters_vector_mult_vf.581, niters.579,
	fadd	v25.2d, v25.2d, v21.2d	// _1381, vect_s_1567.591, vect_s_1567.591
	fadd	v30.2d, v30.2d, v22.2d	// _5032, _1388, vect_s_1567.591
	fadd	v25.2d, v25.2d, v23.2d	// _5033, _1381, vect_s_1567.591
	fadd	v30.2d, v30.2d, v24.2d	// _5034, _5032, vect_s_1567.591
	fadd	v25.2d, v25.2d, v26.2d	// _5035, _5033, vect_s_1361.583
	fadd	v30.2d, v30.2d, v25.2d	// _5137, _5034, _5035
	faddp	d30, v30.2d	// s, _5137
	tst	x1, 15	// niters.579,
	beq	.L405		//,
.L406:
	add	x3, x3, x5, lsl 2	// vectp.597, _1561, niters_vector_mult_vf.581,
// src/cpp/include/utils.h:129:     double s = 0.0;
	mov	x0, 0	// ivtmp_5101,
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	cntw	x2		// tmp1819
// src/cpp/include/utils.h:129:     double s = 0.0;
	mov	z29.d, #0	// vect_s_5203.595
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	ptrue	p5.b, all	// tmp1812
	sub	x1, x1, x5	// bnd.594, niters.579, niters_vector_mult_vf.581
	whilelo	p7.s, xzr, x1	// next_mask_5099,, bnd.594
.L410:
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	punpklo	p6.h, p7.b	// loop_mask_5109, next_mask_5099
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	ld1w	z31.s, p7/z, [x3, x0, lsl 2]	// vect_v_5202.598, next_mask_5099,* vectp.597
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	add	x0, x0, x2	// ivtmp_5101, ivtmp_5101, tmp1819
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	zip1	z28.s, z31.s, z31.s	// tmp1811, vect_v_5202.598, vect_v_5202.598
	zip2	z31.s, z31.s, z31.s	// tmp1815, vect_v_5202.598, vect_v_5202.598
	fcvt	z28.d, p5/m, z28.s	// vect__5201.599_5111, tmp1812, tmp1811
	fcvt	z31.d, p5/m, z31.s	// vect__5201.599_5110, tmp1812, tmp1815
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	fadd	z29.d, p6/m, z29.d, z28.d	// vect_s_5200.600, loop_mask_5109, vect__5201.599_5111
	punpkhi	p6.h, p7.b	// loop_mask_5108, next_mask_5099
	whilelo	p7.s, x0, x1	// next_mask_5099, ivtmp_5101, bnd.594
	fadd	z29.d, p6/m, z29.d, z31.d	// vect_s_5203.595, loop_mask_5108, vect__5201.599_5110
	b.any	.L410	//,
	faddv	d29, p5, z29.d	// _5104, tmp1812, vect_s_5203.595
	fadd	d30, d30, d29	// s, s, _5104
.L405:
// src/cpp/benchmark.cpp:66:         benchmark_global_sink += checksum_tensor(ctx.final_logits);
	ldr	d31, [x23]	// benchmark_global_sink.115_1569, benchmark_global_sink
// src/cpp/benchmark.cpp:52:     for (int i = 0; i < num_runs; ++i) {
	add	x22, x22, 8	// ivtmp.768, ivtmp.768,
// src/cpp/benchmark.cpp:66:         benchmark_global_sink += checksum_tensor(ctx.final_logits);
	fadd	d30, d30, d31	// _1570, s, benchmark_global_sink.115_1569
	str	d30, [x23]	// _1570, benchmark_global_sink
// src/cpp/benchmark.cpp:52:     for (int i = 0; i < num_runs; ++i) {
	cmp	x22, 80	// ivtmp.768,
	bne	.L412		//,
// src/cpp/include/timing.h:101:     qsort(arr, n, sizeof(double), _cmp_double);
	ldr	x23, [sp, 144]	// _1629, %sfp
	adrp	x22, _ZL11_cmp_doublePKvS0_	// tmp2340,
	mov	x2, 8	//,
	add	x22, x22, :lo12:_ZL11_cmp_doublePKvS0_	// tmp2331, tmp2340,
	mov	x1, 10	//,
	mov	x3, x22	//, tmp2331
	mov	x0, x23	//, _1629
	bl	qsort		//
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	add	x4, x23, 32	// tmp2853, _1629,
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	fmov	d31, 5.0e-1	// tmp1830,
// src/cpp/include/timing.h:101:     qsort(arr, n, sizeof(double), _cmp_double);
	mov	x3, x22	//, tmp2331
	ldr	x22, [sp, 152]	// _1650, %sfp
	mov	x2, 8	//,
	mov	x1, 10	//,
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	ldp	d12, d30, [x4]	// MEM[(double *)_1629 + 32B], MEM[(double *)_1629 + 40B],
// src/cpp/include/timing.h:101:     qsort(arr, n, sizeof(double), _cmp_double);
	mov	x0, x22	//, _1650
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	fadd	d12, d12, d30	// _1702, MEM[(double *)_1629 + 32B], MEM[(double *)_1629 + 40B]
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	fmul	d12, d12, d31	// _1703, _1702, tmp1830
// src/cpp/include/timing.h:101:     qsort(arr, n, sizeof(double), _cmp_double);
	bl	qsort		//
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	add	x0, x22, 32	// tmp2857, _1650,
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	fmov	d31, 5.0e-1	// tmp1836,
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	ldp	d13, d30, [x0]	// MEM[(double *)_1650 + 32B], MEM[(double *)_1650 + 40B],
// src/cpp/benchmark.cpp:73:     std::cout << Color::GREEN << "=== Results ===" << Color::RESET << "\n";
	adrp	x0, _ZN5Color5GREENE	// tmp2859,
	ldr	x22, [x0, #:lo12:_ZN5Color5GREENE]	// GREEN.116_1576, GREEN
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	fadd	d13, d13, d30	// _1722, MEM[(double *)_1650 + 32B], MEM[(double *)_1650 + 40B]
// src/cpp/include/timing.h:103:         return (arr[n/2 - 1] + arr[n/2]) / 2.0;
	fmul	d13, d13, d31	// _1723, _1722, tmp1836
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:670:       if (!__s)
	cbz	x22, .L886	// GREEN.116_1576,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x22	//, GREEN.116_1576
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, x0	//, tmp2514
	mov	x1, x22	//, GREEN.116_1576
	mov	x0, x28	//, tmp2355
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
.L414:
	adrp	x1, .LC88	// tmp1851,
	mov	x2, 15	//,
	add	x1, x1, :lo12:.LC88	//, tmp1851,
	mov	x0, x28	//, tmp2355
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:73:     std::cout << Color::GREEN << "=== Results ===" << Color::RESET << "\n";
	adrp	x0, _ZN5Color5RESETE	// tmp2860,
	ldr	x22, [x0, #:lo12:_ZN5Color5RESETE]	// RESET.117_1578, RESET
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:670:       if (!__s)
	cbz	x22, .L887	// RESET.117_1578,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x22	//, RESET.117_1578
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, x0	//, tmp2515
	mov	x1, x22	//, RESET.117_1578
	mov	x0, x28	//, tmp2355
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
.L416:
	ldr	x22, [sp, 168]	// tmp2327, %sfp
	mov	x2, 1	//,
	mov	x0, x28	//, tmp2355
	mov	x1, x22	//, tmp2327
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	adrp	x1, .LC89	// tmp1872,
	mov	x2, 18	//,
	add	x1, x1, :lo12:.LC89	//, tmp1872,
	mov	x0, x28	//, tmp2355
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	fmov	d0, d12	//, _1703
	mov	x0, x28	//, tmp2355
	bl	_ZNSo9_M_insertIdEERSoT_		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, 1	//,
	mov	x1, x22	//, tmp2327
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	adrp	x1, .LC90	// tmp1880,
	mov	x2, 18	//,
	add	x1, x1, :lo12:.LC90	//, tmp1880,
	mov	x0, x28	//, tmp2355
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	fmov	d0, d13	//, _1723
	mov	x0, x28	//, tmp2355
	bl	_ZNSo9_M_insertIdEERSoT_		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC91	// tmp1886,
	mov	x2, 4	//,
	add	x1, x1, :lo12:.LC91	//, tmp1886,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	mov	x22, x0	// _1581, tmp2516
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	fmul	d0, d13, d14	//, _1723, tmp2435
	mov	x0, x22	//, _1581
	bl	_ZNSo9_M_insertIdEERSoT_		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC92	// tmp1890,
	mov	x2, 6	//,
	add	x1, x1, :lo12:.LC92	//, tmp1890,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
.LEHE110:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	ldr	x0, [sp, 152]	//, %sfp
	mov	x1, 80	//,
	bl	_ZdlPvm		//
	ldr	x0, [sp, 144]	//, %sfp
	mov	x1, 80	//,
	bl	_ZdlPvm		//
// src/cpp/benchmark.cpp:300:         results[i] = benchmark_cnn(implementations[i].function, ctx, pmu, num_runs, num_warmup);
	mov	w2, 1	// tmp1891,
	ldr	x5, [sp, 200]	// ivtmp.795, %sfp
// src/cpp/benchmark.cpp:301:         std::cout << "\n\n";
	adrp	x1, .LC46	// tmp1897,
	mov	x0, x28	//, tmp2355
	add	x1, x1, :lo12:.LC46	//, tmp1897,
// src/cpp/benchmark.cpp:300:         results[i] = benchmark_cnn(implementations[i].function, ctx, pmu, num_runs, num_warmup);
	strb	w2, [sp, 256]	// tmp1891, MEM <unsigned char> [(struct bench_result_t *)&D.111033 + 16B]
	ldr	x4, [sp, 256]	// D.111033, D.111033
	str	x4, [x5, 16]	// D.111033, MEM[(struct value_type *)_5133]
	stp	d12, d13, [sp, 240]	// _1703, _1723,
	ldp	x2, x3, [sp, 240]	// D.111033, D.111033
	stp	x2, x3, [x5]	// D.111033, MEM[(struct value_type *)_5133]
.LEHB111:
// src/cpp/benchmark.cpp:301:         std::cout << "\n\n";
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.L391:
// src/cpp/benchmark.cpp:294:     for (int i = 0; i < num_impls; ++i) {
	ldr	x0, [sp, 176]	// ivtmp.794, %sfp
	ldr	x1, [sp, 200]	// ivtmp.795, %sfp
	add	x0, x0, 16	// ivtmp.794, ivtmp.794,
	add	x1, x1, 24	// ivtmp.795, ivtmp.795,
	str	x0, [sp, 176]	// ivtmp.794, %sfp
	str	x1, [sp, 200]	// ivtmp.795, %sfp
	cmp	x24, x0	// tmp2344, ivtmp.794
	bne	.L420		//,
// src/cpp/include/timing.h:153:     close(ctx->fd);
	ldr	w0, [sp, 224]	//, %sfp
	bl	close		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1150: 	return *(this->_M_impl._M_start + __n);
	ldr	x1, [sp, 1016]	// _829, MEM[(const struct vector *)&final_logits].D.57594._M_impl.D.56895._M_start
// src/cpp/include/utils.h:57:     int   best_class = 0;
	mov	w22, 0	// best_class,
// src/cpp/include/utils.h:59:     for (int i = 1; i < logits.width; ++i) {
	ldr	w2, [sp, 1052]	//, final_logits.width
// src/cpp/include/utils.h:58:     float max_score  = logits.data[0];
	ldr	s31, [x1]	//, MEM[(const value_type &)_829]
// src/cpp/include/utils.h:59:     for (int i = 1; i < logits.width; ++i) {
	cmp	w2, 1	// _1348,
	ble	.L421		//,
	mov	x0, 1	// ivtmp.739,
	.p2align 5,,15
.L424:
// src/cpp/include/utils.h:60:         if (logits.data[i] > max_score) {
	ldr	s30, [x1, x0, lsl 2]	//, MEM[(const value_type &)_829 + ivtmp.739_5317 * 4]
// src/cpp/include/utils.h:61:             max_score  = logits.data[i];
	fcmpe	s30, s31	//,
	csel	w22, w22, w0, ls	// best_class, best_class, ivtmp.739,
// src/cpp/include/utils.h:59:     for (int i = 1; i < logits.width; ++i) {
	add	x0, x0, 1	// ivtmp.739, ivtmp.739,
// src/cpp/include/utils.h:61:             max_score  = logits.data[i];
	fcsel	s31, s30, s31, gt	//,,,
// src/cpp/include/utils.h:59:     for (int i = 1; i < logits.width; ++i) {
	cmp	w2, w0	// _1348, ivtmp.739
	bgt	.L424		//,
.L421:
// src/cpp/benchmark.cpp:307:     std::cout << Color::BOLD_GREEN << "The network has successfully predicted the digit: "
	adrp	x1, _ZN5Color10BOLD_GREENE	// tmp1907,
	mov	x0, x28	//, tmp2355
	ldr	x1, [x1, #:lo12:_ZN5Color10BOLD_GREENE]	//, BOLD_GREEN
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC93	// tmp1911,
	mov	x2, 50	//,
	add	x1, x1, :lo12:.LC93	//, tmp1911,
// src/cpp/benchmark.cpp:307:     std::cout << Color::BOLD_GREEN << "The network has successfully predicted the digit: "
	mov	x21, x0	// _446, tmp2519
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:308:               << predicted_digit << Color::RESET << "\n\n";
	mov	w1, w22	//, best_class
	mov	x0, x21	//, _446
	bl	_ZNSolsEi		//
// src/cpp/benchmark.cpp:308:               << predicted_digit << Color::RESET << "\n\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2868,
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:308:               << predicted_digit << Color::RESET << "\n\n";
	adrp	x1, .LC46	// tmp2343,
	add	x1, x1, :lo12:.LC46	// tmp2338, tmp2343,
	str	x1, [sp, 136]	// tmp2338, %sfp
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:310:     if (verbose) {
	ldr	x0, [sp, 208]	// tmp2870, %sfp
// src/cpp/benchmark.cpp:311:         std::cout << Color::BOLD_YELLOW << "Raw Logits (Computational Verification):" << Color::RESET << "\n";
	adrp	x25, _ZN5Color11BOLD_YELLOWE	// tmp2378,
// src/cpp/benchmark.cpp:310:     if (verbose) {
	tbz	x0, 0, .L425	// tmp2870,,
// src/cpp/benchmark.cpp:311:         std::cout << Color::BOLD_YELLOW << "Raw Logits (Computational Verification):" << Color::RESET << "\n";
	ldr	x1, [x25, #:lo12:_ZN5Color11BOLD_YELLOWE]	//, BOLD_YELLOW
	mov	x0, x28	//, tmp2355
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC94	// tmp1922,
	mov	x2, 40	//,
	add	x1, x1, :lo12:.LC94	//, tmp1922,
// src/cpp/benchmark.cpp:311:         std::cout << Color::BOLD_YELLOW << "Raw Logits (Computational Verification):" << Color::RESET << "\n";
	mov	x21, x0	// _453, tmp2520
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:311:         std::cout << Color::BOLD_YELLOW << "Raw Logits (Computational Verification):" << Color::RESET << "\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2871,
	mov	x0, x21	//, _453
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:311:         std::cout << Color::BOLD_YELLOW << "Raw Logits (Computational Verification):" << Color::RESET << "\n";
	ldr	x1, [sp, 168]	//, %sfp
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
	adrp	x23, .LC95	// tmp2370,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x24, .LC96	// tmp2418,
// src/cpp/benchmark.cpp:311:         std::cout << Color::BOLD_YELLOW << "Raw Logits (Computational Verification):" << Color::RESET << "\n";
	mov	x21, 0	// ivtmp.731,
	add	x23, x23, :lo12:.LC95	// tmp2359, tmp2370,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	add	x24, x24, :lo12:.LC96	// tmp2419, tmp2418,
	.p2align 5,,15
.L426:
	mov	x2, 6	//,
	mov	x1, x23	//, tmp2359
	mov	x0, x28	//, tmp2355
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:313:             std::cout << "Class " << i << ": " << final_logits.data[i] << "\n";
	mov	w1, w21	//, ivtmp.731
	mov	x0, x28	//, tmp2355
	bl	_ZNSolsEi		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, 2	//,
	mov	x1, x24	//, tmp2419
// src/cpp/benchmark.cpp:313:             std::cout << "Class " << i << ": " << final_logits.data[i] << "\n";
	mov	x22, x0	// _459, tmp2521
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:313:             std::cout << "Class " << i << ": " << final_logits.data[i] << "\n";
	ldr	x1, [sp, 1016]	// final_logits.D.57594._M_impl.D.56895._M_start, final_logits.D.57594._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:236: 	return _M_insert(static_cast<double>(__f));
	mov	x0, x22	//, _459
	ldr	s0, [x1, w21, uxtw 2]	// *_846, *_846
	fcvt	d0, s0	//, *_846
	bl	_ZNSo9_M_insertIdEERSoT_		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	ldr	x1, [sp, 168]	//, %sfp
	mov	x2, 1	//,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:312:         for (int i = 0; i < num_classes; ++i)
	add	x21, x21, 1	// ivtmp.731, ivtmp.731,
	cmp	x21, 10	// ivtmp.731,
	bne	.L426		//,
// src/cpp/benchmark.cpp:314:         std::cout << "\n";
	ldr	x1, [sp, 168]	//, %sfp
	mov	x0, x28	//, tmp2355
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.L425:
// src/cpp/benchmark.cpp:320:     std::cout << Color::BOLD_YELLOW << "=== Final Performance Summary ===" << Color::RESET << "\n\n";
	ldr	x1, [x25, #:lo12:_ZN5Color11BOLD_YELLOWE]	//, BOLD_YELLOW
	mov	x0, x28	//, tmp2355
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC97	// tmp1951,
	mov	x2, 33	//,
	add	x1, x1, :lo12:.LC97	//, tmp1951,
// src/cpp/benchmark.cpp:320:     std::cout << Color::BOLD_YELLOW << "=== Final Performance Summary ===" << Color::RESET << "\n\n";
	mov	x21, x0	// _463, tmp2522
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:320:     std::cout << Color::BOLD_YELLOW << "=== Final Performance Summary ===" << Color::RESET << "\n\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2872,
	mov	x0, x21	//, _463
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:320:     std::cout << Color::BOLD_YELLOW << "=== Final Performance Summary ===" << Color::RESET << "\n\n";
	ldr	x1, [sp, 136]	//, %sfp
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:321:     printf("%-35s | %-12s | %-20s | %-14s\n",
	adrp	x4, .LC98	// tmp1957,
	adrp	x3, .LC99	// tmp1959,
	adrp	x2, .LC100	// tmp1961,
	adrp	x1, .LC101	// tmp1963,
	adrp	x0, .LC102	// tmp1965,
	add	x4, x4, :lo12:.LC98	//, tmp1957,
	add	x3, x3, :lo12:.LC99	//, tmp1959,
	add	x2, x2, :lo12:.LC100	//, tmp1961,
	add	x1, x1, :lo12:.LC101	//, tmp1963,
	add	x0, x0, :lo12:.LC102	//, tmp1965,
	bl	printf		//
// src/cpp/benchmark.cpp:323:     printf("---------------------------------------------------------------------------------------\n");
	adrp	x0, .LC103	// tmp1967,
	add	x0, x0, :lo12:.LC103	//, tmp1967,
	bl	puts		//
.LEHE111:
// src/cpp/benchmark.cpp:328:     double best_cycles = 0.0;
	movi	d14, #0	// best_cycles
// src/cpp/benchmark.cpp:323:     printf("---------------------------------------------------------------------------------------\n");
	ldr	x23, [sp, 232]	// ivtmp.727, %sfp
// src/cpp/benchmark.cpp:326:     int best_idx = -1;
	mov	w25, -1	// best_idx,
// src/cpp/benchmark.cpp:333:         if (std::strcmp(implementations[i].name, "Baseline Nested-Loop") == 0) {
	adrp	x27, .LC0	// tmp2415,
	add	x0, x27, :lo12:.LC0	// tmp2416, tmp2415,
// src/cpp/benchmark.cpp:330:     for (int i = 0; i < num_impls; ++i) {
	mov	w22, 0	// i,
// src/cpp/benchmark.cpp:325:     int baseline_idx = -1;
	mov	w26, w25	// baseline_idx, best_idx
// src/cpp/benchmark.cpp:323:     printf("---------------------------------------------------------------------------------------\n");
	add	x24, sp, 1056	// tmp2874,,
// src/cpp/benchmark.cpp:333:         if (std::strcmp(implementations[i].name, "Baseline Nested-Loop") == 0) {
	str	x0, [sp, 128]	// tmp2416, %sfp
// src/cpp/benchmark.cpp:327:     double baseline_cycles = 0.0;
	fmov	d15, d14	// baseline_cycles, best_cycles
	.p2align 5,,15
.L436:
// src/cpp/benchmark.cpp:331:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	ldr	x21, [x24]	// _114, MEM[(const char * *)_5245]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	str	x20, [sp, 1304]	// tmp2347, MEM[(struct _Alloc_hider *)_1195]._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:650: 	if (__s == 0)
	cbz	x21, .L888	// _114,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x21	//, _114
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:225: 	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
	str	x0, [sp, 336]	// _1750, MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x27, x0	// _1750, tmp2523
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:227: 	if (__dnew > size_type(_S_local_capacity))
	cmp	x0, 15	// _1750,
	bhi	.L889		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:432: 	if (__n == 1)
	cmp	x0, 1	// _1750,
	beq	.L890		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	cbnz	x0, .L891	// _1750,
.L432:
	mov	x2, x20	// pretmp_5557, tmp2347
.L431:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x27, [sp, 1312]	// _1750, MEM[(struct basic_string *)_1195]._M_string_length
// src/cpp/benchmark.cpp:331:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	mov	x0, x19	//, tmp2346
	ldr	x1, [sp, 160]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x2, x27]	//, MEM[(char_type &)_1760]
.LEHB112:
// src/cpp/benchmark.cpp:331:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	bl	_ZL29implementation_matches_filterRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKc		//
.LEHE112:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x3, [sp, 1304]	// _1761, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// src/cpp/benchmark.cpp:331:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	and	w27, w0, 255	// _513, tmp2525
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x3, x20	// _1761, tmp2347
	beq	.L433		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 1320]	// MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity, MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x0, x3	//, _1761
	add	x1, x1, 1	//, MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L433:
// src/cpp/benchmark.cpp:331:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	tbz	x27, 0, .L434	// _513,,
// src/cpp/benchmark.cpp:332:         if (!results[i].has_data) continue;
	ldrb	w0, [x23, 16]	// MEM[(bool *)_5243 + 16B], MEM[(bool *)_5243 + 16B]
	tbz	x0, 0, .L434	// MEM[(bool *)_5243 + 16B],,
// src/cpp/benchmark.cpp:333:         if (std::strcmp(implementations[i].name, "Baseline Nested-Loop") == 0) {
	ldr	x1, [sp, 128]	//, %sfp
	mov	x0, x21	//, _114
	bl	strcmp		//
// src/cpp/benchmark.cpp:335:             baseline_cycles = results[i].cycles_median;
	ldr	d31, [x23]	// pretmp_5665, MEM[(double *)_5243]
// src/cpp/benchmark.cpp:334:             baseline_idx = i;
	cmp	w0, 0	// tmp2526,
	csel	w26, w22, w26, eq	// baseline_idx, i, baseline_idx,
	fcsel	d15, d31, d15, eq	// baseline_cycles, pretmp_5665, baseline_cycles,
// src/cpp/benchmark.cpp:337:         if (best_idx < 0 || results[i].cycles_median < best_cycles) {
	tbnz	w25, #31, .L586	// best_idx,
// src/cpp/benchmark.cpp:337:         if (best_idx < 0 || results[i].cycles_median < best_cycles) {
	fcmpe	d14, d31	// best_cycles, pretmp_5665
	bls	.L434		//,
.L586:
// src/cpp/benchmark.cpp:339:             best_cycles = results[i].cycles_median;
	fmov	d14, d31	// best_cycles, pretmp_5665
// src/cpp/benchmark.cpp:338:             best_idx = i;
	mov	w25, w22	// best_idx, i
.L434:
// src/cpp/benchmark.cpp:330:     for (int i = 0; i < num_impls; ++i) {
	add	w22, w22, 1	// i, i,
// src/cpp/benchmark.cpp:330:     for (int i = 0; i < num_impls; ++i) {
	add	x24, x24, 16	// ivtmp.726, ivtmp.726,
	add	x23, x23, 24	// ivtmp.727, ivtmp.727,
	cmp	w22, 7	// i,
	bne	.L436		//,
// src/cpp/benchmark.cpp:343:     for (int i = 0; i < num_impls; ++i) {
	mov	w22, 0	// i,
	cmn	w26, #1	// baseline_idx,
	beq	.L892		//,
// src/cpp/benchmark.cpp:363:         printf("%-35s | %-12s | %-20s | %-14s",
	adrp	x24, .LC104	// tmp2410,
// src/cpp/benchmark.cpp:348:             snprintf(speedup_text, sizeof(speedup_text), "%.2fx",
	adrp	x26, .LC107	// tmp2412,
// src/cpp/benchmark.cpp:363:         printf("%-35s | %-12s | %-20s | %-14s",
	add	x24, x24, :lo12:.LC104	// tmp2411, tmp2410,
// src/cpp/benchmark.cpp:348:             snprintf(speedup_text, sizeof(speedup_text), "%.2fx",
	add	x26, x26, :lo12:.LC107	// tmp2413, tmp2412,
	.p2align 5,,15
.L437:
// src/cpp/benchmark.cpp:344:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	ldr	x0, [sp, 192]	// ivtmp.718, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	str	x20, [sp, 1304]	// tmp2347, MEM[(struct _Alloc_hider *)_1195]._M_p
// src/cpp/benchmark.cpp:344:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	ldr	x21, [x0]	// _119, MEM[(const char * *)_1534]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:650: 	if (__s == 0)
	cbz	x21, .L438	// _119,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x21	//, _119
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:225: 	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
	str	x0, [sp, 336]	// _1768, MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x23, x0	// _1768, tmp2530
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:227: 	if (__dnew > size_type(_S_local_capacity))
	cmp	x0, 15	// _1768,
	bhi	.L893		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:432: 	if (__n == 1)
	cmp	x0, 1	// _1768,
	beq	.L894		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	cbnz	x0, .L895	// _1768,
.L460:
	mov	x2, x20	// prephitmp_5562, tmp2347
.L459:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x23, [sp, 1312]	// _1768, MEM[(struct basic_string *)_1195]._M_string_length
// src/cpp/benchmark.cpp:344:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	mov	x0, x19	//, tmp2346
	ldr	x1, [sp, 160]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x2, x23]	//, MEM[(char_type &)_1778]
.LEHB113:
// src/cpp/benchmark.cpp:344:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	bl	_ZL29implementation_matches_filterRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKc		//
.LEHE113:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x2, [sp, 1304]	// _1779, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// src/cpp/benchmark.cpp:344:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	and	w23, w0, 255	// _484, tmp2532
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x2, x20	// _1779, tmp2347
	beq	.L461		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 1320]	// MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity, MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x0, x2	//, _1779
	add	x1, x1, 1	//, MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L461:
// src/cpp/benchmark.cpp:344:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	tbz	x23, 0, .L469	// _484,,
// src/cpp/benchmark.cpp:347:         if (baseline_idx >= 0 && results[i].has_data && results[i].cycles_median > 0.0) {
	ldr	x1, [sp, 184]	// ivtmp.719, %sfp
// src/cpp/benchmark.cpp:346:         char speedup_text[32] = "";
	add	x2, sp, 1136	// tmp2902,,
// src/cpp/benchmark.cpp:347:         if (baseline_idx >= 0 && results[i].has_data && results[i].cycles_median > 0.0) {
	ldrb	w0, [x1, 16]	// MEM[(bool *)_5092 + 16B], MEM[(bool *)_5092 + 16B]
// src/cpp/benchmark.cpp:346:         char speedup_text[32] = "";
	stp	xzr, xzr, [x2, 168]	// MEM[(char[32] *)_1195]
	stp	xzr, xzr, [x2, 184]	// MEM[(char[32] *)_1195]
// src/cpp/benchmark.cpp:347:         if (baseline_idx >= 0 && results[i].has_data && results[i].cycles_median > 0.0) {
	tbz	x0, 0, .L463	// MEM[(bool *)_5092 + 16B],,
// src/cpp/benchmark.cpp:347:         if (baseline_idx >= 0 && results[i].has_data && results[i].cycles_median > 0.0) {
	ldr	d0, [x1]	// _122, MEM[(double *)_5092]
// src/cpp/benchmark.cpp:347:         if (baseline_idx >= 0 && results[i].has_data && results[i].cycles_median > 0.0) {
	fcmpe	d0, #0.0	// _122
	bgt	.L464		//,
.L466:
// src/cpp/benchmark.cpp:354:             snprintf(cyc, sizeof(cyc), "%.2f", results[i].cycles_median);
	ldr	x27, [sp, 184]	// ivtmp.719, %sfp
	adrp	x2, .LC105	// tmp2068,
	add	x23, sp, 304	// tmp2364,,
	add	x2, x2, :lo12:.LC105	//, tmp2068,
	mov	x1, 24	//,
	mov	x0, x23	//, tmp2364
	ldr	d0, [x27]	//, MEM[(double *)_5092]
	bl	snprintf		//
// src/cpp/benchmark.cpp:355:             snprintf(sec, sizeof(sec), "%.6f", results[i].seconds_median);
	adrp	x2, .LC106	// tmp2071,
	ldr	d0, [x27, 8]	//, MEM[(double *)_5092 + 8B]
	add	x0, sp, 336	// tmp2908,,
	add	x2, x2, :lo12:.LC106	//, tmp2071,
	mov	x1, 24	//,
	bl	snprintf		//
.L465:
// src/cpp/benchmark.cpp:362:         if (is_best) std::cout << Color::GREEN;
	cmp	w25, w22	// best_idx, i
	beq	.L467		//,
// src/cpp/benchmark.cpp:363:         printf("%-35s | %-12s | %-20s | %-14s",
	mov	x3, x23	//, tmp2364
	mov	x1, x21	//, _119
	add	x4, sp, 336	// tmp2910,,
	mov	x2, x19	//, tmp2346
	mov	x0, x24	//, tmp2411
.LEHB114:
	bl	printf		//
.L468:
// src/cpp/benchmark.cpp:366:         std::cout << "\n";
	ldr	x1, [sp, 168]	//, %sfp
	mov	x0, x28	//, tmp2355
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.L469:
// src/cpp/benchmark.cpp:343:     for (int i = 0; i < num_impls; ++i) {
	ldr	x0, [sp, 192]	// ivtmp.718, %sfp
// src/cpp/benchmark.cpp:343:     for (int i = 0; i < num_impls; ++i) {
	add	w22, w22, 1	// i, i,
// src/cpp/benchmark.cpp:343:     for (int i = 0; i < num_impls; ++i) {
	add	x0, x0, 16	// ivtmp.718, ivtmp.718,
	str	x0, [sp, 192]	// ivtmp.718, %sfp
	ldr	x0, [sp, 184]	// ivtmp.719, %sfp
	add	x0, x0, 24	// ivtmp.719, ivtmp.719,
	str	x0, [sp, 184]	// ivtmp.719, %sfp
	cmp	w22, 7	// i,
	bne	.L437		//,
.L454:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC108	// tmp2106,
	mov	x2, 17	//,
	add	x1, x1, :lo12:.LC108	//, tmp2106,
	mov	x0, x28	//, tmp2355
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:369:     std::cout << "\nBenchmark sink: " << benchmark_global_sink << "\n\n";
	adrp	x1, .LANCHOR1	// tmp2110,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	mov	x0, x28	//, tmp2355
// src/cpp/benchmark.cpp:369:     std::cout << "\nBenchmark sink: " << benchmark_global_sink << "\n\n";
	ldr	d0, [x1, #:lo12:.LANCHOR1]	//, benchmark_global_sink
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	bl	_ZNSo9_M_insertIdEERSoT_		//
// src/cpp/benchmark.cpp:369:     std::cout << "\nBenchmark sink: " << benchmark_global_sink << "\n\n";
	ldr	x1, [sp, 136]	//, %sfp
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:373:     if (best_idx >= 0) {
	tbz	w25, #31, .L470	// best_idx,
.L472:
// src/cpp/benchmark.cpp:384:     return 0;
	mov	w19, 0	// <retval>,
.L382:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	ldr	x0, [sp, 232]	//, %sfp
	mov	x1, 168	//,
	bl	_ZdlPvm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:738:       }
	b	.L371		//
.L891:
	mov	x0, x20	// _1309, tmp2347
.L429:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	mov	x2, x27	//, _1750
	mov	x1, x21	//, _114
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:251: 	_M_set_length(__dnew);
	ldr	x27, [sp, 336]	// _1750, MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x2, [sp, 1304]	// pretmp_5557, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	b	.L431		//
.L464:
// src/cpp/benchmark.cpp:348:             snprintf(speedup_text, sizeof(speedup_text), "%.2fx",
	fdiv	d0, d15, d0	//, baseline_cycles, _122
	mov	x2, x26	//, tmp2413
	mov	x1, 32	//,
	mov	x0, x19	//, tmp2346
	bl	snprintf		//
// src/cpp/benchmark.cpp:353:         if (results[i].has_data) {
	ldr	x0, [sp, 184]	// ivtmp.719, %sfp
	ldrb	w0, [x0, 16]	// MEM[(bool *)_5092 + 16B], MEM[(bool *)_5092 + 16B]
	tbnz	x0, 0, .L466	// MEM[(bool *)_5092 + 16B],,
.L463:
// src/cpp/benchmark.cpp:357:             snprintf(cyc, sizeof(cyc), "-");
	mov	w0, 45	// tmp2080,
	add	x23, sp, 304	// tmp2364,,
	strh	w0, [sp, 304]	// tmp2080, MEM <char[1:2]> [(void *)_807]
// src/cpp/benchmark.cpp:358:             snprintf(sec, sizeof(sec), "-");
	strh	w0, [sp, 336]	// tmp2080, MEM <char[1:2]> [(void *)_2140]
	b	.L465		//
	.p2align 2,,3
.L581:
// src/cpp/include/utils.h:129:     double s = 0.0;
	movi	d30, #0	// s
	b	.L405		//
	.p2align 2,,3
.L577:
	movi	d30, #0	// s
	b	.L396		//
.L875:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldrb	w0, [x26]	// _1493, MEM[(const char_type &)_95]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	w0, [sp, 1320]	// _1493, MEM[(char_type &)_1195 + 16]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:351:       }
	b	.L377		//
	.p2align 2,,3
.L874:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	mov	x2, 0	//,
	add	x1, sp, 336	// tmp2810,,
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	ldr	x1, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:218:       { _M_dataplus._M_p = __p; }
	str	x0, [sp, 1304]	// _1490, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	str	x1, [sp, 1320]	// MEM[(long unsigned int *)_2140], MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity
	b	.L374		//
	.p2align 2,,3
.L578:
// src/cpp/include/utils.h:129:     double s = 0.0;
	movi	d30, #0	// s
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	mov	x2, 0	// niters_vector_mult_vf.604,
	b	.L397		//
	.p2align 2,,3
.L582:
// src/cpp/include/utils.h:129:     double s = 0.0;
	movi	d30, #0	// s
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	mov	x5, 0	// niters_vector_mult_vf.581,
	b	.L406		//
.L890:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldrb	w0, [x21]	// _1757, MEM[(const char_type &)_114]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	w0, [sp, 1320]	// _1757, MEM[(char_type &)_1195 + 16]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:351:       }
	b	.L432		//
.L881:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldrb	w0, [x22]	// _1511, MEM[(const char_type &)_101]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	w0, [sp, 1320]	// _1511, MEM[(char_type &)_1195 + 16]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:351:       }
	b	.L389		//
	.p2align 2,,3
.L880:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	mov	x2, 0	//,
	add	x1, sp, 336	//,,
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	ldr	x1, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:218:       { _M_dataplus._M_p = __p; }
	str	x0, [sp, 1304]	// _1508, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	str	x1, [sp, 1320]	// MEM[(long unsigned int *)_2140], MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity
	b	.L386		//
	.p2align 2,,3
.L889:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	mov	x2, 0	//,
	add	x1, sp, 336	// tmp2875,,
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	ldr	x1, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:218:       { _M_dataplus._M_p = __p; }
	str	x0, [sp, 1304]	// _1309, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	str	x1, [sp, 1320]	// MEM[(long unsigned int *)_2140], MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity
	b	.L429		//
.L894:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldrb	w0, [x21]	// _1775, MEM[(const char_type &)_119]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	w0, [sp, 1320]	// _1775, MEM[(char_type &)_1195 + 16]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:351:       }
	b	.L460		//
.L893:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	mov	x2, 0	//,
	add	x1, sp, 336	// tmp2895,,
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm		//
.LEHE114:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	ldr	x1, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:218:       { _M_dataplus._M_p = __p; }
	str	x0, [sp, 1304]	// _1423, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	str	x1, [sp, 1320]	// MEM[(long unsigned int *)_2140], MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity
.L457:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	mov	x2, x23	//, _1768
	mov	x1, x21	//, _119
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:251: 	_M_set_length(__dnew);
	ldr	x23, [sp, 336]	// _1768, MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x2, [sp, 1304]	// prephitmp_5562, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	b	.L459		//
.L573:
// src/cpp/include/utils.h:129:     double s = 0.0;
	movi	d19, #0	//
	b	.L350		//
.L872:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC66	// tmp1506,
	mov	x2, 9	//,
	add	x1, x1, :lo12:.LC66	//, tmp1506,
	mov	x0, x28	//, tmp2355
.LEHB115:
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	fmov	d0, d15	//, total
	mov	x0, x28	//, tmp2355
	bl	_ZNSo9_M_insertIdEERSoT_		//
// src/cpp/benchmark.cpp:263:         std::cout << "  total: " << total << " cycles\n\n";
	adrp	x1, .LC67	// tmp1512,
	add	x1, x1, :lo12:.LC67	//, tmp1512,
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/include/timing.h:153:     close(ctx->fd);
	ldr	w0, [sp, 224]	//, %sfp
	bl	close		//
.LEHE115:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	ldr	x0, [sp, 128]	// medians$_M_impl$D100808$_M_end_of_storage, %sfp
	sub	x1, x0, x25	//, medians$_M_impl$D100808$_M_end_of_storage, medians$D101473$_M_impl$D100808$_M_start
	mov	x0, x25	//, medians$D101473$_M_impl$D100808$_M_start
	bl	_ZdlPvm		//
	.p2align 5,,15
.L370:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [x20, -24]	// _817, MEM[(double * *)_608 + 8B]
// src/cpp/benchmark.cpp:208:         };
	sub	x20, x20, #32	// _608, _608,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L367	// _817,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [x20, 24]	// MEM[(double * *)_608 + 24B], MEM[(double * *)_608 + 24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(double * *)_608 + 24B], _817
	bl	_ZdlPvm		//
// src/cpp/benchmark.cpp:267:     }
	cmp	x20, x19	// _608, tmp2346
	bne	.L370		//,
.L319:
// src/cpp/benchmark.cpp:189:         return 0;
	mov	w19, 0	// <retval>,
.L371:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 1016]	// _923, MEM[(struct _Vector_base *)&final_logits]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L473	// _923,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 1032]	// MEM[(struct _Vector_base *)&final_logits]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&final_logits]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&final_logits]._M_impl.D.56895._M_end_of_storage, _923
	bl	_ZdlPvm		//
.L473:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 976]	// _919, MEM[(struct _Vector_base *)&avgpool_out]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L474	// _919,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 992]	// MEM[(struct _Vector_base *)&avgpool_out]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&avgpool_out]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&avgpool_out]._M_impl.D.56895._M_end_of_storage, _919
	bl	_ZdlPvm		//
.L474:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 936]	// _915, MEM[(struct _Vector_base *)&conv3_out]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L475	// _915,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 952]	// MEM[(struct _Vector_base *)&conv3_out]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv3_out]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&conv3_out]._M_impl.D.56895._M_end_of_storage, _915
	bl	_ZdlPvm		//
.L475:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 896]	// _911, MEM[(struct _Vector_base *)&pool2_out]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L476	// _911,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 912]	// MEM[(struct _Vector_base *)&pool2_out]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&pool2_out]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&pool2_out]._M_impl.D.56895._M_end_of_storage, _911
	bl	_ZdlPvm		//
.L476:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 856]	// _907, MEM[(struct _Vector_base *)&conv2_out]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L477	// _907,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 872]	// MEM[(struct _Vector_base *)&conv2_out]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv2_out]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&conv2_out]._M_impl.D.56895._M_end_of_storage, _907
	bl	_ZdlPvm		//
.L477:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 816]	// _903, MEM[(struct _Vector_base *)&pool1_out]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L478	// _903,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 832]	// MEM[(struct _Vector_base *)&pool1_out]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&pool1_out]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&pool1_out]._M_impl.D.56895._M_end_of_storage, _903
	bl	_ZdlPvm		//
.L478:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 776]	// _899, MEM[(struct _Vector_base *)&conv1_out]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L479	// _899,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 792]	// MEM[(struct _Vector_base *)&conv1_out]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv1_out]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&conv1_out]._M_impl.D.56895._M_end_of_storage, _899
	bl	_ZdlPvm		//
.L479:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 736]	// _895, MEM[(struct _Vector_base *)&fc_bias]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L480	// _895,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 752]	// MEM[(struct _Vector_base *)&fc_bias]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&fc_bias]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&fc_bias]._M_impl.D.56895._M_end_of_storage, _895
	bl	_ZdlPvm		//
.L480:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 696]	// _891, MEM[(struct _Vector_base *)&fc_weight]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L481	// _891,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 712]	// MEM[(struct _Vector_base *)&fc_weight]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&fc_weight]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&fc_weight]._M_impl.D.56895._M_end_of_storage, _891
	bl	_ZdlPvm		//
.L481:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 656]	// _887, MEM[(struct _Vector_base *)&conv3_bias]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L482	// _887,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 672]	// MEM[(struct _Vector_base *)&conv3_bias]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv3_bias]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&conv3_bias]._M_impl.D.56895._M_end_of_storage, _887
	bl	_ZdlPvm		//
.L482:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 608]	// _883, MEM[(struct _Vector_base *)&conv3_weight]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L483	// _883,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 624]	// MEM[(struct _Vector_base *)&conv3_weight]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv3_weight]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&conv3_weight]._M_impl.D.56895._M_end_of_storage, _883
	bl	_ZdlPvm		//
.L483:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 560]	// _879, MEM[(struct _Vector_base *)&conv2_bias]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L484	// _879,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 576]	// MEM[(struct _Vector_base *)&conv2_bias]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv2_bias]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&conv2_bias]._M_impl.D.56895._M_end_of_storage, _879
	bl	_ZdlPvm		//
.L484:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 512]	// _875, MEM[(struct _Vector_base *)&conv2_weight]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L485	// _875,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 528]	// MEM[(struct _Vector_base *)&conv2_weight]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv2_weight]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&conv2_weight]._M_impl.D.56895._M_end_of_storage, _875
	bl	_ZdlPvm		//
.L485:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 464]	// _871, MEM[(struct _Vector_base *)&conv1_bias]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L486	// _871,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 480]	// MEM[(struct _Vector_base *)&conv1_bias]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv1_bias]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&conv1_bias]._M_impl.D.56895._M_end_of_storage, _871
	bl	_ZdlPvm		//
.L486:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 416]	// _867, MEM[(struct _Vector_base *)&conv1_weight]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L487	// _867,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 432]	// MEM[(struct _Vector_base *)&conv1_weight]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv1_weight]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&conv1_weight]._M_impl.D.56895._M_end_of_storage, _867
	bl	_ZdlPvm		//
.L487:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	add	x0, sp, 272	// tmp2917,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
.L493:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 368]	// _927, MEM[(struct _Vector_base *)&input_batch]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L296	// _927,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 384]	// MEM[(struct _Vector_base *)&input_batch]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&input_batch]._M_impl.D.56895._M_end_of_storage
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	sub	x1, x1, x0	//, MEM[(struct _Vector_base *)&input_batch]._M_impl.D.56895._M_end_of_storage, _927
	bl	_ZdlPvm		//
.L296:
// src/cpp/benchmark.cpp:385: }
	ldp	d12, d13, [sp, 96]	//,,
	mov	w0, w19	//, <retval>
	ldp	x29, x30, [sp]	//,,
	ldp	x19, x20, [sp, 16]	//,,
	ldp	x21, x22, [sp, 32]	//,,
	ldp	x23, x24, [sp, 48]	//,,
	ldp	x25, x26, [sp, 64]	//,,
	ldp	x27, x28, [sp, 80]	//,,
	ldp	d14, d15, [sp, 112]	//,,
	add	sp, sp, 1824	//,,
	.cfi_remember_state
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
	.cfi_restore 29
	.cfi_restore 30
	.cfi_restore 78
	.cfi_restore 79
	.cfi_restore 76
	.cfi_restore 77
	.cfi_def_cfa_offset 0
	ret	
	.p2align 2,,3
.L367:
	.cfi_restore_state
// src/cpp/benchmark.cpp:267:     }
	cmp	x20, x19	// _608, tmp2346
	bne	.L370		//,
// src/cpp/benchmark.cpp:208:         };
	b	.L319		//
.L574:
// src/cpp/include/utils.h:129:     double s = 0.0;
	movi	d19, #0	//
// src/cpp/include/utils.h:130:     for (float v : t.data) s += (double)v;
	mov	x4, 0	// niters_vector_mult_vf.558,
	b	.L351		//
.L884:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:671: 	__out.setstate(ios_base::badbit);
	ldr	x0, [x22]	// _1523->_vptr.basic_ostream, _1523->_vptr.basic_ostream
	ldr	x0, [x0, -24]	// MEM[(long int *)_1608 + -24B], MEM[(long int *)_1608 + -24B]
	add	x0, x22, x0	// _1611, _1523, MEM[(long int *)_1608 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:187:   { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
	ldr	w1, [x0, 32]	//, MEM[(const struct basic_ios *)_1611].D.81517._M_streambuf_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	orr	w1, w1, 1	//, MEM[(const struct basic_ios *)_1611].D.81517._M_streambuf_state,
.LEHB116:
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
	b	.L395		//
.L883:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:671: 	__out.setstate(ios_base::badbit);
	ldr	x0, [x28]	// cout._vptr.basic_ostream, cout._vptr.basic_ostream
	ldr	x0, [x0, -24]	// MEM[(long int *)_1600 + -24B], MEM[(long int *)_1600 + -24B]
	add	x0, x28, x0	// _1603, tmp2355, MEM[(long int *)_1600 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:187:   { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
	ldr	w1, [x0, 32]	//, MEM[(const struct basic_ios *)_1603].D.81517._M_streambuf_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	orr	w1, w1, 1	//, MEM[(const struct basic_ios *)_1603].D.81517._M_streambuf_state,
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
.LEHE116:
	b	.L393		//
.L887:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:671: 	__out.setstate(ios_base::badbit);
	ldr	x0, [x28]	// cout._vptr.basic_ostream, cout._vptr.basic_ostream
	ldr	x0, [x0, -24]	// MEM[(long int *)_1735 + -24B], MEM[(long int *)_1735 + -24B]
	add	x0, x28, x0	// _1738, tmp2355, MEM[(long int *)_1735 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:187:   { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
	ldr	w1, [x0, 32]	//, MEM[(const struct basic_ios *)_1738].D.81517._M_streambuf_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	orr	w1, w1, 1	//, MEM[(const struct basic_ios *)_1738].D.81517._M_streambuf_state,
.LEHB117:
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
	b	.L416		//
.L886:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:671: 	__out.setstate(ios_base::badbit);
	ldr	x0, [x28]	// cout._vptr.basic_ostream, cout._vptr.basic_ostream
	ldr	x0, [x0, -24]	// MEM[(long int *)_1727 + -24B], MEM[(long int *)_1727 + -24B]
	add	x0, x28, x0	// _1730, tmp2355, MEM[(long int *)_1727 + -24B]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/ios_base.h:187:   { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
	ldr	w1, [x0, 32]	//, MEM[(const struct basic_ios *)_1730].D.81517._M_streambuf_state
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_ios.h:162:       { this->clear(this->rdstate() | __state); }
	orr	w1, w1, 1	//, MEM[(const struct basic_ios *)_1730].D.81517._M_streambuf_state,
	bl	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate		//
.LEHE117:
	b	.L414		//
.L868:
// src/cpp/include/timing.h:132:         fprintf(stderr, "perf_event_open failed: %s\n", strerror(errno));
	adrp	x0, stderr	// tmp1014,
	ldr	x22, [x0, #:lo12:stderr]	// stderr.68_1335, stderr
// src/cpp/include/timing.h:132:         fprintf(stderr, "perf_event_open failed: %s\n", strerror(errno));
	bl	__errno_location		//
// src/cpp/include/timing.h:132:         fprintf(stderr, "perf_event_open failed: %s\n", strerror(errno));
	ldr	w0, [x0]	//, *_1336
	bl	strerror		//
// src/cpp/include/timing.h:132:         fprintf(stderr, "perf_event_open failed: %s\n", strerror(errno));
	adrp	x1, .LC28	// tmp1016,
// src/cpp/include/timing.h:132:         fprintf(stderr, "perf_event_open failed: %s\n", strerror(errno));
	mov	x2, x0	// tmp2450,
// src/cpp/include/timing.h:132:         fprintf(stderr, "perf_event_open failed: %s\n", strerror(errno));
	add	x1, x1, :lo12:.LC28	//, tmp1016,
	mov	x0, x22	//, stderr.68_1335
.LEHB118:
	bl	fprintf		//
// src/cpp/benchmark.cpp:103:         std::cerr << Color::RED << "PMU init failed — cycle counts will be invalid.\n" << Color::RESET;
	adrp	x1, _ZN5Color3REDE	// tmp1018,
	adrp	x0, _ZSt4cerr	// tmp1020,
	ldr	x1, [x1, #:lo12:_ZN5Color3REDE]	//, RED
	add	x0, x0, :lo12:_ZSt4cerr	//, tmp1020,
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, 50	//,
// src/cpp/benchmark.cpp:103:         std::cerr << Color::RED << "PMU init failed — cycle counts will be invalid.\n" << Color::RESET;
	mov	x22, x0	// _233, tmp2451
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC29	// tmp1022,
	add	x1, x1, :lo12:.LC29	//, tmp1022,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:103:         std::cerr << Color::RED << "PMU init failed — cycle counts will be invalid.\n" << Color::RESET;
	adrp	x1, _ZN5Color5RESETE	// tmp2685,
	mov	x0, x22	//, _233
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.LEHE118:
	b	.L317		//
.L467:
// src/cpp/benchmark.cpp:362:         if (is_best) std::cout << Color::GREEN;
	adrp	x0, _ZN5Color5GREENE	// tmp2911,
	ldr	x1, [x0, #:lo12:_ZN5Color5GREENE]	//, GREEN
	mov	x0, x28	//, tmp2355
.LEHB119:
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:363:         printf("%-35s | %-12s | %-20s | %-14s",
	add	x4, sp, 336	// tmp2912,,
	mov	x3, x23	//, tmp2364
	mov	x2, x19	//, tmp2346
	mov	x1, x21	//, _119
	mov	x0, x24	//, tmp2411
	bl	printf		//
// src/cpp/benchmark.cpp:365:         if (is_best) std::cout << Color::RESET;
	adrp	x0, _ZN5Color5RESETE	// tmp2913,
	ldr	x1, [x0, #:lo12:_ZN5Color5RESETE]	//, RESET
	mov	x0, x28	//, tmp2355
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.LEHE119:
	b	.L468		//
.L869:
// src/cpp/benchmark.cpp:177:         std::string banner = std::string("  VERIFYING IMPLEMENTATION: ") + latest.name + "  ";
	adrp	x1, .LC44	// tmp1212,
	add	x0, sp, 336	// tmp2761,,
	add	x1, x1, :lo12:.LC44	//, tmp1212,
.LEHB120:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_.constprop.0		//
.LEHE120:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3735:     { return std::move(__lhs.append(__rhs)); }
	adrp	x21, .LC6	// tmp2373,
	add	x0, sp, 336	// tmp2762,,
	add	x21, x21, :lo12:.LC6	// tmp2365, tmp2373,
	mov	x1, x21	//, tmp2365
.LEHB121:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc		//
.LEHE121:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3735:     { return std::move(__lhs.append(__rhs)); }
	mov	x1, x0	//, tmp2455
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3735:     { return std::move(__lhs.append(__rhs)); }
	adrp	x1, .LC45	// tmp1219,
	mov	x0, x19	//, tmp2346
	add	x1, x1, :lo12:.LC45	//, tmp1219,
.LEHB122:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc		//
.LEHE122:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:3735:     { return std::move(__lhs.append(__rhs)); }
	add	x23, sp, 304	// tmp2364,,
	mov	x1, x0	//, tmp2456
	mov	x0, x23	//, tmp2364
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	add	x0, sp, 336	// tmp2763,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	add	x3, sp, 352	// tmp1225,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:671:       { _M_construct(__n, __c); }
	ldr	x1, [sp, 312]	//, MEM[(struct basic_string *)_807]._M_string_length
	add	x0, sp, 336	// tmp2764,,
	mov	w2, 61	//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	str	x3, [sp, 336]	// tmp1225, MEM[(struct _Alloc_hider *)_2140]._M_p
.LEHB123:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:671:       { _M_construct(__n, __c); }
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructEmc		//
.LEHE123:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	ldr	x24, [sp, 168]	// tmp2327, %sfp
	mov	x2, 1	//,
	mov	x0, x28	//, tmp2355
	mov	x1, x24	//, tmp2327
.LEHB124:
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:180:                   << border << Color::RESET << "\n\n";
	adrp	x1, _ZN5Color11BOLD_YELLOWE	// tmp1232,
	mov	x0, x28	//, tmp2355
	ldr	x1, [x1, #:lo12:_ZN5Color11BOLD_YELLOWE]	//, BOLD_YELLOW
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:4082:       return __ostream_insert(__os, __str.data(), __str.size());
	ldp	x1, x2, [sp, 336]	//,,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, 1	//,
	mov	x1, x24	//, tmp2327
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:4082:       return __ostream_insert(__os, __str.data(), __str.size());
	mov	x22, x0	// _754, tmp2457
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:4082:       return __ostream_insert(__os, __str.data(), __str.size());
	ldp	x1, x2, [sp, 304]	//,,
	mov	x0, x22	//, _754
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	mov	x2, 1	//,
	mov	x1, x24	//, tmp2327
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:4082:       return __ostream_insert(__os, __str.data(), __str.size());
	mov	x22, x0	// _770, tmp2458
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:4082:       return __ostream_insert(__os, __str.data(), __str.size());
	ldp	x1, x2, [sp, 336]	//,,
	mov	x0, x22	//, _770
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:180:                   << border << Color::RESET << "\n\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2765,
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:180:                   << border << Color::RESET << "\n\n";
	adrp	x1, .LC46	// tmp1242,
	add	x1, x1, :lo12:.LC46	//, tmp1242,
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:182:         latest.function(ctx);
	add	x0, sp, 1168	//,,
	bl	_Z21specialized_maxpool2dR10CNNContext		//
// src/cpp/benchmark.cpp:183:         const std::string out_path = "../python/weights_cpp/cpp_logits.bin";
	adrp	x1, .LC47	// tmp1245,
	mov	x0, x19	//, tmp2346
	add	x1, x1, :lo12:.LC47	//, tmp1245,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_.constprop.0		//
.LEHE124:
// src/cpp/benchmark.cpp:184:         save_binary(out_path, final_logits.data);
	mov	x1, x20	//, tmp2339
	mov	x0, x19	//, tmp2346
.LEHB125:
	bl	_Z11save_binaryRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIfSaIfEE		//
// src/cpp/benchmark.cpp:185:         std::cout << Color::BOLD_GREEN << "Wrote logits for " << input_batch.batches
	adrp	x1, _ZN5Color10BOLD_GREENE	// tmp1250,
	mov	x0, x28	//, tmp2355
	ldr	x1, [x1, #:lo12:_ZN5Color10BOLD_GREENE]	//, BOLD_GREEN
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC48	// tmp1254,
	mov	x2, 17	//,
	add	x1, x1, :lo12:.LC48	//, tmp1254,
// src/cpp/benchmark.cpp:185:         std::cout << Color::BOLD_GREEN << "Wrote logits for " << input_batch.batches
	mov	x20, x0	// _720, tmp2459
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:187:                   << Color::RESET << "\n";
	ldr	w1, [sp, 392]	//, input_batch.batches
	mov	x0, x20	//, _720
	bl	_ZNSolsEi		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC49	// tmp1256,
	mov	x2, 25	//,
	add	x1, x1, :lo12:.LC49	//, tmp1256,
// src/cpp/benchmark.cpp:187:                   << Color::RESET << "\n";
	mov	x20, x0	// _722, tmp2460
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:187:                   << Color::RESET << "\n";
	mov	x1, x21	//, tmp2365
	mov	x0, x20	//, _722
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC50	// tmp1260,
	mov	x2, 5	//,
	add	x1, x1, :lo12:.LC50	//, tmp1260,
// src/cpp/benchmark.cpp:187:                   << Color::RESET << "\n";
	mov	x20, x0	// _724, tmp2461
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:4082:       return __ostream_insert(__os, __str.data(), __str.size());
	ldr	x1, [sp, 1304]	//, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
	mov	x0, x20	//, _724
	ldr	x2, [sp, 1312]	//, MEM[(struct basic_string *)_1195]._M_string_length
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:187:                   << Color::RESET << "\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2766,
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:187:                   << Color::RESET << "\n";
	mov	x1, x24	//, tmp2327
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/include/timing.h:153:     close(ctx->fd);
	ldr	w0, [sp, 224]	//, %sfp
	bl	close		//
.LEHE125:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	add	x0, sp, 336	// tmp2767,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	mov	x0, x23	//, tmp2364
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	b	.L319		//
.L895:
	mov	x0, x20	// _1423, tmp2347
	b	.L457		//
.L470:
// src/cpp/benchmark.cpp:374:         std::ofstream timing_file("../python/weights_cpp/cpp_timing.json");
	adrp	x1, .LC109	// tmp2116,
	mov	w2, 16	//,
	add	x1, x1, :lo12:.LC109	//, tmp2116,
	mov	x0, x19	//, tmp2346
.LEHB126:
	bl	_ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode		//
.LEHE126:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/fstream:276:       { return _M_file.is_open(); }
	add	x0, sp, 1424	//,,
	bl	_ZNKSt12__basic_fileIcE7is_openEv		//
// src/cpp/benchmark.cpp:375:         if (timing_file.is_open()) {
	tbz	x0, 0, .L471	// tmp2533,,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC110	// tmp2123,
	mov	x2, 2	//,
	add	x1, x1, :lo12:.LC110	//, tmp2123,
	mov	x0, x19	//, tmp2346
.LEHB127:
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	adrp	x1, .LC111	// tmp2126,
	mov	x2, 21	//,
	add	x1, x1, :lo12:.LC111	//, tmp2126,
	mov	x0, x19	//, tmp2346
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:377:                         << "  \"implementation\": \"" << implementations[best_idx].name << "\",\n"
	sbfiz	x1, x25, 4, 32	// tmp2129, best_idx,,
// src/cpp/benchmark.cpp:378:                         << "  \"median_time_sec\": " << results[best_idx].seconds_median << ",\n"
	add	x2, sp, 1056	// tmp2914,,
	ldr	x1, [x2, x1]	//, implementations[best_idx_159].name
	mov	x0, x19	//, tmp2346
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC112	// tmp2133,
	mov	x2, 3	//,
	add	x1, x1, :lo12:.LC112	//, tmp2133,
// src/cpp/benchmark.cpp:378:                         << "  \"median_time_sec\": " << results[best_idx].seconds_median << ",\n"
	mov	x20, x0	// _473, tmp2534
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	adrp	x1, .LC113	// tmp2135,
	mov	x2, 21	//,
	add	x1, x1, :lo12:.LC113	//, tmp2135,
	mov	x0, x20	//, _473
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	ldr	x21, [sp, 232]	// _1475, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	mov	w1, 24	// tmp2138,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	mov	x0, x20	//, _473
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	umull	x25, w25, w1	// _860, best_idx, tmp2138
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:1131: 	return *(this->_M_impl._M_start + __n);
	add	x1, x21, x25	// _861, _1475, _860
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	ldr	d0, [x1, 8]	//, _861->seconds_median
	bl	_ZNSo9_M_insertIdEERSoT_		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC114	// tmp2140,
	mov	x2, 2	//,
	add	x1, x1, :lo12:.LC114	//, tmp2140,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	mov	x20, x0	// _858, tmp2535
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
	adrp	x1, .LC115	// tmp2142,
	mov	x2, 19	//,
	add	x1, x1, :lo12:.LC115	//, tmp2142,
	mov	x0, x20	//, _858
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	ldr	d0, [x21, x25]	//, _861->cycles_median
	mov	x0, x20	//, _858
	bl	_ZNSo9_M_insertIdEERSoT_		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	ldr	x1, [sp, 168]	//, %sfp
	mov	x2, 1	//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:229:       { return _M_insert(__f); }
	mov	x20, x0	// _862, tmp2536
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:380:                         << "}\n";
	adrp	x1, .LC116	// tmp2146,
	mov	x0, x20	//, _862
	add	x1, x1, :lo12:.LC116	//, tmp2146,
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.LEHE127:
.L471:
// src/cpp/benchmark.cpp:382:     }
	mov	x0, x19	//, tmp2346
	bl	_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev		//
	b	.L472		//
.L878:
// src/cpp/benchmark.cpp:284:                   << " (" << num_selected << " of " << num_impls << " implementations)"
	adrp	x0, _ZN5Color9BOLD_CYANE	// tmp2815,
	ldr	x1, [x0, #:lo12:_ZN5Color9BOLD_CYANE]	//, BOLD_CYAN
	mov	x0, x28	//, tmp2355
.LEHB128:
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC70	// tmp1590,
	mov	x2, 18	//,
	add	x1, x1, :lo12:.LC70	//, tmp1590,
// src/cpp/benchmark.cpp:284:                   << " (" << num_selected << " of " << num_impls << " implementations)"
	mov	x22, x0	// _427, tmp2494
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:284:                   << " (" << num_selected << " of " << num_impls << " implementations)"
	ldr	x1, [sp, 160]	//, %sfp
	mov	x0, x22	//, _427
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC71	// tmp1592,
	mov	x2, 2	//,
	add	x1, x1, :lo12:.LC71	//, tmp1592,
// src/cpp/benchmark.cpp:284:                   << " (" << num_selected << " of " << num_impls << " implementations)"
	mov	x22, x0	// _429, tmp2495
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:284:                   << " (" << num_selected << " of " << num_impls << " implementations)"
	mov	w1, w23	//, num_selected
	mov	x0, x22	//, _429
	bl	_ZNSolsEi		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC72	// tmp1594,
	mov	x2, 4	//,
	add	x1, x1, :lo12:.LC72	//, tmp1594,
// src/cpp/benchmark.cpp:284:                   << " (" << num_selected << " of " << num_impls << " implementations)"
	mov	x22, x0	// _431, tmp2496
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:285:                   << Color::RESET << "\n\n";
	mov	x0, x22	//, _431
	mov	w1, 7	//,
	bl	_ZNSolsEi		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x1, .LC73	// tmp1596,
	mov	x2, 17	//,
	add	x1, x1, :lo12:.LC73	//, tmp1596,
// src/cpp/benchmark.cpp:285:                   << Color::RESET << "\n\n";
	mov	x22, x0	// _433, tmp2497
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:285:                   << Color::RESET << "\n\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2816,
	mov	x0, x22	//, _433
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:285:                   << Color::RESET << "\n\n";
	adrp	x1, .LC46	// tmp1600,
	add	x1, x1, :lo12:.LC46	//, tmp1600,
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.LEHE128:
	b	.L383		//
.L892:
// src/cpp/benchmark.cpp:354:             snprintf(cyc, sizeof(cyc), "%.2f", results[i].cycles_median);
	adrp	x24, .LC105	// tmp2405,
// src/cpp/benchmark.cpp:355:             snprintf(sec, sizeof(sec), "%.6f", results[i].seconds_median);
	adrp	x26, .LC106	// tmp2407,
// src/cpp/benchmark.cpp:354:             snprintf(cyc, sizeof(cyc), "%.2f", results[i].cycles_median);
	add	x24, x24, :lo12:.LC105	// tmp2406, tmp2405,
// src/cpp/benchmark.cpp:355:             snprintf(sec, sizeof(sec), "%.6f", results[i].seconds_median);
	add	x26, x26, :lo12:.LC106	// tmp2408, tmp2407,
	.p2align 5,,15
.L453:
// src/cpp/benchmark.cpp:344:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	ldr	x0, [sp, 192]	// ivtmp.718, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:194: 	: allocator_type(__a), _M_p(__dat) { }
	str	x20, [sp, 1304]	// tmp2347, MEM[(struct _Alloc_hider *)_1195]._M_p
// src/cpp/benchmark.cpp:344:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	ldr	x21, [x0]	// _5433, MEM[(const char * *)_1901]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:650: 	if (__s == 0)
	cbz	x21, .L438	// _5433,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x0, x21	//, _5433
	bl	strlen		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:225: 	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
	str	x0, [sp, 336]	// prephitmp_5414, MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:391: 	return __builtin_strlen(__s);
	mov	x23, x0	// prephitmp_5414, tmp2527
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:227: 	if (__dnew > size_type(_S_local_capacity))
	cmp	x0, 15	// prephitmp_5414,
	bhi	.L896		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:432: 	if (__n == 1)
	cmp	x0, 1	// prephitmp_5414,
	beq	.L442		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	cbnz	x0, .L897	// prephitmp_5414,
.L443:
	mov	x0, x20	// prephitmp_5413, tmp2347
.L441:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:223:       { _M_string_length = __length; }
	str	x23, [sp, 1312]	// prephitmp_5414, MEM[(struct basic_string *)_1195]._M_string_length
// src/cpp/benchmark.cpp:344:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	ldr	x1, [sp, 160]	//, %sfp
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	wzr, [x0, x23]	//, MEM[(char_type &)_5411]
// src/cpp/benchmark.cpp:344:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	mov	x0, x19	//, tmp2346
.LEHB129:
	bl	_ZL29implementation_matches_filterRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKc		//
.LEHE129:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x2, [sp, 1304]	// _5406, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// src/cpp/benchmark.cpp:344:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	and	w23, w0, 255	// _5407, tmp2529
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:269: 	if (_M_data() == _M_local_data())
	cmp	x2, x20	// _5406, tmp2347
	beq	.L445		//,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:294:       { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
	ldr	x1, [sp, 1320]	// MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity, MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x0, x2	//, _5406
	add	x1, x1, 1	//, MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity,
	bl	_ZdlPvm		//
.L445:
// src/cpp/benchmark.cpp:344:         if (!implementation_matches_filter(implementations[i].name, bench_filter)) continue;
	tbz	x23, 0, .L446	// _5407,,
// src/cpp/benchmark.cpp:353:         if (results[i].has_data) {
	ldr	x0, [sp, 184]	// ivtmp.719, %sfp
// src/cpp/benchmark.cpp:346:         char speedup_text[32] = "";
	add	x1, sp, 1136	// tmp2880,,
// src/cpp/benchmark.cpp:353:         if (results[i].has_data) {
	ldrb	w0, [x0, 16]	// MEM[(bool *)_2011 + 16B], MEM[(bool *)_2011 + 16B]
// src/cpp/benchmark.cpp:346:         char speedup_text[32] = "";
	stp	xzr, xzr, [x1, 168]	// MEM[(char[32] *)_1195]
	stp	xzr, xzr, [x1, 184]	// MEM[(char[32] *)_1195]
// src/cpp/benchmark.cpp:353:         if (results[i].has_data) {
	tbz	x0, 0, .L898	// MEM[(bool *)_2011 + 16B],,
// src/cpp/benchmark.cpp:354:             snprintf(cyc, sizeof(cyc), "%.2f", results[i].cycles_median);
	add	x23, sp, 304	// tmp2364,,
	mov	x2, x24	//, tmp2406
	ldr	x27, [sp, 184]	// ivtmp.719, %sfp
	mov	x1, 24	//,
	mov	x0, x23	//, tmp2364
	ldr	d0, [x27]	//, MEM[(double *)_2011]
	bl	snprintf		//
// src/cpp/benchmark.cpp:355:             snprintf(sec, sizeof(sec), "%.6f", results[i].seconds_median);
	ldr	d0, [x27, 8]	//, MEM[(double *)_2011 + 8B]
	mov	x2, x26	//, tmp2408
	mov	x1, 24	//,
	add	x0, sp, 336	// tmp2889,,
	bl	snprintf		//
.L452:
// src/cpp/benchmark.cpp:362:         if (is_best) std::cout << Color::GREEN;
	cmp	w25, w22	// best_idx, i
	beq	.L899		//,
// src/cpp/benchmark.cpp:363:         printf("%-35s | %-12s | %-20s | %-14s",
	adrp	x0, .LC104	// tmp2034,
	mov	x3, x23	//, tmp2364
	mov	x1, x21	//, _5433
	add	x0, x0, :lo12:.LC104	//, tmp2034,
	add	x4, sp, 336	// tmp2886,,
	mov	x2, x19	//, tmp2346
.LEHB130:
	bl	printf		//
.L451:
// src/cpp/benchmark.cpp:366:         std::cout << "\n";
	ldr	x1, [sp, 168]	//, %sfp
	mov	x0, x28	//, tmp2355
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.L446:
// src/cpp/benchmark.cpp:343:     for (int i = 0; i < num_impls; ++i) {
	ldr	x0, [sp, 192]	// ivtmp.718, %sfp
// src/cpp/benchmark.cpp:343:     for (int i = 0; i < num_impls; ++i) {
	add	w22, w22, 1	// i, i,
// src/cpp/benchmark.cpp:343:     for (int i = 0; i < num_impls; ++i) {
	add	x0, x0, 16	// ivtmp.718, ivtmp.718,
	str	x0, [sp, 192]	// ivtmp.718, %sfp
	ldr	x0, [sp, 184]	// ivtmp.719, %sfp
	add	x0, x0, 24	// ivtmp.719, ivtmp.719,
	str	x0, [sp, 184]	// ivtmp.719, %sfp
	cmp	w22, 7	// i,
	bne	.L453		//,
	b	.L454		//
.L897:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:421: 	if (__n == 0)
	mov	x0, x20	// _5419, tmp2347
.L444:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:427: 	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
	mov	x2, x23	//, prephitmp_5414
	mov	x1, x21	//, _5433
	bl	memcpy		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:251: 	_M_set_length(__dnew);
	ldr	x23, [sp, 336]	// prephitmp_5414, MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:228:       { return _M_dataplus._M_p; }
	ldr	x0, [sp, 1304]	// prephitmp_5413, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
	b	.L441		//
.L442:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	ldrb	w0, [x21]	// _5422, MEM[(const char_type &)_5433]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/char_traits.h:350: 	__c1 = __c2;
	strb	w0, [sp, 1320]	// _5422, MEM[(char_type &)_1195 + 16]
	b	.L443		//
.L898:
// src/cpp/benchmark.cpp:357:             snprintf(cyc, sizeof(cyc), "-");
	mov	w0, 45	// tmp2036,
	add	x23, sp, 304	// tmp2364,,
	strh	w0, [sp, 304]	// tmp2036, MEM <char[1:2]> [(void *)_807]
// src/cpp/benchmark.cpp:358:             snprintf(sec, sizeof(sec), "-");
	strh	w0, [sp, 336]	// tmp2036, MEM <char[1:2]> [(void *)_2140]
	b	.L452		//
.L896:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.tcc:229: 	    _M_data(_M_create(__dnew, size_type(0)));
	mov	x2, 0	//,
	add	x1, sp, 336	// tmp2877,,
	mov	x0, x19	//, tmp2346
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	ldr	x1, [sp, 336]	// MEM[(long unsigned int *)_2140], MEM[(long unsigned int *)_2140]
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:218:       { _M_dataplus._M_p = __p; }
	str	x0, [sp, 1304]	// _5419, MEM[(struct basic_string *)_1195]._M_dataplus._M_p
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:255:       { _M_allocated_capacity = __capacity; }
	str	x1, [sp, 1320]	// MEM[(long unsigned int *)_2140], MEM[(struct basic_string *)_1195].D.64159._M_allocated_capacity
	b	.L444		//
.L899:
// src/cpp/benchmark.cpp:362:         if (is_best) std::cout << Color::GREEN;
	adrp	x0, _ZN5Color5GREENE	// tmp2883,
	ldr	x1, [x0, #:lo12:_ZN5Color5GREENE]	//, GREEN
	mov	x0, x28	//, tmp2355
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:363:         printf("%-35s | %-12s | %-20s | %-14s",
	adrp	x0, .LC104	// tmp2025,
	mov	x3, x23	//, tmp2364
	mov	x1, x21	//, _5433
	add	x0, x0, :lo12:.LC104	//, tmp2025,
	add	x4, sp, 336	// tmp2884,,
	mov	x2, x19	//, tmp2346
	bl	printf		//
// src/cpp/benchmark.cpp:365:         if (is_best) std::cout << Color::RESET;
	adrp	x0, _ZN5Color5RESETE	// tmp2885,
	ldr	x1, [x0, #:lo12:_ZN5Color5RESETE]	//, RESET
	mov	x0, x28	//, tmp2355
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
	b	.L451		//
.L877:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/ostream:673: 	__ostream_insert(__out, __s,
	adrp	x19, _ZSt4cerr	// tmp2324,
	adrp	x1, .LC69	// tmp1577,
	add	x19, x19, :lo12:_ZSt4cerr	// tmp2376, tmp2324,
	add	x1, x1, :lo12:.LC69	//, tmp1577,
	mov	x0, x19	//, tmp2376
	mov	x2, 52	//,
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l		//
// src/cpp/benchmark.cpp:278:                   << (bench_filter ? bench_filter : "") << "\n";
	ldr	x0, [sp, 160]	// bench_filter, %sfp
	cbz	x0, .L900	// bench_filter,
.L381:
// src/cpp/benchmark.cpp:278:                   << (bench_filter ? bench_filter : "") << "\n";
	ldr	x1, [sp, 160]	//, %sfp
	mov	x0, x19	//, tmp2376
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:278:                   << (bench_filter ? bench_filter : "") << "\n";
	ldr	x1, [sp, 168]	//, %sfp
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/include/timing.h:153:     close(ctx->fd);
	ldr	w0, [sp, 224]	//, %sfp
	bl	close		//
.LEHE130:
// src/cpp/benchmark.cpp:280:         return 1;
	mov	w19, 1	// <retval>,
	b	.L382		//
.L900:
// src/cpp/benchmark.cpp:278:                   << (bench_filter ? bench_filter : "") << "\n";
	adrp	x0, .LC23	// tmp936,
	add	x0, x0, :lo12:.LC23	// bench_filter, tmp936,
	str	x0, [sp, 160]	// bench_filter, %sfp
	b	.L381		//
.L625:
.L857:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x1, x0	// tmp2564,
	mov	x0, x19	//, tmp2346
	mov	x19, x1	// tmp2275, tmp2564
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	b	.L419		//
.L597:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	mov	x19, x0	// tmp2311, tmp2580
.L549:
	ldr	x0, [sp, 512]	// _995, MEM[(struct _Vector_base *)&conv2_weight]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 528]	// MEM[(struct _Vector_base *)&conv2_weight]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv2_weight]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _997, MEM[(struct _Vector_base *)&conv2_weight]._M_impl.D.56895._M_end_of_storage, _995
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L496	// _995,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L496:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 464]	// _999, MEM[(struct _Vector_base *)&conv1_bias]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 480]	// MEM[(struct _Vector_base *)&conv1_bias]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv1_bias]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _1001, MEM[(struct _Vector_base *)&conv1_bias]._M_impl.D.56895._M_end_of_storage, _999
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L552	// _999,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L552:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 416]	// _1003, MEM[(struct _Vector_base *)&conv1_weight]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 432]	// MEM[(struct _Vector_base *)&conv1_weight]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv1_weight]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _1005, MEM[(struct _Vector_base *)&conv1_weight]._M_impl.D.56895._M_end_of_storage, _1003
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbnz	x0, .L901	// _1003,
.L554:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	add	x0, sp, 272	// tmp2923,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
.L492:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 368]	// _1007, MEM[(struct _Vector_base *)&input_batch]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 384]	// MEM[(struct _Vector_base *)&input_batch]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&input_batch]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _1009, MEM[(struct _Vector_base *)&input_batch]._M_impl.D.56895._M_end_of_storage, _1007
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L555	// _1007,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L555:
	mov	x0, x19	//, tmp2187
.LEHB131:
	bl	_Unwind_Resume		//
.LEHE131:
.L630:
// src/cpp/benchmark.cpp:382:     }
	mov	x1, x0	// tmp2566,
	mov	x0, x19	//, tmp2346
	mov	x19, x1	// tmp2281, tmp2566
	bl	_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev		//
.L419:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	ldr	x0, [sp, 232]	//, %sfp
	mov	x1, 168	//,
	bl	_ZdlPvm		//
.L509:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 1016]	// _947, MEM[(struct _Vector_base *)&final_logits]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 1032]	// MEM[(struct _Vector_base *)&final_logits]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&final_logits]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _949, MEM[(struct _Vector_base *)&final_logits]._M_impl.D.56895._M_end_of_storage, _947
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L530	// _947,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L530:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 976]	// _951, MEM[(struct _Vector_base *)&avgpool_out]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 992]	// MEM[(struct _Vector_base *)&avgpool_out]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&avgpool_out]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _953, MEM[(struct _Vector_base *)&avgpool_out]._M_impl.D.56895._M_end_of_storage, _951
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L532	// _951,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L532:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 936]	// _955, MEM[(struct _Vector_base *)&conv3_out]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 952]	// MEM[(struct _Vector_base *)&conv3_out]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv3_out]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _957, MEM[(struct _Vector_base *)&conv3_out]._M_impl.D.56895._M_end_of_storage, _955
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L534	// _955,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L534:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 896]	// _959, MEM[(struct _Vector_base *)&pool2_out]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 912]	// MEM[(struct _Vector_base *)&pool2_out]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&pool2_out]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _961, MEM[(struct _Vector_base *)&pool2_out]._M_impl.D.56895._M_end_of_storage, _959
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L536	// _959,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L536:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 856]	// _963, MEM[(struct _Vector_base *)&conv2_out]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 872]	// MEM[(struct _Vector_base *)&conv2_out]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv2_out]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _965, MEM[(struct _Vector_base *)&conv2_out]._M_impl.D.56895._M_end_of_storage, _963
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L538	// _963,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L538:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 816]	// _967, MEM[(struct _Vector_base *)&pool1_out]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 832]	// MEM[(struct _Vector_base *)&pool1_out]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&pool1_out]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _969, MEM[(struct _Vector_base *)&pool1_out]._M_impl.D.56895._M_end_of_storage, _967
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L540	// _967,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L540:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 776]	// _971, MEM[(struct _Vector_base *)&conv1_out]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 792]	// MEM[(struct _Vector_base *)&conv1_out]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv1_out]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _973, MEM[(struct _Vector_base *)&conv1_out]._M_impl.D.56895._M_end_of_storage, _971
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L505	// _971,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L505:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 736]	// _975, MEM[(struct _Vector_base *)&fc_bias]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 752]	// MEM[(struct _Vector_base *)&fc_bias]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&fc_bias]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _977, MEM[(struct _Vector_base *)&fc_bias]._M_impl.D.56895._M_end_of_storage, _975
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L543	// _975,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L543:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 696]	// _979, MEM[(struct _Vector_base *)&fc_weight]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 712]	// MEM[(struct _Vector_base *)&fc_weight]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&fc_weight]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _981, MEM[(struct _Vector_base *)&fc_weight]._M_impl.D.56895._M_end_of_storage, _979
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L502	// _979,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L502:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 656]	// _983, MEM[(struct _Vector_base *)&conv3_bias]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 672]	// MEM[(struct _Vector_base *)&conv3_bias]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv3_bias]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _985, MEM[(struct _Vector_base *)&conv3_bias]._M_impl.D.56895._M_end_of_storage, _983
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L546	// _983,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L546:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 608]	// _987, MEM[(struct _Vector_base *)&conv3_weight]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 624]	// MEM[(struct _Vector_base *)&conv3_weight]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv3_weight]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _989, MEM[(struct _Vector_base *)&conv3_weight]._M_impl.D.56895._M_end_of_storage, _987
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L499	// _987,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L499:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [sp, 560]	// _991, MEM[(struct _Vector_base *)&conv2_bias]._M_impl.D.56895._M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 576]	// MEM[(struct _Vector_base *)&conv2_bias]._M_impl.D.56895._M_end_of_storage, MEM[(struct _Vector_base *)&conv2_bias]._M_impl.D.56895._M_end_of_storage
	sub	x1, x1, x0	// _993, MEM[(struct _Vector_base *)&conv2_bias]._M_impl.D.56895._M_end_of_storage, _991
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L549	// _991,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/alloc_traits.h:513:       { __a.deallocate(__p, __n); }
	b	.L549		//
.L901:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/alloc_traits.h:513:       { __a.deallocate(__p, __n); }
	b	.L554		//
.L592:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x19, x0	// tmp2318, tmp2583
	b	.L554		//
.L590:
.L862:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	mov	x19, x0	// tmp2187, tmp2584
	b	.L492		//
.L588:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x2, x0	// tmp2537,
	mov	x0, x19	//, tmp2346
	mov	x20, x1	// tmp2186, tmp2538
	mov	x19, x2	// tmp2184, tmp2537
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	mov	x0, x19	// tmp2183, tmp2184
	mov	x1, x20	// tmp2185, tmp2186
.L490:
// src/cpp/benchmark.cpp:113:     } catch (const std::exception& e) {
	cmp	x1, 1	// tmp2185,
	bne	.L862		//,
// src/cpp/benchmark.cpp:113:     } catch (const std::exception& e) {
	bl	__cxa_begin_catch		//
// src/cpp/benchmark.cpp:114:         std::cerr << Color::RED << e.what() << Color::RESET << "\n";
	adrp	x1, _ZN5Color3REDE	// tmp2190,
	adrp	x2, _ZSt4cerr	// tmp2192,
// src/cpp/benchmark.cpp:113:     } catch (const std::exception& e) {
	mov	x19, x0	// _776, tmp2541
// src/cpp/benchmark.cpp:114:         std::cerr << Color::RED << e.what() << Color::RESET << "\n";
	ldr	x1, [x1, #:lo12:_ZN5Color3REDE]	//, RED
	add	x0, x2, :lo12:_ZSt4cerr	//, tmp2192,
.LEHB132:
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:114:         std::cerr << Color::RED << e.what() << Color::RESET << "\n";
	ldr	x2, [x19]	// MEM[(const struct exception *)_776]._vptr.exception, MEM[(const struct exception *)_776]._vptr.exception
// src/cpp/benchmark.cpp:114:         std::cerr << Color::RED << e.what() << Color::RESET << "\n";
	mov	x1, x0	// tmp2542,
// src/cpp/benchmark.cpp:114:         std::cerr << Color::RED << e.what() << Color::RESET << "\n";
	mov	x0, x19	//, _776
// src/cpp/benchmark.cpp:114:         std::cerr << Color::RED << e.what() << Color::RESET << "\n";
	mov	x19, x1	// _778, tmp2542
// src/cpp/benchmark.cpp:114:         std::cerr << Color::RED << e.what() << Color::RESET << "\n";
	ldr	x1, [x2, 16]	// MEM[(int (*) () *)_13 + 16B], MEM[(int (*) () *)_13 + 16B]
	blr	x1		// MEM[(int (*) () *)_13 + 16B]
	mov	x1, x0	//, tmp2543
	mov	x0, x19	//, _778
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:114:         std::cerr << Color::RED << e.what() << Color::RESET << "\n";
	adrp	x1, _ZN5Color5RESETE	// tmp2918,
	ldr	x1, [x1, #:lo12:_ZN5Color5RESETE]	//, RESET
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
// src/cpp/benchmark.cpp:114:         std::cerr << Color::RED << e.what() << Color::RESET << "\n";
	ldr	x1, [sp, 168]	//, %sfp
	bl	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc		//
.LEHE132:
// src/cpp/benchmark.cpp:115:         return 1;
	mov	w19, 1	// <retval>,
// src/cpp/benchmark.cpp:116:     }
	bl	__cxa_end_catch		//
	b	.L493		//
.L613:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	mov	x19, x0	// tmp2287, tmp2570
	b	.L532		//
.L612:
	mov	x19, x0	// tmp2290, tmp2571
	b	.L534		//
.L610:
	mov	x19, x0	// tmp2296, tmp2573
	b	.L538		//
.L609:
	mov	x19, x0	// tmp2299, tmp2574
	b	.L540		//
.L611:
	mov	x19, x0	// tmp2293, tmp2572
	b	.L536		//
.L608:
.L858:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x1, x0	// tmp2552,
	mov	x0, x19	//, tmp2346
	mov	x19, x1	// tmp2231, tmp2552
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	b	.L505		//
.L600:
.L860:
	mov	x1, x0	// tmp2548,
	mov	x0, x19	//, tmp2346
	mov	x19, x1	// tmp2215, tmp2548
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	b	.L499		//
.L589:
	b	.L490		//
.L616:
	mov	x1, x0	// tmp2553,
	mov	x0, x19	//, tmp2346
	mov	x19, x1	// tmp2235, tmp2553
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
.L508:
	add	x0, sp, 336	// tmp2919,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	b	.L509		//
.L617:
	mov	x19, x0	// tmp2247, tmp2557
.L512:
	mov	x0, x23	//, tmp2364
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	b	.L509		//
.L618:
	mov	x19, x0	// tmp2242, tmp2556
.L511:
	add	x0, sp, 336	// tmp2920,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	b	.L512		//
.L627:
	b	.L857		//
.L599:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	mov	x19, x0	// tmp2210, tmp2579
	b	.L499		//
.L615:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x19, x0	// tmp2234, tmp2554
	b	.L508		//
.L598:
	b	.L860		//
.L607:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	mov	x19, x0	// tmp2226, tmp2575
	b	.L505		//
.L614:
	mov	x19, x0	// tmp2284, tmp2569
	b	.L530		//
.L606:
	b	.L858		//
.L591:
// src/cpp/benchmark.cpp:116:     }
	mov	x19, x0	// tmp2199, tmp2544
	bl	__cxa_end_catch		//
	b	.L492		//
.L603:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	mov	x19, x0	// tmp2218, tmp2577
	b	.L502		//
.L601:
	mov	x19, x0	// tmp2307, tmp2578
	b	.L546		//
.L602:
.L859:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x1, x0	// tmp2550,
	mov	x0, x19	//, tmp2346
	mov	x19, x1	// tmp2223, tmp2550
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	b	.L502		//
.L596:
.L861:
	mov	x1, x0	// tmp2546,
	mov	x0, x19	//, tmp2346
	mov	x19, x1	// tmp2207, tmp2546
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	b	.L496		//
.L594:
	b	.L861		//
.L595:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	mov	x19, x0	// tmp2202, tmp2581
	b	.L496		//
.L593:
	mov	x19, x0	// tmp2315, tmp2582
	b	.L552		//
.L624:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x21, x0	// tmp2559,
	add	x0, sp, 336	// tmp2921,,
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
.L518:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x0, [sp, 128]	// medians$_M_impl$D100808$_M_end_of_storage, %sfp
	sub	x1, x0, x25	// _937, medians$_M_impl$D100808$_M_end_of_storage, medians$D101473$_M_impl$D100808$_M_start
.L556:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x0, x25	//, medians$D101473$_M_impl$D100808$_M_start
	bl	_ZdlPvm		//
.L523:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [x20, -24]	// _939, MEM[(double * *)_694 + 8B]
// src/cpp/benchmark.cpp:208:         };
	sub	x20, x20, #32	// _694, _694,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [x20, 24]	// MEM[(double * *)_694 + 24B], MEM[(double * *)_694 + 24B]
	sub	x1, x1, x0	// _941, MEM[(double * *)_694 + 24B], _939
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L522	// _939,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
.L522:
// src/cpp/benchmark.cpp:267:     }
	cmp	x20, x19	// _694, tmp2346
	bne	.L523		//,
	mov	x19, x21	// tmp2239, tmp2264
	b	.L509		//
.L626:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x19, x0	// tmp1904, tmp2567
	b	.L419		//
.L879:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:651: 	  std::__throw_logic_error(__N("basic_string: "
	adrp	x0, .LC11	// tmp1616,
	add	x0, x0, :lo12:.LC11	//, tmp1616,
.LEHB133:
	bl	_ZSt19__throw_logic_errorPKc		//
.L873:
	adrp	x0, .LC11	// tmp1560,
	add	x0, x0, :lo12:.LC11	//, tmp1560,
	bl	_ZSt19__throw_logic_errorPKc		//
.L628:
	b	.L857		//
.L438:
	adrp	x0, .LC11	// tmp2048,
	add	x0, x0, :lo12:.LC11	//, tmp2048,
	bl	_ZSt19__throw_logic_errorPKc		//
.LEHE133:
.L619:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:809:       { _M_dispose(); }
	mov	x1, x0	// tmp2555,
	mov	x0, x19	//, tmp2346
	mov	x19, x1	// tmp2243, tmp2555
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv		//
	b	.L511		//
.L642:
// src/cpp/benchmark.cpp:208:         };
	mov	x1, 1	// _142,
	mov	x20, x0	// tmp2257, tmp2471
.L323:
	mov	x21, 9	// tmp2252,
	sub	w21, w21, w1	// _40, tmp2252, _142
	add	x21, x19, x21, lsl 5	// _143, tmp2346, _40,
.L516:
// src/cpp/benchmark.cpp:208:         };
	cmp	x21, x19	// _143, tmp2346
	bne	.L902		//,
	mov	x19, x20	// tmp2239, tmp2257
	b	.L509		//
.L621:
// src/cpp/benchmark.cpp:208:         };
	mov	x1, 0	// _142,
	mov	x20, x0	// tmp2257, tmp2472
	b	.L323		//
.L633:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [sp, 128]	// medians$_M_impl$D100808$_M_end_of_storage, %sfp
	mov	x21, x0	// tmp2260, tmp2560
	add	x20, sp, 1624	// _694,,
	sub	x1, x1, x25	// _937, medians$_M_impl$D100808$_M_end_of_storage, medians$D101473$_M_impl$D100808$_M_start
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x25, .L523	// medians$D101473$_M_impl$D100808$_M_start,
	b	.L556		//
.L623:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	mov	x21, x0	// tmp2260, tmp2585
	b	.L518		//
.L622:
	add	x20, sp, 1624	// _694,,
// src/cpp/benchmark.cpp:208:         };
	mov	x21, x0	// tmp2264, tmp2561
	b	.L523		//
.L631:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x19, x0	// tmp1900, tmp2518
	b	.L418		//
.L639:
// src/cpp/benchmark.cpp:208:         };
	mov	x1, 4	// _142,
	mov	x20, x0	// tmp2257, tmp2468
	b	.L323		//
.L635:
	mov	x1, 8	// _142,
	mov	x20, x0	// tmp2257, tmp2464
	b	.L323		//
.L636:
	mov	x1, 7	// _142,
	mov	x20, x0	// tmp2257, tmp2465
	b	.L323		//
.L637:
	mov	x1, 6	// _142,
	mov	x20, x0	// tmp2257, tmp2466
	b	.L323		//
.L638:
	mov	x1, 5	// _142,
	mov	x20, x0	// tmp2257, tmp2467
	b	.L323		//
.L632:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	mov	x19, x0	// tmp2517,
	ldr	x0, [sp, 152]	//, %sfp
	mov	x1, 80	//,
	bl	_ZdlPvm		//
.L418:
	ldr	x0, [sp, 144]	//, %sfp
	mov	x1, 80	//,
	bl	_ZdlPvm		//
	b	.L419		//
.L634:
// src/cpp/benchmark.cpp:208:         };
	mov	x1, 9	// _142,
	mov	x20, x0	// tmp2257, tmp2558
	b	.L323		//
.L863:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:651: 	  std::__throw_logic_error(__N("basic_string: "
	adrp	x0, .LC11	// tmp953,
	add	x0, x0, :lo12:.LC11	//, tmp953,
.LEHB134:
	bl	_ZSt19__throw_logic_errorPKc		//
.LEHE134:
.L604:
	b	.L859		//
.L620:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	mov	x19, x0	// tmp2239, tmp2568
	b	.L509		//
.L641:
// src/cpp/benchmark.cpp:208:         };
	mov	x1, 2	// _142,
	mov	x20, x0	// tmp2257, tmp2470
	b	.L323		//
.L640:
	mov	x1, 3	// _142,
	mov	x20, x0	// tmp2257, tmp2469
	b	.L323		//
.L605:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	mov	x19, x0	// tmp2303, tmp2576
	b	.L543		//
.L629:
	b	.L857		//
.L888:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/basic_string.h:651: 	  std::__throw_logic_error(__N("basic_string: "
	adrp	x0, .LC11	// tmp1971,
	add	x0, x0, :lo12:.LC11	//, tmp1971,
.LEHB135:
	bl	_ZSt19__throw_logic_errorPKc		//
.LEHE135:
.L902:
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:368: 	_M_deallocate(_M_impl._M_start,
	ldr	x0, [x21, -24]	// _931, MEM[(double * *)_697 + 8B]
// src/cpp/benchmark.cpp:208:         };
	sub	x21, x21, #32	// _143, _143,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:369: 		      _M_impl._M_end_of_storage - _M_impl._M_start);
	ldr	x1, [x21, 24]	// MEM[(double * *)_697 + 24B], MEM[(double * *)_697 + 24B]
	sub	x1, x1, x0	// _933, MEM[(double * *)_697 + 24B], _931
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/stl_vector.h:388: 	if (__p)
	cbz	x0, .L516	// _931,
// /user-environment/linux-neoverse_v2/gcc-14.2.0-ovez6ffqfi4256sigothvjlis6onlz4h/lib/gcc/aarch64-unknown-linux-gnu/14.2.0/../../../../include/c++/14.2.0/bits/new_allocator.h:172: 	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
	bl	_ZdlPvm		//
// src/cpp/benchmark.cpp:196:         struct LayerStat { const char *name; std::vector<double> cycles; };
	b	.L516		//
	.cfi_endproc
.LFE7375:
	.section	.gcc_except_table
	.align	2
.LLSDA7375:
	.byte	0xff
	.byte	0x9b
	.uleb128 .LLSDATT7375-.LLSDATTD7375
.LLSDATTD7375:
	.byte	0x1
	.uleb128 .LLSDACSE7375-.LLSDACSB7375
.LLSDACSB7375:
	.uleb128 .LEHB51-.LFB7375
	.uleb128 .LEHE51-.LEHB51
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB52-.LFB7375
	.uleb128 .LEHE52-.LEHB52
	.uleb128 .L589-.LFB7375
	.uleb128 0x3
	.uleb128 .LEHB53-.LFB7375
	.uleb128 .LEHE53-.LEHB53
	.uleb128 .L588-.LFB7375
	.uleb128 0x3
	.uleb128 .LEHB54-.LFB7375
	.uleb128 .LEHE54-.LEHB54
	.uleb128 .L589-.LFB7375
	.uleb128 0x3
	.uleb128 .LEHB55-.LFB7375
	.uleb128 .LEHE55-.LEHB55
	.uleb128 .L590-.LFB7375
	.uleb128 0
	.uleb128 .LEHB56-.LFB7375
	.uleb128 .LEHE56-.LEHB56
	.uleb128 .L592-.LFB7375
	.uleb128 0
	.uleb128 .LEHB57-.LFB7375
	.uleb128 .LEHE57-.LEHB57
	.uleb128 .L593-.LFB7375
	.uleb128 0
	.uleb128 .LEHB58-.LFB7375
	.uleb128 .LEHE58-.LEHB58
	.uleb128 .L595-.LFB7375
	.uleb128 0
	.uleb128 .LEHB59-.LFB7375
	.uleb128 .LEHE59-.LEHB59
	.uleb128 .L594-.LFB7375
	.uleb128 0
	.uleb128 .LEHB60-.LFB7375
	.uleb128 .LEHE60-.LEHB60
	.uleb128 .L595-.LFB7375
	.uleb128 0
	.uleb128 .LEHB61-.LFB7375
	.uleb128 .LEHE61-.LEHB61
	.uleb128 .L596-.LFB7375
	.uleb128 0
	.uleb128 .LEHB62-.LFB7375
	.uleb128 .LEHE62-.LEHB62
	.uleb128 .L595-.LFB7375
	.uleb128 0
	.uleb128 .LEHB63-.LFB7375
	.uleb128 .LEHE63-.LEHB63
	.uleb128 .L597-.LFB7375
	.uleb128 0
	.uleb128 .LEHB64-.LFB7375
	.uleb128 .LEHE64-.LEHB64
	.uleb128 .L599-.LFB7375
	.uleb128 0
	.uleb128 .LEHB65-.LFB7375
	.uleb128 .LEHE65-.LEHB65
	.uleb128 .L598-.LFB7375
	.uleb128 0
	.uleb128 .LEHB66-.LFB7375
	.uleb128 .LEHE66-.LEHB66
	.uleb128 .L599-.LFB7375
	.uleb128 0
	.uleb128 .LEHB67-.LFB7375
	.uleb128 .LEHE67-.LEHB67
	.uleb128 .L600-.LFB7375
	.uleb128 0
	.uleb128 .LEHB68-.LFB7375
	.uleb128 .LEHE68-.LEHB68
	.uleb128 .L599-.LFB7375
	.uleb128 0
	.uleb128 .LEHB69-.LFB7375
	.uleb128 .LEHE69-.LEHB69
	.uleb128 .L601-.LFB7375
	.uleb128 0
	.uleb128 .LEHB70-.LFB7375
	.uleb128 .LEHE70-.LEHB70
	.uleb128 .L603-.LFB7375
	.uleb128 0
	.uleb128 .LEHB71-.LFB7375
	.uleb128 .LEHE71-.LEHB71
	.uleb128 .L602-.LFB7375
	.uleb128 0
	.uleb128 .LEHB72-.LFB7375
	.uleb128 .LEHE72-.LEHB72
	.uleb128 .L603-.LFB7375
	.uleb128 0
	.uleb128 .LEHB73-.LFB7375
	.uleb128 .LEHE73-.LEHB73
	.uleb128 .L604-.LFB7375
	.uleb128 0
	.uleb128 .LEHB74-.LFB7375
	.uleb128 .LEHE74-.LEHB74
	.uleb128 .L603-.LFB7375
	.uleb128 0
	.uleb128 .LEHB75-.LFB7375
	.uleb128 .LEHE75-.LEHB75
	.uleb128 .L605-.LFB7375
	.uleb128 0
	.uleb128 .LEHB76-.LFB7375
	.uleb128 .LEHE76-.LEHB76
	.uleb128 .L607-.LFB7375
	.uleb128 0
	.uleb128 .LEHB77-.LFB7375
	.uleb128 .LEHE77-.LEHB77
	.uleb128 .L606-.LFB7375
	.uleb128 0
	.uleb128 .LEHB78-.LFB7375
	.uleb128 .LEHE78-.LEHB78
	.uleb128 .L607-.LFB7375
	.uleb128 0
	.uleb128 .LEHB79-.LFB7375
	.uleb128 .LEHE79-.LEHB79
	.uleb128 .L608-.LFB7375
	.uleb128 0
	.uleb128 .LEHB80-.LFB7375
	.uleb128 .LEHE80-.LEHB80
	.uleb128 .L607-.LFB7375
	.uleb128 0
	.uleb128 .LEHB81-.LFB7375
	.uleb128 .LEHE81-.LEHB81
	.uleb128 .L609-.LFB7375
	.uleb128 0
	.uleb128 .LEHB82-.LFB7375
	.uleb128 .LEHE82-.LEHB82
	.uleb128 .L610-.LFB7375
	.uleb128 0
	.uleb128 .LEHB83-.LFB7375
	.uleb128 .LEHE83-.LEHB83
	.uleb128 .L611-.LFB7375
	.uleb128 0
	.uleb128 .LEHB84-.LFB7375
	.uleb128 .LEHE84-.LEHB84
	.uleb128 .L612-.LFB7375
	.uleb128 0
	.uleb128 .LEHB85-.LFB7375
	.uleb128 .LEHE85-.LEHB85
	.uleb128 .L613-.LFB7375
	.uleb128 0
	.uleb128 .LEHB86-.LFB7375
	.uleb128 .LEHE86-.LEHB86
	.uleb128 .L614-.LFB7375
	.uleb128 0
	.uleb128 .LEHB87-.LFB7375
	.uleb128 .LEHE87-.LEHB87
	.uleb128 .L620-.LFB7375
	.uleb128 0
	.uleb128 .LEHB88-.LFB7375
	.uleb128 .LEHE88-.LEHB88
	.uleb128 .L634-.LFB7375
	.uleb128 0
	.uleb128 .LEHB89-.LFB7375
	.uleb128 .LEHE89-.LEHB89
	.uleb128 .L635-.LFB7375
	.uleb128 0
	.uleb128 .LEHB90-.LFB7375
	.uleb128 .LEHE90-.LEHB90
	.uleb128 .L636-.LFB7375
	.uleb128 0
	.uleb128 .LEHB91-.LFB7375
	.uleb128 .LEHE91-.LEHB91
	.uleb128 .L637-.LFB7375
	.uleb128 0
	.uleb128 .LEHB92-.LFB7375
	.uleb128 .LEHE92-.LEHB92
	.uleb128 .L638-.LFB7375
	.uleb128 0
	.uleb128 .LEHB93-.LFB7375
	.uleb128 .LEHE93-.LEHB93
	.uleb128 .L639-.LFB7375
	.uleb128 0
	.uleb128 .LEHB94-.LFB7375
	.uleb128 .LEHE94-.LEHB94
	.uleb128 .L640-.LFB7375
	.uleb128 0
	.uleb128 .LEHB95-.LFB7375
	.uleb128 .LEHE95-.LEHB95
	.uleb128 .L641-.LFB7375
	.uleb128 0
	.uleb128 .LEHB96-.LFB7375
	.uleb128 .LEHE96-.LEHB96
	.uleb128 .L642-.LFB7375
	.uleb128 0
	.uleb128 .LEHB97-.LFB7375
	.uleb128 .LEHE97-.LEHB97
	.uleb128 .L621-.LFB7375
	.uleb128 0
	.uleb128 .LEHB98-.LFB7375
	.uleb128 .LEHE98-.LEHB98
	.uleb128 .L622-.LFB7375
	.uleb128 0
	.uleb128 .LEHB99-.LFB7375
	.uleb128 .LEHE99-.LEHB99
	.uleb128 .L633-.LFB7375
	.uleb128 0
	.uleb128 .LEHB100-.LFB7375
	.uleb128 .LEHE100-.LEHB100
	.uleb128 .L623-.LFB7375
	.uleb128 0
	.uleb128 .LEHB101-.LFB7375
	.uleb128 .LEHE101-.LEHB101
	.uleb128 .L624-.LFB7375
	.uleb128 0
	.uleb128 .LEHB102-.LFB7375
	.uleb128 .LEHE102-.LEHB102
	.uleb128 .L623-.LFB7375
	.uleb128 0
	.uleb128 .LEHB103-.LFB7375
	.uleb128 .LEHE103-.LEHB103
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB104-.LFB7375
	.uleb128 .LEHE104-.LEHB104
	.uleb128 .L620-.LFB7375
	.uleb128 0
	.uleb128 .LEHB105-.LFB7375
	.uleb128 .LEHE105-.LEHB105
	.uleb128 .L625-.LFB7375
	.uleb128 0
	.uleb128 .LEHB106-.LFB7375
	.uleb128 .LEHE106-.LEHB106
	.uleb128 .L626-.LFB7375
	.uleb128 0
	.uleb128 .LEHB107-.LFB7375
	.uleb128 .LEHE107-.LEHB107
	.uleb128 .L627-.LFB7375
	.uleb128 0
	.uleb128 .LEHB108-.LFB7375
	.uleb128 .LEHE108-.LEHB108
	.uleb128 .L626-.LFB7375
	.uleb128 0
	.uleb128 .LEHB109-.LFB7375
	.uleb128 .LEHE109-.LEHB109
	.uleb128 .L631-.LFB7375
	.uleb128 0
	.uleb128 .LEHB110-.LFB7375
	.uleb128 .LEHE110-.LEHB110
	.uleb128 .L632-.LFB7375
	.uleb128 0
	.uleb128 .LEHB111-.LFB7375
	.uleb128 .LEHE111-.LEHB111
	.uleb128 .L626-.LFB7375
	.uleb128 0
	.uleb128 .LEHB112-.LFB7375
	.uleb128 .LEHE112-.LEHB112
	.uleb128 .L628-.LFB7375
	.uleb128 0
	.uleb128 .LEHB113-.LFB7375
	.uleb128 .LEHE113-.LEHB113
	.uleb128 .L629-.LFB7375
	.uleb128 0
	.uleb128 .LEHB114-.LFB7375
	.uleb128 .LEHE114-.LEHB114
	.uleb128 .L626-.LFB7375
	.uleb128 0
	.uleb128 .LEHB115-.LFB7375
	.uleb128 .LEHE115-.LEHB115
	.uleb128 .L623-.LFB7375
	.uleb128 0
	.uleb128 .LEHB116-.LFB7375
	.uleb128 .LEHE116-.LEHB116
	.uleb128 .L626-.LFB7375
	.uleb128 0
	.uleb128 .LEHB117-.LFB7375
	.uleb128 .LEHE117-.LEHB117
	.uleb128 .L632-.LFB7375
	.uleb128 0
	.uleb128 .LEHB118-.LFB7375
	.uleb128 .LEHE118-.LEHB118
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB119-.LFB7375
	.uleb128 .LEHE119-.LEHB119
	.uleb128 .L626-.LFB7375
	.uleb128 0
	.uleb128 .LEHB120-.LFB7375
	.uleb128 .LEHE120-.LEHB120
	.uleb128 .L620-.LFB7375
	.uleb128 0
	.uleb128 .LEHB121-.LFB7375
	.uleb128 .LEHE121-.LEHB121
	.uleb128 .L615-.LFB7375
	.uleb128 0
	.uleb128 .LEHB122-.LFB7375
	.uleb128 .LEHE122-.LEHB122
	.uleb128 .L616-.LFB7375
	.uleb128 0
	.uleb128 .LEHB123-.LFB7375
	.uleb128 .LEHE123-.LEHB123
	.uleb128 .L617-.LFB7375
	.uleb128 0
	.uleb128 .LEHB124-.LFB7375
	.uleb128 .LEHE124-.LEHB124
	.uleb128 .L618-.LFB7375
	.uleb128 0
	.uleb128 .LEHB125-.LFB7375
	.uleb128 .LEHE125-.LEHB125
	.uleb128 .L619-.LFB7375
	.uleb128 0
	.uleb128 .LEHB126-.LFB7375
	.uleb128 .LEHE126-.LEHB126
	.uleb128 .L626-.LFB7375
	.uleb128 0
	.uleb128 .LEHB127-.LFB7375
	.uleb128 .LEHE127-.LEHB127
	.uleb128 .L630-.LFB7375
	.uleb128 0
	.uleb128 .LEHB128-.LFB7375
	.uleb128 .LEHE128-.LEHB128
	.uleb128 .L626-.LFB7375
	.uleb128 0
	.uleb128 .LEHB129-.LFB7375
	.uleb128 .LEHE129-.LEHB129
	.uleb128 .L629-.LFB7375
	.uleb128 0
	.uleb128 .LEHB130-.LFB7375
	.uleb128 .LEHE130-.LEHB130
	.uleb128 .L626-.LFB7375
	.uleb128 0
	.uleb128 .LEHB131-.LFB7375
	.uleb128 .LEHE131-.LEHB131
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB132-.LFB7375
	.uleb128 .LEHE132-.LEHB132
	.uleb128 .L591-.LFB7375
	.uleb128 0
	.uleb128 .LEHB133-.LFB7375
	.uleb128 .LEHE133-.LEHB133
	.uleb128 .L626-.LFB7375
	.uleb128 0
	.uleb128 .LEHB134-.LFB7375
	.uleb128 .LEHE134-.LEHB134
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB135-.LFB7375
	.uleb128 .LEHE135-.LEHB135
	.uleb128 .L626-.LFB7375
	.uleb128 0
.LLSDACSE7375:
	.byte	0
	.byte	0
	.byte	0x1
	.byte	0x7d
	.align	2
	.4byte	DW.ref._ZTISt9exception-.
.LLSDATT7375:
	.section	.text.startup
	.size	main, .-main
	.section	.rodata.str1.8
	.align	3
.LC118:
	.string	"\033[0m"
	.align	3
.LC119:
	.string	"\033[1m"
	.align	3
.LC120:
	.string	"\033[2m"
	.align	3
.LC121:
	.string	"\033[31m"
	.align	3
.LC122:
	.string	"\033[32m"
	.align	3
.LC123:
	.string	"\033[33m"
	.align	3
.LC124:
	.string	"\033[36m"
	.align	3
.LC125:
	.string	"\033[1;32m"
	.align	3
.LC126:
	.string	"\033[1;36m"
	.align	3
.LC127:
	.string	"\033[1;33m"
	.align	3
.LC128:
	.string	"\033[2;32m"
	.section	.text.startup
	.align	2
	.p2align 5,,15
	.type	_GLOBAL__sub_I_main, %function
_GLOBAL__sub_I_main:
.LFB8375:
	.cfi_startproc
// src/cpp/include/utils.h:20:     inline const bool _tty       = (isatty(STDOUT_FILENO) != 0);
	adrp	x1, _ZGVN5Color4_ttyE	// tmp171,
// src/cpp/benchmark.cpp:385: }
	stp	x29, x30, [sp, -16]!	//,,,
	.cfi_def_cfa_offset 16
	.cfi_offset 29, -16
	.cfi_offset 30, -8
	mov	x29, sp	//,
// src/cpp/include/utils.h:20:     inline const bool _tty       = (isatty(STDOUT_FILENO) != 0);
	ldr	x0, [x1, #:lo12:_ZGVN5Color4_ttyE]	// _ZGVN5Color4_ttyE, _ZGVN5Color4_ttyE
	tbz	x0, 0, .L939	// _ZGVN5Color4_ttyE,,
.L904:
// src/cpp/include/utils.h:21:     inline const char* const RESET       = _tty ? "\033[0m"    : "";
	adrp	x1, _ZGVN5Color5RESETE	// tmp182,
	ldr	x0, [x1, #:lo12:_ZGVN5Color5RESETE]	// _ZGVN5Color5RESETE, _ZGVN5Color5RESETE
	tbnz	x0, 0, .L905	// _ZGVN5Color5RESETE,,
// src/cpp/include/utils.h:21:     inline const char* const RESET       = _tty ? "\033[0m"    : "";
	adrp	x0, _ZN5Color4_ttyE	// tmp189,
// src/cpp/include/utils.h:21:     inline const char* const RESET       = _tty ? "\033[0m"    : "";
	adrp	x3, _ZN5Color5RESETE	// tmp193,
// src/cpp/include/utils.h:21:     inline const char* const RESET       = _tty ? "\033[0m"    : "";
	ldrb	w4, [x0, #:lo12:_ZN5Color4_ttyE]	// _tty, _tty
// src/cpp/include/utils.h:21:     inline const char* const RESET       = _tty ? "\033[0m"    : "";
	mov	x5, 1	// tmp187,
// src/cpp/include/utils.h:21:     inline const char* const RESET       = _tty ? "\033[0m"    : "";
	adrp	x2, .LC118	// tmp148,
	adrp	x0, .LC23	// tmp149,
	add	x2, x2, :lo12:.LC118	// tmp324, tmp148,
	add	x0, x0, :lo12:.LC23	// tmp325, tmp149,
// src/cpp/include/utils.h:21:     inline const char* const RESET       = _tty ? "\033[0m"    : "";
	str	x5, [x1, #:lo12:_ZGVN5Color5RESETE]	// tmp187, _ZGVN5Color5RESETE
// src/cpp/include/utils.h:21:     inline const char* const RESET       = _tty ? "\033[0m"    : "";
	tst	x4, 1	// _tty,
	csel	x0, x0, x2, eq	// iftmp.135_10, tmp325, tmp324,
// src/cpp/include/utils.h:21:     inline const char* const RESET       = _tty ? "\033[0m"    : "";
	str	x0, [x3, #:lo12:_ZN5Color5RESETE]	// iftmp.135_10, RESET
.L905:
// src/cpp/include/utils.h:22:     inline const char* const BOLD        = _tty ? "\033[1m"    : "";
	adrp	x1, _ZGVN5Color4BOLDE	// tmp195,
	ldr	x0, [x1, #:lo12:_ZGVN5Color4BOLDE]	// _ZGVN5Color4BOLDE, _ZGVN5Color4BOLDE
	tbnz	x0, 0, .L907	// _ZGVN5Color4BOLDE,,
// src/cpp/include/utils.h:22:     inline const char* const BOLD        = _tty ? "\033[1m"    : "";
	adrp	x0, _ZN5Color4_ttyE	// tmp202,
// src/cpp/include/utils.h:22:     inline const char* const BOLD        = _tty ? "\033[1m"    : "";
	adrp	x3, _ZN5Color4BOLDE	// tmp206,
// src/cpp/include/utils.h:22:     inline const char* const BOLD        = _tty ? "\033[1m"    : "";
	ldrb	w4, [x0, #:lo12:_ZN5Color4_ttyE]	// _tty, _tty
// src/cpp/include/utils.h:22:     inline const char* const BOLD        = _tty ? "\033[1m"    : "";
	mov	x5, 1	// tmp200,
// src/cpp/include/utils.h:22:     inline const char* const BOLD        = _tty ? "\033[1m"    : "";
	adrp	x2, .LC119	// tmp150,
	adrp	x0, .LC23	// tmp151,
	add	x2, x2, :lo12:.LC119	// tmp326, tmp150,
	add	x0, x0, :lo12:.LC23	// tmp327, tmp151,
// src/cpp/include/utils.h:22:     inline const char* const BOLD        = _tty ? "\033[1m"    : "";
	str	x5, [x1, #:lo12:_ZGVN5Color4BOLDE]	// tmp200, _ZGVN5Color4BOLDE
// src/cpp/include/utils.h:22:     inline const char* const BOLD        = _tty ? "\033[1m"    : "";
	tst	x4, 1	// _tty,
	csel	x0, x0, x2, eq	// iftmp.138_14, tmp327, tmp326,
// src/cpp/include/utils.h:22:     inline const char* const BOLD        = _tty ? "\033[1m"    : "";
	str	x0, [x3, #:lo12:_ZN5Color4BOLDE]	// iftmp.138_14, BOLD
.L907:
// src/cpp/include/utils.h:23:     inline const char* const DIM         = _tty ? "\033[2m"    : "";
	adrp	x1, _ZGVN5Color3DIME	// tmp208,
	ldr	x0, [x1, #:lo12:_ZGVN5Color3DIME]	// _ZGVN5Color3DIME, _ZGVN5Color3DIME
	tbnz	x0, 0, .L909	// _ZGVN5Color3DIME,,
// src/cpp/include/utils.h:23:     inline const char* const DIM         = _tty ? "\033[2m"    : "";
	adrp	x0, _ZN5Color4_ttyE	// tmp215,
// src/cpp/include/utils.h:23:     inline const char* const DIM         = _tty ? "\033[2m"    : "";
	adrp	x3, _ZN5Color3DIME	// tmp219,
// src/cpp/include/utils.h:23:     inline const char* const DIM         = _tty ? "\033[2m"    : "";
	ldrb	w4, [x0, #:lo12:_ZN5Color4_ttyE]	// _tty, _tty
// src/cpp/include/utils.h:23:     inline const char* const DIM         = _tty ? "\033[2m"    : "";
	mov	x5, 1	// tmp213,
// src/cpp/include/utils.h:23:     inline const char* const DIM         = _tty ? "\033[2m"    : "";
	adrp	x2, .LC120	// tmp152,
	adrp	x0, .LC23	// tmp153,
	add	x2, x2, :lo12:.LC120	// tmp328, tmp152,
	add	x0, x0, :lo12:.LC23	// tmp329, tmp153,
// src/cpp/include/utils.h:23:     inline const char* const DIM         = _tty ? "\033[2m"    : "";
	str	x5, [x1, #:lo12:_ZGVN5Color3DIME]	// tmp213, _ZGVN5Color3DIME
// src/cpp/include/utils.h:23:     inline const char* const DIM         = _tty ? "\033[2m"    : "";
	tst	x4, 1	// _tty,
	csel	x0, x0, x2, eq	// iftmp.141_18, tmp329, tmp328,
// src/cpp/include/utils.h:23:     inline const char* const DIM         = _tty ? "\033[2m"    : "";
	str	x0, [x3, #:lo12:_ZN5Color3DIME]	// iftmp.141_18, DIM
.L909:
// src/cpp/include/utils.h:24:     inline const char* const RED         = _tty ? "\033[31m"   : "";
	adrp	x1, _ZGVN5Color3REDE	// tmp221,
	ldr	x0, [x1, #:lo12:_ZGVN5Color3REDE]	// _ZGVN5Color3REDE, _ZGVN5Color3REDE
	tbnz	x0, 0, .L911	// _ZGVN5Color3REDE,,
// src/cpp/include/utils.h:24:     inline const char* const RED         = _tty ? "\033[31m"   : "";
	adrp	x0, _ZN5Color4_ttyE	// tmp228,
// src/cpp/include/utils.h:24:     inline const char* const RED         = _tty ? "\033[31m"   : "";
	adrp	x3, _ZN5Color3REDE	// tmp232,
// src/cpp/include/utils.h:24:     inline const char* const RED         = _tty ? "\033[31m"   : "";
	ldrb	w4, [x0, #:lo12:_ZN5Color4_ttyE]	// _tty, _tty
// src/cpp/include/utils.h:24:     inline const char* const RED         = _tty ? "\033[31m"   : "";
	mov	x5, 1	// tmp226,
// src/cpp/include/utils.h:24:     inline const char* const RED         = _tty ? "\033[31m"   : "";
	adrp	x2, .LC121	// tmp154,
	adrp	x0, .LC23	// tmp155,
	add	x2, x2, :lo12:.LC121	// tmp330, tmp154,
	add	x0, x0, :lo12:.LC23	// tmp331, tmp155,
// src/cpp/include/utils.h:24:     inline const char* const RED         = _tty ? "\033[31m"   : "";
	str	x5, [x1, #:lo12:_ZGVN5Color3REDE]	// tmp226, _ZGVN5Color3REDE
// src/cpp/include/utils.h:24:     inline const char* const RED         = _tty ? "\033[31m"   : "";
	tst	x4, 1	// _tty,
	csel	x0, x0, x2, eq	// iftmp.144_22, tmp331, tmp330,
// src/cpp/include/utils.h:24:     inline const char* const RED         = _tty ? "\033[31m"   : "";
	str	x0, [x3, #:lo12:_ZN5Color3REDE]	// iftmp.144_22, RED
.L911:
// src/cpp/include/utils.h:25:     inline const char* const GREEN       = _tty ? "\033[32m"   : "";
	adrp	x1, _ZGVN5Color5GREENE	// tmp234,
	ldr	x0, [x1, #:lo12:_ZGVN5Color5GREENE]	// _ZGVN5Color5GREENE, _ZGVN5Color5GREENE
	tbnz	x0, 0, .L913	// _ZGVN5Color5GREENE,,
// src/cpp/include/utils.h:25:     inline const char* const GREEN       = _tty ? "\033[32m"   : "";
	adrp	x0, _ZN5Color4_ttyE	// tmp241,
// src/cpp/include/utils.h:25:     inline const char* const GREEN       = _tty ? "\033[32m"   : "";
	adrp	x3, _ZN5Color5GREENE	// tmp245,
// src/cpp/include/utils.h:25:     inline const char* const GREEN       = _tty ? "\033[32m"   : "";
	ldrb	w4, [x0, #:lo12:_ZN5Color4_ttyE]	// _tty, _tty
// src/cpp/include/utils.h:25:     inline const char* const GREEN       = _tty ? "\033[32m"   : "";
	mov	x5, 1	// tmp239,
// src/cpp/include/utils.h:25:     inline const char* const GREEN       = _tty ? "\033[32m"   : "";
	adrp	x2, .LC122	// tmp156,
	adrp	x0, .LC23	// tmp157,
	add	x2, x2, :lo12:.LC122	// tmp332, tmp156,
	add	x0, x0, :lo12:.LC23	// tmp333, tmp157,
// src/cpp/include/utils.h:25:     inline const char* const GREEN       = _tty ? "\033[32m"   : "";
	str	x5, [x1, #:lo12:_ZGVN5Color5GREENE]	// tmp239, _ZGVN5Color5GREENE
// src/cpp/include/utils.h:25:     inline const char* const GREEN       = _tty ? "\033[32m"   : "";
	tst	x4, 1	// _tty,
	csel	x0, x0, x2, eq	// iftmp.147_26, tmp333, tmp332,
// src/cpp/include/utils.h:25:     inline const char* const GREEN       = _tty ? "\033[32m"   : "";
	str	x0, [x3, #:lo12:_ZN5Color5GREENE]	// iftmp.147_26, GREEN
.L913:
// src/cpp/include/utils.h:26:     inline const char* const YELLOW      = _tty ? "\033[33m"   : "";
	adrp	x1, _ZGVN5Color6YELLOWE	// tmp247,
	ldr	x0, [x1, #:lo12:_ZGVN5Color6YELLOWE]	// _ZGVN5Color6YELLOWE, _ZGVN5Color6YELLOWE
	tbnz	x0, 0, .L915	// _ZGVN5Color6YELLOWE,,
// src/cpp/include/utils.h:26:     inline const char* const YELLOW      = _tty ? "\033[33m"   : "";
	adrp	x0, _ZN5Color4_ttyE	// tmp254,
// src/cpp/include/utils.h:26:     inline const char* const YELLOW      = _tty ? "\033[33m"   : "";
	adrp	x3, _ZN5Color6YELLOWE	// tmp258,
// src/cpp/include/utils.h:26:     inline const char* const YELLOW      = _tty ? "\033[33m"   : "";
	ldrb	w4, [x0, #:lo12:_ZN5Color4_ttyE]	// _tty, _tty
// src/cpp/include/utils.h:26:     inline const char* const YELLOW      = _tty ? "\033[33m"   : "";
	mov	x5, 1	// tmp252,
// src/cpp/include/utils.h:26:     inline const char* const YELLOW      = _tty ? "\033[33m"   : "";
	adrp	x2, .LC123	// tmp158,
	adrp	x0, .LC23	// tmp159,
	add	x2, x2, :lo12:.LC123	// tmp334, tmp158,
	add	x0, x0, :lo12:.LC23	// tmp335, tmp159,
// src/cpp/include/utils.h:26:     inline const char* const YELLOW      = _tty ? "\033[33m"   : "";
	str	x5, [x1, #:lo12:_ZGVN5Color6YELLOWE]	// tmp252, _ZGVN5Color6YELLOWE
// src/cpp/include/utils.h:26:     inline const char* const YELLOW      = _tty ? "\033[33m"   : "";
	tst	x4, 1	// _tty,
	csel	x0, x0, x2, eq	// iftmp.150_30, tmp335, tmp334,
// src/cpp/include/utils.h:26:     inline const char* const YELLOW      = _tty ? "\033[33m"   : "";
	str	x0, [x3, #:lo12:_ZN5Color6YELLOWE]	// iftmp.150_30, YELLOW
.L915:
// src/cpp/include/utils.h:27:     inline const char* const CYAN        = _tty ? "\033[36m"   : "";
	adrp	x1, _ZGVN5Color4CYANE	// tmp260,
	ldr	x0, [x1, #:lo12:_ZGVN5Color4CYANE]	// _ZGVN5Color4CYANE, _ZGVN5Color4CYANE
	tbnz	x0, 0, .L917	// _ZGVN5Color4CYANE,,
// src/cpp/include/utils.h:27:     inline const char* const CYAN        = _tty ? "\033[36m"   : "";
	adrp	x0, _ZN5Color4_ttyE	// tmp267,
// src/cpp/include/utils.h:27:     inline const char* const CYAN        = _tty ? "\033[36m"   : "";
	adrp	x3, _ZN5Color4CYANE	// tmp271,
// src/cpp/include/utils.h:27:     inline const char* const CYAN        = _tty ? "\033[36m"   : "";
	ldrb	w4, [x0, #:lo12:_ZN5Color4_ttyE]	// _tty, _tty
// src/cpp/include/utils.h:27:     inline const char* const CYAN        = _tty ? "\033[36m"   : "";
	mov	x5, 1	// tmp265,
// src/cpp/include/utils.h:27:     inline const char* const CYAN        = _tty ? "\033[36m"   : "";
	adrp	x2, .LC124	// tmp160,
	adrp	x0, .LC23	// tmp161,
	add	x2, x2, :lo12:.LC124	// tmp336, tmp160,
	add	x0, x0, :lo12:.LC23	// tmp337, tmp161,
// src/cpp/include/utils.h:27:     inline const char* const CYAN        = _tty ? "\033[36m"   : "";
	str	x5, [x1, #:lo12:_ZGVN5Color4CYANE]	// tmp265, _ZGVN5Color4CYANE
// src/cpp/include/utils.h:27:     inline const char* const CYAN        = _tty ? "\033[36m"   : "";
	tst	x4, 1	// _tty,
	csel	x0, x0, x2, eq	// iftmp.153_34, tmp337, tmp336,
// src/cpp/include/utils.h:27:     inline const char* const CYAN        = _tty ? "\033[36m"   : "";
	str	x0, [x3, #:lo12:_ZN5Color4CYANE]	// iftmp.153_34, CYAN
.L917:
// src/cpp/include/utils.h:28:     inline const char* const BOLD_GREEN  = _tty ? "\033[1;32m" : "";
	adrp	x1, _ZGVN5Color10BOLD_GREENE	// tmp273,
	ldr	x0, [x1, #:lo12:_ZGVN5Color10BOLD_GREENE]	// _ZGVN5Color10BOLD_GREENE, _ZGVN5Color10BOLD_GREENE
	tbnz	x0, 0, .L919	// _ZGVN5Color10BOLD_GREENE,,
// src/cpp/include/utils.h:28:     inline const char* const BOLD_GREEN  = _tty ? "\033[1;32m" : "";
	adrp	x0, _ZN5Color4_ttyE	// tmp280,
// src/cpp/include/utils.h:28:     inline const char* const BOLD_GREEN  = _tty ? "\033[1;32m" : "";
	adrp	x3, _ZN5Color10BOLD_GREENE	// tmp284,
// src/cpp/include/utils.h:28:     inline const char* const BOLD_GREEN  = _tty ? "\033[1;32m" : "";
	ldrb	w4, [x0, #:lo12:_ZN5Color4_ttyE]	// _tty, _tty
// src/cpp/include/utils.h:28:     inline const char* const BOLD_GREEN  = _tty ? "\033[1;32m" : "";
	mov	x5, 1	// tmp278,
// src/cpp/include/utils.h:28:     inline const char* const BOLD_GREEN  = _tty ? "\033[1;32m" : "";
	adrp	x2, .LC125	// tmp162,
	adrp	x0, .LC23	// tmp163,
	add	x2, x2, :lo12:.LC125	// tmp338, tmp162,
	add	x0, x0, :lo12:.LC23	// tmp339, tmp163,
// src/cpp/include/utils.h:28:     inline const char* const BOLD_GREEN  = _tty ? "\033[1;32m" : "";
	str	x5, [x1, #:lo12:_ZGVN5Color10BOLD_GREENE]	// tmp278, _ZGVN5Color10BOLD_GREENE
// src/cpp/include/utils.h:28:     inline const char* const BOLD_GREEN  = _tty ? "\033[1;32m" : "";
	tst	x4, 1	// _tty,
	csel	x0, x0, x2, eq	// iftmp.156_38, tmp339, tmp338,
// src/cpp/include/utils.h:28:     inline const char* const BOLD_GREEN  = _tty ? "\033[1;32m" : "";
	str	x0, [x3, #:lo12:_ZN5Color10BOLD_GREENE]	// iftmp.156_38, BOLD_GREEN
.L919:
// src/cpp/include/utils.h:29:     inline const char* const BOLD_CYAN   = _tty ? "\033[1;36m" : "";
	adrp	x1, _ZGVN5Color9BOLD_CYANE	// tmp286,
	ldr	x0, [x1, #:lo12:_ZGVN5Color9BOLD_CYANE]	// _ZGVN5Color9BOLD_CYANE, _ZGVN5Color9BOLD_CYANE
	tbnz	x0, 0, .L921	// _ZGVN5Color9BOLD_CYANE,,
// src/cpp/include/utils.h:29:     inline const char* const BOLD_CYAN   = _tty ? "\033[1;36m" : "";
	adrp	x0, _ZN5Color4_ttyE	// tmp293,
// src/cpp/include/utils.h:29:     inline const char* const BOLD_CYAN   = _tty ? "\033[1;36m" : "";
	adrp	x3, _ZN5Color9BOLD_CYANE	// tmp297,
// src/cpp/include/utils.h:29:     inline const char* const BOLD_CYAN   = _tty ? "\033[1;36m" : "";
	ldrb	w4, [x0, #:lo12:_ZN5Color4_ttyE]	// _tty, _tty
// src/cpp/include/utils.h:29:     inline const char* const BOLD_CYAN   = _tty ? "\033[1;36m" : "";
	mov	x5, 1	// tmp291,
// src/cpp/include/utils.h:29:     inline const char* const BOLD_CYAN   = _tty ? "\033[1;36m" : "";
	adrp	x2, .LC126	// tmp164,
	adrp	x0, .LC23	// tmp165,
	add	x2, x2, :lo12:.LC126	// tmp340, tmp164,
	add	x0, x0, :lo12:.LC23	// tmp341, tmp165,
// src/cpp/include/utils.h:29:     inline const char* const BOLD_CYAN   = _tty ? "\033[1;36m" : "";
	str	x5, [x1, #:lo12:_ZGVN5Color9BOLD_CYANE]	// tmp291, _ZGVN5Color9BOLD_CYANE
// src/cpp/include/utils.h:29:     inline const char* const BOLD_CYAN   = _tty ? "\033[1;36m" : "";
	tst	x4, 1	// _tty,
	csel	x0, x0, x2, eq	// iftmp.159_42, tmp341, tmp340,
// src/cpp/include/utils.h:29:     inline const char* const BOLD_CYAN   = _tty ? "\033[1;36m" : "";
	str	x0, [x3, #:lo12:_ZN5Color9BOLD_CYANE]	// iftmp.159_42, BOLD_CYAN
.L921:
// src/cpp/include/utils.h:30:     inline const char* const BOLD_YELLOW = _tty ? "\033[1;33m" : "";
	adrp	x1, _ZGVN5Color11BOLD_YELLOWE	// tmp299,
	ldr	x0, [x1, #:lo12:_ZGVN5Color11BOLD_YELLOWE]	// _ZGVN5Color11BOLD_YELLOWE, _ZGVN5Color11BOLD_YELLOWE
	tbnz	x0, 0, .L923	// _ZGVN5Color11BOLD_YELLOWE,,
// src/cpp/include/utils.h:30:     inline const char* const BOLD_YELLOW = _tty ? "\033[1;33m" : "";
	adrp	x0, _ZN5Color4_ttyE	// tmp306,
// src/cpp/include/utils.h:30:     inline const char* const BOLD_YELLOW = _tty ? "\033[1;33m" : "";
	adrp	x3, _ZN5Color11BOLD_YELLOWE	// tmp310,
// src/cpp/include/utils.h:30:     inline const char* const BOLD_YELLOW = _tty ? "\033[1;33m" : "";
	ldrb	w4, [x0, #:lo12:_ZN5Color4_ttyE]	// _tty, _tty
// src/cpp/include/utils.h:30:     inline const char* const BOLD_YELLOW = _tty ? "\033[1;33m" : "";
	mov	x5, 1	// tmp304,
// src/cpp/include/utils.h:30:     inline const char* const BOLD_YELLOW = _tty ? "\033[1;33m" : "";
	adrp	x2, .LC127	// tmp166,
	adrp	x0, .LC23	// tmp167,
	add	x2, x2, :lo12:.LC127	// tmp342, tmp166,
	add	x0, x0, :lo12:.LC23	// tmp343, tmp167,
// src/cpp/include/utils.h:30:     inline const char* const BOLD_YELLOW = _tty ? "\033[1;33m" : "";
	str	x5, [x1, #:lo12:_ZGVN5Color11BOLD_YELLOWE]	// tmp304, _ZGVN5Color11BOLD_YELLOWE
// src/cpp/include/utils.h:30:     inline const char* const BOLD_YELLOW = _tty ? "\033[1;33m" : "";
	tst	x4, 1	// _tty,
	csel	x0, x0, x2, eq	// iftmp.162_46, tmp343, tmp342,
// src/cpp/include/utils.h:30:     inline const char* const BOLD_YELLOW = _tty ? "\033[1;33m" : "";
	str	x0, [x3, #:lo12:_ZN5Color11BOLD_YELLOWE]	// iftmp.162_46, BOLD_YELLOW
.L923:
// src/cpp/include/utils.h:31:     inline const char* const DIM_GREEN   = _tty ? "\033[2;32m" : "";
	adrp	x1, _ZGVN5Color9DIM_GREENE	// tmp312,
	ldr	x0, [x1, #:lo12:_ZGVN5Color9DIM_GREENE]	// _ZGVN5Color9DIM_GREENE, _ZGVN5Color9DIM_GREENE
	tbnz	x0, 0, .L903	// _ZGVN5Color9DIM_GREENE,,
// src/cpp/include/utils.h:31:     inline const char* const DIM_GREEN   = _tty ? "\033[2;32m" : "";
	adrp	x0, _ZN5Color4_ttyE	// tmp319,
// src/cpp/include/utils.h:31:     inline const char* const DIM_GREEN   = _tty ? "\033[2;32m" : "";
	adrp	x3, _ZN5Color9DIM_GREENE	// tmp323,
// src/cpp/include/utils.h:31:     inline const char* const DIM_GREEN   = _tty ? "\033[2;32m" : "";
	ldrb	w4, [x0, #:lo12:_ZN5Color4_ttyE]	// _tty, _tty
// src/cpp/include/utils.h:31:     inline const char* const DIM_GREEN   = _tty ? "\033[2;32m" : "";
	mov	x5, 1	// tmp317,
// src/cpp/include/utils.h:31:     inline const char* const DIM_GREEN   = _tty ? "\033[2;32m" : "";
	adrp	x2, .LC128	// tmp168,
	adrp	x0, .LC23	// tmp169,
	add	x2, x2, :lo12:.LC128	// tmp344, tmp168,
	add	x0, x0, :lo12:.LC23	// tmp345, tmp169,
// src/cpp/include/utils.h:31:     inline const char* const DIM_GREEN   = _tty ? "\033[2;32m" : "";
	str	x5, [x1, #:lo12:_ZGVN5Color9DIM_GREENE]	// tmp317, _ZGVN5Color9DIM_GREENE
// src/cpp/include/utils.h:31:     inline const char* const DIM_GREEN   = _tty ? "\033[2;32m" : "";
	tst	x4, 1	// _tty,
	csel	x0, x0, x2, eq	// iftmp.165_50, tmp345, tmp344,
// src/cpp/include/utils.h:31:     inline const char* const DIM_GREEN   = _tty ? "\033[2;32m" : "";
	str	x0, [x3, #:lo12:_ZN5Color9DIM_GREENE]	// iftmp.165_50, DIM_GREEN
.L903:
// src/cpp/benchmark.cpp:385: }
	ldp	x29, x30, [sp], 16	//,,,
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret	
.L939:
	.cfi_restore_state
// src/cpp/include/utils.h:20:     inline const bool _tty       = (isatty(STDOUT_FILENO) != 0);
	mov	x2, 1	// tmp176,
// src/cpp/include/utils.h:20:     inline const bool _tty       = (isatty(STDOUT_FILENO) != 0);
	mov	w0, w2	//, tmp176
// src/cpp/include/utils.h:20:     inline const bool _tty       = (isatty(STDOUT_FILENO) != 0);
	str	x2, [x1, #:lo12:_ZGVN5Color4_ttyE]	// tmp176, _ZGVN5Color4_ttyE
// src/cpp/include/utils.h:20:     inline const bool _tty       = (isatty(STDOUT_FILENO) != 0);
	bl	isatty		//
// src/cpp/include/utils.h:20:     inline const bool _tty       = (isatty(STDOUT_FILENO) != 0);
	adrp	x1, _ZN5Color4_ttyE	// tmp178,
// src/cpp/include/utils.h:20:     inline const bool _tty       = (isatty(STDOUT_FILENO) != 0);
	cmp	w0, 0	// tmp346,
	cset	w0, ne	// tmp179,
	strb	w0, [x1, #:lo12:_ZN5Color4_ttyE]	// tmp179, _tty
	b	.L904		//
	.cfi_endproc
.LFE8375:
	.size	_GLOBAL__sub_I_main, .-_GLOBAL__sub_I_main
	.section	.init_array,"aw"
	.align	3
	.xword	_GLOBAL__sub_I_main
	.section	.rodata.str1.8
	.align	3
.LC1:
	.string	"Restructured Nested-Loop"
	.align	3
.LC2:
	.string	"Hoisted Vars + restrict"
	.align	3
.LC3:
	.string	"Reordered Nested-Loop"
	.align	3
.LC4:
	.string	"Specialized Implementation"
	.align	3
.LC5:
	.string	"NEON Blocked Implementation"
	.weak	_ZGVN5Color9DIM_GREENE
	.section	.bss._ZGVN5Color9DIM_GREENE,"awG",@nobits,_ZGVN5Color9DIM_GREENE,comdat
	.align	3
	.type	_ZGVN5Color9DIM_GREENE, %gnu_unique_object
	.size	_ZGVN5Color9DIM_GREENE, 8
_ZGVN5Color9DIM_GREENE:
	.zero	8
	.weak	_ZGVN5Color11BOLD_YELLOWE
	.section	.bss._ZGVN5Color11BOLD_YELLOWE,"awG",@nobits,_ZGVN5Color11BOLD_YELLOWE,comdat
	.align	3
	.type	_ZGVN5Color11BOLD_YELLOWE, %gnu_unique_object
	.size	_ZGVN5Color11BOLD_YELLOWE, 8
_ZGVN5Color11BOLD_YELLOWE:
	.zero	8
	.weak	_ZGVN5Color9BOLD_CYANE
	.section	.bss._ZGVN5Color9BOLD_CYANE,"awG",@nobits,_ZGVN5Color9BOLD_CYANE,comdat
	.align	3
	.type	_ZGVN5Color9BOLD_CYANE, %gnu_unique_object
	.size	_ZGVN5Color9BOLD_CYANE, 8
_ZGVN5Color9BOLD_CYANE:
	.zero	8
	.weak	_ZGVN5Color10BOLD_GREENE
	.section	.bss._ZGVN5Color10BOLD_GREENE,"awG",@nobits,_ZGVN5Color10BOLD_GREENE,comdat
	.align	3
	.type	_ZGVN5Color10BOLD_GREENE, %gnu_unique_object
	.size	_ZGVN5Color10BOLD_GREENE, 8
_ZGVN5Color10BOLD_GREENE:
	.zero	8
	.weak	_ZGVN5Color4CYANE
	.section	.bss._ZGVN5Color4CYANE,"awG",@nobits,_ZGVN5Color4CYANE,comdat
	.align	3
	.type	_ZGVN5Color4CYANE, %gnu_unique_object
	.size	_ZGVN5Color4CYANE, 8
_ZGVN5Color4CYANE:
	.zero	8
	.weak	_ZGVN5Color6YELLOWE
	.section	.bss._ZGVN5Color6YELLOWE,"awG",@nobits,_ZGVN5Color6YELLOWE,comdat
	.align	3
	.type	_ZGVN5Color6YELLOWE, %gnu_unique_object
	.size	_ZGVN5Color6YELLOWE, 8
_ZGVN5Color6YELLOWE:
	.zero	8
	.weak	_ZGVN5Color5GREENE
	.section	.bss._ZGVN5Color5GREENE,"awG",@nobits,_ZGVN5Color5GREENE,comdat
	.align	3
	.type	_ZGVN5Color5GREENE, %gnu_unique_object
	.size	_ZGVN5Color5GREENE, 8
_ZGVN5Color5GREENE:
	.zero	8
	.weak	_ZGVN5Color3REDE
	.section	.bss._ZGVN5Color3REDE,"awG",@nobits,_ZGVN5Color3REDE,comdat
	.align	3
	.type	_ZGVN5Color3REDE, %gnu_unique_object
	.size	_ZGVN5Color3REDE, 8
_ZGVN5Color3REDE:
	.zero	8
	.weak	_ZGVN5Color3DIME
	.section	.bss._ZGVN5Color3DIME,"awG",@nobits,_ZGVN5Color3DIME,comdat
	.align	3
	.type	_ZGVN5Color3DIME, %gnu_unique_object
	.size	_ZGVN5Color3DIME, 8
_ZGVN5Color3DIME:
	.zero	8
	.weak	_ZGVN5Color4BOLDE
	.section	.bss._ZGVN5Color4BOLDE,"awG",@nobits,_ZGVN5Color4BOLDE,comdat
	.align	3
	.type	_ZGVN5Color4BOLDE, %gnu_unique_object
	.size	_ZGVN5Color4BOLDE, 8
_ZGVN5Color4BOLDE:
	.zero	8
	.weak	_ZGVN5Color5RESETE
	.section	.bss._ZGVN5Color5RESETE,"awG",@nobits,_ZGVN5Color5RESETE,comdat
	.align	3
	.type	_ZGVN5Color5RESETE, %gnu_unique_object
	.size	_ZGVN5Color5RESETE, 8
_ZGVN5Color5RESETE:
	.zero	8
	.weak	_ZGVN5Color4_ttyE
	.section	.bss._ZGVN5Color4_ttyE,"awG",@nobits,_ZGVN5Color4_ttyE,comdat
	.align	3
	.type	_ZGVN5Color4_ttyE, %gnu_unique_object
	.size	_ZGVN5Color4_ttyE, 8
_ZGVN5Color4_ttyE:
	.zero	8
	.weak	_ZN5Color9DIM_GREENE
	.section	.bss._ZN5Color9DIM_GREENE,"awG",@nobits,_ZN5Color9DIM_GREENE,comdat
	.align	3
	.type	_ZN5Color9DIM_GREENE, %gnu_unique_object
	.size	_ZN5Color9DIM_GREENE, 8
_ZN5Color9DIM_GREENE:
	.zero	8
	.weak	_ZN5Color11BOLD_YELLOWE
	.section	.bss._ZN5Color11BOLD_YELLOWE,"awG",@nobits,_ZN5Color11BOLD_YELLOWE,comdat
	.align	3
	.type	_ZN5Color11BOLD_YELLOWE, %gnu_unique_object
	.size	_ZN5Color11BOLD_YELLOWE, 8
_ZN5Color11BOLD_YELLOWE:
	.zero	8
	.weak	_ZN5Color9BOLD_CYANE
	.section	.bss._ZN5Color9BOLD_CYANE,"awG",@nobits,_ZN5Color9BOLD_CYANE,comdat
	.align	3
	.type	_ZN5Color9BOLD_CYANE, %gnu_unique_object
	.size	_ZN5Color9BOLD_CYANE, 8
_ZN5Color9BOLD_CYANE:
	.zero	8
	.weak	_ZN5Color10BOLD_GREENE
	.section	.bss._ZN5Color10BOLD_GREENE,"awG",@nobits,_ZN5Color10BOLD_GREENE,comdat
	.align	3
	.type	_ZN5Color10BOLD_GREENE, %gnu_unique_object
	.size	_ZN5Color10BOLD_GREENE, 8
_ZN5Color10BOLD_GREENE:
	.zero	8
	.weak	_ZN5Color4CYANE
	.section	.bss._ZN5Color4CYANE,"awG",@nobits,_ZN5Color4CYANE,comdat
	.align	3
	.type	_ZN5Color4CYANE, %gnu_unique_object
	.size	_ZN5Color4CYANE, 8
_ZN5Color4CYANE:
	.zero	8
	.weak	_ZN5Color6YELLOWE
	.section	.bss._ZN5Color6YELLOWE,"awG",@nobits,_ZN5Color6YELLOWE,comdat
	.align	3
	.type	_ZN5Color6YELLOWE, %gnu_unique_object
	.size	_ZN5Color6YELLOWE, 8
_ZN5Color6YELLOWE:
	.zero	8
	.weak	_ZN5Color5GREENE
	.section	.bss._ZN5Color5GREENE,"awG",@nobits,_ZN5Color5GREENE,comdat
	.align	3
	.type	_ZN5Color5GREENE, %gnu_unique_object
	.size	_ZN5Color5GREENE, 8
_ZN5Color5GREENE:
	.zero	8
	.weak	_ZN5Color3REDE
	.section	.bss._ZN5Color3REDE,"awG",@nobits,_ZN5Color3REDE,comdat
	.align	3
	.type	_ZN5Color3REDE, %gnu_unique_object
	.size	_ZN5Color3REDE, 8
_ZN5Color3REDE:
	.zero	8
	.weak	_ZN5Color3DIME
	.section	.bss._ZN5Color3DIME,"awG",@nobits,_ZN5Color3DIME,comdat
	.align	3
	.type	_ZN5Color3DIME, %gnu_unique_object
	.size	_ZN5Color3DIME, 8
_ZN5Color3DIME:
	.zero	8
	.weak	_ZN5Color4BOLDE
	.section	.bss._ZN5Color4BOLDE,"awG",@nobits,_ZN5Color4BOLDE,comdat
	.align	3
	.type	_ZN5Color4BOLDE, %gnu_unique_object
	.size	_ZN5Color4BOLDE, 8
_ZN5Color4BOLDE:
	.zero	8
	.weak	_ZN5Color5RESETE
	.section	.bss._ZN5Color5RESETE,"awG",@nobits,_ZN5Color5RESETE,comdat
	.align	3
	.type	_ZN5Color5RESETE, %gnu_unique_object
	.size	_ZN5Color5RESETE, 8
_ZN5Color5RESETE:
	.zero	8
	.weak	_ZN5Color4_ttyE
	.section	.bss._ZN5Color4_ttyE,"awG",@nobits,_ZN5Color4_ttyE,comdat
	.type	_ZN5Color4_ttyE, %gnu_unique_object
	.size	_ZN5Color4_ttyE, 1
_ZN5Color4_ttyE:
	.zero	1
	.section	.rodata.cst8,"aM",@progbits,8
	.align	3
.LC117:
	.word	-400107883
	.word	1041313291
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC43:
// name:
	.xword	.LC0
// function:
	.xword	_Z12cnn_baselineR10CNNContext
// name:
	.xword	.LC1
// function:
	.xword	_Z16cnn_restructuredR10CNNContext
// name:
	.xword	.LC2
// function:
	.xword	_Z18cnn_hoist_restrictR10CNNContext
// name:
	.xword	.LC3
// function:
	.xword	_Z11cnn_reorderR10CNNContext
// name:
	.xword	.LC4
// function:
	.xword	_Z15cnn_specializedR10CNNContext
// name:
	.xword	.LC5
// function:
	.xword	_Z23cnn_specialized_blockedR10CNNContext
// name:
	.xword	.LC6
// function:
	.xword	_Z21specialized_maxpool2dR10CNNContext
.LC8:
// cycles_median:
	.word	0
	.word	2146959360
// seconds_median:
	.word	0
	.word	2146959360
// has_data:
	.byte	0
	.zero	7
	.bss
	.align	3
	.set	.LANCHOR1,. + 0
	.type	_ZL21benchmark_global_sink, %object
	.size	_ZL21benchmark_global_sink, 8
_ZL21benchmark_global_sink:
	.zero	8
	.hidden	DW.ref._ZTISt9exception
	.weak	DW.ref._ZTISt9exception
	.section	.data.DW.ref._ZTISt9exception,"awG",@progbits,DW.ref._ZTISt9exception,comdat
	.align	3
	.type	DW.ref._ZTISt9exception, %object
	.size	DW.ref._ZTISt9exception, 8
DW.ref._ZTISt9exception:
	.xword	_ZTISt9exception
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
