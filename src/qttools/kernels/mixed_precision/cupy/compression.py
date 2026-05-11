# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import cupy as cp
import numpy as np

from qttools import NDArray

BLOCK_SIZE = 256

# Configure the custom floating-point format here.

# For standard precision, use 8 exponent bits and bias 128.
# NUM_EXPONENT_BITS = 8
# EXPONENT_BIAS = 128

# For 7 bit exponent bits and bias 117.
# NUM_EXPONENT_BITS = 7
# EXPONENT_BIAS = 117

# For 6 bit exponent bits and bias 53.
NUM_EXPONENT_BITS = 6
EXPONENT_BIAS = 53

# For 5 bit exponent bits and bias 21.
# NUM_EXPONENT_BITS = 5
# EXPONENT_BIAS = 21



cuda_source = f"""
#include <cupy/complex.cuh>
#define NUM_EXP_BITS {NUM_EXPONENT_BITS}
#define EXPONENT_BIAS {EXPONENT_BIAS}

template<int T>
__global__
void _compress_impl(unsigned char* out, const complex<double>* inp, const size_t N) {{
    // For T bits per component, total bits = 2*T, bytes = (2*T + 7) / 8
    static const int num_bits_total = T * 2;
    static const int num_bytes = (num_bits_total + 7) / 8;
    __shared__ unsigned char s_out[{BLOCK_SIZE} * num_bytes];

    size_t tile = blockIdx.x;
    size_t idx = tile * {BLOCK_SIZE} + threadIdx.x;

    for (size_t i = 0; i < num_bytes; ++i) {{
        s_out[threadIdx.x * num_bytes + i] = 0;
    }}

    if (idx < N) {{
        complex<double> r_inp = inp[idx];
        double vals[2] = {{r_inp.real(), r_inp.imag()}};
        unsigned long long packed_vals[2];

        for(int v=0; v<2; ++v) {{
            unsigned long long bits_64 = __double_as_longlong(vals[v]);
            unsigned long long sign = (bits_64 >> 63) & 0x1ULL;
            unsigned long long exp_64 = (bits_64 >> 52) & 0x7FFULL;
            unsigned long long mant_64 = bits_64 & 0xFFFFFFFFFFFFFULL;

            const int num_mantissa_bits = T - NUM_EXP_BITS - 1;
            const int shift = 52 - num_mantissa_bits;
            
            unsigned int exp_f;
            unsigned long long mant_trunc = mant_64 >> shift;

            if (exp_64 == 0x7FFULL) {{
                exp_f = (1U << NUM_EXP_BITS) - 1;  // All 1s for exponent
                if (mant_64 != 0) {{
                    mant_trunc = (1ULL << (num_mantissa_bits - 1));
                }}
            }} else {{
                int unbiased_exp = (int)exp_64 - 1023;
                int biased_f = unbiased_exp + EXPONENT_BIAS;

                if (biased_f <= 0) {{
                    // UNDERFLOW: Number is too small to represent.
                    // Cut to zero (±0 depending on sign bit, which is preserved separately).
                    // No rounding applied - ensures exact zero encoding.
                    exp_f = 0;
                    mant_trunc = 0;
                }} else if (biased_f >= (1 << NUM_EXP_BITS) - 1) {{
                    exp_f = (1U << NUM_EXP_BITS) - 2;  // Max normal exponent (avoid infinity)
                    mant_trunc = (1ULL << num_mantissa_bits) - 1;  // Max mantissa
                }} else {{
                    exp_f = (unsigned int)biased_f;
                    // Round to Nearest Even
                    unsigned long long guard_bit = 1ULL << (shift - 1);
                    unsigned long long sticky_mask = guard_bit - 1;
                    if ((bits_64 & guard_bit) && ((bits_64 & sticky_mask) || (mant_trunc & 1))) {{
                        mant_trunc++;
                        if (mant_trunc >= (1ULL << num_mantissa_bits)) {{
                            mant_trunc = 0; exp_f++;
                            if (exp_f >= (1U << NUM_EXP_BITS) - 1) {{ exp_f = (1U << NUM_EXP_BITS) - 2; mant_trunc = (1ULL << num_mantissa_bits) - 1; }}
                        }}
                    }}
                }}
            }}
            // Ensure exp_f never reaches all-1s (which encodes inf/nan)
            exp_f = min(exp_f, (1U << NUM_EXP_BITS) - 2);
            unsigned int sgn_exp = (static_cast<unsigned int>(sign) << NUM_EXP_BITS) | (exp_f & ((1U << NUM_EXP_BITS) - 1));
            // Sign bit is always included to preserve ±0 distinction when underflowing to zero.
            packed_vals[v] = ((unsigned long long)sgn_exp << num_mantissa_bits) | (mant_trunc & ((1ULL << num_mantissa_bits) - 1));
        }}

        // Pack both components into a bit stream without assuming <= 64 total bits.
        for (int v = 0; v < 2; ++v) {{
            size_t bit_base = static_cast<size_t>(v) * T;
            for (int bit = 0; bit < T; ++bit) {{
                if ((packed_vals[v] >> bit) & 1ULL) {{
                    size_t global_bit = bit_base + static_cast<size_t>(bit);
                    size_t byte_index = global_bit / 8;
                    size_t bit_offset = global_bit % 8;
                    s_out[threadIdx.x * num_bytes + byte_index] |= (1U << bit_offset);
                }}
            }}
        }}
    }}
    __syncthreads();

    size_t start_idx = tile * {BLOCK_SIZE} * num_bytes;
    size_t end_idx = min(N * num_bytes, start_idx + {BLOCK_SIZE} * num_bytes);
    for (size_t i = start_idx + threadIdx.x; i < end_idx; i += {BLOCK_SIZE}) {{
        out[i] = s_out[i - start_idx];
    }}
}}

template<int T>
__global__
void _decompress_impl(complex<double>* out, const unsigned char* inp, const size_t N) {{
    // For T bits per component, total bits = 2*T, bytes = (2*T + 7) / 8
    static const int num_bits_total = T * 2;
    static const int num_bytes = (num_bits_total + 7) / 8;
    __shared__ unsigned char s_inp[{BLOCK_SIZE} * num_bytes];

    size_t tile = blockIdx.x;
    size_t idx = tile * {BLOCK_SIZE} + threadIdx.x;
    size_t start_idx = tile * {BLOCK_SIZE} * num_bytes;
    size_t end_idx = min(N * num_bytes, start_idx + {BLOCK_SIZE} * num_bytes);
    
    for (size_t i = threadIdx.x; (start_idx + i) < end_idx; i += {BLOCK_SIZE}) {{
        s_inp[i] = inp[start_idx + i];
    }}
    __syncthreads();

    if (idx < N) {{
        // Read back the packed bit stream.
        auto read_bits = [&](int component) -> unsigned long long {{
            unsigned long long value = 0ULL;
            size_t bit_base = static_cast<size_t>(component) * T;
            for (int bit = 0; bit < T; ++bit) {{
                size_t global_bit = bit_base + static_cast<size_t>(bit);
                size_t byte_index = global_bit / 8;
                size_t bit_offset = global_bit % 8;
                unsigned char byte_value = s_inp[threadIdx.x * num_bytes + byte_index];
                if ((byte_value >> bit_offset) & 1U) {{
                    value |= (1ULL << bit);
                }}
            }}
            return value;
        }};

        unsigned long long real_bits = read_bits(0);
        unsigned long long imag_bits = read_bits(1);

        // Helper lambda to unpack T-bit value to double
        auto unpack = [](unsigned long long bits) -> double {{
            const int num_mantissa_bits = T - NUM_EXP_BITS - 1;
            const int shift = 52 - num_mantissa_bits;

            unsigned int sgn_exp_f = (unsigned int)(bits >> num_mantissa_bits);
            unsigned long long sign = (sgn_exp_f >> NUM_EXP_BITS) & 1;
            unsigned long long exp_f = sgn_exp_f & ((1U << NUM_EXP_BITS) - 1);
            unsigned long long mant_bits = bits & ((1ULL << num_mantissa_bits) - 1);

            unsigned long long exp_d;
            unsigned int inf_exp_f = (1U << NUM_EXP_BITS) - 1;  // All 1s = inf/nan
            
            if (exp_f == inf_exp_f) {{
                exp_d = 0x7FF;
                if (mant_bits != 0) {{
                    mant_bits = (1ULL << (num_mantissa_bits - 1));
                }} else {{
                    mant_bits = 0;
                }}
            }}
            else{{
                if (exp_f == 0) {{
                    // ZERO/UNDERFLOW: Exact zero representation (±0 depending on sign bit).
                    // Decompresses to ±0.0 in IEEE 754 double precision.
                    exp_d = 0;
                }} else {{
                    exp_d = exp_f + (1023 - EXPONENT_BIAS);
                    // Clamp to valid double exponent range (max normal exponent is 1023, field value 2046)
                    if (exp_d >= 2047) exp_d = 2046;  // Prevent infinity
                }}
            }}
            // Shift bits back to the high-order position of the 52-bit mantissa
            unsigned long long mant_d = mant_bits << shift;
            unsigned long long res = (sign << 63) | (exp_d << 52) | mant_d;
            return __longlong_as_double(res);
        }};
        
        out[idx] = complex<double>(unpack(real_bits), unpack(imag_bits));
    }}
}}

extern "C" {{
    __global__ void _compress_fp16(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress_impl<16>(out, inp, N); }}
    __global__ void _compress_fp20(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress_impl<20>(out, inp, N); }}
    __global__ void _compress_fp24(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress_impl<24>(out, inp, N); }}
    __global__ void _compress_fp28(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress_impl<28>(out, inp, N); }}
    __global__ void _compress_fp32(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress_impl<32>(out, inp, N); }}
    __global__ void _compress_fp36(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress_impl<36>(out, inp, N); }}
    __global__ void _compress_fp40(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress_impl<40>(out, inp, N); }}
    __global__ void _compress_fp48(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress_impl<48>(out, inp, N); }}
    __global__ void _compress_fp56(unsigned char* out, const complex<double>* inp, const size_t N) {{ _compress_impl<56>(out, inp, N); }}

    __global__ void _decompress_fp16(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress_impl<16>(out, inp, N); }}
    __global__ void _decompress_fp20(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress_impl<20>(out, inp, N); }}
    __global__ void _decompress_fp24(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress_impl<24>(out, inp, N); }}
    __global__ void _decompress_fp28(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress_impl<28>(out, inp, N); }}
    __global__ void _decompress_fp32(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress_impl<32>(out, inp, N); }}
    __global__ void _decompress_fp36(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress_impl<36>(out, inp, N); }}
    __global__ void _decompress_fp40(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress_impl<40>(out, inp, N); }}
    __global__ void _decompress_fp48(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress_impl<48>(out, inp, N); }}
    __global__ void _decompress_fp56(complex<double>* out, const unsigned char* inp, const size_t N) {{ _decompress_impl<56>(out, inp, N); }}
}}
"""

module = cp.RawModule(code=cuda_source, options=("--std=c++17",))

_kernels = {
    "compress": {
        b: module.get_function(f"_compress_fp{b}") for b in [16, 20, 24, 28, 32, 36, 40, 48, 56]
    },
    "decompress": {
        b: module.get_function(f"_decompress_fp{b}") for b in [16, 20, 24, 28, 32, 36, 40, 48, 56]
    },
}


def compress(inp: NDArray, bits: int, out: NDArray | None = None) -> NDArray:
    """Compresses complex128 data to a custom floating point format.
    It is specified by the number of bits where
    1 bit is for the sign, the exponent bits depend on the precision mode (8 bits for standard, 7 bits for narrow),
    and the rest of the bits are for the mantissa (taken from fp64).

    Parameters
    ----------
    inp : NDArray
        Input array of complex128 values to be compressed.
    bits : int
        Number of bits per component (real or imaginary) for the custom floating point format 
        (e.g., 16, 20, 24, 28, 32, 36, 40, 48, 56).
        The total bits per complex number is 2*bits, stored in (2*bits + 7) // 8 bytes.
    out : NDArray, optional
        Pre-allocated output array to store the compressed data. If None, a new array will
        be created. The shape of the output array will be inp.shape + (num_output_bytes,).
        The dtype of the output array must be cp.uint8.

    Returns
    -------
    NDArray
        The compressed data as an array of unsigned bytes.

    """

    # check input is complex128
    if inp.dtype != cp.complex128:
        raise ValueError(
            f"Input array must have dtype cp.complex128 but got {inp.dtype}."
        )

    if bits not in _kernels["compress"].keys():
        raise ValueError(
            f"Unsupported bit width: {bits}. Supported values are {list(_kernels['compress'].keys())}."
        )

    inp = cp.ascontiguousarray(inp)

    N = np.prod(inp.shape)
    
    # Calculate output size: (2*bits + 7) // 8 bytes per complex number
    num_output_bytes = (2 * bits + 7) // 8

    if out is None:
        out = cp.empty(inp.shape + (num_output_bytes,), dtype=cp.uint8)
    else:
        if out.shape != inp.shape + (num_output_bytes,):
            raise ValueError(
                f"Output array must have shape {inp.shape + (num_output_bytes,)} but got {out.shape}."
            )
        if out.dtype != cp.uint8:
            raise ValueError(
                f"Output array must have dtype cp.uint8 but got {out.dtype}."
            )

    # check if the output array is contiguous
    if not out.flags["C_CONTIGUOUS"]:
        _out = cp.empty(out.shape, dtype=cp.uint8)
    else:
        _out = out

    if not _out.flags["C_CONTIGUOUS"]:
        raise ValueError("Must be contiguous")

    if not inp.flags["C_CONTIGUOUS"]:
        raise ValueError("Must be contiguous")
    _kernels["compress"][bits](
        ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,), (BLOCK_SIZE,), (_out, inp, N)
    )

    if not out.flags["C_CONTIGUOUS"]:
        out[:] = _out

    return out


def decompress(inp: NDArray, bits: int, out: NDArray | None = None) -> NDArray:
    """Decompresses data from a custom floating point format to complex128.
    The custom floating point format is specified by the number of bits where
    1 bit is for the sign, the exponent bits depend on the precision mode (8 bits for standard, 7 bits for narrow),
    and the rest of the bits are for the mantissa (taken from fp64).

    Parameters
    ----------
    inp : NDArray
        Input array of unsigned bytes representing the compressed data.
        The shape of the input array should be (..., (2*bits + 7) // 8)
        where bits is the bit width per component of the custom floating point format.
    bits : int
        Number of bits per component (real or imaginary) for the custom floating point format 
        (e.g., 16, 20, 24, 28, 32, 36, 40, 48, 56).
    out : NDArray, optional
        Pre-allocated output array to store the decompressed complex128 data.
        If None, a new array will be created.
        The shape of the output array will be inp.shape[:-1] and dtype will be cp.complex128.

    Returns
    -------
    NDArray
        The decompressed data as an array of complex128 values.

    """

    if bits not in _kernels["decompress"].keys():
        raise ValueError(
            f"Unsupported bit width: {bits}. Supported values are {list(_kernels['decompress'].keys())}."
        )

    if inp.dtype != cp.uint8:
        raise ValueError(f"Input array must have dtype cp.uint8 but got {inp.dtype}.")

    inp = cp.ascontiguousarray(inp)

    N = np.prod(inp.shape[:-1])
    
    # Calculate expected input size: (2*bits + 7) // 8 bytes per complex number
    num_input_bytes = (2 * bits + 7) // 8

    if out is None:
        out = cp.empty(inp.shape[:-1], dtype=cp.complex128)
    else:
        if out.shape != inp.shape[:-1]:
            raise ValueError(
                f"Output array must have shape {inp.shape[:-1]} but got {out.shape}."
            )
        if out.dtype != cp.complex128:
            raise ValueError(
                f"Output array must have dtype cp.complex128 but got {out.dtype}."
            )

    if inp.shape[-1] != num_input_bytes:
        raise ValueError(
            f"Last dimension of input array must be {num_input_bytes} but got {inp.shape[-1]}."
        )

    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("Must be contiguous")

    if not inp.flags["C_CONTIGUOUS"]:
        raise ValueError("Must be contiguous")

    _kernels["decompress"][bits](
        ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,), (BLOCK_SIZE,), (out, inp, N)
    )

    return out
