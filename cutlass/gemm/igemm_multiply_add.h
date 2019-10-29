/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Implements matrix multiply accumulate operation of 8-bit integer data using DP4A
   instruction.
*/
#pragma once

#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 610))

#include "cutlass/fragment.h"
#include "cutlass/gemm/thread_multiply_add.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ int device_grammar_size;
__constant__ unsigned char device_grammar_body[1000];
__constant__ unsigned int device_grammar_tail[1000];


/// Template performing matrix multiply-add operation within a thread
template <typename ThreadGemmShape_, typename ThreadsPerWarp_>
struct ThreadMultiplyAdd<ThreadGemmShape_, ThreadsPerWarp_, int8_t, int8_t, int> {
  /// The shape of the instruction.
  typedef Shape<4, 1, 1> InstructionShape;
  /// Shape of the thread-level GEMM (K-by-N-by-M)
  typedef ThreadGemmShape_ ThreadGemmShape;
  
  /// Thread-level GEMM (N-by-M) must be a multiple of 32.
  static_assert((ThreadGemmShape::kH * ThreadGemmShape::kW) % 32 == 0, 
          "Thread-level GEMM (N-by-M) must be multiple of 32");

  /// Aliased for compatibility. Will be removed in CUTLASS v2.0
  typedef ThreadGemmShape AccumulatorsPerThread;
  /// The number of threads per warp.
  typedef ThreadsPerWarp_ ThreadsPerWarp;
  /// The number of accumulators per warp.
  typedef typename ShapeMul<ThreadGemmShape, ThreadsPerWarp>::Shape AccumulatorsPerWarp;
  /// The type for A.
  typedef int8_t ScalarA;
  /// The fragment for A.
  typedef Fragment<ScalarA, AccumulatorsPerThread::kW * 4> FragmentA;
  /// The type for B.
  typedef int8_t ScalarB;
  /// The fragment for B.
  typedef Fragment<ScalarB, AccumulatorsPerThread::kH * 4> FragmentB;
  /// The type for C and D.
  typedef int ScalarC;
  /// The accumulators.
  typedef Fragment<ScalarC, AccumulatorsPerThread::kH * AccumulatorsPerThread::kW> Accumulators;

  /// Ctor.
  CUTLASS_DEVICE ThreadMultiplyAdd() {}

  /// Multiply : d = a*b + c.
  CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                   FragmentB const& b,
                                   Accumulators const& c,
                                   Accumulators& d) {

    // The inputs.
    int const* a_int = reinterpret_cast<int const*>(&a[0]);
    int const* b_int = reinterpret_cast<int const*>(&b[0]);

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < AccumulatorsPerThread::kH; ++j) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < AccumulatorsPerThread::kW; ++i) {
        asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                     : "=r"(d[j * AccumulatorsPerThread::kW + i])
                     : "r"(a_int[i]), "r"(b_int[j]), "r"(c[j * AccumulatorsPerThread::kW + i]));
      }
    }
  }
};


/// Template performing matrix multiply-add operation within a thread
template <typename ThreadGemmShape_, typename ThreadsPerWarp_>
struct ThreadMultiplyAddBoolVector<ThreadGemmShape_, ThreadsPerWarp_, int8_t, int8_t, int> {
    /// The shape of the instruction.
    typedef Shape<4, 1, 1> InstructionShape;
    /// Shape of the thread-level GEMM (K-by-N-by-M)
    typedef ThreadGemmShape_ ThreadGemmShape;

    /// Thread-level GEMM (N-by-M) must be a multiple of 32.
    static_assert((ThreadGemmShape::kH * ThreadGemmShape::kW) % 32 == 0,
                  "Thread-level GEMM (N-by-M) must be multiple of 32");

    /// Aliased for compatibility. Will be removed in CUTLASS v2.0
    typedef ThreadGemmShape AccumulatorsPerThread;
    /// The number of threads per warp.
    typedef ThreadsPerWarp_ ThreadsPerWarp;
    /// The number of accumulators per warp.
    typedef typename ShapeMul<ThreadGemmShape, ThreadsPerWarp>::Shape AccumulatorsPerWarp;
    /// The type for A.
    typedef int8_t ScalarA;
    /// The fragment for A.
    typedef Fragment<ScalarA, AccumulatorsPerThread::kW * 4> FragmentA;
    /// The type for B.
    typedef int8_t ScalarB;
    /// The fragment for B.
    typedef Fragment<ScalarB, AccumulatorsPerThread::kH * 4> FragmentB;
    /// The type for C and D.
    typedef int ScalarC;
    /// The accumulators.
    typedef Fragment<ScalarC, AccumulatorsPerThread::kH * AccumulatorsPerThread::kW> Accumulators;

    /// Ctor.
    CUTLASS_DEVICE ThreadMultiplyAddBoolVector() {}

    /// Multiply : d = a*b + c.
    CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                     FragmentB const& b,
                                     Accumulators const& c,
                                     Accumulators& d) {

        // The inputs.
        unsigned int const* a_int = reinterpret_cast<unsigned int const*>(&a[0]);
        unsigned int const* b_int = reinterpret_cast<unsigned int const*>(&b[0]);

        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < AccumulatorsPerThread::kH; ++j) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < AccumulatorsPerThread::kW; ++i) {
                unsigned int mult = 0;
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < 4; k++) {
                    unsigned int left = (a_int[i] >> k * 2) & 0xFF;
                    left <<= 8;
                    unsigned int right = (b_int[j] >> k * 2) & 0xFF;
                    unsigned int conc = left | right;
                    if (left != 0 || right != 0) printf("%p %p %p\n", left >> 8, right, conc);

                    for (int t = 0; t < device_grammar_size; t++) {
                        if ((device_grammar_tail[t] & conc) == device_grammar_tail[t]) {
                            mult |= device_grammar_body[t];
                        }
                    }
                    // we should move as we work with 4 numbers in one
                    if (k != 3) {
                        mult <<= 8;
                    }
                }
                d[j * AccumulatorsPerThread::kW + i] = mult | c[j * AccumulatorsPerThread::kW + i];
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass

#endif // if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 610))
