#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

cxx_compiler_flags = []
if os.name == 'nt':
    cxx_compiler_flags.append("/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")

setup(
    name="diff_gaussian_rasterization_fastgs",
    packages=['diff_gaussian_rasterization_fastgs'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_fastgs._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "cuda_rasterizer/adam.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "nvcc": [
                    "-allow-unsupported-compiler",
                    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
                ],
                "cxx": cxx_compiler_flags
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
