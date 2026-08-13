from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []
if os.name == 'nt':
    cxx_compiler_flags.append("/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=[
            "ssim.cu",
            "ext.cpp"],
            extra_compile_args={
                "nvcc": ["-allow-unsupported-compiler", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"],
                "cxx": cxx_compiler_flags
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
