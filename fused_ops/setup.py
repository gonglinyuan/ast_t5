import os
import site
import subprocess
import sys

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

site.ENABLE_USER_SITE = True

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


if not torch.cuda.is_available():
    print('\nWarning: Torch did not find available GPUs on this system.\n',
          'If your intention is to cross-compile, this is not an error.\n'
          'By default, it will cross-compile for Volta (compute capability 7.0), Turing (compute capability 7.5),\n'
          'and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n'
          'If you wish to cross-compile for a single specific architecture,\n'
          'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n')
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        _, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
        if int(bare_metal_major) == 11:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5"

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 0 or TORCH_MINOR < 12:
    raise RuntimeError("Requires Pytorch 1.12.0 or newer.\n" +
                       "The latest stable release can be obtained from https://pytorch.org/")

cmdclass = {}
ext_modules = []
extras = {}


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
        raise RuntimeError("Cuda extensions are being compiled with a version of Cuda that does " +
                           "not match the version used to compile Pytorch binaries.  " +
                           "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda))


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ['-DVERSION_GE_1_1']
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ['-DVERSION_GE_1_3']
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ['-DVERSION_GE_1_5']
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

cmdclass['build_ext'] = BuildExtension

if torch.utils.cpp_extension.CUDA_HOME is None:
    raise RuntimeError(
        "Nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")

check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)

cc_flag = []
_, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
if int(bare_metal_major) >= 11:
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")

if "--fused_softmax_dropout" in sys.argv:
    sys.argv.remove("--fused_softmax_dropout")

    ext_modules.append(
        CUDAExtension(
            name='fused_softmax_dropout_fast_cuda',
            sources=[
                'csrc/fused_softmax_dropout/interface.cpp',
                'csrc/fused_softmax_dropout/softmax_dropout_kernel.cu'
            ],
            include_dirs=[os.path.join(this_dir, 'csrc')],
            extra_compile_args={
                'cxx': ['-O3', ] + version_dependent_macros,
                'nvcc': append_nvcc_threads(
                    [
                        '-O3', '--use_fast_math',
                        '-gencode', 'arch=compute_70,code=sm_70',
                        '-U__CUDA_NO_HALF_OPERATORS__',
                        '-U__CUDA_NO_HALF_CONVERSIONS__',
                        '-U__CUDA_NO_BFLOAT16_OPERATORS__',
                        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                        '-U__CUDA_NO_BFLOAT162_OPERATORS__',
                        '-U__CUDA_NO_BFLOAT162_CONVERSIONS__',
                        '--expt-relaxed-constexpr',
                        '--expt-extended-lambda'
                    ]
                    + version_dependent_macros
                    + cc_flag
                )
            }
        )
    )

if "--fused_layer_norm" in sys.argv:
    sys.argv.remove("--fused_layer_norm")

    ext_modules.append(
        CUDAExtension(
            name='fused_layer_norm_fast_cuda',
            sources=[
                'csrc/fused_layer_norm/interface.cpp',
                'csrc/fused_layer_norm/layernorm.cu'
            ],
            include_dirs=[os.path.join(this_dir, 'csrc')],
            extra_compile_args={
                'cxx': ['-O3', ] + version_dependent_macros,
                'nvcc': append_nvcc_threads(
                    [
                        '-O3', '--use_fast_math',
                        '-gencode', 'arch=compute_70,code=sm_70',
                        '-U__CUDA_NO_HALF_OPERATORS__',
                        '-U__CUDA_NO_HALF_CONVERSIONS__',
                        '-U__CUDA_NO_BFLOAT16_OPERATORS__',
                        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                        '-U__CUDA_NO_BFLOAT162_OPERATORS__',
                        '-U__CUDA_NO_BFLOAT162_CONVERSIONS__',
                        '--expt-relaxed-constexpr',
                        '--expt-extended-lambda'
                    ]
                    + version_dependent_macros
                    + cc_flag
                )
            }
        )
    )

    ext_modules.append(
        CUDAExtension(
            name='fused_layer_norm_backward_gamma_beta_cuda',
            sources=[
                'csrc/fused_layer_norm/interface_gamma_beta.cpp',
                'csrc/fused_layer_norm/layernorm_backward.cu'
            ],
            include_dirs=[os.path.join(this_dir, 'csrc')],
            extra_compile_args={
                'cxx': ['-O3', ] + version_dependent_macros,
                'nvcc': append_nvcc_threads(
                    [
                        '-O3', '--use_fast_math', '-maxrregcount=50',
                        '-gencode', 'arch=compute_70,code=sm_70',
                        '-U__CUDA_NO_HALF_OPERATORS__',
                        '-U__CUDA_NO_HALF_CONVERSIONS__',
                        '-U__CUDA_NO_BFLOAT16_OPERATORS__',
                        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                        '-U__CUDA_NO_BFLOAT162_OPERATORS__',
                        '-U__CUDA_NO_BFLOAT162_CONVERSIONS__',
                        '--expt-relaxed-constexpr',
                        '--expt-extended-lambda'
                    ]
                    + version_dependent_macros
                    + cc_flag
                )
            }
        )
    )

setup(
    name='fused_ops',
    version='0.2',
    packages=find_packages(
        exclude=(
            'build',
            'csrc',
            'include',
            'tests',
            'dist',
            'docs',
            'tests',
            'examples',
        )
    ),
    description='Fused ops',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    extras_require=extras,
)
