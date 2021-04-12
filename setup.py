#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Feng Wang

import glob
import os
import sys

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"

os_name = sys.platform

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "dl_lib", "layers")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        if sys.platform == 'win32':
            extra_compile_args["nvcc"].append("-D _WIN64")

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "dl_lib._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


cur_dir = os.getcwd()

if os_name == "win32":
    dl_train_name = "tools/dl_train.bat"
    dl_test_name = "tools/dl_test.bat"
    head = f"set OMP_NUM_THREADS=1\n"
    python_command = "python"
    parameters = "%*"
elif os_name == "linux":
    dl_train_name = "tools/dl_train"
    dl_test_name = "tools/dl_test"
    head = f"#!/bin/bash\n\nexport OMP_NUM_THREADS=1\n"
    python_command = "python3"
    parameters = "$@"
else:
    raise Exception("Target OS not support")

with open(dl_train_name, "w") as dl_lib_train:
    dl_lib_train.write(
        head + f"{python_command} {os.path.join(cur_dir, 'tools', 'train_net.py')} {parameters}")
with open(dl_test_name, "w") as dl_lib_test:
    dl_lib_test.write(
        head + f"{python_command} {os.path.join(cur_dir, 'tools', 'test_net.py')} {parameters}")

setup(
    name="dl_lib",
    version="0.1",
    author="Feng Wang",
    url="https://github.com/Fatescript/centernet",
    description="Deep Learning lib(dl_lib) is Feng Wang's"
    "platform for object detection based on Detectron2.",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    install_requires=[
        "termcolor>=1.1",
        "Pillow>=6.0",
        "tabulate",
        "cloudpickle",
        "matplotlib",
        "tqdm>4.29.0",
        "Shapely",
        "tensorboard",
        "portalocker",
        "pycocotools",
        "easydict",
        "imagesize",
    ],
    extras_require={"all": ["shapely", "psutil"]},
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    scripts=["tools/dl_train", "tools/dl_test"] if os_name == 'linux'
    else ["tools/dl_train.bat", "tools/dl_test.bat"],
)
