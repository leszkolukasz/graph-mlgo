import multiprocessing

import jax
import llvmlite.binding as llvm
import llvmlite.binding.ffi as ffi
from loguru import logger

if multiprocessing.current_process().name == "MainProcess":
    logger.info(f"llvmlite LLVM version: {llvm.llvm_version_info}")
    logger.info(f"llvmlite library path: {ffi.lib._name}")
    logger.info(f"JAX devices: {jax.devices()}")

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
