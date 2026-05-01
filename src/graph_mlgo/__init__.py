from loguru import logger
import llvmlite.binding as llvm
import llvmlite.binding.ffi as ffi
import multiprocessing
import jax

if multiprocessing.current_process().name == "MainProcess":
    logger.info(f"llvmlite LLVM version: {llvm.llvm_version_info}")
    logger.info(f"llvmlite library path: {ffi.lib._name}")
    logger.info(f"JAX devices: {jax.devices()}")

llvm.initialize_native_target()
llvm.initialize_native_asmprinter()