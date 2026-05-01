import llvmlite.binding as llvm
import ctypes
from loguru import logger

from graph_mlgo import cpp_bindings # ty: ignore

def test_inline():
    llvm_code = """
    define i32 @callee(i32 %x) {
        ret i32 %x
    }
    define i32 @caller() {
        %1 = call i32 @callee(i32 5)
        ret i32 %1
    }

    define i32 @callee2(i32 %x) {
        ret i32 %x
    }
    define i32 @caller2() {
        %1 = call i32 @callee2(i32 5)
        ret i32 %1
    }
    """
    
    mod = llvm.parse_assembly(llvm_code)
    mod.verify()
    
    logger.info("=== BEFORE INLINING ===")
    logger.info(str(mod))
    
    ptr_address = ctypes.cast(mod._ptr, ctypes.c_void_p).value
    
    success_count = cpp_bindings.inline_edges(ptr_address, "caller", "callee")
    
    logger.info(f"\nNumber of successful inlinings: {success_count}")
    
    logger.info("=== AFTER INLINING ===")
    logger.info(str(mod))

if __name__ == "__main__":
    test_inline()