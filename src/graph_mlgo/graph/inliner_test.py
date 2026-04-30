import llvmlite.binding as llvm
import ctypes

from graph_mlgo import llvm_inliner # ty: ignore

def test_inline():
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

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
    
    print("=== BEFORE INLINING ===")
    print(str(mod))
    
    ptr_address = ctypes.cast(mod._ptr, ctypes.c_void_p).value
    
    success_count = llvm_inliner.inline_edges(ptr_address, "caller", "callee")
    
    print(f"\nNumber of successful inlinings: {success_count}")
    
    print("=== AFTER INLINING ===")
    print(str(mod))

if __name__ == "__main__":
    test_inline()