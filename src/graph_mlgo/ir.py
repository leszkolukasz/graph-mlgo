import llvmlite.binding as llvm

from loguru import logger

def compile_module(module: llvm.ModuleRef, enable_inlining: bool) -> tuple[bytes, str]:
        module_clone = module.clone()

        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine(opt=2)
        
        pto = llvm.create_pipeline_tuning_options(speed_level=2, size_level=2)

        if not enable_inlining:
            pto.inlining_threshold = 0

        pass_builder = llvm.create_pass_builder(target_machine, pto)

        mpm = pass_builder.getModulePassManager()
        mpm.run(module_clone, pass_builder)

        return target_machine.emit_object(module_clone), str(module_clone)

def test_inline():
    llvm_code = """
    define internal i32 @callee(i32 %x) {
        %a = add i32 %x, 10
        %b = mul i32 %a, 20
        %c = sub i32 %b, 5
        ret i32 %c
    }
    
    define i32 @caller() {
        %1 = call i32 @callee(i32 5)
        ret i32 %1
    }
    """
    
    mod = llvm.parse_assembly(llvm_code)
    mod.verify()
    
    with_inlining_bytes, with_inlining_ir = compile_module(mod, enable_inlining=True)
    without_inlining_bytes, without_inlining_ir = compile_module(mod, enable_inlining=False)

    logger.info(f"Size without inlining: {len(without_inlining_bytes)} bytes")
    logger.info(f"IR without inlining:\n{without_inlining_ir}")

    logger.info(f"Size with inlining: {len(with_inlining_bytes)} bytes")
    logger.info(f"IR with inlining:\n{with_inlining_ir}")

if __name__ == "__main__":
    test_inline()