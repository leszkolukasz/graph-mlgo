import subprocess
import tempfile
import os
from pathlib import Path
from loguru import logger

def compile_module(ir_text: str, enable_inlining: bool) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_ll = tmp_path / "input.ll"
        optimized_ll = tmp_path / "opt_after.ll"
        object_file = tmp_path / "final.o"
        text_bin = tmp_path / "text_only.bin"

        input_ll.write_text(ir_text)

        if enable_inlining:
            pass_string = "module(function(sroa),cgscc(inline),function(instcombine<no-verify-fixpoint>,simplifycfg,adce))"
        else:
            pass_string = "function(sroa,instcombine<no-verify-fixpoint>,simplifycfg,adce)"
        
        opt_cmd = [
            "opt",
            "-S",
            f"-passes={pass_string}",
            str(input_ll),
            "-o", str(optimized_ll)
        ]

        llc_cmd = [
            "llc",
            str(optimized_ll),
            "-o", str(object_file),
            "-filetype=obj",
            "-O2"
        ]

        objcopy_cmd = [
            "llvm-objcopy",
            "--only-section=.text",
            "--output-target=binary",
            str(object_file),
            str(text_bin)
        ]

        try:
            subprocess.run(opt_cmd, check=True, capture_output=True)
            ir_after = optimized_ll.read_text()
            
            subprocess.run(llc_cmd, check=True, capture_output=True)
            subprocess.run(objcopy_cmd, check=True, capture_output=True)

            return os.path.getsize(text_bin), ir_after

        except subprocess.CalledProcessError as e:
            logger.error(f"LLVM Tool Error: {e.stderr.decode()}")
            raise

def test_benchmark():
    llvm_code = """
    define internal i32 @callee(i32 %x) {
        %a = add i32 %x, 10
        %b = mul i32 %a, 20
        %c = sub i32 %b, 5
        ret i32 %c
    }
    
    define i32 @caller(i32 %input) {
        %1 = call i32 @callee(i32 %input)
        ret i32 %1
    }
    """
    
    size_without, ir_without = compile_module(llvm_code, enable_inlining=False)
    size_with, ir_with = compile_module(llvm_code, enable_inlining=True)

    logger.info(f"Before inlining: {size_without} bytes")
    logger.info(f"With inlining: {size_with} bytes")
    logger.success(f"Gain: {size_without - size_with} bytes")

    logger.info("IR without inlining:")
    logger.info(ir_without)

    logger.info("IR with inlining:")
    logger.info(ir_with)

if __name__ == "__main__":
    test_benchmark()