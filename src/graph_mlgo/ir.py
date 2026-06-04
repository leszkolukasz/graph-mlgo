import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from loguru import logger

Oz_PASS_STRING = "annotation2metadata,forceattrs,inferattrs,coro-early,function<eager-inv>(ee-instrument<>,lower-expect,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;no-switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,sroa<modify-cfg>,early-cse<>),openmp-opt,ipsccp,called-value-propagation,globalopt,function<eager-inv>(mem2reg,instcombine<max-iterations=1;no-verify-fixpoint>,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>),always-inline,require<globals-aa>,function(invalidate<aa>),require<profile-summary>,cgscc(devirt<4>(inline,function-attrs<skip-non-recursive-function-attrs>,function<eager-inv;no-rerun>(sroa<modify-cfg>,early-cse<memssa>,speculative-execution<only-if-divergent-target>,jump-threading,correlated-propagation,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,instcombine<max-iterations=1;no-verify-fixpoint>,aggressive-instcombine,tailcallelim,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,reassociate,constraint-elimination,loop-mssa(loop-instsimplify,loop-simplifycfg,licm<no-allowspeculation>,loop-rotate<no-header-duplication;no-prepare-for-lto>,licm<allowspeculation>,simple-loop-unswitch<no-nontrivial;trivial>),simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,instcombine<max-iterations=1;no-verify-fixpoint>,loop(loop-idiom,indvars,extra-simple-loop-unswitch-passes,loop-deletion,loop-unroll-full),sroa<modify-cfg>,vector-combine,mldst-motion<no-split-footer-bb>,gvn<>,sccp,bdce,instcombine<max-iterations=1;no-verify-fixpoint>,jump-threading,correlated-propagation,adce,memcpyopt,dse,move-auto-init,loop-mssa(licm<allowspeculation>),coro-elide,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,instcombine<max-iterations=1;no-verify-fixpoint>),function-attrs,function(require<should-not-run-function-passes>),coro-split,coro-annotation-elide)),deadargelim,coro-cleanup,globalopt,globaldce,elim-avail-extern,rpo-function-attrs,recompute-globalsaa,function<eager-inv>(float2int,lower-constant-intrinsics,loop(loop-rotate<no-header-duplication;no-prepare-for-lto>,loop-deletion),loop-distribute,inject-tli-mappings,loop-vectorize<no-interleave-forced-only;vectorize-forced-only;>,infer-alignment,loop-load-elim,instcombine<max-iterations=1;no-verify-fixpoint>,simplifycfg<bonus-inst-threshold=1;forward-switch-cond;switch-range-to-icmp;switch-to-lookup;no-keep-loops;hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,vector-combine,instcombine<max-iterations=1;no-verify-fixpoint>,loop-unroll<O2>,transform-warning,sroa<preserve-cfg>,infer-alignment,instcombine<max-iterations=1;no-verify-fixpoint>,loop-mssa(licm<allowspeculation>),alignment-from-assumptions,loop-sink,instsimplify,div-rem-pairs,tailcallelim,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;speculate-unpredictables>),globaldce,constmerge,cg-profile,rel-lookup-table-converter,function(annotation-remarks),verify"


def compile_module_no_opt(ir_text: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_ll = tmp_path / "input.ll"
        object_file = tmp_path / "final.o"
        text_bin = tmp_path / "text_only.bin"

        input_ll.write_text(ir_text)

        llc_cmd = [
            "llc",
            str(input_ll),
            "-o",
            str(object_file),
            "-filetype=obj",
            "-O0",
        ]

        objcopy_cmd = [
            "llvm-objcopy",
            "--only-section=.text",
            "--output-target=binary",
            str(object_file),
            str(text_bin),
        ]

        try:
            subprocess.run(llc_cmd, check=True, capture_output=True)
            subprocess.run(objcopy_cmd, check=True, capture_output=True)

            return os.path.getsize(text_bin), ir_text

        except subprocess.CalledProcessError as e:
            logger.error(f"LLVM Tool Error (No Opt): {e.stderr.decode()}")
            raise


def compile_module_benchmark(
    ir_text: str, mode: Literal["baseline", "llvm", "agent"]
) -> tuple[int, str]:
    """
    baseline -> Oz pass without inlining
    llvm -> first inlining then Oz without inlining
    agent -> just Oz with inlining (agent should have done inlining)
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_ll = tmp_path / "input.ll"
        inlined_ll = tmp_path / "inlined.ll"
        optimized_ll = tmp_path / "opt_after.ll"
        object_file = tmp_path / "final.o"
        text_bin = tmp_path / "text_only.bin"

        input_ll.write_text(ir_text)
        current_ir_path = input_ll

        if mode == "llvm":
            inline_cmd = [
                "opt",
                "-S",
                "-passes=inline",
                str(current_ir_path),
                "-o",
                str(inlined_ll),
            ]
            try:
                subprocess.run(inline_cmd, check=True, capture_output=True)
                current_ir_path = inlined_ll
            except subprocess.CalledProcessError as e:
                logger.error(f"LLVM Inline Error: {e.stderr.decode()}")
                raise

        cleanup_passes = Oz_PASS_STRING.replace("always-inline,", "").replace(
            "devirt<4>(inline,", "devirt<4>("
        )

        opt_cmd = [
            "opt",
            "-S",
            f"-passes={cleanup_passes}",
            str(current_ir_path),
            "-o",
            str(optimized_ll),
        ]
        llc_cmd = [
            "llc",
            str(optimized_ll),
            "-o",
            str(object_file),
            "-filetype=obj",
            "-O2",
        ]
        objcopy_cmd = [
            "llvm-objcopy",
            "--only-section=.text",
            "--output-target=binary",
            str(object_file),
            str(text_bin),
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


def compile_module(ir_text: str, enable_inlining: bool) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_ll = tmp_path / "input.ll"
        optimized_ll = tmp_path / "opt_after.ll"
        object_file = tmp_path / "final.o"
        text_bin = tmp_path / "text_only.bin"

        input_ll.write_text(ir_text)

        # Result of: opt -Oz -print-pipeline-passes < /dev/null
        pass_string = Oz_PASS_STRING

        if enable_inlining:
            pass_string = "default<Oz>"
            # pass_string = "module(function(sroa),cgscc(inline),function(instcombine<no-verify-fixpoint>,simplifycfg,adce),globaldce)"
        # else:
        # pass_string = "module(function(sroa,instcombine<no-verify-fixpoint>,simplifycfg,adce),globaldce)"

        if not enable_inlining:
            pass_string = pass_string.replace("devirt<4>(inline,", "devirt<4>(")

        # pass_string = "default<Oz>"

        opt_cmd = [
            "opt",
            "-S",
            f"-passes={pass_string}",
        ]

        # if enable_inlining:
        #     opt_cmd.extend(["-mllvm", "-enable-ml-inliner=release"])

        # if enable_inlining:
        #     opt_cmd.extend(["-enable-ml-inliner=release"])

        # if not enable_inlining:
        # opt_cmd.append("-disable-inlining")
        # opt_cmd.append("-inline-threshold=0")

        opt_cmd.extend([str(input_ll), "-o", str(optimized_ll)])

        llc_cmd = [
            "llc",
            str(optimized_ll),
            "-o",
            str(object_file),
            "-filetype=obj",
            "-O2",
        ]

        objcopy_cmd = [
            "llvm-objcopy",
            "--only-section=.text",
            "--output-target=binary",
            str(object_file),
            str(text_bin),
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
