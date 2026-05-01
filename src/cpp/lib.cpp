#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include "llvm/Support/raw_ostream.h"
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IR/LLVMContext.h>  
#include <string>
#include <vector>
#include <cstdint>

namespace py = pybind11;
using namespace llvm;

int inline_edges(size_t mod_ptr, const std::string& caller_name, const std::string& callee_name) {
    llvm::Module* M = reinterpret_cast<llvm::Module*>(mod_ptr);
    
    llvm::Function* caller = M->getFunction(caller_name);
    llvm::Function* callee = M->getFunction(callee_name);

    if (!caller || !callee) {
        llvm::errs() << "[C++] ERROR: Caller or Callee not found!\n";
        
        int func_count = 0;
        llvm::errs() << "[C++] Dumping available functions (first 5):\n";
        for (auto& F : *M) {
            if (func_count < 5) {
                llvm::errs() << "      - '" << F.getName() << "'\n";
            }
            func_count++;
        }
        llvm::errs() << "[C++] Total functions inside this specific module pointer: " << func_count << "\n";
        llvm::errs() << "[C++] ====================\n";
        return 0;
    }

    std::vector<llvm::CallBase*> calls_to_inline;
    for (auto& BB : *caller) {
        for (auto& I : BB) {
            if (auto* cb = llvm::dyn_cast<llvm::CallBase>(&I)) {
                if (cb->getCalledFunction() == callee) {
                    calls_to_inline.push_back(cb);
                }
            }
        }
    }

    int inlined_count = 0;
    for (auto* cb : calls_to_inline) {
        llvm::InlineFunctionInfo IFI;
        
        llvm::InlineResult res = llvm::InlineFunction(*cb, IFI);
        
        if (res.isSuccess()) {
            inlined_count++;
        } else {
            llvm::errs() << "[C++] Inline failed for " << callee_name 
                         << ": " << res.getFailureReason() << "\n";
        }
    }

    return inlined_count;
}

std::pair<std::string, int> inline_edges_safe(const std::string& ir_text, const std::string& caller_name, const std::string& callee_name) {
    llvm::LLVMContext Context;
    llvm::SMDiagnostic Err;
    
    std::unique_ptr<llvm::MemoryBuffer> MemBuf = llvm::MemoryBuffer::getMemBuffer(ir_text);
    std::unique_ptr<llvm::Module> M = llvm::parseIR(*MemBuf, Err, Context);
    
    if (!M) {
        return {ir_text, 0}; 
    }

    llvm::Function* caller = M->getFunction(caller_name);
    llvm::Function* callee = M->getFunction(callee_name);

    if (!caller || !callee) {
        return {ir_text, 0};
    }

    std::vector<llvm::CallBase*> calls_to_inline;
    for (auto& BB : *caller) {
        for (auto& I : BB) {
            if (auto* cb = llvm::dyn_cast<llvm::CallBase>(&I)) {
                if (cb->getCalledFunction() == callee) {
                    calls_to_inline.push_back(cb);
                }
            }
        }
    }

    int inlined_count = 0;
    for (auto* cb : calls_to_inline) {
        llvm::InlineFunctionInfo IFI;
        llvm::InlineResult res = llvm::InlineFunction(*cb, IFI);
        if (res.isSuccess()) {
            inlined_count++;
        }
    }

    std::string output_ir = ir_text;
    if (inlined_count > 0) {
        output_ir.clear();
        llvm::raw_string_ostream os(output_ir);
        M->print(os, nullptr);
    }
    
    return {output_ir, inlined_count};
}

PYBIND11_MODULE(cpp_bindings, m) {
    m.doc() = "C++ Bindings for LLVM Graph MLGO";
    
    m.def("inline_edges", &inline_edges, 
          py::arg("module_ptr_addr"), py::arg("caller_name"), py::arg("callee_name"),
          "Inline call graph edges using unsafe memory pointer");

    m.def("inline_edges_safe", &inline_edges_safe, 
          py::arg("ir_text"), py::arg("caller_name"), py::arg("callee_name"),
          "Safely inline call graph edges using string serialization");
}