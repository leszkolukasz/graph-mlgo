#include <pybind11/pybind11.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <vector>
#include <cstdint>

namespace py = pybind11;
using namespace llvm;

int inline_edges(uintptr_t module_addr, const std::string &callerName, const std::string &calleeName) {
    Module *M = reinterpret_cast<Module*>(module_addr);
    if (!M) return 0;

    Function *caller = M->getFunction(callerName);
    if (!caller) return 0;

    std::vector<CallBase*> callsToInline;
    for (BasicBlock &BB : *caller) {
        for (Instruction &I : BB) {
            if (auto *call = dyn_cast<CallBase>(&I)) {
                Function *calledFunc = call->getCalledFunction();
                if (calledFunc && calledFunc->getName() == calleeName) {
                    callsToInline.push_back(call);
                }
            }
        }
    }

    int successCount = 0;
    InlineFunctionInfo IFI;
    for (CallBase *call : callsToInline) {
        if (InlineFunction(*call, IFI).isSuccess()) {
            successCount++;
        }
    }

    return successCount;
}

PYBIND11_MODULE(cpp_bindings, m) {
    m.doc() = "C++ Bindings";
    
    m.def("inline_edges", &inline_edges, 
          py::arg("module_ptr_addr"), py::arg("caller_name"), py::arg("callee_name"),
          "Inline call graph edges");
}