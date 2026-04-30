cmake_minimum_required(VERSION 3.15)
project(mlgo_cpp_ext)

if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Conda detected: $ENV{CONDA_PREFIX}")
    
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    
    set(LLVM_DIR "$ENV{CONDA_PREFIX}/lib/cmake/llvm")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
link_directories(${LLVM_LIBRARY_DIRS})

pybind11_add_module(llvm_inliner inliner.cpp)

llvm_map_components_to_libnames(llvm_libs 
    Core 
    Support 
    Analysis 
    TransformUtils 
    IPO
)

target_link_libraries(llvm_inliner PRIVATE pybind11::module ${llvm_libs})
target_compile_features(llvm_inliner PRIVATE cxx_std_17)

install(TARGETS llvm_inliner LIBRARY DESTINATION graph_mlgo)