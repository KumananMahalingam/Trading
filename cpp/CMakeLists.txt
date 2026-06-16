# === FILE: cpp/CMakeLists.txt ===
cmake_minimum_required(VERSION 3.20)

project(trading_lstm_cpp LANGUAGES CXX)

# --------------------------------------------------------------------------
# Standard & warnings
# --------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --------------------------------------------------------------------------
# Dependencies
#
# LibTorch is NOT a vcpkg package; point CMAKE_PREFIX_PATH at your unpacked
# libtorch distribution, e.g.:
#   cmake -B build -DCMAKE_PREFIX_PATH=C:/libtorch \
#         -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
# --------------------------------------------------------------------------
find_package(Torch REQUIRED)
find_package(CURL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(xlnt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# --------------------------------------------------------------------------
# Logging library (ISOLATED) -- the key fix for the fmt conflict.
#
# LibTorch bundles its own (newer) fmt under <libtorch>/include/fmt. The
# precompiled vcpkg spdlog was built against a different fmt, so any TU that
# includes BOTH <torch/torch.h> and spdlog headers fails (Torch's fmt headers
# shadow vcpkg's on the include path). We therefore confine spdlog to this one
# library, which links spdlog PRIVATELY and does NOT link Torch -- so its only
# fmt on the include path is the vcpkg one spdlog expects. No other translation
# unit includes spdlog.
# --------------------------------------------------------------------------
add_library(trading_logging STATIC
  src/util/logging.cpp
)
target_include_directories(trading_logging PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# PRIVATE: spdlog (and its fmt) are an implementation detail and do NOT
# propagate to consumers of trading_logging.
target_link_libraries(trading_logging PRIVATE spdlog::spdlog)
target_compile_features(trading_logging PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(trading_logging PRIVATE /utf-8 /W4 /O2)
else()
  target_compile_options(trading_logging PRIVATE -Wall -Wextra -O2)
endif()

# --------------------------------------------------------------------------
# Core library (all modules) -- links Torch but NOT spdlog.
# --------------------------------------------------------------------------
add_library(trading_core
  src/config/secret_key.cpp
  src/data/yahoo_client.cpp
  src/features/technical_indicators.cpp
  src/features/scaler.cpp
  src/features/alpha_formula.cpp
  src/features/alpha_generator.cpp
  src/models/lstm_modules.cpp
  src/models/return_predictor.cpp
  src/models/losses.cpp
  src/models/dataset.cpp
  src/models/metrics.cpp
  src/models/trainer.cpp
  src/storage/excel_handler.cpp
)

target_include_directories(trading_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(trading_core PUBLIC
  ${TORCH_LIBRARIES}
  CURL::libcurl
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  xlnt::xlnt
  trading_logging
)

target_compile_features(trading_core PUBLIC cxx_std_20)

# /utf-8 is required by LibTorch's bundled fmt headers on MSVC.
if(MSVC)
  target_compile_options(trading_core PRIVATE /utf-8 /W4 /O2)
else()
  target_compile_options(trading_core PRIVATE -Wall -Wextra -O2)
endif()

# --------------------------------------------------------------------------
# Executable
# --------------------------------------------------------------------------
add_executable(trading_lstm src/main.cpp)
target_link_libraries(trading_lstm PRIVATE trading_core)

if(MSVC)
  target_compile_options(trading_lstm PRIVATE /utf-8 /W4 /O2)
else()
  target_compile_options(trading_lstm PRIVATE -Wall -Wextra -O2)
endif()

# --------------------------------------------------------------------------
# On Windows, copy the LibTorch DLLs next to the executable.
# --------------------------------------------------------------------------
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET trading_lstm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${TORCH_DLLS}
      $<TARGET_FILE_DIR:trading_lstm>)
endif()
