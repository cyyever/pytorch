# Bootstrap the Intel oneAPI SYCL/device toolchain before configuring PyTorch.
# PyTorch keeps the host compiler in CMAKE_CXX_COMPILER and uses icpx as the
# SYCL driver. This is a separate CMake invocation because ExternalProject
# runs during the build and cannot change compiler selection after project().

include(ExternalProject)

set(ONEAPI_VERSION "2026.1" CACHE STRING "Intel oneAPI version to install")
set(ONEAPI_URL
    "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/5996e26b-f48a-42b1-8db0-b002ad0bd8d7/intel-oneapi-toolkit-2026.1.1.33_offline.sh"
    CACHE STRING "URL of the Intel oneAPI offline installer")
set(ONEAPI_SHA256 "" CACHE STRING "SHA256 of the Intel oneAPI offline installer")
set(ONEAPI_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/oneapi/${ONEAPI_VERSION}" CACHE PATH "oneAPI installation directory")

if(NOT ONEAPI_SHA256)
  message(WARNING
    "ONEAPI_SHA256 is not set; the installer will be downloaded without a hash check.")
  set(ONEAPI_URL_HASH)
else()
  set(ONEAPI_URL_HASH "URL_HASH;SHA256=${ONEAPI_SHA256}")
endif()

ExternalProject_Add(oneapi-${ONEAPI_VERSION}
  URL "${ONEAPI_URL}"
  ${ONEAPI_URL_HASH}
  DOWNLOAD_NO_EXTRACT TRUE
  DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/oneapi/download"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/oneapi/source"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND
    "${CMAKE_COMMAND}" -E env ACCEPT_EULA=accept bash
    "<DOWNLOADED_FILE>" -a --silent --cli --eula accept
    --install-dir "${ONEAPI_INSTALL_PREFIX}"
  BUILD_BYPRODUCTS "${ONEAPI_INSTALL_PREFIX}/compiler/2026.1/linux/bin/icpx"
)

message(STATUS "oneAPI ${ONEAPI_VERSION} will be installed under ${ONEAPI_INSTALL_PREFIX}")
