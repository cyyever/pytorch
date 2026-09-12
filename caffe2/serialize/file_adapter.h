#pragma once

#include <c10/macros/Macros.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "caffe2/serialize/istream_adapter.h"
#include "caffe2/serialize/read_adapter_interface.h"


namespace caffe2::serialize {

class TORCH_API FileAdapter final : public ReadAdapterInterface {
 public:
  C10_DISABLE_COPY_AND_ASSIGN(FileAdapter);
  explicit FileAdapter(const std::string& file_name);
  size_t size() const override;
  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override;
  // A second descriptor on the same file, opened with O_DIRECT so that readers
  // can bypass the page cache. cuFile refuses to register a handle built from a
  // descriptor that lacks the flag, so the buffered FILE* used for the zip
  // metadata cannot be reused for it. Opened on first use, owned by the
  // adapter, and -1 when the platform or the file system does not support the
  // flag.
  int directFd();
  ~FileAdapter() override;

 private:
  // An RAII Wrapper for a FILE pointer. Closes on destruction.
  struct RAIIFile {
    FILE* fp_;
    explicit RAIIFile(const std::string& file_name);
    ~RAIIFile();
  };

  RAIIFile file_;
  std::string file_name_;
  std::once_flag direct_fd_once_;
  int direct_fd_{-1};
  // The size of the opened file in bytes
  uint64_t size_;
};

} // namespace caffe2::serialize
