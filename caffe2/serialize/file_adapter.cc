#include "caffe2/serialize/file_adapter.h"
#include <c10/util/Exception.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <string>

namespace caffe2 {
namespace serialize {

FileAdapter::RAIIFile::RAIIFile(const std::string& file_name) {
  fd_ = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd_ < 0) {
    auto old_errno = errno;
    auto error_msg =
        std::system_category().default_error_condition(old_errno).message();
    TORCH_CHECK(
        false,
        "open file failed because of errno ",
        old_errno,
        " on open: ",
        error_msg,
        ", file path: ",
        file_name);
  }
}

FileAdapter::RAIIFile::~RAIIFile() {
  if (fd_ >= 0) {
    close(fd_);
  }
}

// Reads go through pread, so the descriptor carries no shared file position
// and concurrent reads of different records need no synchronization.
FileAdapter::FileAdapter(const std::string& file_name) : file_(file_name) {
  struct stat file_stat{};
  const int fstat_ret = fstat(file_.fd_, &file_stat);
  TORCH_CHECK(fstat_ret == 0, "fstat returned ", fstat_ret);
  size_ = file_stat.st_size;
}

size_t FileAdapter::size() const {
  return size_;
}

size_t FileAdapter::read(uint64_t pos, void* buf, size_t n, const char* what)
    const {
  // Ensure that pos doesn't exceed size_.
  pos = std::min(pos, size_);
  // If pos doesn't exceed size_, then size_ - pos can never be negative (in
  // signed math) or since these are unsigned values, a very large value.
  // Clamp 'n' to the smaller of 'size_ - pos' and 'n' itself. i.e. if the
  // user requested to read beyond the end of the file, we clamp to just the
  // end of the file.
  n = std::min(static_cast<size_t>(size_ - pos), n);
  size_t done = 0;
  while (done < n) {
    const ssize_t got = pread(
        file_.fd_, static_cast<char*>(buf) + done, n - done, pos + done);
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      TORCH_CHECK(
          false, "pread failed with errno ", errno, ", context: ", what);
    }
    if (got == 0) {
      break;
    }
    done += static_cast<size_t>(got);
  }
  return done;
}

FileAdapter::~FileAdapter() = default;

} // namespace serialize
} // namespace caffe2
