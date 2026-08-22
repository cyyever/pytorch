#include <ATen/MapAllocator.h>

#include <atomic>
#include <random>
#include <string>
#if ATOMIC_INT_LOCK_FREE == 2
#define AT_ATOMIC_IPC_REFCOUNT 1
#endif


#include <c10/util/error.h>

#if defined(HAVE_MMAP)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#if !defined(_MSC_VER) || defined(HAVE_MMAP)
#include <sys/types.h>
#include <unistd.h>
#endif
#include <fmt/format.h>

namespace at {

static constexpr int64_t map_alloc_alignment = 64;

std::string NewProcessWideShmHandle() {
  static std::atomic<uint64_t> counter{0};
  static std::random_device rd;
  return fmt::format(
      "/torch_{}_{}_{}",
      getpid(),
      rd(),
      counter.fetch_add(1, std::memory_order_relaxed));
}
#ifdef HAVE_MMAP

namespace {
struct MapInfo {
  std::atomic<int> refcount;
};

constexpr const char* unknown_filename = "filename not specified";
}  // namespace (anonymous)

MapAllocator::MapAllocator(WithFd /*unused*/, std::string_view filename, int fd, int flags, size_t size)
  : filename_(filename.empty() ? unknown_filename : filename)
  , size_(0) // to be filled later
  , fd_(fd)
{

  if (!(flags & ALLOCATOR_MAPPED_SHARED) && !(flags & ALLOCATOR_MAPPED_SHAREDMEM)) {
    flags &= ~ALLOCATOR_MAPPED_NOCREATE;
  }
  if ((flags ^ ALLOCATOR_MAPPED_EXCLUSIVE) == 0) {
    TORCH_CHECK(false, "ALLOCATOR_MAPPED_EXCLUSIVE flag requires opening the file in shared mode");
  }
  flags_ = flags;

  // OK, now do the allocation

  if (size == 0) {
    return;
  }

  {
    /* open file */
    int fd{-1};
    int flags{}; // shadow

    if (flags_ & (ALLOCATOR_MAPPED_SHARED | ALLOCATOR_MAPPED_SHAREDMEM)) {
      flags = O_RDWR | O_CREAT;
    } else {
      flags = O_RDONLY;
    }

    if (flags_ & ALLOCATOR_MAPPED_EXCLUSIVE) {
      flags |= O_EXCL;
    }
    if (flags_ & ALLOCATOR_MAPPED_NOCREATE) {
      flags &= ~O_CREAT;
    }

    if (!(flags_ & ALLOCATOR_MAPPED_FROMFD)) {
      if (flags_ & ALLOCATOR_MAPPED_SHARED) {
        // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
        if ((fd = open(filename_.c_str(), flags, static_cast<mode_t>(0600))) == -1) {
          TORCH_CHECK(false, "unable to open file <", filename_, "> in read-write mode: ", c10::utils::str_error(errno), " (", errno, ")");
        }
      } else if (flags_ & ALLOCATOR_MAPPED_SHAREDMEM) {
#ifdef HAVE_SHM_OPEN
        // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
        if((fd = shm_open(filename_.c_str(), flags, static_cast<mode_t>(0600))) == -1) {
          TORCH_CHECK(false, "unable to open shared memory object <", filename_, "> in read-write mode: ", c10::utils::str_error(errno), " (", errno, ")");
        }
#else
        TORCH_CHECK(false, "unable to open file <", filename_, "> in sharedmem mode, shm_open unavailable on this platform");
#endif
      } else {
        // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
        if ((fd = open(filename_.c_str(), O_RDONLY)) == -1) {
          TORCH_CHECK(false, "unable to open file <", filename_, "> in read-only mode: ", c10::utils::str_error(errno), " (", errno, ")");
        }
      }
    } else {
      fd = fd_;
      TORCH_INTERNAL_ASSERT(fd >= 0);
    }

    struct stat file_stat{};
    if (fstat(fd, &file_stat) == -1) {
#ifndef STRIP_ERROR_MESSAGES
      int last_err = errno;
#endif
      if (!(flags_ & ALLOCATOR_MAPPED_FROMFD)) {
        ::close(fd);
      }
      TORCH_CHECK(false, "unable to stat the file <", filename_, ">: ", c10::utils::str_error(last_err), " (", last_err, ")");
    }

    if (size > 0) {
      if (static_cast<int64_t>(size) > file_stat.st_size) {
        if (flags_) {
          if (ftruncate(fd, static_cast<off_t>(size)) == -1) {
            TORCH_CHECK(false, "unable to resize file <", filename_, "> to the right size: ", c10::utils::str_error(errno), " (", errno, ")");
          }

#ifdef HAVE_POSIX_FALLOCATE
          if (flags_ & ALLOCATOR_MAPPED_SHAREDMEM) {
            for (;;) {
              int err = posix_fallocate(fd, 0, static_cast<off_t>(size));
              if (err == 0) {
                break;
              }

              if (err == EINTR) {
                continue;
              }

              if (err == EINVAL || err == EOPNOTSUPP) {
                // the underlying filesystem does not support the operation
                break;
              }

              TORCH_CHECK(false, "unable to allocate shared memory(shm) for file <", filename_, ">: ", c10::utils::str_error(err), " (", err, ")");
            }
          }
#endif

          if (fstat(fd, &file_stat) == -1 || file_stat.st_size < static_cast<int64_t>(size)) {
#ifndef STRIP_ERROR_MESSAGES
            int last_err = errno;
#endif
            ::close(fd);
            TORCH_CHECK(false, "unable to stretch file <", filename_, "> to the right size: ", c10::utils::str_error(last_err), " (", last_err, ")");
          }
/* on macOS write returns with errno 45 (Operation not supported) when used
 * with a file descriptor obtained via shm_open
 */
#ifndef __APPLE__
          if ((write(fd, "", 1)) != 1) /* note that the string "" contains the '\0' byte ... */ {
#ifndef STRIP_ERROR_MESSAGES
            int last_err = errno;
#endif
            ::close(fd);
            TORCH_CHECK(false, "unable to write to file <", filename_, ">: ", c10::utils::str_error(last_err), " (", last_err, ")");
          }
#endif
        } else {
          ::close(fd);
          TORCH_CHECK(false, "file <", filename_, "> size <",  file_stat.st_size, "> is smaller than the required mapping size <", size, ">");
        }
      }
    } else {
      size = file_stat.st_size;
    }

    size_ = static_cast<ptrdiff_t>(size); /* if we are here, it must be the right size */

    /* map it */
    if (flags_ & (ALLOCATOR_MAPPED_SHARED | ALLOCATOR_MAPPED_SHAREDMEM)) {
      base_ptr_ = mmap(nullptr, size_, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    } else {
      base_ptr_ = mmap(nullptr, size_, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
    }

    if (base_ptr_ == MAP_FAILED) {
      base_ptr_ = nullptr; /* let's be sure it is NULL */
      TORCH_CHECK(false, "unable to mmap ", size_, " bytes from file <", filename_, ">: ", c10::utils::str_error(errno), " (", errno, ")");
    }

#if !defined(__APPLE__) && !defined(__ANDROID__)
    /* attempt to use larger block size on Linux, which is important for getting better CUDA upload speed */
    posix_fadvise(fd, 0, static_cast<off_t>(size), POSIX_FADV_SEQUENTIAL);
#endif

    if (flags_ & ALLOCATOR_MAPPED_KEEPFD) {
      fd_ = fd;
    } else {
      if (::close(fd) == -1) {
        TORCH_CHECK(false, "Error closing file <", filename_, ">: ", c10::utils::str_error(errno), " (", errno, ")");
      }
      fd_ = -1;
    }

    if (flags_ & ALLOCATOR_MAPPED_UNLINK) {
      if (flags_ & ALLOCATOR_MAPPED_SHAREDMEM) {
#ifdef HAVE_SHM_UNLINK
        if (shm_unlink(filename_.c_str()) == -1) {
          TORCH_CHECK(false, "could not unlink the shared memory file ", filename_, " : ", c10::utils::str_error(errno), " (", errno, ")");
        }
#else
        TORCH_CHECK(false, "could not unlink the shared memory file ", filename_, ", shm_unlink not available on platform");
#endif
      } else {
        if (unlink(filename_.c_str()) == -1)
          TORCH_CHECK(false, "could not unlink file ", filename_, " : ", c10::utils::str_error(errno), " (", errno, ")");
      }
    }

    if (base_ptr_ == MAP_FAILED) {
      TORCH_CHECK(false, "$ Torch: unable to mmap memory: you tried to mmap ", size_/1073741824, " GB.");
    }
  }
  c10::reportMemoryUsageToProfiler(base_ptr_, size_, 0, size_, c10::Device(c10::DeviceType::CPU));
}

MapAllocator::MapAllocator(std::string_view filename, int flags, size_t size)
  : MapAllocator(WITH_FD, filename, -1, flags, size)
{}


void MapAllocator::close() {
  if (closed_) {
    return;
  }
  closed_ = true;
  if (base_ptr_ == nullptr) {
    return;
  }
  if (flags_ & ALLOCATOR_MAPPED_KEEPFD) {
    if (::close(fd_) == -1) {
      TORCH_CHECK(false, "could not close file descriptor ", fd_, " :", c10::utils::str_error(errno), " (", errno, ")" );
    }
  }

  if (munmap(base_ptr_, size_)) {
    TORCH_CHECK(false, "could not unmap the shared memory file: ", c10::utils::str_error(errno), " (", errno, ")");
  }

  if (!(flags_ & (ALLOCATOR_MAPPED_FROMFD | ALLOCATOR_MAPPED_UNLINK))) {
    if (flags_ & ALLOCATOR_MAPPED_SHAREDMEM) {
#ifdef HAVE_SHM_UNLINK
      if (shm_unlink(filename_.c_str()) == -1) {
        TORCH_CHECK(false, "could not unlink the shared memory file ", filename_, " : ", c10::utils::str_error(errno), " (", errno, ")");
      }
#else
      TORCH_CHECK(false, "could not unlink the shared memory file ", filename_, ", shm_unlink not available on platform");
#endif
    }
  }
}

#else /* defined(HAVE_MMAP) */

MapAllocator::MapAllocator(std::string_view filename, int flags, size_t size) {
  TORCH_CHECK(false, "file mapping not supported on your system");
}

MapAllocator::MapAllocator(WithFd, std::string_view filename, int fd, int flags, size_t size) {
  TORCH_CHECK(false, "file mapping not supported on your system");
}

void MapAllocator::close() { }

#endif

#if defined(HAVE_MMAP) && defined(AT_ATOMIC_IPC_REFCOUNT)

RefcountedMapAllocatorArgCheck::RefcountedMapAllocatorArgCheck(int flags) {
  if (flags & ALLOCATOR_MAPPED_FROMFD) {
    TORCH_CHECK(false, "RefcountedMapAllocator doesn't support ALLOCATOR_MAPPED_FROMFD flag");
  }
  if (flags & ALLOCATOR_MAPPED_KEEPFD) {
    TORCH_CHECK(false, "RefcountedMapAllocator doesn't support ALLOCATOR_MAPPED_KEEPFD flag");
  }
  if (flags & ALLOCATOR_MAPPED_UNLINK) {
    TORCH_CHECK(false, "RefcountedMapAllocator doesn't support ALLOCATOR_MAPPED_UNLINK flag");
  }
  if (!(flags & ALLOCATOR_MAPPED_SHAREDMEM)) {
    TORCH_CHECK(false, "RefcountedMapAllocator requires ALLOCATOR_MAPPED_SHAREDMEM flag");
  }
}

RefcountedMapAllocator::RefcountedMapAllocator(const char *filename, int flags, size_t size)
  : RefcountedMapAllocatorArgCheck(flags)
  , MapAllocator(filename, flags, size + map_alloc_alignment) {

    initializeAlloc();
}
RefcountedMapAllocator::RefcountedMapAllocator(WithFd /*unused*/, const char *filename, int fd, int flags, size_t size)
  : RefcountedMapAllocatorArgCheck(flags)
  , MapAllocator(WITH_FD, filename, flags, fd, size + map_alloc_alignment) {

    initializeAlloc();
}

void RefcountedMapAllocator::initializeAlloc() {
  TORCH_CHECK(base_ptr_, "base_ptr_ is null");
  MapInfo *map_info = static_cast<MapInfo*>(base_ptr_);


  if (flags_ & ALLOCATOR_MAPPED_EXCLUSIVE) {
    new (&map_info->refcount) std::atomic<int>(1);
  } else {
    map_info->refcount++;
  }
}

void RefcountedMapAllocator::close() {
  if (closed_) {
    return;
  }
  closed_ = true;

  void* data = base_ptr_;


  MapInfo *info = static_cast<MapInfo*>(data);
  if (--info->refcount == 0) {
#ifdef HAVE_SHM_UNLINK
    if (shm_unlink(filename_.c_str()) == -1) {
      TORCH_CHECK(false, "could not unlink the shared memory file ", filename_);
    }
#else
    TORCH_CHECK(false, "could not unlink the shared memory file ", filename_, ", shm_unlink not available on platform");
#endif /* HAVE_SHM_UNLINK */
  }
  if (munmap(info, size_)) {
    TORCH_CHECK(false, "could not unmap the shared memory file ", filename_);
  }
}

void RefcountedMapAllocator::incref()
{
  MapInfo *map_info = static_cast<MapInfo*>(base_ptr_);
  ++map_info->refcount;
}

int RefcountedMapAllocator::decref()
{
  MapInfo *map_info = static_cast<MapInfo*>(base_ptr_);
  return --map_info->refcount == 0;
}

#else


RefcountedMapAllocatorArgCheck::RefcountedMapAllocatorArgCheck(int flags) {}

RefcountedMapAllocator::RefcountedMapAllocator(const char *filename, int flags, size_t size)
  : RefcountedMapAllocatorArgCheck(flags),
    MapAllocator(filename, flags, size + map_alloc_alignment)
{
  TORCH_CHECK(false, "refcounted file mapping not supported on your system");
}

RefcountedMapAllocator::RefcountedMapAllocator(WithFd, const char *filename, int fd, int flags, size_t size)
  : RefcountedMapAllocatorArgCheck(flags),
    MapAllocator(WITH_FD, filename, flags, fd, size + map_alloc_alignment)
{
  TORCH_CHECK(false, "refcounted file mapping not supported on your system");
}

void RefcountedMapAllocator::initializeAlloc() {}

void RefcountedMapAllocator::close() {}

#endif

static void deleteMapAllocator(void* ptr) {
  delete static_cast<MapAllocator*>(ptr);
}

static void deleteRefcountedMapAllocator(void* ptr) {
  delete static_cast<RefcountedMapAllocator*>(ptr);
}

MapAllocator* MapAllocator::fromDataPtr(const at::DataPtr& dptr) {
  return dptr.cast_context<MapAllocator>(&deleteMapAllocator);
}

RefcountedMapAllocator* RefcountedMapAllocator::fromDataPtr(const at::DataPtr& dptr) {
  return dptr.cast_context<RefcountedMapAllocator>(&deleteRefcountedMapAllocator);
}

at::DataPtr MapAllocator::makeDataPtr(std::string_view filename, int flags, size_t size, size_t* actual_size_out) {
  auto* context = new MapAllocator(filename, flags, size);
  if (actual_size_out) *actual_size_out = context->size();
  return {context->data(), context, &deleteMapAllocator, at::DeviceType::CPU};
}

at::DataPtr MapAllocator::makeDataPtr(WithFd /*unused*/, const char *filename, int fd, int flags, size_t size, size_t* actual_size_out) {
  auto* context = new MapAllocator(WITH_FD, filename, fd, flags, size);
  if (actual_size_out) *actual_size_out = context->size();
  return {context->data(), context, &deleteMapAllocator, at::DeviceType::CPU};
}

at::DataPtr RefcountedMapAllocator::makeDataPtr(const char *filename, int flags, size_t size, size_t* actual_size_out) {
  auto* context = new RefcountedMapAllocator(filename, flags, size);
  if (actual_size_out) *actual_size_out = context->size() - map_alloc_alignment;
  return {context->data(), context, &deleteRefcountedMapAllocator, at::DeviceType::CPU};
}

at::DataPtr RefcountedMapAllocator::makeDataPtr(WithFd /*unused*/, const char *filename, int fd, int flags, size_t size, size_t* actual_size_out) {
  auto* context = new RefcountedMapAllocator(WITH_FD, filename, fd, flags, size);
  if (actual_size_out) *actual_size_out = context->size() - map_alloc_alignment;
  return {context->data(), context, &deleteRefcountedMapAllocator, at::DeviceType::CPU};
}

void* RefcountedMapAllocator::data() const {
  return static_cast<void*>(static_cast<char*>(base_ptr_) + map_alloc_alignment);
}

MapAllocator::~MapAllocator() {
  MapAllocator::close();
  c10::reportMemoryUsageToProfiler(base_ptr_, -size_, 0, 0, c10::Device(c10::DeviceType::CPU));
}

}  // namespace at
