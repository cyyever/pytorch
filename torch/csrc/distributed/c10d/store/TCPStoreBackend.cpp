
#include <c10/util/irange.h>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <utility>
#include <vector>

#include <c10/util/thread_name.h>
#include <torch/csrc/distributed/c10d/logging.h>
#include <torch/csrc/distributed/c10d/store/TCPStoreBackend.hpp>

#include <torch/csrc/distributed/c10d/socket.h>

namespace c10d::detail {

// Background thread parent class methods
BackgroundThread::BackgroundThread() = default;

BackgroundThread::~BackgroundThread() = default;

// WARNING:
// Since we rely on the subclass for the daemon thread clean-up, we cannot
// destruct our member variables in the destructor. The subclass must call
// dispose() in its own destructor.
void BackgroundThread::dispose() {
  // Stop the run
  stop();
  // Join the thread
  daemonThread_.join();
}

void BackgroundThread::start() {
  daemonThread_ = std::thread{&BackgroundThread::run, this};
  is_running_.store(true);
}

} // namespace c10d::detail
