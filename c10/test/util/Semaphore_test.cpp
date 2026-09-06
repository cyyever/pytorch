#include <c10/util/Semaphore.h>
#include <c10/util/irange.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <semaphore>
#include <thread>

using namespace ::testing;

TEST(SemaphoreTest, TestConcurrency) {
  auto num_threads = std::thread::hardware_concurrency();
  auto num_incr = 10000;

  c10::Semaphore sem;

  std::vector<std::thread> threads;
  for ([[maybe_unused]] const auto _ : c10::irange(num_threads)) {
    threads.emplace_back([num_incr = num_incr, &sem]() {
      for ([[maybe_unused]] const auto _ : c10::irange(num_incr)) {
        sem.release();
      }
      for ([[maybe_unused]] const auto _ : c10::irange(num_incr)) {
        sem.acquire();
      }
      sem.release(num_incr);
      for ([[maybe_unused]] const auto _ : c10::irange(num_incr)) {
        sem.acquire();
      }
    });
  }

  std::ranges::for_each(
      threads, [](std::thread& t) { t.join(); });

  EXPECT_FALSE(sem.tryAcquire());
}

// c10::Semaphore refuses std::counting_semaphore on libstdc++ (see the
// __GLIBCXX__ term in Semaphore.h) because of gcc bug 98033: _M_release only
// notifies when the counter was zero, so a waiter that failed its CAS and saw
// zero could sleep through a release. This drives that shape directly and says
// whether the standard library in use still drops one.
//
// The waiters block in acquire(), so a lost wakeup would hang rather than
// fail. They are detached and hold the round's state through shared_ptr, which
// lets the deadline below report a failure while a stranded waiter keeps what
// it is parked on alive.
//
// The rendezvous is a counter and a yield rather than std::latch on purpose:
// this asks about a libstdc++ synchronization primitive, so it leans on as few
// of them as it can.
TEST(SemaphoreTest, StlSemaphoreWakesEveryWaiter) {
  constexpr int kRounds = 50;
  const unsigned num_waiters =
      std::max(4u, std::thread::hardware_concurrency());

  for ([[maybe_unused]] const auto round : c10::irange(kRounds)) {
    auto sem = std::make_shared<std::counting_semaphore<>>(0);
    auto woken = std::make_shared<std::atomic<unsigned>>(0);
    auto ready = std::make_shared<std::atomic<unsigned>>(0);

    for ([[maybe_unused]] const auto _ : c10::irange(num_waiters)) {
      std::thread([sem, woken, ready] {
        ready->fetch_add(1, std::memory_order_release);
        sem->acquire();
        woken->fetch_add(1, std::memory_order_release);
      }).detach();
    }

    // The counter says every waiter reached acquire(); the pause gives them
    // time to park in it, which is the state the releases below need to find.
    while (ready->load(std::memory_order_acquire) < num_waiters) {
      std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    // Release one at a time. Waiters that wake early consume permits as fast as
    // they appear, so later releases keep finding a positive counter -- the
    // case _M_release declines to notify.
    for ([[maybe_unused]] const auto _ : c10::irange(num_waiters)) {
      sem->release();
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (woken->load(std::memory_order_acquire) < num_waiters &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    ASSERT_EQ(woken->load(std::memory_order_acquire), num_waiters)
        << "a waiter was not woken after every permit was released; "
        << "std::counting_semaphore dropped a wakeup on round " << round;
  }
}
