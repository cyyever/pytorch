#include <gtest/gtest.h>

#include <ATen/DeviceAccelerator.h>
#include <ATen/accelerator/Graph.h>

TEST(AcceleratorGraphTest, graphRegistration) {
  const auto device_type = at::accelerator::getAccelerator(false);
  if (!device_type.has_value()) {
    GTEST_SKIP() << "No accelerator is available";
  }

  ASSERT_TRUE(at::has_graph_impl(*device_type));
  EXPECT_NO_THROW({
    at::accelerator::Graph graph(true);
    graph.enable_debug_mode();
  });
}
