#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <c10/util/Logging.h>
#include "c10/util/irange.h"
#include "caffe2/serialize/file_adapter.h"
#include "caffe2/serialize/in_memory_adapter.h"
#include "caffe2/serialize/inline_container.h"

namespace caffe2 {
namespace serialize {
namespace {

TEST(PyTorchStreamWriterAndReader, SaveAndLoad) {
  int64_t kFieldAlignment = 4096L;

  std::ostringstream oss;
  // write records through writers
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 127> data1;
  // Inplace memory buffer
  std::vector<uint8_t> buf(data1.size());

  for (auto i : c10::irange(data1.size())) {
    data1[i] = data1.size() - i;
  }
  writer.writeRecord("key1", data1.data(), data1.size());

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 64> data2;
  for (auto i : c10::irange(data2.size())) {
    data2[i] = data2.size() - i;
  }
  writer.writeRecord("key2", data2.data(), data2.size());

  const std::unordered_set<std::string>& written_records =
      writer.getAllWrittenRecords();
  ASSERT_EQ(written_records.size(), 2);
  ASSERT_EQ(written_records.count("key1"), 1);
  ASSERT_EQ(written_records.count("key2"), 1);

  writer.writeEndOfFile();
  ASSERT_EQ(written_records.count(kSerializationIdRecordName), 1);

  std::string the_file = std::move(oss).str();
  const char* file_name = "output.zip";
  std::ofstream foo(file_name);
  foo.write(the_file.c_str(), the_file.size());
  foo.close();

  std::istringstream iss(the_file);

  // read records through readers
  PyTorchStreamReader reader(&iss);
  ASSERT_TRUE(reader.hasRecord("key1"));
  ASSERT_TRUE(reader.hasRecord("key2"));
  ASSERT_FALSE(reader.hasRecord("key2000"));
  at::DataPtr data_ptr;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t size;
  std::tie(data_ptr, size) = reader.getRecord("key1");
  size_t off1 = reader.getRecordOffset("key1");
  ASSERT_EQ(size, data1.size());
  ASSERT_EQ(memcmp(data_ptr.get(), data1.data(), data1.size()), 0);
  ASSERT_EQ(memcmp(the_file.c_str() + off1, data1.data(), data1.size()), 0);
  ASSERT_EQ(off1 % kFieldAlignment, 0);
  // inplace getRecord() test
  std::vector<uint8_t> dst(size);
  size_t ret = reader.getRecord("key1", dst.data(), size);
  ASSERT_EQ(ret, size);
  ASSERT_EQ(memcmp(dst.data(), data1.data(), size), 0);
  // chunked getRecord() test
  ret = reader.getRecord(
      "key1",
      dst.data(),
      size,
      3,
      buf.data(),
      [](void* dst, const void* src, size_t n) { memcpy(dst, src, n); });
  ASSERT_EQ(ret, size);
  ASSERT_EQ(memcmp(dst.data(), data1.data(), size), 0);

  std::tie(data_ptr, size) = reader.getRecord("key2");
  size_t off2 = reader.getRecordOffset("key2");
  ASSERT_EQ(off2 % kFieldAlignment, 0);

  ASSERT_EQ(size, data2.size());
  ASSERT_EQ(memcmp(data_ptr.get(), data2.data(), data2.size()), 0);
  ASSERT_EQ(memcmp(the_file.c_str() + off2, data2.data(), data2.size()), 0);
  // inplace getRecord() test
  dst.resize(size);
  ret = reader.getRecord("key2", dst.data(), size);
  ASSERT_EQ(ret, size);
  ASSERT_EQ(memcmp(dst.data(), data2.data(), size), 0);
  // chunked getRecord() test
  ret = reader.getRecord(
      "key2",
      dst.data(),
      size,
      3,
      buf.data(),
      [](void* dst, const void* src, size_t n) { memcpy(dst, src, n); });
  ASSERT_EQ(ret, size);
  ASSERT_EQ(memcmp(dst.data(), data2.data(), size), 0);
  // clean up
  remove(file_name);
}

TEST(PyTorchStreamWriterAndReader, LoadFromFile) {
  std::ostringstream oss;
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });

  std::array<char, 127> data1{};
  std::array<char, 64> data2{};
  for (auto i : c10::irange(data1.size())) {
    data1[i] = data1.size() - i;
  }
  writer.writeRecord("key1", data1.data(), data1.size());
  for (auto i : c10::irange(data2.size())) {
    data2[i] = data2.size() - i;
  }
  writer.writeRecord("key2", data2.data(), data2.size());
  writer.writeEndOfFile();

  const std::string the_file = std::move(oss).str();
  const char* file_name = "output_from_file.zip";
  {
    std::ofstream out(file_name, std::ios::binary);
    out.write(the_file.c_str(), the_file.size());
  }

  // Read through the file path, which is the one that goes via FileAdapter.
  PyTorchStreamReader reader(file_name);
  at::DataPtr data_ptr;
  size_t size = 0;

  std::tie(data_ptr, size) = reader.getRecord("key1");
  EXPECT_EQ(size, data1.size());
  EXPECT_EQ(memcmp(data_ptr.get(), data1.data(), data1.size()), 0);

  // Out of order, and back again: each read carries its own offset, so the
  // record read before it cannot leave the descriptor somewhere else.
  std::tie(data_ptr, size) = reader.getRecord("key2");
  EXPECT_EQ(size, data2.size());
  EXPECT_EQ(memcmp(data_ptr.get(), data2.data(), data2.size()), 0);

  std::tie(data_ptr, size) = reader.getRecord("key1");
  EXPECT_EQ(memcmp(data_ptr.get(), data1.data(), data1.size()), 0);

  // In-place variant over the same file.
  std::vector<uint8_t> dst(data2.size());
  EXPECT_EQ(reader.getRecord("key2", dst.data(), dst.size()), data2.size());
  EXPECT_EQ(memcmp(dst.data(), data2.data(), data2.size()), 0);

  remove(file_name);
}

TEST(PytorchStreamWriterAndReader, GetNonexistentRecordThrows) {
  std::ostringstream oss;
  // write records through writers
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 127> data1;

  // Inplace memory buffer
  std::vector<uint8_t> buf;

  for (auto i : c10::irange(data1.size())) {
    data1[i] = data1.size() - i;
  }
  writer.writeRecord("key1", data1.data(), data1.size());

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 64> data2;
  for (auto i : c10::irange(data2.size())) {
    data2[i] = data2.size() - i;
  }
  writer.writeRecord("key2", data2.data(), data2.size());

  const std::unordered_set<std::string>& written_records =
      writer.getAllWrittenRecords();
  ASSERT_EQ(written_records.size(), 2);
  ASSERT_EQ(written_records.count("key1"), 1);
  ASSERT_EQ(written_records.count("key2"), 1);

  writer.writeEndOfFile();
  ASSERT_EQ(written_records.count(kSerializationIdRecordName), 1);

  std::string the_file = std::move(oss).str();
  const char* file_name = "output2.zip";
  std::ofstream foo(file_name);
  foo.write(the_file.c_str(), the_file.size());
  foo.close();

  std::istringstream iss(the_file);

  // read records through readers
  PyTorchStreamReader reader(&iss);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(reader.getRecord("key3"), c10::Error);
  std::vector<uint8_t> dst(data1.size());
  EXPECT_THROW(reader.getRecord("key3", dst.data(), data1.size()), c10::Error);
  EXPECT_THROW(
      reader.getRecord(
          "key3",
          dst.data(),
          data1.size(),
          3,
          buf.data(),
          [](void* dst, const void* src, size_t n) { memcpy(dst, src, n); }),
      c10::Error);

  // Reader should still work after throwing
  EXPECT_TRUE(reader.hasRecord("key1"));
  // clean up
  remove(file_name);
}

TEST(PytorchStreamWriterAndReader, SkipDebugRecords) {
  std::ostringstream oss;
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 127> data1;
  // Inplace memory buffer
  std::vector<uint8_t> buf(data1.size());

  for (auto i : c10::irange(data1.size())) {
    data1[i] = data1.size() - i;
  }
  writer.writeRecord("key1.debug_pkl", data1.data(), data1.size());

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 64> data2;
  for (auto i : c10::irange(data2.size())) {
    data2[i] = data2.size() - i;
  }
  writer.writeRecord("key2.debug_pkl", data2.data(), data2.size());

  const std::unordered_set<std::string>& written_records =
      writer.getAllWrittenRecords();
  ASSERT_EQ(written_records.size(), 2);
  ASSERT_EQ(written_records.count("key1.debug_pkl"), 1);
  ASSERT_EQ(written_records.count("key2.debug_pkl"), 1);
  writer.writeEndOfFile();
  ASSERT_EQ(written_records.count(kSerializationIdRecordName), 1);

  std::string the_file = std::move(oss).str();
  const char* file_name = "output3.zip";
  std::ofstream foo(file_name);
  foo.write(the_file.c_str(), the_file.size());
  foo.close();

  std::istringstream iss(the_file);

  // read records through readers
  PyTorchStreamReader reader(&iss);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)

  reader.setShouldLoadDebugSymbol(false);
  EXPECT_FALSE(reader.hasRecord("key1.debug_pkl"));
  at::DataPtr ptr;
  size_t size;
  std::tie(ptr, size) = reader.getRecord("key1.debug_pkl");
  EXPECT_EQ(size, 0);
  std::vector<uint8_t> dst(data1.size());
  size_t ret = reader.getRecord("key1.debug_pkl", dst.data(), data1.size());
  EXPECT_EQ(ret, 0);
  ret = reader.getRecord(
      "key1.debug_pkl",
      dst.data(),
      data1.size(),
      3,
      buf.data(),
      [](void* dst, const void* src, size_t n) { memcpy(dst, src, n); });
  EXPECT_EQ(ret, 0);
  // clean up
  remove(file_name);
}

TEST(PytorchStreamWriterAndReader, ValidSerializationId) {
  std::ostringstream oss;
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-avoid-magic-numbers)
  std::array<char, 127> data1;

  for (auto i : c10::irange(data1.size())) {
    data1[i] = data1.size() - i;
  }
  writer.writeRecord("key1.debug_pkl", data1.data(), data1.size());
  writer.writeEndOfFile();
  auto writer_serialization_id = writer.serializationId();

  std::string the_file = oss.str();

  std::istringstream iss(the_file);

  // read records through readers
  PyTorchStreamReader reader(&iss);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)

  EXPECT_EQ(reader.serializationId(), writer_serialization_id);

  // write a second time
  PyTorchStreamWriter writer2([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });
  writer2.writeRecord("key1.debug_pkl", data1.data(), data1.size());
  writer2.writeEndOfFile();
  auto writer2_serialization_id = writer2.serializationId();

  EXPECT_EQ(writer_serialization_id, writer2_serialization_id);
}

TEST(PytorchStreamWriterAndReader, SkipDuplicateSerializationIdRecords) {
  std::ostringstream oss;
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });

  std::string dup_serialization_id = "dup-serialization-id";
  writer.writeRecord(
      kSerializationIdRecordName,
      dup_serialization_id.c_str(),
      dup_serialization_id.size());

  const std::unordered_set<std::string>& written_records =
      writer.getAllWrittenRecords();
  ASSERT_EQ(written_records.size(), 0);
  writer.writeEndOfFile();
  ASSERT_EQ(written_records.count(kSerializationIdRecordName), 1);
  auto writer_serialization_id = writer.serializationId();

  std::string the_file = std::move(oss).str();
  const char* file_name = "output4.zip";
  std::ofstream foo(file_name);
  foo.write(the_file.c_str(), the_file.size());
  foo.close();

  std::istringstream iss(the_file);

  // read records through readers
  PyTorchStreamReader reader(&iss);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)

  EXPECT_EQ(reader.serializationId(), writer_serialization_id);
  // clean up
  remove(file_name);
}

TEST(PytorchStreamWriterAndReader, LogAPIUsageMetadata) {
  std::map<std::string, std::map<std::string, std::string>> logs;

  SetAPIUsageMetadataLogger(
      [&](const std::string& context,
          const std::map<std::string, std::string>& metadata_map) {
        logs.insert({context, metadata_map});
      });
  std::ostringstream oss;
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });
  writer.writeEndOfFile();

  std::istringstream iss(oss.str());
  // read records through readers
  PyTorchStreamReader reader(&iss);

  ASSERT_EQ(logs.size(), 2);
  std::map<std::string, std::map<std::string, std::string>> expected_logs = {
      {"pytorch.stream.writer.metadata",
       {{"serialization_id", writer.serializationId()},
        {"file_name", "archive"},
        {"file_size", str(std::move(oss).str().length())}}},
      {"pytorch.stream.reader.metadata",
       {{"serialization_id", writer.serializationId()},
        {"file_name", "archive"},
        {"file_size", str(iss.str().length())}}}};
  ASSERT_EQ(expected_logs, logs);

  // reset logger
  SetAPIUsageMetadataLogger(
      [&](const std::string& context,
          const std::map<std::string, std::string>& metadata_map) {});
}





class ChunkRecordIteratorTest : public ::testing::TestWithParam<int64_t> {};
INSTANTIATE_TEST_SUITE_P(
    ChunkRecordIteratorTestGroup,
    ChunkRecordIteratorTest,
    testing::Values(100, 150, 1010));

TEST_P(ChunkRecordIteratorTest, ChunkRead) {
  auto chunkSize = GetParam();
  std::string zipFileName =
      "output_chunk_" + std::to_string(chunkSize) + ".zip";
  const char* fileName = zipFileName.c_str();
  const std::string recordName = "key1";
  const size_t tensorDataSizeInBytes = 1000;

  // write records through writers
  std::ostringstream oss(std::ios::binary);
  PyTorchStreamWriter writer([&](const void* b, size_t n) -> size_t {
    oss.write(static_cast<const char*>(b), n);
    return oss ? n : 0;
  });

  auto tensorData = std::vector<uint8_t>(tensorDataSizeInBytes, 1);
  auto dataPtr = tensorData.data();
  writer.writeRecord(recordName, dataPtr, tensorDataSizeInBytes);
  const std::unordered_set<std::string>& written_records =
      writer.getAllWrittenRecords();
  ASSERT_EQ(written_records.size(), 1);
  ASSERT_EQ(written_records.count(recordName), 1);
  writer.writeEndOfFile();
  ASSERT_EQ(written_records.count(kSerializationIdRecordName), 1);

  std::string the_file = std::move(oss).str();
  std::ofstream foo(fileName, std::ios::binary);
  foo.write(the_file.c_str(), the_file.size());
  foo.close();
  LOG(INFO) << "Finished saving tensor into zip file " << fileName;

  LOG(INFO) << "Testing chunk size " << chunkSize;
  PyTorchStreamReader reader(fileName);
  ASSERT_TRUE(reader.hasRecord(recordName));
  auto chunkIterator = reader.createChunkReaderIter(
      recordName, tensorDataSizeInBytes, chunkSize);
  std::vector<uint8_t> buffer(chunkSize);
  size_t totalReadSize = 0;
  while (auto readSize = chunkIterator.next(buffer.data())) {
    auto expectedData = std::vector<uint8_t>(readSize, 1);
    ASSERT_EQ(memcmp(expectedData.data(), buffer.data(), readSize), 0);
    totalReadSize += readSize;
  }
  ASSERT_EQ(totalReadSize, tensorDataSizeInBytes);
  // clean up
  remove(fileName);
}

TEST(MemoryReadAdapterTest, ClampsReadsToBufferSize) {
  constexpr size_t kBufSize = 64;
  std::vector<uint8_t> buf(kBufSize, 0xAA);
  MemoryReadAdapter adapter(buf.data(), static_cast<off_t>(kBufSize));
  ASSERT_EQ(adapter.size(), kBufSize);

  std::array<uint8_t, 32> out{};

  // pos straddles end: read starts at 48, only 16 bytes available.
  out.fill(0);
  EXPECT_EQ(adapter.read(48, out.data(), out.size()), 16u);
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(out[i], 0xAA);
  }

  // pos at end: zero bytes available.
  out.fill(0);
  EXPECT_EQ(adapter.read(kBufSize, out.data(), out.size()), 0u);

  // pos past end: still zero; no OOB memcpy.
  out.fill(0);
  EXPECT_EQ(adapter.read(kBufSize + 1024, out.data(), out.size()), 0u);

  // In-bounds read returns full count.
  out.fill(0);
  EXPECT_EQ(adapter.read(0, out.data(), out.size()), out.size());
}

TEST(FileAdapterTest, ReadsAtOffsetsIndependently) {
  constexpr size_t kSize = 100000;
  const char* file_name = "file_adapter.bin";
  std::vector<uint8_t> ref(kSize);
  for (const auto i : c10::irange(kSize)) {
    ref[i] = static_cast<uint8_t>(i * 31 + 7);
  }
  {
    std::ofstream out(file_name, std::ios::binary);
    out.write(reinterpret_cast<const char*>(ref.data()), ref.size());
  }

  FileAdapter adapter(file_name);
  ASSERT_EQ(adapter.size(), kSize);
  std::vector<uint8_t> buf(kSize);

  EXPECT_EQ(adapter.read(0, buf.data(), kSize), kSize);
  EXPECT_EQ(memcmp(buf.data(), ref.data(), kSize), 0);

  EXPECT_EQ(adapter.read(12345, buf.data(), 5000), 5000u);
  EXPECT_EQ(memcmp(buf.data(), ref.data() + 12345, 5000), 0);

  // pos straddles end: only the tail is available.
  EXPECT_EQ(adapter.read(kSize - 10, buf.data(), 999), 10u);
  EXPECT_EQ(memcmp(buf.data(), ref.data() + kSize - 10, 10), 0);

  // pos past end: zero bytes, no read past the file.
  EXPECT_EQ(adapter.read(kSize + 500, buf.data(), 10), 0u);

  // Reads carry their own offset, so one does not move the next. This is what
  // the fseeko+fread implementation could not do on a single descriptor.
  EXPECT_EQ(adapter.read(0, buf.data(), 4), 4u);
  EXPECT_EQ(adapter.read(50000, buf.data() + 4, 4), 4u);
  EXPECT_EQ(memcmp(buf.data(), ref.data(), 4), 0);
  EXPECT_EQ(memcmp(buf.data() + 4, ref.data() + 50000, 4), 0);

  remove(file_name);
}

} // namespace
} // namespace serialize
} // namespace caffe2
