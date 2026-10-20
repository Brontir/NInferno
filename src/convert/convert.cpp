#include "convert/convert.h"

#include "convert/files.h"
#include "core/clock.h"
#include "core/hex.h"

#include "artifact/formats.h"
#include "convert/consume.h"
#include "convert/dense.h"
#include "convert/gguf_model.h"
#include "convert/drafters.h"
#include "convert/formats.h"
#include "convert/gemma4.h"
#include "convert/ggml_quants.h"
#include "convert/qwen3_5.h"
#include "convert/numerics.h"
#include "convert/planner.h"
#include "convert/writer.h"
#include "core/ggml_frag.h"

#include "core/host_memory.h"
#include "core/sys.h"
#include "core/fnv.h"
#include "core/units.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_set>

namespace ninferno::convert {

// The ranking a proposal head is restricted by when --ranking names none
// (tools/freq_corpus/fixtures/ranking, installed with the registry's rankings).
constexpr const char* kDefaultRanking = "ranking.train.counts.i64";
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kTaskBytes      = kMiB; // per task: bounds scratch memory per thread
constexpr std::uint64_t kReadAheadBytes = 256ULL << 20;
constexpr std::size_t kUnmapTasks       = 64;

GroupedFormat grouped_format(QType format) {
    switch (format) {
    case QType::Q4_G64_FP16:
        return kQ4;
    case QType::Q5_G64_FP16:
        return kQ5;
    case QType::Q6_G64_FP16:
        return kQ6;
    case QType::Q8_G32_FP16:
        return kQ8;
    default:
        throw std::logic_error("not a grouped integer format");
    }
}

std::size_t direct_bytes(QType format) { return format == QType::BF16 ? 2 : 4; }

// cast_direct: torch's `.to(dtype)` from the source's value dtype.
void cast_words(ValueType from, const std::byte* in, std::uint64_t count, QType to,
                std::byte* out) {
    if (to == QType::INT32) {
        if (from == ValueType::I32) {
            std::memcpy(out, in, count * 4);
            return;
        }
        if (from != ValueType::I64) {
            throw std::runtime_error("int32 conversion from a non-integer source is not native");
        }
        for (std::uint64_t i = 0; i < count; ++i) {
            std::int64_t v;
            std::memcpy(&v, in + 8 * i, 8);
            if (v < -(std::int64_t(1) << 31) || v > (std::int64_t(1) << 31) - 1) {
                throw std::runtime_error(
                    "int32 conversion source is outside the representable range");
            }
            const auto w = static_cast<std::int32_t>(v);
            std::memcpy(out + 4 * i, &w, 4);
        }
        return;
    }
    if (to == QType::BF16 && from == ValueType::BF16) {
        std::memcpy(out, in, count * 2);
        return;
    }
    if (to == QType::FP32 && from == ValueType::F32) {
        std::memcpy(out, in, count * 4);
        return;
    }
    for (std::uint64_t i = 0; i < count; ++i) {
        float value = 0.0F;
        switch (from) {
        case ValueType::BF16: {
            std::uint16_t w;
            std::memcpy(&w, in + 2 * i, 2);
            value = bf16_to_f32(w);
            break;
        }
        case ValueType::F16: {
            std::uint16_t w;
            std::memcpy(&w, in + 2 * i, 2);
            value = f16_to_f32(w);
            break;
        }
        case ValueType::F32:
            std::memcpy(&value, in + 4 * i, 4);
            break;
        case ValueType::F64: {
            double v;
            std::memcpy(&v, in + 8 * i, 8);
            value = static_cast<float>(v);
            break;
        }
        case ValueType::I32: {
            std::int32_t v;
            std::memcpy(&v, in + 4 * i, 4);
            value = static_cast<float>(v);
            break;
        }
        case ValueType::I64: {
            std::int64_t v;
            std::memcpy(&v, in + 8 * i, 8);
            value = static_cast<float>(v);
            break;
        }
        case ValueType::I8:
            value = static_cast<float>(static_cast<std::int8_t>(in[i]));
            break;
        case ValueType::U8:
            value = static_cast<float>(static_cast<std::uint8_t>(in[i]));
            break;
        }
        if (to == QType::BF16) {
            const auto w = f32_to_bf16(value);
            std::memcpy(out + 2 * i, &w, 2);
        } else {
            std::memcpy(out + 4 * i, &value, 4);
        }
    }
}

struct Task {
    std::size_t job      = 0;
    std::size_t position = 0; // of the job in the production order
    std::uint64_t begin  = 0; // rows for matrix methods, elements for cast_direct
    std::uint64_t end    = 0;
};

// Tasks for the given jobs, in the given order.
std::vector<Task> make_tasks(const Plan& plan, const std::vector<std::uint32_t>& order) {
    std::vector<Task> tasks;
    for (std::size_t position = 0; position < order.size(); ++position) {
        const std::size_t j = order[position];
        const auto& job     = plan.weights[j];
        switch (job.method) {
        case Method::CastDirect: {
            const auto elements = shape_elements(job.shape);
            const auto step     = std::max<std::uint64_t>(1, kTaskBytes / direct_bytes(job.format));
            for (std::uint64_t e = 0; e < elements; e += step) {
                tasks.push_back({j, position, e, std::min(elements, e + step)});
            }
            break;
        }
        case Method::Fp8RowMaxabs: {
            const auto rows = job.shape[0];
            const auto step = std::max<std::uint64_t>(1, kTaskBytes / (job.shape[1] * 4));
            for (std::uint64_t r = 0; r < rows; r += step) {
                tasks.push_back({j, position, r, std::min(rows, r + step)});
            }
            break;
        }
        case Method::GroupedAbsmax: {
            const auto rows = job.shape[0];
            const auto step =
                std::max<std::uint64_t>(1, kTaskBytes / (job.geometry.padded_columns * 4));
            for (std::uint64_t r = 0; r < rows; r += step) {
                tasks.push_back({j, position, r, std::min(rows, r + step)});
            }
            break;
        }
        case Method::ImportEncoded: {
            const auto k    = job.shape[1];
            const auto step = std::max<std::uint64_t>(128, kTaskBytes / (k / 2) / 128 * 128);
            // A stacked plane restarts its row blocks at every source boundary: one block, one
            // divisor.
            std::vector<std::uint64_t> bounds;
            if (job.divisors > 1) {
                std::uint64_t edge = 0;
                for (const auto& source : job.sources) {
                    edge += source->shape()[0];
                    bounds.push_back(edge);
                }
            } else {
                bounds.push_back(job.shape[0]);
            }
            std::uint64_t cursor = 0;
            for (const auto edge : bounds) {
                for (std::uint64_t r = cursor; r < edge; r += step) {
                    tasks.push_back({j, position, r, std::min(edge, r + step)});
                }
                cursor = edge;
            }
            break;
        }
        }
    }
    return tasks;
}

// Visit the (source, local rows) spans a parent row range covers.
template <class F>
void for_rows(const WeightJob& job, std::uint64_t first, std::uint64_t last, F&& visit) {
    std::uint64_t cursor = 0;
    for (const auto& source : job.sources) {
        const auto rows = source->shape()[0];
        const auto low  = std::max(first, cursor);
        const auto high = std::min(last, cursor + rows);
        if (low < high) { visit(*source, low - cursor, high - cursor, low - first); }
        cursor += rows;
    }
}

template <class F>
void for_elements(const WeightJob& job, std::uint64_t begin, std::uint64_t end, F&& visit) {
    std::uint64_t cursor = 0;
    for (const auto& source : job.sources) {
        const auto count = source->elements();
        const auto low   = std::max(begin, cursor);
        const auto high  = std::min(end, cursor + count);
        if (low < high) { visit(*source, low - cursor, high - cursor, low - begin); }
        cursor += count;
    }
}

// Produces weight objects on a thread pool, in a given order. With a Committer (a
// source-consuming run), objects are admitted in order while their output fits a byte window,
// and a commit thread makes each completed prefix durable before its sources are punched.
class Executor {
public:
    Executor(std::vector<const Checkpoint*> stores, const Plan& plan, ArtifactWriter& writer,
             std::vector<std::uint32_t> order, unsigned threads, bool keep_cache,
             Committer* committer = nullptr, std::uint64_t window = 0)
        : stores_(std::move(stores)), plan_(plan), writer_(writer), order_(std::move(order)),
          tasks_(make_tasks(plan, order_)),
          threads_(threads ? threads : std::clamp(std::thread::hardware_concurrency(), 1U, 16U)),
          keep_cache_(keep_cache), committer_(committer), window_(window),
          remaining_(plan.weights.size()) {
        for (const auto& task : tasks_) { ++remaining_[task.job]; }
        // A checkpoint tensor read by two parents (Gemma's tied table is both the embedding and
        // the head) stays cached until the end; every other page is dropped after its read.
        std::map<const TensorInfo*, int> readers;
        for (const auto j : order_) {
            std::set<const TensorInfo*> mine;
            for (const auto& source : plan.weights[j].sources) {
                std::vector<const TensorInfo*> tensors;
                source->tensors(*stores_.front(), tensors);
                mine.insert(tensors.begin(), tensors.end());
            }
            for (const auto* tensor : mine) { ++readers[tensor]; }
        }
        for (const auto& [tensor, count] : readers) {
            if (count > 1) { shared_.insert(tensor); }
        }
    }

    void run() {
        // Read-ahead window, in tasks: each picked task schedules the one `ahead_` places later,
        // so about ahead_ tasks' checkpoint bytes are in flight at any time.
        ahead_ = std::max<std::size_t>(threads_, kReadAheadBytes / kTaskBytes);
        {
            ReadLog ahead;
            for (std::size_t i = 0; i < std::min(ahead_, tasks_.size()); ++i) {
                prefetch(i, ahead);
            }
        }
        std::thread committing;
        if (committer_) {
            committing = std::thread([this] { commit_loop(); });
        }
        std::vector<std::thread> workers;
        for (unsigned t = 0; t < threads_; ++t) {
            workers.emplace_back([this] { work(); });
        }
        for (auto& worker : workers) { worker.join(); }
        {
            std::lock_guard lock(state_mutex_);
            workers_done_ = true;
        }
        state_changed_.notify_all();
        if (committing.joinable()) { committing.join(); }
        if (error_) { std::rethrow_exception(error_); }
    }

    [[nodiscard]] std::size_t task_count() const noexcept { return tasks_.size(); }

    [[nodiscard]] unsigned threads() const noexcept { return threads_; }

private:
    struct Scratch {
        std::vector<float> values;
        std::vector<std::uint8_t> base;
        std::vector<std::uint8_t> high;
        std::vector<std::uint16_t> scales;
        std::vector<std::byte> raw;
        std::vector<std::byte> out;
        std::vector<std::uint8_t> swizzled;
        EncodedRows encoded;
        ReadLog log;
        ReadLog ahead;
    };

    void fail() {
        std::lock_guard lock(error_mutex_);
        if (!error_) { error_ = std::current_exception(); }
        failed_ = true;
        state_changed_.notify_all();
    }

    void work() {
        Scratch scratch;
        while (!failed_.load(std::memory_order_relaxed)) {
            const auto index = next_.fetch_add(1);
            if (index >= tasks_.size()) { return; }
            try {
                const auto& task = tasks_[index];
                if (committer_ && !admit(task.position)) { return; }
                prefetch(index + ahead_, scratch.ahead);
                run(task, scratch);
                release(scratch.log);
                // Pages two tasks share stay mapped after each release; drop them periodically.
                if ((index + 1) % kUnmapTasks == 0) {
                    for (const auto* store : stores_) { store->unmap_all(); }
                }
                if (--remaining_[task.job] == 0) { complete(task.job); }
            } catch (...) { fail(); }
        }
    }

    // Every task of an object is written; the object's divisor words close it.
    void complete(std::size_t j) {
        const auto& job = plan_.weights[j];
        for (std::size_t i = 0; i < job.divisor_words.size(); ++i) {
            emit(job.offset + job.geometry.divisor_offset + 4 * i, &job.divisor_words[i], 4);
        }
        if (job.order) {
            emit(job.offset + job.geometry.order_offset, job.order->data(),
                 job.order->size() * sizeof(std::uint32_t));
        }
        if (committer_) {
            committer_->complete(static_cast<std::uint32_t>(j));
            std::lock_guard lock(state_mutex_);
            ++completions_;
            state_changed_.notify_all();
        }
    }

    // Objects enter production in order, while the output of uncommitted objects fits the
    // window (an object larger than the window runs alone).
    bool admit(std::size_t position) {
        std::unique_lock lock(state_mutex_);
        while (position >= admitted_) {
            if (failed_) { return false; }
            const auto bytes = plan_.weights[order_[admitted_]].geometry.bytes;
            if (inflight_ == 0 || inflight_ + bytes <= window_) {
                inflight_ += bytes;
                ++admitted_;
                state_changed_.notify_all();
                continue;
            }
            state_changed_.wait(lock);
        }
        return !failed_;
    }

    void commit_loop() {
        try {
            std::size_t seen = 0;
            while (true) {
                {
                    std::unique_lock lock(state_mutex_);
                    state_changed_.wait(
                        lock, [&] { return completions_ != seen || failed_ || workers_done_; });
                    if (failed_) { return; }
                    seen = completions_;
                }
                const auto before = committer_->committed();
                if (committer_->commit()) {
                    std::uint64_t freed = 0;
                    for (auto p = before; p < committer_->committed(); ++p) {
                        freed += plan_.weights[order_[p]].geometry.bytes;
                    }
                    std::lock_guard lock(state_mutex_);
                    inflight_ -= freed;
                    state_changed_.notify_all();
                }
                if (committer_->done()) { return; }
                std::lock_guard lock(state_mutex_);
                if (workers_done_ && completions_ == seen) {
                    throw std::logic_error("objects completed out of reach of the commit order");
                }
            }
        } catch (...) { fail(); }
    }

    // Release what a task read, merging adjacent ranges of one tensor first: rows are logged one
    // at a time and a page is only released when a merged range covers it whole.
    void release(ReadLog& log) const {
        auto& ranges = log.ranges;
        std::sort(
            ranges.begin(), ranges.end(),
            [](const ReadLog::Range& a, const ReadLog::Range& b) { return a.begin < b.begin; });
        std::size_t out = 0;
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            if (out && ranges[out - 1].tensor == ranges[i].tensor &&
                ranges[out - 1].begin + ranges[out - 1].bytes >= ranges[i].begin) {
                auto& last = ranges[out - 1];
                const auto end =
                    std::max(last.begin + last.bytes, ranges[i].begin + ranges[i].bytes);
                last.bytes = static_cast<std::size_t>(end - last.begin);
            } else {
                ranges[out++] = ranges[i];
            }
        }
        ranges.resize(out);
        for (const auto& range : ranges) {
            // Each checkpoint ignores addresses outside its own shards.
            for (const auto* store : stores_) {
                store->release(range.begin, range.bytes,
                               !keep_cache_ && !shared_.contains(range.tensor));
            }
        }
        ranges.clear();
    }

    void prefetch(std::size_t index, ReadLog& ahead) const {
        if (index >= tasks_.size()) { return; }
        const auto& task = tasks_[index];
        const auto& job  = plan_.weights[task.job];
        ahead.ranges.clear();
        if (job.method == Method::CastDirect) {
            for_elements(job, task.begin, task.end,
                         [&](const Source& source, std::uint64_t low, std::uint64_t high,
                             std::uint64_t) { source.ranges(low, high, ahead); });
        } else {
            const auto k = job.shape[1];
            for_rows(job, task.begin, task.end,
                     [&](const Source& source, std::uint64_t low, std::uint64_t high,
                         std::uint64_t) { source.ranges(low * k, high * k, ahead); });
        }
        for (const auto& range : ahead.ranges) {
            for (const auto* store : stores_) { store->prefetch(range.begin, range.bytes); }
        }
    }

    void emit(std::uint64_t offset, const void* data, std::size_t bytes) {
        writer_.write(offset, data, bytes);
    }

    void run(const Task& task, Scratch& s) {
        const auto& job = plan_.weights[task.job];
        switch (job.method) {
        case Method::CastDirect:
            run_direct(job, task, s);
            return;
        case Method::GroupedAbsmax:
            run_grouped(job, task, s);
            return;
        case Method::Fp8RowMaxabs:
            run_fp8_maxabs(job, task, s);
            return;
        case Method::ImportEncoded:
            run_encoded(job, task, s);
            return;
        }
    }

    void run_direct(const WeightJob& job, const Task& task, Scratch& s) {
        const auto word  = direct_bytes(job.format);
        const auto count = task.end - task.begin;
        s.out.resize(count * word);
        for_elements(
            job, task.begin, task.end,
            [&](const Source& source, std::uint64_t low, std::uint64_t high, std::uint64_t at) {
                const auto type = source.value_type();
                s.raw.resize((high - low) * value_bytes(type));
                source.values_raw(low, high, s.raw.data(), &s.log);
                cast_words(type, s.raw.data(), high - low, job.format, s.out.data() + at * word);
            });
        emit(job.offset + task.begin * word, s.out.data(), s.out.size());
    }

    void run_grouped(const WeightJob& job, const Task& task, Scratch& s) {
        const auto& g    = job.geometry;
        const auto rows  = task.end - task.begin;
        const auto k     = job.shape[1];
        const auto k_pad = g.padded_columns;
        s.values.assign(rows * k_pad, 0.0F);
        for_rows(
            job, task.begin, task.end,
            [&](const Source& source, std::uint64_t low, std::uint64_t high, std::uint64_t at) {
                if (k == k_pad) {
                    source.values_f32(low * k, high * k, s.values.data() + at * k, &s.log);
                    return;
                }
                for (auto r = low; r < high; ++r) {
                    source.values_f32(r * k, (r + 1) * k, s.values.data() + (at + r - low) * k_pad,
                                      &s.log);
                }
            });
        s.base.resize(rows * g.code_bytes_per_row);
        s.high.resize(rows * g.high_bytes_per_row);
        s.scales.resize(rows * g.scale_bytes_per_row / 2);
        quantize_grouped_rows(grouped_format(job.format), s.values.data(), rows, k_pad,
                              s.base.data(), s.high.data(), s.scales.data());
        emit(job.offset + task.begin * g.code_bytes_per_row, s.base.data(), s.base.size());
        if (g.high_bytes_per_row) {
            emit(job.offset + g.high_offset + task.begin * g.high_bytes_per_row, s.high.data(),
                 s.high.size());
        }
        emit(job.offset + g.scale_offset + task.begin * g.scale_bytes_per_row, s.scales.data(),
             s.scales.size() * 2);
    }

    // fp8_row_maxabs: values rounded to BF16, quantized row by row to FP8 with a BF16 scale.
    void run_fp8_maxabs(const WeightJob& job, const Task& task, Scratch& s) {
        const auto& g   = job.geometry;
        const auto rows = task.end - task.begin;
        const auto k    = job.shape[1];
        s.values.resize(rows * k);
        for_rows(
            job, task.begin, task.end,
            [&](const Source& source, std::uint64_t low, std::uint64_t high, std::uint64_t at) {
                source.values_f32(low * k, high * k, s.values.data() + at * k, &s.log);
            });
        s.base.resize(rows * k);
        s.scales.resize(rows);
        quantize_fp8_rows(s.values.data(), rows, k, s.base.data(), s.scales.data());
        emit(job.offset + task.begin * k, s.base.data(), s.base.size());
        emit(job.offset + g.scale_offset + task.begin * 2, s.scales.data(), rows * 2);
    }

    // Row-scaled FP8: codes at row * K, BF16 row scales in the scale plane.
    void run_fp8(const WeightJob& job, const Task& task, Scratch& s) {
        const auto& g = job.geometry;
        const auto k  = job.shape[1];
        s.encoded.codes.clear();
        s.encoded.scales.clear();
        for_rows(job, task.begin, task.end,
                 [&](const Source& source, std::uint64_t low, std::uint64_t high, std::uint64_t) {
                     source.encoded_rows(low, high, s.encoded, &s.log);
                 });
        emit(job.offset + task.begin * k, s.encoded.codes.data(), s.encoded.codes.size());
        if (job.format == QType::FP8_E4M3FN_B128_F32) {
            // Block rows task.begin/128 onward (tasks start at multiples of 128).
            const auto block_rows = (task.end - task.begin + 127) / 128;
            if (task.begin % 128 || s.encoded.scales.size() != block_rows * ((k + 127) / 128) * 4) {
                throw std::runtime_error(job.id + ": FP8 block scales of the wrong size");
            }
            emit(job.offset + g.scale_offset + task.begin / 128 * ((k + 127) / 128) * 4,
                 s.encoded.scales.data(), s.encoded.scales.size());
            return;
        }
        emit(job.offset + g.scale_offset + task.begin * g.scale_bytes_per_row,
             s.encoded.scales.data(), s.encoded.scales.size());
    }

    // MXFP4: the checkpoint's codes and E8M0 exponents, both row-major, unchanged.
    void run_mxfp4(const WeightJob& job, const Task& task, Scratch& s) {
        const auto& g = job.geometry;
        const auto k  = job.shape[1];
        s.encoded.codes.clear();
        s.encoded.scales.clear();
        for_rows(job, task.begin, task.end,
                 [&](const Source& source, std::uint64_t low, std::uint64_t high, std::uint64_t) {
                     source.encoded_rows(low, high, s.encoded, &s.log);
                 });
        const auto rows = task.end - task.begin;
        if (s.encoded.codes.size() != rows * (k / 2) ||
            s.encoded.scales.size() != rows * (k / 32)) {
            throw std::runtime_error(job.id + ": MXFP4 rows of the wrong size");
        }
        for (const auto e : s.encoded.scales) {
            if (e == 0xFF) { throw std::runtime_error(job.id + ": E8M0 NaN exponent"); }
        }
        emit(job.offset + task.begin * (k / 2), s.encoded.codes.data(), s.encoded.codes.size());
        emit(job.offset + g.scale_offset + task.begin * (k / 32), s.encoded.scales.data(),
             s.encoded.scales.size());
    }

    // Groupwise Q8 words (GGUF Q8_0): int8 codes and binary16 group scales laid into the
    // row_split planes, the columns past K padded with zero codes under zero scales (what
    // grouped_absmax writes for an all-zero group).
    void run_q8(const WeightJob& job, const Task& task, Scratch& s) {
        const auto& g   = job.geometry;
        const auto rows = task.end - task.begin;
        const auto k    = job.shape[1];
        s.encoded.codes.clear();
        s.encoded.scales.clear();
        for_rows(job, task.begin, task.end,
                 [&](const Source& source, std::uint64_t low, std::uint64_t high, std::uint64_t) {
                     source.encoded_rows(low, high, s.encoded, &s.log);
                 });
        if (k % 32 || s.encoded.codes.size() != rows * k ||
            s.encoded.scales.size() != rows * (k / 32) * 2 || g.high_bytes_per_row ||
            g.code_bytes_per_row != g.padded_columns ||
            g.scale_bytes_per_row != g.padded_columns / 32 * 2) {
            throw std::runtime_error(job.id + ": Q8 rows of the wrong size");
        }
        s.base.assign(rows * g.code_bytes_per_row, 0);
        s.scales.assign(rows * g.scale_bytes_per_row / 2, 0);
        for (std::uint64_t r = 0; r < rows; ++r) {
            std::memcpy(s.base.data() + r * g.code_bytes_per_row, s.encoded.codes.data() + r * k,
                        k);
            std::memcpy(s.scales.data() + r * (g.scale_bytes_per_row / 2),
                        s.encoded.scales.data() + r * (k / 32) * 2, (k / 32) * 2);
        }
        emit(job.offset + task.begin * g.code_bytes_per_row, s.base.data(), s.base.size());
        emit(job.offset + g.scale_offset + task.begin * g.scale_bytes_per_row, s.scales.data(),
             s.scales.size() * 2);
    }

    // Row-major planes of whole rows: codes, then (high plane) zero points or scales, then
    // scales, each at its row's bytes per the geometry (u4_g128_zp_fp16: packed codes, one zero
    // point byte and one binary16 scale per group; ggml_q4_k/q5_k: the blocks alone; ggml_q6_k:
    // ql/qh, int8 scales, binary16 d).
    void run_row_planes(const WeightJob& job, const Task& task, Scratch& s) {
        const auto& g   = job.geometry;
        const auto rows = task.end - task.begin;
        s.encoded.codes.clear();
        s.encoded.zeros.clear();
        s.encoded.scales.clear();
        for_rows(job, task.begin, task.end,
                 [&](const Source& source, std::uint64_t low, std::uint64_t high, std::uint64_t) {
                     s.encoded.order.reset();
                     source.encoded_rows(low, high, s.encoded, &s.log);
                     // Codes in one column order under another plane's order would be wrong
                     // without any value looking so.
                     if (!same_order(s.encoded.order, job.order)) {
                         throw std::runtime_error(job.id + ": " + source.label() +
                                                  " yields its columns in an order the plane "
                                                  "does not store");
                     }
                 });
        if (s.encoded.codes.size() != rows * g.code_bytes_per_row ||
            s.encoded.zeros.size() != rows * g.high_bytes_per_row ||
            s.encoded.scales.size() != rows * g.scale_bytes_per_row) {
            throw std::runtime_error(job.id + ": " +
                                     std::string(artifact::format_name(job.format)) +
                                     " rows of the wrong size");
        }
        if (job.layout == QuantLayout::GgmlFrag) {
            // ggml_frag_v1: each row's published code bytes in fragment order (core/ggml_frag.h).
            s.base.resize(s.encoded.codes.size());
            for (std::uint64_t r = 0; r < rows; ++r) {
                ggml_frag_row(job.format, job.shape[1],
                              s.encoded.codes.data() + r * g.code_bytes_per_row,
                              s.base.data() + r * g.code_bytes_per_row, true);
            }
            s.encoded.codes.swap(s.base);
        }
        emit(job.offset + task.begin * g.code_bytes_per_row, s.encoded.codes.data(),
             s.encoded.codes.size());
        if (g.high_bytes_per_row) {
            emit(job.offset + g.high_offset + task.begin * g.high_bytes_per_row,
                 s.encoded.zeros.data(), s.encoded.zeros.size());
        }
        if (g.scale_bytes_per_row) {
            emit(job.offset + g.scale_offset + task.begin * g.scale_bytes_per_row,
                 s.encoded.scales.data(), s.encoded.scales.size());
        }
    }

    void run_encoded(const WeightJob& job, const Task& task, Scratch& s) {
        if (u4_zp_group(job.format) || q3_group(job.format) || is_ggml_block(job.format)) {
            run_row_planes(job, task, s);
            return;
        }
        if (job.format == QType::Q8_G32_FP16) {
            run_q8(job, task, s);
            return;
        }
        if (job.format == QType::FP8_E4M3FN_ROW_BF16 || job.format == QType::FP8_E4M3FN_ROW_F32 ||
            job.format == QType::FP8_E4M3FN_B128_F32) {
            run_fp8(job, task, s);
            return;
        }
        if (job.format == QType::MXFP4) {
            run_mxfp4(job, task, s);
            return;
        }
        const auto& g   = job.geometry;
        const auto rows = task.end - task.begin;
        const auto k    = job.shape[1];
        s.encoded.codes.clear();
        s.encoded.scales.clear();
        std::optional<std::uint32_t> divisor;
        for_rows(job, task.begin, task.end,
                 [&](const Source& source, std::uint64_t low, std::uint64_t high, std::uint64_t) {
                     source.encoded_rows(low, high, s.encoded, &s.log); // appends
                     if (divisor && *divisor != s.encoded.divisor) {
                         throw std::runtime_error(
                             "encoded inputs cannot share one parent format/divisor");
                     }
                     divisor = s.encoded.divisor;
                 });
        const auto rows_per_divisor = job.shape[0] / job.divisors;
        const auto index            = task.begin / rows_per_divisor;
        if (task.end - index * rows_per_divisor > rows_per_divisor) {
            throw std::runtime_error(job.id +
                                     ": a row block may not span two separately quantised sources");
        }
        if (!divisor || *divisor != job.divisor_words[index]) {
            throw std::runtime_error(job.id + ": weight divisor changed between row blocks");
        }
        // Tasks start on 128-row tiles; the last one of a plane whose rows end inside a tile
        // writes that whole tile (its rows past the plane zero), which the geometry holds.
        s.swizzled.resize((rows + 127) / 128 * 128 * (k / 16));
        swizzle_nvfp4_scales(s.encoded.scales.data(), rows, k, s.swizzled.data());
        emit(job.offset + task.begin * (k / 2), s.encoded.codes.data(), s.encoded.codes.size());
        emit(job.offset + g.scale_offset + task.begin * (k / 16), s.swizzled.data(),
             s.swizzled.size());
    }

    std::vector<const Checkpoint*> stores_;
    const Plan& plan_;
    ArtifactWriter& writer_;
    std::vector<std::uint32_t> order_;
    std::vector<Task> tasks_;
    unsigned threads_;
    bool keep_cache_;
    Committer* committer_;
    std::uint64_t window_;
    std::vector<std::atomic<std::size_t>> remaining_; // unfinished tasks per job
    std::size_t ahead_ = 1;

    std::unordered_set<const TensorInfo*> shared_;
    std::atomic<std::size_t> next_{0};
    std::atomic<bool> failed_{false};
    std::mutex error_mutex_;
    std::exception_ptr error_;

    // Admission and commit state (source-consuming runs).
    std::mutex state_mutex_;
    std::condition_variable state_changed_;
    std::size_t admitted_    = 0; // positions admitted
    std::uint64_t inflight_  = 0; // output bytes admitted and not committed
    std::size_t completions_ = 0;
    bool workers_done_       = false;
};

void write_report(const std::string& report_path, const OJson& report) {
    const auto temporary = report_path + ".tmp";
    std::remove(temporary.c_str()); // a leftover of an interrupted run
    const int fd = sys::open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) { throw std::runtime_error("cannot create " + temporary); }
    const auto text  = py_dumps_indent2(report) + "\n";
    std::size_t done = 0;
    bool ok          = true;
    while (done < text.size()) {
        const auto count = sys::write(fd, text.data() + done, text.size() - done);
        if (count <= 0) {
            ok = false;
            break;
        }
        done += static_cast<std::size_t>(count);
    }
    ok = ok && sys::fsync(fd) == 0;
    sys::close(fd);
    std::error_code renamed;
    if (ok) { std::filesystem::rename(temporary, report_path, renamed); } // replaces it on Windows too
    if (!ok || renamed) {
        std::remove(temporary.c_str());
        throw std::runtime_error("cannot write " + report_path);
    }
}

std::uint64_t free_bytes(const std::filesystem::path& file, const ConvertOptions& options) {
    if (options.assume_free_bytes) { return *options.assume_free_bytes; }
    create_parent_directories(file);
    return free_bytes_beside(file);
}

// Every payload range the conversion stores: resources, each object's planes (not the zero
// padding between them) and divisor words, and the auxiliaries.
std::vector<PayloadRange> written_ranges(const Plan& plan) {
    std::vector<PayloadRange> out;
    const auto add = [&](std::uint64_t begin, std::uint64_t bytes) {
        if (bytes) { out.push_back({begin, begin + bytes}); }
    };
    for (const auto& resource : plan.resources) { add(resource.offset, resource.bytes->size()); }
    for (const auto& job : plan.weights) {
        const auto& g = job.geometry;
        switch (job.layout) {
        case QuantLayout::Contiguous:
            add(job.offset, g.bytes);
            break;
        case QuantLayout::GgmlBlocks:
            add(job.offset, g.code_bytes);
            break;
        case QuantLayout::RowSplit:
        case QuantLayout::RowGroupZp:
        case QuantLayout::RowGroupZpOrdered:
        case QuantLayout::GgmlPlanes:
        case QuantLayout::GgmlFrag: // the planes of Q4_K's blocks are empty past the codes
            add(job.offset, g.code_bytes);
            add(job.offset + g.high_offset, g.high_bytes);
            add(job.offset + g.scale_offset, g.scale_bytes);
            if (g.order_offset) { add(job.offset + g.order_offset, g.bytes - g.order_offset); }
            break;
        case QuantLayout::BlockScaleK16M128x4:
            add(job.offset, g.code_bytes);
            add(job.offset + g.scale_offset, g.scale_bytes);
            add(job.offset + g.divisor_offset, g.bytes - g.divisor_offset);
            break;
        case QuantLayout::RowScale:
        case QuantLayout::BlockScaleK32E8M0:
        case QuantLayout::BlockScale128F32:
            add(job.offset, g.code_bytes);
            add(job.offset + g.scale_offset, g.scale_bytes);
            break;
        }
    }
    for (const auto& auxiliary : plan.auxiliaries) { add(auxiliary.offset, 4); }
    return out;
}

// Bytes the report and filesystem metadata may add beyond the artifact files.
constexpr std::uint64_t kSpaceMargin = 64ULL << 20;

// The checkpoint as consumption segments: per shard its header (pinned), every tensor (read
// by the objects whose sources name it; pinned if the plan read it), and the gaps between.
ConsumptionLedger consumption_ledger(const Checkpoint& base, const Plan& plan) {
    std::map<const TensorInfo*, std::vector<std::uint32_t>> readers;
    for (std::size_t j = 0; j < plan.weights.size(); ++j) {
        for (const auto& source : plan.weights[j].sources) {
            std::vector<const TensorInfo*> tensors;
            source->tensors(base, tensors);
            for (const auto* tensor : tensors) {
                readers[tensor].push_back(static_cast<std::uint32_t>(j));
            }
        }
    }
    std::vector<std::vector<const TensorInfo*>> by_shard(base.shard_count());
    for (const auto& [name, tensor] : base.tensors()) {
        by_shard.at(static_cast<std::size_t>(tensor.file)).push_back(&tensor);
    }
    std::vector<SourceSegment> segments;
    std::vector<std::uint64_t> sizes;
    for (std::size_t s = 0; s < base.shard_count(); ++s) {
        sizes.push_back(base.shard_bytes(s));
        auto& tensors = by_shard[s];
        std::sort(tensors.begin(), tensors.end(), [&](const TensorInfo* a, const TensorInfo* b) {
            return base.shard_offset(*a) < base.shard_offset(*b);
        });
        const int shard = static_cast<int>(s);
        segments.push_back({shard, 0, base.shard_header_bytes(s), {}, true});
        std::uint64_t cursor = base.shard_header_bytes(s);
        for (const auto* tensor : tensors) {
            const auto begin = base.shard_offset(*tensor);
            if (begin > cursor) { segments.push_back({shard, cursor, begin, {}, false}); }
            SourceSegment segment{shard, begin, begin + tensor->bytes, {}, base.pinned(*tensor)};
            if (const auto it = readers.find(tensor); it != readers.end()) {
                segment.readers = it->second;
            }
            segments.push_back(std::move(segment));
            cursor = std::max(cursor, begin + tensor->bytes);
        }
        if (cursor < base.shard_bytes(s)) {
            segments.push_back({shard, cursor, base.shard_bytes(s), {}, false});
        }
    }
    return ConsumptionLedger(std::move(sizes), 4096, std::move(segments), plan.weights.size());
}

// What of the base checkpoint the plan leaves unread. A tensor counts as read when a source of
// the plan names it, and a quantization companion (divisors, activation scales, zero points,
// group indices) when the matrix it belongs to is read.
struct Unconverted {
    std::vector<std::string> kv_scales;             // static KV-cache scales
    std::map<std::string, std::uint64_t> by_prefix; // everything else, by name prefix
};

Unconverted unconverted_tensors(const Checkpoint& base, const Plan& plan) {
    std::set<std::string> read;
    for (const auto& job : plan.weights) {
        for (const auto& source : job.sources) {
            std::vector<const TensorInfo*> tensors;
            source->tensors(base, tensors);
            for (const auto* tensor : tensors) { read.insert(tensor->name); }
        }
    }
    static const std::regex companion(
        R"((.*?)(\.weight_scale_2|\.weight_global_scale|\.input_scale|\.input_global_scale|)"
        R"(\.weight_scale_inv|\.weight_scale|\.weight_zero_point|\.weight_g_idx|\.weight_shape|)"
        R"(\.qzeros|\.scales|\.g_idx|_scales)$)");
    static const std::regex kv_scale(
        R"(.*(\.k_scale|\.v_scale|\.kv_scale|\.q_scale|\.prob_scale|k_proj\.output_scale|)"
        R"(v_proj\.output_scale)$)");
    const auto matrix_read = [&](const std::string& prefix) {
        for (const char* suffix : {".weight", ".weight_packed", ".qweight", "_blocks"}) {
            if (read.contains(prefix + suffix)) { return true; }
        }
        return false;
    };
    Unconverted out;
    for (const auto& [name, tensor] : base.tensors()) {
        if (read.contains(name)) { continue; }
        std::smatch match;
        if (std::regex_match(name, kv_scale)) {
            out.kv_scales.push_back(name);
            continue;
        }
        if (std::regex_match(name, match, companion) && matrix_read(match[1].str())) { continue; }
        // model.vision_tower.encoder... -> model.vision_tower.*; model.layers.3.x ->
        // model.layers.3.*
        std::string prefix;
        std::size_t at = 0;
        for (int part = 0; part < 3; ++part) {
            const auto dot = name.find('.', at);
            if (dot == std::string::npos) { break; }
            prefix = name.substr(0, dot);
            at     = dot + 1;
            if (std::isdigit(static_cast<unsigned char>(name[at - 2]))) { break; }
        }
        ++out.by_prefix[(prefix.empty() ? name : prefix + ".*")];
    }
    return out;
}

// Refuse checkpoints whose space would not come back, or that something else still shares.
void check_consumable(const Checkpoint& base, const std::filesystem::path& out) {
    const auto dir = directory_of(out);
#ifdef _WIN32
    // The same checks through std::filesystem; a volume is the drive (or share) of the full path.
    const auto volume = std::filesystem::absolute(dir).root_name();
    for (std::size_t s = 0; s < base.shard_count(); ++s) {
        const auto& path = base.shard_path(s);
        std::error_code error;
        if (std::filesystem::is_symlink(std::filesystem::symlink_status(path, error))) {
            throw std::runtime_error("--consume-source: " + path.string() +
                                     " is a symbolic link; consuming it would destroy its target");
        }
        if (std::filesystem::hard_link_count(path, error) > 1) {
            throw std::runtime_error("--consume-source: " + path.string() +
                                     " has other hard links that would lose their data too");
        }
        if (std::filesystem::absolute(path).root_name() != volume) {
            throw std::runtime_error("--consume-source: " + path.string() +
                                     " is on another filesystem than the output; consuming it "
                                     "frees no space there");
        }
    }
#else
    struct stat output_dir{};
    if (stat(dir.c_str(), &output_dir) != 0) {
        throw std::runtime_error("cannot stat " + dir.string());
    }
    for (std::size_t s = 0; s < base.shard_count(); ++s) {
        const auto& path = base.shard_path(s);
        struct stat link{};
        if (lstat(path.c_str(), &link) != 0) {
            throw std::runtime_error("cannot stat " + path.string());
        }
        if (S_ISLNK(link.st_mode)) {
            throw std::runtime_error("--consume-source: " + path.string() +
                                     " is a symbolic link; consuming it would destroy its target");
        }
        if (link.st_nlink > 1) {
            throw std::runtime_error("--consume-source: " + path.string() +
                                     " has other hard links that would lose their data too");
        }
        if (link.st_dev != output_dir.st_dev) {
            throw std::runtime_error("--consume-source: " + path.string() +
                                     " is on another filesystem than the output; consuming it "
                                     "frees no space there");
        }
    }
#endif
}

class DriverIo final : public CommitIo {
public:
    DriverIo(ArtifactWriter& writer, Journal& journal, const Checkpoint& base)
        : writer_(writer), journal_(journal), base_(base) {}

    void sync_output() override { writer_.sync(); }

    void journal(std::span<const std::uint32_t> ids) override { journal_.append(ids); }

    void punch(const Punch& punch) override {
        base_.punch(static_cast<std::size_t>(punch.shard), punch.offset, punch.bytes);
    }

private:
    ArtifactWriter& writer_;
    Journal& journal_;
    const Checkpoint& base_;
};

// The encodings a conversion keeps as published (natively executed or mapped onto an engine
// format without changing one value); the refusal names them.
constexpr const char* kNativeEncodings =
    "NVFP4 (ModelOpt, compressed-tensors, GGUF), MXFP4 (gpt-oss, compressed-tensors, GGUF), "
    "FP8 E4M3 with per-tensor, per-channel or 128x128-block scales, GGUF Q8_0, "
    "int4 with 32-, 64- or 128-value groups or per channel, symmetric or with zero points (AWQ, "
    "GPTQ with or without act-order, AutoRound, compressed-tensors), int8 weights (GPTQ, "
    "compressed-tensors W8A16), GGUF Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1, "
    "IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M, TQ1_0, TQ2_0, Q1_0 and "
    "Q2_0";

// What a plan does to the checkpoint's numbers beyond keeping or casting them. A published
// quantization is kept as published, never dequantized. NInferno quantizes plain (BF16, F16, F32)
// values itself only where an official recipe's table assigns a format to what a quantized
// checkpoint's publisher left plain (the vocabulary, projections kept in BF16 next to a quantized
// body) and in the drafters' official layouts; the projections of a checkpoint without any
// published quantization stay as published.
struct Fidelity {
    std::map<std::string, std::uint64_t> foreign; // "<encoding>: <handling>" -> matrices
    std::map<std::string, std::vector<std::string>> examples; // foreign key -> first parameters
    std::map<std::string, std::uint64_t> own;     // format -> projections NInferno quantized
    // Decodes a source's own transform needs ("<encoding>: <handling> <reason>" -> tensors):
    // reported, not refused.
    std::map<std::string, std::uint64_t> declared;
    // Parameters whose source reads a quantized stored tensor as values without saying so (a
    // source that forgot dequantizes()): named in the refusal.
    std::vector<std::string> undeclared;
    std::uint64_t kept = 0; // projections kept in their published encoding
};

// How a job treats a source read as values from a quantized encoding: "dequantized to <format>"
// or "dequantized and quantized again to <format>".
std::string decode_handling(Method method, const std::string& format) {
    return method == Method::CastDirect ? "dequantized to " + format
                                        : "dequantized and quantized again to " + format;
}

Fidelity fidelity(const Checkpoint& base, const Model& model, const Plan& plan) {
    Fidelity out;
    for (const auto& job : plan.weights) {
        const std::string format(artifact::format_name(job.format));
        for (std::size_t j = 0; j < job.sources.size(); ++j) {
            const auto& source    = *job.sources[j];
            const auto& parameter = model.parameters[job.parameters[j]];
            const bool body = parameter.projection() && parameter.name.starts_with("text/layers/");
            if (job.method == Method::ImportEncoded) {
                if (source.keeps_losslessly()) {
                    out.kept += body ? 1 : 0;
                } else if (source.dequantizes()) {
                    ++out.foreign[source.encoding() + ": kept as " + format +
                                  " with its scales rounded"];
                } else if (decodes_quantized(base, source)) {
                    const auto handling = decode_handling(Method::GroupedAbsmax, format);
                    ++out.foreign[source.encoding() + ": " + handling];
                    out.undeclared.push_back(parameter.name);
                } else if (body) {
                    ++out.own[format];
                }
                continue;
            }
            if (decodes_quantized(base, source)) {
                const auto handling =
                    source.encoding() + ": " + decode_handling(job.method, format);
                if (!source.dequantizes()) {
                    ++out.foreign[handling];
                    out.undeclared.push_back(parameter.name);
                } else if (const auto reason = source.decode_reason(); !reason.empty()) {
                    ++out.declared[handling + " " + reason];
                } else {
                    ++out.foreign[handling];
                    if (out.examples[handling].size() < 3) {
                        out.examples[handling].push_back(parameter.name);
                    }
                }
            } else if (body && (job.method == Method::GroupedAbsmax ||
                                job.method == Method::Fp8RowMaxabs)) {
                ++out.own[format];
            }
        }
    }
    return out;
}

// A published encoding an engine format holds word for word is kept wherever the recipe's table
// would decode it: the table is written for unquantized or vendor checkpoints, and decoding
// published words only to quantize them again (or to widen them) adds error and bytes. Only the
// parameter's own source qualifies (a recipe's derived source - scaled, gathered - is its
// choice), and only groupwise Q8 (GGUF Q8_0), 3-bit groupwise and zero-point int4 (compressed-
// tensors, GPTQ, AWQ) so far: NVFP4, FP8 and MXFP4 sites carry packing,
// divisor and activation rules the recipes decide. A kept int8-code parameter records AllowA8
// (kept_block_policy) except the output head, whose logits keep BF16 activations; others keep the
// recipe's policy. Returns the parameters it changed.
std::map<QType, std::size_t> keep_published(const Model& model, Recipe& recipe) {
    std::map<QType, std::size_t> kept;
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        const auto& parameter = model.parameters[i];
        const auto& selection = recipe.selections[i];
        if (recipe.alias[i] || selection.method == Method::ImportEncoded ||
            parameter.shape.size() != 2 || !selection.source ||
            selection.source != parameter.source) {
            continue;
        }
        const auto& source = *selection.source;
        const auto format  = source.encoded_format();
        if (!source.dequantizes() || !format || !kept_where_decoded(*format) ||
            !source.keeps_losslessly()) {
            continue;
        }
        recipe.assign(i, *format, Method::ImportEncoded, selection.source);
        if (kept_block_policy(*format) == ActivationPolicy::AllowA8 &&
            parameter.name != "text/output_head") {
            recipe.set_policy(i, ActivationPolicy::AllowA8);
        }
        ++kept[*format];
    }
    return kept;
}

// The drafter architectures NInferno may quantize itself: a drafter
// only proposes tokens and the target verifies every one, so what is served is the target's
// output whatever the drafter's precision. Targets always run as published.
bool is_drafter_type(const std::string& type) {
    return type == "gemma4_assistant" || type == "gemma4_dflash" || type == "dflash";
}

// A companion drafter's parameter in a target's conversion (Qwen DFlash and DFlash2 ride as the
// target's dflash / dflash2 components): a drafter like a separate one, whatever the target.
bool companion_parameter(const std::string& name) {
    return name.starts_with("dflash/") || name.starts_with("dflash2/");
}

// --drafter-quant: every projection of a drafter that the recipe left in its direct BF16 form
// (of a target, only its companion drafters' projections),
// read from plain values (a published encoding is never quantized again), in NInferno's own
// quantization: q8 (q8_g32_fp16), q4 (q4_g64_fp16), or nvfp4 (round-to-nearest NVFP4, weight
// only: the drafter keeps BF16 activations). A packing group (gate/up) takes one format, and for
// nvfp4 one divisor; it takes q8 when one of its matrices is not whole 128-row tiles with K a
// multiple of 64. Returns the matrices quantized per format.
std::map<std::string, std::size_t> quantize_drafter(const Model& model, Recipe& recipe,
                                                    const std::string& mode) {
    const auto type     = model.config().value("model_type", std::string());
    const bool drafter  = is_drafter_type(type);
    const bool carrying = std::any_of(model.parameters.begin(), model.parameters.end(),
                                      [](const Parameter& p) { return companion_parameter(p.name); });
    if (!drafter && !carrying) {
        throw std::runtime_error("--drafter-quant applies to drafters (a Gemma 4 assistant or a "
                                 "DFlash drafter, separate or a target's companion); a " +
                                 (type.empty() ? std::string("target") : type) +
                                 " runs as published");
    }
    if (mode != "q8" && mode != "q4" && mode != "nvfp4") {
        throw std::runtime_error("--drafter-quant takes q8, q4 or nvfp4, not '" + mode + "'");
    }
    const auto plain = [&](std::size_t i) {
        const auto& p         = model.parameters[i];
        const auto& selection = recipe.selections[i];
        if (!drafter && !companion_parameter(p.name)) { return false; } // the target as published
        if (!p.projection() || p.shape.size() != 2 || recipe.alias[i] ||
            selection.method != Method::CastDirect || selection.format != QType::BF16) {
            return false;
        }
        const auto& source = selection.source ? selection.source : p.source;
        if (!source) { return false; }
        source->resolve();
        return !source->dequantizes();
    };
    std::vector<std::vector<std::size_t>> units;
    std::vector<bool> grouped(model.parameters.size(), false);
    for (const auto& group : model.packing_groups) {
        if (std::all_of(group.begin(), group.end(), plain)) { units.push_back(group); }
        for (const auto i : group) { grouped[i] = true; }
    }
    for (std::size_t i = 0; i < model.parameters.size(); ++i) {
        if (!grouped[i] && plain(i)) { units.push_back({i}); }
    }
    std::map<std::string, std::size_t> counts;
    for (const auto& unit : units) {
        const bool tiles = std::all_of(unit.begin(), unit.end(), [&](std::size_t i) {
            const auto& shape = model.parameters[i].shape;
            return shape[0] % 128 == 0 && shape[1] % 64 == 0;
        });
        if (mode == "nvfp4" && tiles) {
            std::vector<SourcePtr> divisor_group;
            for (const auto i : unit) {
                const auto& s = recipe.selections[i].source;
                divisor_group.push_back(s ? s : model.parameters[i].source);
            }
            for (std::size_t m = 0; m < unit.size(); ++m) {
                recipe.assign(unit[m], QType::NVFP4, Method::ImportEncoded,
                              nvfp4_rounded_source(divisor_group[m], divisor_group));
                recipe.set_policy(unit[m], ActivationPolicy::A16Only);
                ++counts["nvfp4"];
            }
            continue;
        }
        const auto format = mode == "q4" ? QType::Q4_G64_FP16 : QType::Q8_G32_FP16;
        for (const auto i : unit) {
            recipe.assign(i, format, Method::GroupedAbsmax);
            ++counts[std::string(artifact::format_name(format))];
        }
    }
    return counts;
}

// Throws (or, under --maintainer-quantize, logs) what would make the artifact differ from the
// published model. `bf16_recipe` is the family's recipe that keeps plain projections unquantized
// (empty when the family has none).
void check_fidelity(const Checkpoint& base, const Model& model, const Plan& plan,
                    const ConvertOptions& options, const std::string& bf16_recipe,
                    const std::function<void(const std::string&)>& log) {
    const auto found = fidelity(base, model, plan);
    // A decode the source's transform needs (a scale folded into an embedding table): changed on
    // purpose and said so, in the log here and in the report's native notes.
    for (const auto& [handling, count] : found.declared) {
        log("decoded on purpose: " + handling + " (" + std::to_string(count) + " tensors)");
    }
    const bool drafter =
        is_drafter_type(model.config().value("model_type", std::string()));
    const bool own_refused = !found.own.empty() && found.kept == 0 && !drafter;
    if (found.foreign.empty() && !own_refused) { return; }
    std::string what;
    for (const auto& [handling, count] : found.foreign) {
        what += "\n  " + handling + " (" + std::to_string(count) + " matrices";
        const auto listed = found.examples.find(handling);
        if (listed != found.examples.end()) {
            for (std::size_t i = 0; i < listed->second.size(); ++i) {
                what += (i == 0 ? ": " : ", ") + listed->second[i];
            }
            if (count > listed->second.size()) { what += ", ..."; }
        }
        what += ")";
    }
    if (!found.undeclared.empty()) {
        what += "\n  read as plain values from a quantized encoding:";
        for (const auto& name : found.undeclared) { what += " " + name; }
    }
    if (own_refused) {
        for (const auto& [format, count] : found.own) {
            what += "\n  " + std::to_string(count) +
                    " unquantized projections quantized by NInferno to " + format;
        }
    }
    if (options.maintainer_quantize) {
        log("--maintainer-quantize: this artifact is NInferno's own quantization, not the "
            "published "
            "model:" +
            what);
        return;
    }
    std::string advice;
    if (!found.foreign.empty()) {
        advice = "NInferno runs a published quantization only as published, and this one has no "
                 "native execution. It keeps " +
                 std::string(kNativeEncodings) +
                 ". Use a checkpoint of this model in one of those, or its BF16 original.";
    } else if (!bf16_recipe.empty()) {
        advice = "This checkpoint publishes no quantization of its projections, so NInferno runs "
                 "them as published: --recipe " +
                 bf16_recipe +
                 " (the default). For a smaller model take an official quantized checkpoint "
                 "(NVFP4, FP8) of it.";
    } else {
        advice = "This checkpoint publishes no quantization of its projections and this "
                 "architecture has no unquantized recipe: take an official quantized checkpoint "
                 "(NVFP4, FP8) of the model.";
    }
    throw std::runtime_error(
        "recipe " + options.recipe + " would not run " + options.model.string() +
        " as published:" + what + "\n" + advice +
        " A maintainer can force the conversion with --maintainer-quantize (the artifact is then "
        "NInferno's own quantization).");
}

} // namespace

ConvertResult convert(const ConvertOptions& options) {
    const auto begin = Clock::now();
    set_round_tiny_scales(options.round_tiny_scales);
    const auto log   = [&](const std::string& line) {
        if (options.log) { options.log(line); }
    };
    std::vector<std::string> components;
    for (std::size_t at = 0;;) {
        const auto comma = options.components.find(',', at);
        components.push_back(options.components.substr(at, comma - at));
        if (comma == std::string::npos) { break; }
        at = comma + 1;
    }
    if (std::set<std::string>(components.begin(), components.end()).size() != components.size()) {
        throw std::runtime_error("--components " + options.components + " names a component twice");
    }
    // The MTP head alone: a NemotronH GGUF that keeps its NextN block converts it as the drafter.
    const auto request = gguf_request_for(options.components);
    Checkpoint base(options.model, request);
    OJson gguf_summary;
    if (const auto* gguf = base.gguf()) {
        // What the GGUF holds and what reading it as its Hugging Face checkpoint undid.
        std::map<std::string, std::uint64_t> types;
        for (const auto& [name, tensor] : base.tensors()) { ++types[ggml::type_name(tensor.ggml_type)]; }
        OJson type_counts = OJson::object();
        std::string listed;
        for (const auto& [type, count] : types) {
            type_counts[type] = count;
            listed += (listed.empty() ? "" : ", ") + type + " x" + std::to_string(count);
        }
        const auto architecture = gguf->metadata.value("general.architecture", std::string());
        log("gguf " + architecture + " (v" + std::to_string(gguf->version) + ", " +
            std::to_string(base.shard_count()) + " file(s)): " + listed);
        for (const auto& note : base.mapping_notes()) { log("gguf mapping: " + note); }
        // Before the model is built: a tokenizer the metadata cannot rebuild fails there.
        for (const auto& note : base.rebuilt_notes()) { log("gguf resources: " + note); }
        gguf_summary                 = OJson::object();
        gguf_summary["architecture"] = architecture;
        gguf_summary["version"]      = gguf->version;
        gguf_summary["files"]        = base.shard_count();
        gguf_summary["tensor_types"] = std::move(type_counts);
        gguf_summary["mapping"]      = base.mapping_notes();
        if (!base.gguf_mapper().empty()) { gguf_summary["mapper"] = base.gguf_mapper(); }
    }
    if (options.components_auto && is_qwen3_5_architecture(declared_architecture(base.config()))) {
        const auto keep = [&](const std::string& component, const std::string& why) {
            if (std::find(components.begin(), components.end(), component) != components.end()) {
                return;
            }
            components.push_back(component);
            log("component " + component + " kept: " + why);
        };
        if (base.has("mtp.fc.weight")) {
            keep("mtp", "the model carries MTP layers");
        }
        for (const auto& [name, path] : options.sources) {
            if (name == "dflash" || name == "dflash2") { keep(name, "--source names it"); }
        }
    }
    SourceInputs sources(base, options.sources);
    // A drafter companion is opened before the model is built; other named sources when the
    // recipe first asks for them (their provenance keeps that order).
    std::map<std::string, const Checkpoint*> companions;
    for (const char* key : {"dflash", "dflash2"}) {
        if (std::find(components.begin(), components.end(), key) != components.end()) {
            companions[key] = &sources.get(key);
        }
    }
    // A Vision encoder from its own file (a GGUF model's mmproj projector): --source vision=PATH.
    if (std::find(components.begin(), components.end(), "vision") != components.end() &&
        std::any_of(options.sources.begin(), options.sources.end(),
                    [](const auto& source) { return source.first == "vision"; })) {
        companions["vision"] = &sources.get("vision");
    }
    Model model;
    std::string bf16_recipe; // the family's recipe that keeps plain projections unquantized
    if (is_qwen3_5_architecture(declared_architecture(base.config()))) {
        model       = build_qwen3_5_model(base, components, companions, options.resources);
        bf16_recipe = model.config().contains("num_experts") ? "" : "qwen3_5_bf16";
    } else if (is_dense_architecture(declared_architecture(base.config()))) {
        // A NemotronH MTP drafter read from its target's GGUF is that drafter's text component.
        const bool drafter_of_target =
            request == GgufRequest::Mtp &&
            declared_architecture(base.config()) == "NemotronHMtpForCausalLM";
        if (components != std::vector<std::string>{"text"} && !drafter_of_target) {
            throw std::runtime_error("--components " + options.components +
                                     ": a dense decoder has the text component only");
        }
        model       = build_dense_model(base, options.resources);
        const auto type = model.config().value("model_type", std::string());
        bf16_recipe     = type == "moe_decoder" || type == "nemotron_h"     ? "moe_published"
                          : type == "nemotron_h_mtp" ? "nemotron_h_mtp_published"
                                                     : "dense_bf16";
    } else {
        const bool vision =
            std::find(components.begin(), components.end(), "vision") != components.end();
        const std::set<std::string> selected(components.begin(), components.end());
        if (!selected.contains("text") ||
            selected.size() > 1 + std::size_t(selected.contains("vision"))) {
            throw std::runtime_error("--components " + options.components +
                                     ": a Gemma 4 model has the text and vision components only");
        }
        const auto companion = companions.find("vision");
        model = build_model(base, options.resources,
                            !vision                       ? nullptr
                            : companion != companions.end() ? companion->second
                                                            : &base);
    }
    for (const auto& note : model.resource_notes) { log("resource " + note); }
    // The token ranking a proposal head is restricted by: --ranking, else the default one, found
    // as the registry's rankings are (resolve_ranking: the working tree, the rankings installed
    // beside the binary, the tree it was built from).
    std::filesystem::path ranking = options.ranking;
    if (options.proposal && ranking.empty()) {
        const auto found = resolve_ranking(kDefaultRanking, std::filesystem::current_path());
        if (!found) {
            throw std::runtime_error(std::string("the default token ranking ") + kDefaultRanking +
                                     " is neither installed (share/ninferno/rankings) nor in a "
                                     "source tree here; pass --ranking");
        }
        ranking = *found;
    }
    const auto ranking_text =
        options.ranking_label.empty() ? py_path_str(ranking.string()) : options.ranking_label;
    // A Gemma 4 drafter has no separate proposal head: its own head is the one restricted.
    const bool restricted =
        options.proposal && model.config().value("model_type", std::string()) == "gemma4_assistant";
    if (restricted) { restrict_head(model, ranking, ranking_text, options.proposal_rows); }
    Recipe recipe(model);
    std::string vocabulary; // the dense and MoE recipes' --vocab policy, recorded in provenance
    if (!options.vocab.empty() && !is_dense_recipe(options.recipe)) {
        throw std::runtime_error("--vocab applies to the dense and MoE recipes; recipe " +
                                 options.recipe + " fixes its own vocabulary format");
    }
    // A recipe's refusal (the wrong family, a missing source) names the recipe.
    try {
        if (is_qwen3_5_recipe(options.recipe)) {
            apply_qwen3_5_recipe(options.recipe, model, recipe, sources);
        } else if (is_dense_recipe(options.recipe)) {
            vocabulary = apply_dense_recipe(options.recipe, model, recipe, sources.get("base"),
                                            options.vocab);
        } else {
            apply_recipe(options.recipe, model, recipe, sources.get("base"));
        }
    } catch (const std::runtime_error& error) {
        throw std::runtime_error("recipe " + options.recipe + ": " + error.what());
    }
    if (is_dense_recipe(options.recipe)) {
        log("vocabulary " + vocabulary +
            (options.vocab.empty() ? " (the recipe's default)" : " (--vocab)"));
    }
    if (!options.drafter_quant.empty()) {
        std::string counted;
        for (const auto& [format, count] : quantize_drafter(model, recipe, options.drafter_quant)) {
            counted += (counted.empty() ? "" : ", ") + std::to_string(count) + " " + format;
        }
        log("drafter quantization " + options.drafter_quant + ": " +
            (counted.empty() ? std::string("no plain projection left to quantize") : counted));
    }
    if (share_tied_head(model, recipe)) {
        log("tied output head: bound to the token embedding's objects (one copy)");
    }
    if (options.proposal && !restricted) {
        add_official_proposal(model, recipe, ranking, ranking_text, options.proposal_rows);
    }
    // --maintainer-quantize runs the recipe's table as written.
    if (!options.maintainer_quantize) {
        for (const auto& [format, count] : keep_published(model, recipe)) {
            log("kept " + std::to_string(count) +
                (format == QType::Q8_G32_FP16 ? " published groupwise Q8 matrices as q8_g32_fp16"
                                              : " published matrices as " +
                                                    std::string(artifact::format_name(format))) +
                " (the recipe's table decodes them)");
        }
    }
    const auto stores       = sources.opened();
    OJson provenance        = OJson::object();
    provenance["converter"] = "ninferno-v3";
    provenance["recipe"]    = options.recipe;
    if (!options.drafter_quant.empty()) {
        provenance["drafter_quantization"] = options.drafter_quant;
    }
    if (!vocabulary.empty()) { provenance["vocabulary"] = vocabulary; }
    provenance["sources"]   = sources.provenance();
    if (!base.gguf_mapper().empty()) { provenance["gguf_mapper"] = base.gguf_mapper(); }
    if (options.proposal) { provenance["ranking"] = ranking_text; }
    ConvertResult result;
    result.model_seconds = seconds_since(begin);

    // pipeline.convert
    const auto output      = py_path_str(options.out.string());
    const auto report_path = output + ".conversion.json";
    if (std::filesystem::exists(output) || std::filesystem::exists(report_path)) {
        throw std::runtime_error("conversion output already exists: " + output + " or " +
                                 report_path);
    }
    const auto start = Clock::now();
    const Plan plan  = prepare(model, recipe, options.expert_layout);
    check_fidelity(base, model, plan, options, bf16_recipe, log);
    {
        std::uint64_t frag = 0;
        for (const auto& job : plan.weights) { frag += job.layout == QuantLayout::GgmlFrag; }
        if (frag) {
            log("stored " + std::to_string(frag) +
                " routed-expert banks in ggml_frag_v1 (lossless fragment order; --layout " +
                std::string(expert_layout_name(options.expert_layout)) + ")");
        }
    }
    // Planning faulted in thousands of scattered scalar and first-row pages (and their
    // fault-around neighbours); drop them from the process before the bulk pass.
    // Nothing of the checkpoint is dropped silently: what the plan leaves unread is logged and
    // reported. Static KV-cache scales get their own warning: NInferno's quantized KV codecs (int8,
    // k8v4, nvfp4) compute their scales per token and group when a token is appended, so a
    // checkpoint's calibrated k_scale/v_scale are never applied, whatever the KV dtype.
    const Unconverted unconverted = unconverted_tensors(base, plan);
    if (!unconverted.kv_scales.empty()) {
        std::string listed;
        for (std::size_t i = 0; i < unconverted.kv_scales.size() && i < 4; ++i) {
            listed += (i ? ", " : "") + unconverted.kv_scales[i];
        }
        log("warning: " + std::to_string(unconverted.kv_scales.size()) +
            " static KV-cache scale tensors not used (" + listed +
            (unconverted.kv_scales.size() > 4 ? ", ..." : "") +
            "): the int8/k8v4/nvfp4 KV codecs scale every token and group at append");
    }
    if (!unconverted.by_prefix.empty()) {
        std::string listed;
        for (const auto& [prefix, count] : unconverted.by_prefix) {
            listed += (listed.empty() ? "" : ", ") + prefix + " (" + std::to_string(count) + ")";
        }
        log("checkpoint tensors not converted: " + listed);
    }
    for (const auto* store : stores) { store->unmap_all(); }
    const double preparation_seconds = seconds_since(start);
    result.plan_seconds              = preparation_seconds;

    OJson metadata = OJson::object();
    if (options.name) { metadata["name"] = *options.name; }
    OJson report                  = OJson::object();
    report["components"]          = model.components;
    report["name"]                = options.name ? OJson(*options.name) : OJson(nullptr);
    report["output"]              = output;
    report["device"]              = "cpu";
    report["torch_version"]       = nullptr;
    report["cuda_version"]        = nullptr;
    report["rows_per_chunk"]      = options.rows_per_chunk;
    report["provenance"]          = provenance;
    report["preparation_seconds"] = preparation_seconds;
    report["parameters"]          = model.parameters.size();
    report["uses"]                = plan.uses.size();
    report["objects"]             = plan.objects.size();
    OJson formats                 = OJson::object();
    for (const auto& job : plan.weights) {
        const std::string format(artifact::format_name(job.format));
        formats[format] = formats.value(format, 0) + 1;
    }
    report["formats"] = std::move(formats);
    OJson methods     = OJson::array();
    for (const auto& job : plan.weights) {
        OJson item     = OJson::object();
        item["object"] = job.id;
        OJson names    = OJson::array();
        OJson labels   = OJson::array();
        for (std::size_t j = 0; j < job.parameters.size(); ++j) {
            names.push_back(model.parameters[job.parameters[j]].name);
            labels.push_back(job.sources[j]->label());
        }
        item["parameters"]        = std::move(names);
        item["sources"]           = std::move(labels);
        item["method"]            = std::string(method_name(job.method));
        item["method_parameters"] = OJson::object();
        item["format"]            = std::string(artifact::format_name(job.format));
        item["layout"]            = std::string(artifact::layout_name(job.layout));
        item["shape"]             = job.shape;
        methods.push_back(std::move(item));
    }
    report["methods"] = std::move(methods);

    // What happened to every quantized checkpoint encoding: kept as an engine format (and
    // whether exactly), or dequantized and quantized again (never exact). The log always says
    // so; the report too when the checkpoint holds an encoding tools/convert cannot read (its
    // report is otherwise the Python one, byte for byte).
    struct Note {
        std::string encoding;
        std::string handling;
        bool lossless          = false;
        OJson objects          = OJson::array();
        std::uint64_t matrices = 0;
    };

    std::vector<Note> notes;
    bool python_readable = true;
    for (const auto& job : plan.weights) {
        const std::string format(artifact::format_name(job.format));
        for (const auto& source : job.sources) {
            python_readable = python_readable && source->python_readable();
            std::string handling;
            bool lossless = false;
            if (job.method == Method::ImportEncoded) {
                handling = "kept as " + format;
                if (job.layout == QuantLayout::GgmlFrag) { handling += " in ggml_frag_v1 order"; }
                lossless = source->keeps_losslessly();
            } else if (!decodes_quantized(base, *source)) {
                continue; // plain values: the recipe's own quantization or cast
            } else if (job.method == Method::GroupedAbsmax) {
                handling =
                    "dequantized to f32 and requantized to " + format + " (double quantization)";
            } else {
                handling = "dequantized to " + format;
            }
            if (job.method != Method::ImportEncoded && source->dequantizes() &&
                !source->decode_reason().empty()) {
                handling += " " + source->decode_reason();
            }
            auto it = std::find_if(notes.begin(), notes.end(), [&](const Note& note) {
                return note.encoding == source->encoding() && note.handling == handling &&
                       note.lossless == lossless;
            });
            if (it == notes.end()) {
                notes.push_back({source->encoding(), handling, lossless});
                it = notes.end() - 1;
            }
            ++it->matrices;
            if (it->objects.empty() || it->objects.back() != job.id) {
                it->objects.push_back(job.id);
            }
        }
    }
    OJson native_notes = OJson::array();
    for (const auto& note : notes) {
        log("source " + note.encoding + ": " + std::to_string(note.matrices) + " matrices " +
            note.handling + (note.lossless ? ", lossless" : ", LOSSY"));
        OJson item       = OJson::object();
        item["encoding"] = note.encoding;
        item["handling"] = note.handling;
        item["lossless"] = note.lossless;
        item["matrices"] = note.matrices;
        item["objects"]  = note.objects;
        native_notes.push_back(std::move(item));
    }
    if (!python_readable) { report["native_notes"] = std::move(native_notes); }
    // The unconverted towers and optional components stay in the log: the report is otherwise
    // tools/convert's, byte for byte, and tools/convert does not list them.
    if (!unconverted.kv_scales.empty()) {
        report["kv_cache_scales_ignored"] = unconverted.kv_scales;
    }
    if (!gguf_summary.is_null()) { report["gguf"] = std::move(gguf_summary); }
    if (!model.resource_notes.empty()) { report["resource_notes"] = model.resource_notes; }

    OJson description         = OJson::object();
    description["components"] = model.components;
    description["objects"]    = plan.objects;
    description["bindings"]   = plan.bindings;
    description["uses"]       = plan.uses;
    description["metadata"]   = metadata;
    description["provenance"] = provenance;
    auto layout = layout_artifact(options.out.filename().string(), py_dumps_compact(description),
                                  plan.payload_bytes, options.max_file_bytes);

    // Everything written before the first object: headers, directory, resources, auxiliaries.
    std::uint64_t prologue =
        layout.payload_offset + (layout.files.size() - 1) * 4096 + 4 * plan.auxiliaries.size();
    for (const auto& resource : plan.resources) { prologue += resource.bytes->size(); }
    std::vector<std::uint64_t> output_bytes;
    for (const auto& job : plan.weights) { output_bytes.push_back(job.geometry.bytes); }
    const auto available    = free_bytes(options.out, options);
    const auto needed       = layout.file_bytes() + kSpaceMargin;
    const auto journal_path = std::filesystem::path(output + ".journal");
    const bool resuming     = options.consume_source && std::filesystem::exists(journal_path);
    if (!options.consume_source && std::filesystem::exists(journal_path)) {
        throw std::runtime_error("a source-consuming conversion to " + output +
                                 " was interrupted (" + journal_path.string() +
                                 "); run it again with --consume-source to resume it");
    }
    const auto fingerprint = fnv::bytes(layout.directory);

    std::vector<std::uint32_t> order;
    std::optional<ConsumptionLedger> ledger;
    std::optional<Journal> journal;
    std::vector<std::uint32_t> durable;
    if (options.consume_source && stores.size() > 1) {
        throw std::runtime_error("--consume-source converts from the --model checkpoint alone; "
                                 "this recipe also reads other sources");
    }
    if (!options.consume_source) {
        if (needed > available && stores.size() > 1) {
            throw std::runtime_error("not enough disk space for " + output +
                                     ": the artifact needs " + std::to_string(needed) + " bytes (" +
                                     format_gib(needed) + "), " + std::to_string(available) +
                                     " bytes (" + format_gib(available) + ") are free.");
        }
        if (needed > available) {
            auto consumed = consumption_ledger(base, plan);
            const auto consume_peak =
                schedule_consumption(consumed, output_bytes, prologue, options.consume_window)
                    .peak_extra;
            throw std::runtime_error(
                "not enough disk space for " + output + ": the artifact needs " +
                std::to_string(needed) + " bytes (" + format_gib(needed) + "), " +
                std::to_string(available) + " bytes (" + format_gib(available) +
                ") are free. Free space, or pass --consume-source to convert while releasing "
                "the checkpoint's own space; that destroys the checkpoint and needs about " +
                std::to_string(consume_peak + kSpaceMargin) + " bytes (" +
                format_gib(consume_peak + kSpaceMargin) + ") free.");
        }
        for (std::uint32_t j = 0; j < plan.weights.size(); ++j) { order.push_back(j); }
    } else {
        check_consumable(base, options.out);
        ledger.emplace(consumption_ledger(base, plan));
        if (resuming) {
            journal.emplace(journal_path, fingerprint, artifact::ArtifactId{}, plan.weights.size());
            durable                = journal->completed();
            result.resumed         = true;
            result.resumed_objects = durable.size();
        } else {
            // A partial output without a journal never reached its first punch: start over.
            for (std::size_t i = 0; i < layout.files.size(); ++i) {
                const auto target =
                    i == 0 ? options.out : options.out.parent_path() / *layout.files[i].path;
                std::filesystem::remove(ArtifactWriter::partial_path(target));
            }
        }
        auto planning = *ledger;
        (void)planning.durable(durable);
        auto schedule = schedule_consumption(planning, output_bytes, resuming ? 0 : prologue,
                                             options.consume_window);
        order         = std::move(schedule.order);
        result.consume_peak_bound = schedule.peak_extra;
        const auto consume_needed = schedule.peak_extra + kSpaceMargin;
        if (consume_needed > available) {
            throw std::runtime_error(
                "not enough disk space even with --consume-source: converting " + output +
                " while releasing the checkpoint peaks at " + std::to_string(consume_needed) +
                " bytes (" + format_gib(consume_needed) + ") above the checkpoint, " +
                std::to_string(available) + " bytes (" + format_gib(available) + ") are free.");
        }
        log("consume-source: " + std::string(resuming ? "resuming, " : "") +
            std::to_string(durable.size()) + " of " + std::to_string(plan.weights.size()) +
            " objects durable; planned peak " + format_gib(schedule.peak_extra) +
            " above the checkpoint (window " + format_gib(options.consume_window) + ")");
    }

    WriterOptions writer_options;
    writer_options.named_partial = options.consume_source;
    // A resumable conversion needs a mapped output fdatasync can make durable at any object.
    writer_options.direct = !options.consume_source;
    if (writer_options.direct) { writer_options.written = written_ranges(plan); }
    if (journal) { writer_options.resume = journal->artifact_id(); }
    ArtifactWriter writer(options.out, std::move(layout), writer_options);
    if (options.consume_source) { writer.keep_partial(); }
    const auto writing = Clock::now();
    for (const auto& resource : plan.resources) {
        writer.write(resource.offset, resource.bytes->data(), resource.bytes->size());
    }
    for (const auto& auxiliary : plan.auxiliaries) {
        writer.write(auxiliary.offset, &auxiliary.word, 4);
    }
    std::optional<DriverIo> io;
    std::optional<Committer> committer;
    if (options.consume_source) {
        // The prologue and the journal are durable before anything is punched.
        writer.sync();
        sync_directory_of(options.out);
        if (!journal) {
            journal.emplace(journal_path, fingerprint, writer.artifact_id(), plan.weights.size());
        }
        io.emplace(writer, *journal, base);
        committer.emplace(*ledger, *io, order, durable);
    }
    Executor executor(stores, plan, writer, order, options.threads, options.keep_source_cache,
                      committer ? &*committer : nullptr, options.consume_window);
    log("converting " + std::to_string(order.size()) + " parents in " +
        std::to_string(executor.task_count()) + " tasks on " + std::to_string(executor.threads()) +
        " threads");
    try {
        executor.run();
    } catch (...) {
        if (options.consume_source) {
            log("consume-source: stopped; run the same command again to resume from " +
                journal_path.string());
        }
        throw;
    }
    result.write_seconds = seconds_since(writing);
    const auto finishing = Clock::now();
    writer.finish();
    sync_directory_of(options.out);
    if (!options.keep_source_cache) {
        for (const auto* store : stores) { store->release_all(); }
    }
    result.finish_seconds = seconds_since(finishing);

    report["artifact_id"] = hex_encode(writer.artifact_id().data(), writer.artifact_id().size());
    OJson files           = OJson::array();
    const auto parent     = options.out.parent_path().string();
    for (std::size_t i = 0; i < writer.files().size(); ++i) {
        OJson item            = OJson::object();
        item["path"]          = i == 0 ? output
                                       : py_path_str((parent.empty() ? std::string(".") : parent) + "/" +
                                                     *writer.files()[i].path);
        item["payload_bytes"] = writer.files()[i].payload_bytes;
        files.push_back(std::move(item));
    }
    report["files"]         = std::move(files);
    report["payload_bytes"] = writer.payload_bytes();
    report["seconds"]       = seconds_since(start);
    write_report(report_path, report);
    if (options.consume_source) {
        // Every shard is consumed apart from its header and the scalars planning read; the
        // artifact and its report are durable, so the shards and the journal go.
        result.punched_bytes = committer->punched_bytes();
        for (std::size_t s = 0; s < base.shard_count(); ++s) {
            std::filesystem::remove(base.shard_path(s));
        }
        sync_directory_of(base.shard_path(0));
        journal->remove();
    }

    result.seconds = seconds_since(begin);
    result.objects = plan.objects.size();
    result.files   = writer.files().size();
    result.report  = std::move(report);
    return result;
}

// The families with converters of their own are named here; the dense, MoE and Nemotron-H
// architectures and the GGUF ones come from the tables the converters read, so the list
// cannot fall behind them.
const std::string& supported_architectures_text() {
    static const std::string text = [] {
        std::string out =
            "Gemma4ForCausalLM, Gemma4ForConditionalGeneration (Gemma 4 26B-A4B, 31B), "
            "Gemma4UnifiedForConditionalGeneration (Gemma 4 12B), "
            "Gemma4AssistantForCausalLM, Gemma4UnifiedAssistantForCausalLM, DFlashDraftModel "
            "(Gemma 4 drafters), MuseGlimmerAssistantModel (Muse Glimmer's DFlash drafter), "
            "Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration, Qwen3_5MoeForCausalLM, "
            "Qwen3_5MoeForConditionalGeneration; dense, MoE and hybrid decoders:";
        const char* separator = " ";
        for (const std::string_view name : dense_architectures()) {
            out.append(separator).append(name);
            separator = ", ";
        }
        out += "; GGUF general.architecture:";
        separator = " ";
        for (const std::string& name : supported_gguf_architectures()) {
            out.append(separator).append(name);
            separator = ", ";
        }
        return out;
    }();
    return text;
}

namespace {

// What stops a Qwen3.5 recipe from planning on a checkpoint, or (`kept`) from keeping its
// published encodings as convert() would (nothing: nullopt). Planning reads headers and scalars
// only, never the bulk of the weights.
std::optional<std::string> qwen3_5_plan_error(const std::filesystem::path& checkpoint,
                                              const RecipeChoice& choice, bool kept = false) {
    try {
        const Checkpoint base(checkpoint);
        SourceInputs sources(base, choice.sources);
        // The components the default conversion keeps: the checkpoint's own MTP layers too.
        std::vector<std::string> components{"text"};
        if (base.has("mtp.fc.weight")) { components.emplace_back("mtp"); }
        Model model = build_qwen3_5_model(base, components, {}, {});
        Recipe recipe(model);
        apply_qwen3_5_recipe(choice.recipe, model, recipe, sources);
        (void)share_tied_head(model, recipe);
        if (kept) { (void)keep_published(model, recipe); }
        const Plan plan = prepare(model, recipe);
        if (kept) {
            const auto found = fidelity(base, model, plan);
            if (!found.foreign.empty()) {
                const auto& [handling, count] = *found.foreign.begin();
                return "it would change " + std::to_string(count) + " published matrices: " +
                       handling;
            }
        }
    } catch (const std::exception& error) {
        return std::string(error.what());
    }
    return std::nullopt;
}

// Whether the report beside `artifact` records `component` among its converted components.
bool records_component(const std::filesystem::path& artifact, const std::string& component) {
    std::ifstream stream(artifact.string() + ".conversion.json");
    if (!stream) { return false; }
    try {
        const OJson report     = OJson::parse(stream);
        const auto* components = member(report, "components");
        if (components == nullptr || !components->is_array()) { return false; }
        for (const auto& name : *components) {
            if (name.is_string() && name.get<std::string>() == component) { return true; }
        }
    } catch (const std::exception&) { return false; }
    return false;
}

// Whether an artifact of `recipe` was converted from a checkpoint directory: cache.cpp's
// <dir>/.ninferno-cache/[vision.]<recipe>.ninferno, or acquire.cpp's beside a hub snapshot,
// <repo>/ninferno/<commit>/model[.vision].<recipe>.ninferno; with `mtp`, one whose report
// records the checkpoint's MTP layers among its components.
bool converted(const std::filesystem::path& checkpoint, const std::string& recipe,
               bool mtp = false) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto cache = checkpoint / ".ninferno-cache";
    const auto hub   = checkpoint.parent_path().parent_path() / "ninferno" / checkpoint.filename();
    for (const auto& path : {cache / (recipe + ".ninferno"), cache / ("vision." + recipe + ".ninferno"),
                             hub / ("model." + recipe + ".ninferno"),
                             hub / ("model.vision." + recipe + ".ninferno")}) {
        if (fs::exists(path, ec) && (!mtp || records_component(path, "mtp"))) { return true; }
    }
    return false;
}

// Whether a Hugging Face checkpoint still holds its weight files (a consumed one keeps only its
// config, and its recipe choice then stays the configuration's).
bool has_safetensors(const std::filesystem::path& checkpoint) {
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(checkpoint, ec)) {
        if (entry.path().extension() == ".safetensors") { return true; }
    }
    return false;
}

// Whether a checkpoint's weight files hold MTP layers, which qwen3_5_plan_error plans with.
bool has_mtp_weights(const std::filesystem::path& checkpoint) {
    if (!has_safetensors(checkpoint)) { return false; }
    try {
        return Checkpoint(checkpoint).has("mtp.fc.weight");
    } catch (const std::exception&) { return false; }
}

} // namespace

RecipeChoice choose_recipe(const std::filesystem::path& checkpoint, GgufRequest request) {
    OJson config;
    bool nvfp4     = false;
    bool mxfp4     = false; // a GGUF with MXFP4 tensors (llama.cpp's MXFP4_MOE experts)
    bool quantized = false; // a Hugging Face checkpoint with a published quantization
    std::string header_note;
    const bool gguf = !gguf_model_files(checkpoint).empty();
    if (gguf) {
        const Checkpoint opened(checkpoint, request);
        config = opened.config();
        for (const auto& [name, tensor] : opened.tensors()) {
            nvfp4 = nvfp4 || tensor.ggml_type == ggml::NVFP4;
            mxfp4 = mxfp4 || tensor.ggml_type == ggml::MXFP4;
        }
    } else {
        // config.json only, so a consumed checkpoint (whose shards are gone) still answers.
        std::ifstream stream(checkpoint / "config.json");
        if (!stream) {
            throw std::runtime_error("no config.json or GGUF model in " + checkpoint.string() +
                                     "; supported architectures: " + supported_architectures_text());
        }
        config = OJson::parse(stream);
        const auto* text = member(config, "text_config");
        for (const OJson* scope : {static_cast<const OJson*>(&config), text}) {
            const auto* q = scope ? member(*scope, "quantization_config") : nullptr;
            if (!q) { continue; }
            const auto spelled = q->dump();
            nvfp4 = nvfp4 || spelled.find("nvfp4") != std::string::npos ||
                    spelled.find("NVFP4") != std::string::npos;
        }
        // A quantization_config says what a tool meant to write; the tensor headers say what the
        // checkpoint holds, and they decide whenever the shards are there (a consumed checkpoint
        // has only its config). A disagreement is stated in the recipe note, never taken silently.
        bool shards = false;
        for (const auto& entry : std::filesystem::directory_iterator(checkpoint)) {
            shards = shards || entry.path().extension() == ".safetensors";
        }
        if (shards) {
            const Checkpoint opened(checkpoint);
            std::size_t encoded = 0;
            for (const auto& [name, tensor] : opened.tensors()) {
                constexpr std::string_view kScale = ".weight_scale";
                if (name.size() > kScale.size() && name.ends_with(kScale) &&
                    nvfp4_layout(opened, name.substr(0, name.size() - kScale.size()))) {
                    ++encoded;
                }
            }
            if ((encoded > 0) != nvfp4) {
                header_note = nvfp4
                                  ? std::string("quantization_config names NVFP4 but no matrix "
                                                "in the checkpoint is NVFP4: chosen by the tensors")
                                  : "quantization_config does not name NVFP4 but " +
                                        std::to_string(encoded) +
                                        " matrices are NVFP4: chosen by the tensors";
                nvfp4       = encoded > 0;
            }
            // Any word that is not a plain float (FP8, packed integers, E2M1 bytes) is a
            // published quantization, whatever config.json says about it.
            quantized = false;
            for (const auto& [name, tensor] : opened.tensors()) {
                const auto& dtype = tensor.dtype_name;
                quantized = quantized || !(dtype == "BF16" || dtype == "F16" || dtype == "F32" ||
                                           dtype == "F64");
            }
        } else {
            for (const OJson* scope : {static_cast<const OJson*>(&config), text}) {
                quantized = quantized || (scope && member(*scope, "quantization_config"));
            }
        }
    }
    std::string architecture;
    try {
        architecture = declared_architecture(config);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(error.what()) +
                                 "; supported architectures: " + supported_architectures_text());
    }
    RecipeChoice out;
    if (is_qwen3_5_architecture(architecture)) {
        const auto* text = member(config, "text_config");
        const auto& raw  = text ? *text : config;
        const bool moe   = present(member(raw, "num_experts"));
        if (nvfp4) {
            // The NVFP4 recipes import the encoded projections from a quantized source: the
            // checkpoint itself.
            out.recipe  = moe ? "qwen3_6_35b_a3b_nvfp4" : "qwen3_6_27b_nvfp4";
            out.sources = {{"quantized", checkpoint}};
            if (gguf) {
                if (const auto error = qwen3_5_plan_error(checkpoint, out)) {
                    out.note    = out.recipe + " does not fit this GGUF (" + *error + ")";
                    out.recipe  = moe ? "qwen3_6_35b_a3b" : "qwen3_5_bf16";
                    out.sources = {};
                }
            } else if (!converted(checkpoint, out.recipe) ||
                       (!converted(checkpoint, out.recipe, true) && has_mtp_weights(checkpoint))) {
                // Those recipes fit one layout (NVFP4 experts or MLPs, the rest BF16); another
                // publisher's (RedHatAI: NVFP4 attention; nvidia: ModelOpt experts, FP8
                // attention, an NVFP4 head) runs every tensor as published. A converted
                // artifact of either recipe answers without planning (every serve start, and a
                // consumed checkpoint, whose shards are gone), unless it was converted without
                // the checkpoint's MTP layers (--no-drafters): it then proves nothing about
                // them, and a later default conversion keeps them.
                if (converted(checkpoint, "qwen3_5_published")) {
                    out.recipe  = "qwen3_5_published";
                    out.sources = {};
                } else if (!has_safetensors(checkpoint)) {
                    // Nothing to plan on: the configuration's recipe, as before.
                } else if (const auto error = qwen3_5_plan_error(checkpoint, out, true)) {
                    out.note    = out.recipe + " does not fit this checkpoint (" + *error + ")";
                    out.recipe  = "qwen3_5_published";
                    out.sources = {};
                }
            }
        } else if (gguf && mxfp4) {
            // MXFP4 blocks are NVFP4 words with power-of-two scales (gguf_model.cpp): the
            // published recipe keeps them so (weight-only, BF16 activations: no calibrated
            // divisor) and every Q8_0 / K-quant tensor beside them word for word.
            out.recipe = "qwen3_5_published";
            if (const auto error = qwen3_5_plan_error(checkpoint, out)) {
                out.note   = out.recipe + " does not fit this GGUF (" + *error + ")";
                out.recipe = moe ? "qwen3_6_35b_a3b" : "qwen3_5_bf16";
            }
        } else {
            // Unquantized projections run as published; the MoE has no unquantized recipe, and
            // its maintainer recipe is refused unless forced.
            out.recipe = moe ? "qwen3_6_35b_a3b" : "qwen3_5_bf16";
        }
        if (!header_note.empty()) {
            out.note = out.note.empty() ? header_note : header_note + "; " + out.note;
        }
        return out;
    }
    if (is_dense_architecture(architecture)) {
        out.recipe = default_dense_recipe(config, quantized);
        return out;
    }
    try {
        out.recipe = default_recipe(config);
    } catch (const std::exception& error) {
        throw std::runtime_error("no official recipe for architecture '" + architecture +
                                 "'; supported architectures: " + supported_architectures_text());
    }
    return out;
}

} // namespace ninferno::convert
