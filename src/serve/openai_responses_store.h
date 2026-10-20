#pragma once

// Process-local bounded storage for OpenAI Responses objects and their previous_response_id
// context DAG. Stored contexts are flattened when a continuation is submitted; the bounded session
// key is only an Engine lookup/retention hint and this store owns no Program capability.

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninferno::serve {

struct OpenAIResponseContextNode {
    std::shared_ptr<const OpenAIResponseContextNode> parent;
    std::vector<ChatTurn> turns;
    std::size_t owned_bytes      = 0;
    std::size_t cumulative_bytes = 0;
    std::size_t cumulative_turns = 0;
};

using OpenAIResponseContext = std::shared_ptr<const OpenAIResponseContextNode>;

OpenAIResponseContext append_openai_response_context(OpenAIResponseContext parent,
                                                     std::vector<ChatTurn> turns);
std::vector<ChatTurn> flatten_openai_response_context(const OpenAIResponseContext& context);

// The bytes a stored Response will hold for its request side alone: the previous context, the
// node of the new input turns, and the normalized input Items. The stored record also holds the
// response body and output turns, so a request above the store's capacity cannot be stored.
[[nodiscard]] std::size_t openai_response_input_bytes(const OpenAIResponseContext& previous,
                                                      const std::vector<ChatTurn>& input_turns,
                                                      const std::vector<nlohmann::json>& items);

struct StoredOpenAIResponse {
    std::string id;
    std::string session_key;
    nlohmann::json response;
    std::vector<nlohmann::json> input_items;
    OpenAIResponseContext context;
    std::optional<bool> preserve_thinking;
};

class OpenAIResponsesStore {
public:
    OpenAIResponsesStore(std::size_t max_records, std::size_t max_bytes);

    // get() refreshes LRU recency. Returned immutable records remain valid if
    // another request evicts or deletes their public store entry.
    std::shared_ptr<const StoredOpenAIResponse> get(const std::string& id);
    // false when the response alone (its body, input items and context) is more than the store
    // holds: nothing is stored or evicted, and the caller answers with it unstored.
    [[nodiscard]] bool put(StoredOpenAIResponse response);
    // Throws response_store_capacity_exceeded (400) before generation when the request side of a
    // Response (openai_response_input_bytes) already exceeds the store's capacity.
    void admit(std::size_t input_bytes) const;
    bool erase(const std::string& id);

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t bytes() const;

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return max_bytes_; }

private:
    struct Entry {
        std::shared_ptr<const StoredOpenAIResponse> response;
        std::list<std::string>::iterator lru;
        std::size_t envelope_bytes = 0;
    };

    void retain_context_locked(const OpenAIResponseContext& context);
    void release_context_locked(const OpenAIResponseContext& context);
    void erase_locked(const std::string& id);

    std::size_t max_records_ = 0;
    std::size_t max_bytes_   = 0;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> records_;
    std::list<std::string> lru_;
    std::unordered_map<const OpenAIResponseContextNode*, std::size_t> live_context_references_;
    std::size_t current_bytes_ = 0;
};

} // namespace ninferno::serve
