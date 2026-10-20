#include "serve/openai_responses_store.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninferno::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

ChatTurn text_turn(ninferno::ChatRole role, std::string text) {
    ChatTurn turn;
    turn.role = role;
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "input_text";
    part.text     = std::move(text);
    turn.content.push_back(std::move(part));
    return turn;
}

StoredOpenAIResponse record(std::string id, OpenAIResponseContext context) {
    StoredOpenAIResponse value;
    value.id          = std::move(id);
    value.session_key = "session-" + value.id;
    value.response =
        nlohmann::json{{"id", value.id}, {"object", "response"}, {"status", "completed"}};
    value.input_items.push_back(nlohmann::json{{"id", "msg_" + value.id}, {"type", "message"}});
    value.context = std::move(context);
    return value;
}

int test_context_dag() {
    const OpenAIResponseContext first =
        append_openai_response_context({}, {text_turn(ninferno::ChatRole::User, "one"),
                                            text_turn(ninferno::ChatRole::Assistant, "a")});
    const OpenAIResponseContext second =
        append_openai_response_context(first, {text_turn(ninferno::ChatRole::User, "two"),
                                               text_turn(ninferno::ChatRole::Assistant, "b")});
    const std::vector<ChatTurn> flattened = flatten_openai_response_context(second);
    int failures                          = 0;
    failures += check(flattened.size() == 4, "context chain flattened all turns");
    failures += check(flattened[0].content[0].text == "one" && flattened[3].content[0].text == "b",
                      "context chain preserves chronological order");
    failures += check(second->parent.get() == first.get(), "context nodes share their parent");
    return failures;
}

int test_lru_and_delete() {
    OpenAIResponsesStore store(2, 1ULL << 20);
    const OpenAIResponseContext root =
        append_openai_response_context({}, {text_turn(ninferno::ChatRole::User, "root")});
    (void)store.put(record("resp_1", root));
    const OpenAIResponseContext child =
        append_openai_response_context(root, {text_turn(ninferno::ChatRole::Assistant, "child")});
    (void)store.put(record("resp_2", child));
    (void)store.get("resp_1"); // resp_2 becomes the least-recently used entry.
    (void)store.put(record("resp_3", append_openai_response_context(
                                         root, {text_turn(ninferno::ChatRole::User, "fork")})));

    int failures = 0;
    failures += check(store.get("resp_1") != nullptr, "get refreshes LRU recency");
    failures += check(store.get("resp_2") == nullptr, "least-recent response evicted");
    failures += check(store.get("resp_3") != nullptr, "new response retained");
    failures += check(store.size() == 2 && store.bytes() != 0, "store reports bounded usage");
    failures += check(store.erase("resp_1"), "stored response deleted");
    failures += check(!store.erase("resp_1") && store.get("resp_1") == nullptr,
                      "deleted response is no longer addressable");
    // The child/fork context owns a shared parent even when the parent's public
    // response entry is deleted.
    const std::shared_ptr<const StoredOpenAIResponse> fork = store.get("resp_3");
    failures += check(fork && flatten_openai_response_context(fork->context).size() == 2,
                      "descendant context survives parent response deletion");
    return failures;
}

int test_oversized_record() {
    OpenAIResponsesStore store(4, 256);
    StoredOpenAIResponse large = record(
        "resp_large", append_openai_response_context(
                          {}, {text_turn(ninferno::ChatRole::User, std::string(1024, 'x'))}));
    int failures = 0;
    failures += check(!store.put(std::move(large)),
                      "an oversized response is not stored (the caller answers it unstored)");
    failures += check(store.size() == 0, "oversized insertion does not mutate store");
    return failures;
}

// A request whose input alone exceeds the store is refused before generation (400); the bound is
// below what put() needs, so an admitted input can still be too large for put() with its output
// (answered unstored), never the reverse.
int test_input_admission() {
    OpenAIResponsesStore store(4, 2048);
    int failures = 0;
    const std::vector<nlohmann::json> items{nlohmann::json{{"type", "message"}}};
    const std::vector<ChatTurn> small{text_turn(ninferno::ChatRole::User, "hi")};
    const std::vector<ChatTurn> large{text_turn(ninferno::ChatRole::User, std::string(4096, 'x'))};
    try {
        store.admit(openai_response_input_bytes({}, small, items));
    } catch (const ApiException&) { failures += check(false, "a small input is admitted"); }
    ApiError refused;
    try {
        store.admit(openai_response_input_bytes({}, large, items));
    } catch (const ApiException& exception) { refused = exception.error(); }
    failures += check(refused.code == "response_store_capacity_exceeded" && refused.status == 400,
                      "an input above the store's capacity is refused with 400");
    const auto previous = append_openai_response_context({}, large);
    refused             = {};
    try {
        store.admit(openai_response_input_bytes(previous, small, items));
    } catch (const ApiException& exception) { refused = exception.error(); }
    failures += check(refused.code == "response_store_capacity_exceeded",
                      "a previous context above the store's capacity is refused");
    const std::size_t bound     = openai_response_input_bytes({}, small, items);
    StoredOpenAIResponse stored = record("resp_small", append_openai_response_context({}, small));
    stored.input_items          = items;
    failures +=
        check(bound <= sizeof(StoredOpenAIResponse) + stored.id.size() + stored.session_key.size() +
                           stored.response.dump().size() + sizeof(nlohmann::json) +
                           items[0].dump().size() + stored.context->cumulative_bytes,
              "the admission bound does not exceed what put() accounts");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_context_dag();
    failures += test_lru_and_delete();
    failures += test_oversized_record();
    failures += test_input_admission();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
