#pragma once

// Frontend resources rebuilt from GGUF metadata, for a GGUF file that travels without its
// Hugging Face tokenizer files:
//
//   chat_template.jinja     <- tokenizer.chat_template
//   tokenizer.json          <- tokenizer.ggml.{tokens,token_type,merges,scores}, for the
//                              tokenizers the engine reads: "gpt2" (byte-level BPE) with the
//                              pre-tokenizers qwen35, qwen2, llama-bpe, dbrx, smaug-bpe, tekken,
//                              gpt-4o, smollm and refact; "gemma4" (Gemma 4: space-to-U+2581,
//                              byte-fallback BPE); "llama" (SentencePiece BPE: merges from the
//                              piece scores, byte fallback, U+2581 space prefix)
//   tokenizer_config.json   <- added tokens (CONTROL -> special, USER_DEFINED -> not), and the
//                              bos/eos/pad/unk token strings
//   generation_config.json  <- tokenizer.ggml.{bos,eos,eot,eom,padding}_token_id and
//                              general.sampling.*
//
// What llama.cpp's converter did not keep cannot come back: it marks an added token CONTROL when
// it looks special ("<|...|>") even if the Hugging Face tokenizer called it not special, and it
// pads the token list to vocab_size with [PAD<i>] entries (dropped here). Files next to the GGUF
// and --resource overrides always win over these.

#include "convert/gguf.h"

#include <map>
#include <string>
#include <vector>

namespace ninferno::convert {

struct RebuiltResources {
    std::map<std::string, std::string> files; // role -> bytes
    std::vector<std::string> notes;           // what was rebuilt or could not be
};

[[nodiscard]] RebuiltResources gguf_frontend_resources(const GgufFile& file);

} // namespace ninferno::convert
