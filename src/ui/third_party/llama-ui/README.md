# llama-ui (third party)

ninferno-ui's chat window is llama.cpp's web UI, `tools/ui` of https://github.com/ggml-org/llama.cpp
(MIT, copyright the ggml authors; LICENSE here). It is not vendored: src/ui/CMakeLists.txt downloads
the prebuilt bundle llama.cpp's own build uses, from the Hugging Face bucket `ggml-org/llama-ui`,
pinned by version (NINFERNO_UI_CHAT_VERSION) and SHA-256 (NINFERNO_UI_CHAT_SHA256), and copies this
LICENSE beside it. ninferno-ui serves the bundle unchanged under /chat/<id>/ and answers its `./props`
itself (src/ui/chat_ui.h).

To move to a newer bundle: read `https://huggingface.co/buckets/ggml-org/llama-ui/resolve/<bNNNN>/dist.tar.gz.sha256`
and update both variables.
