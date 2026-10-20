#pragma once

// Gemma 4 text (26B-A4B MoE, 31B, 12B and the E2B/E4B models with per-layer embeddings) as its
// own model family.
//
// This is not a variant of qwen3_5: the layer schedule differs at every point (sandwich norms,
// two attention geometries in one model, K reused as V on the full layers, a dense MLP running in
// parallel with the expert bank, a per-layer output scalar, a soft-capped head), and there is no
// recurrent state at all. The family owns its artifact binding, its prepared parameters, its two
// KV pools and its layers' execution; it shares Ops, the artifact reader, the core types and the
// Qwen 3.5 Program's scheduling machinery (ring KV bindings, prefill calls), which the server
// runs every family on.
//
// The applications' single include. Family code includes the narrower header it needs: config.h,
// weights.h, model.h or session.h.

#include "models/gemma4/config.h"
#include "models/gemma4/model.h"
#include "models/gemma4/session.h"
#include "models/gemma4/weights.h"
