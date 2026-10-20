#include "ninferno/types.h"

#include <utility>

namespace ninferno {

CancellationView::CancellationView(std::function<bool()> requested)
    : requested_(std::move(requested)) {}

bool CancellationView::requested() const { return requested_ && requested_(); }

} // namespace ninferno
