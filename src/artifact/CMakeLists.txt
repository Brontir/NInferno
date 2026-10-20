add_library(ninferno_artifact STATIC
  schema.cpp
  formats.cpp
  layouts.cpp
  file_io.cpp
  reader.cpp
  binder.cpp
  materializer.cpp
  expert_groups.cpp
  views.cpp
)
ninferno_internal_includes(ninferno_artifact)
target_link_libraries(ninferno_artifact PUBLIC ninferno_core ninferno::json)
