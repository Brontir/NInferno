add_library(ninferno_media_decode STATIC
  decode/decode.cpp
)
ninferno_internal_includes(ninferno_media_decode)
target_link_libraries(ninferno_media_decode PRIVATE PkgConfig::FFMPEG)

# PDF text through pdfium, loaded at run time: no build dependency.
add_library(ninferno_media_pdf STATIC
  pdf/pdf_text.cpp
)
ninferno_internal_includes(ninferno_media_pdf)
target_link_libraries(ninferno_media_pdf PRIVATE ninferno_core ${CMAKE_DL_LIBS})
