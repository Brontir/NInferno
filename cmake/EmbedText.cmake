# ninferno_embed_text(FILE DELIMITER OUT_VAR): OUT_VAR becomes the initializer of a char array
# holding FILE's text and a terminating NUL, for a header made by configure_file:
#
#   inline constexpr char kText[] = @OUT_VAR@;
#
# A raw string literal R"DELIMITER(...)DELIMITER" where the compiler takes long literals; on MSVC,
# which caps a string literal at 64 KB even after concatenation (C2026), the bytes one by one.
function(ninferno_embed_text file delimiter out_var)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${file})
  if(MSVC)
    file(READ ${file} hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "'\\\\x\\1'," hex "${hex}")
    set(${out_var} "{${hex}'\\0'}" PARENT_SCOPE)
  else()
    file(READ ${file} text)
    set(${out_var} "R\"${delimiter}(${text})${delimiter}\"" PARENT_SCOPE)
  endif()
endfunction()
