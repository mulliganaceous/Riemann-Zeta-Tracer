# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Release")
  file(REMOVE_RECURSE
  "CMakeFiles/securesocketclient_autogen.dir/AutogenUsed.txt"
  "CMakeFiles/securesocketclient_autogen.dir/ParseCache.txt"
  "securesocketclient_autogen"
  )
endif()
