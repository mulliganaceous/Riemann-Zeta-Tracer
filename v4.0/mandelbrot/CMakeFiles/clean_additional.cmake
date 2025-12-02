# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "RelWithDebInfo")
  file(REMOVE_RECURSE
  "CMakeFiles/mandelbrot_autogen.dir/AutogenUsed.txt"
  "CMakeFiles/mandelbrot_autogen.dir/ParseCache.txt"
  "mandelbrot_autogen"
  )
endif()
