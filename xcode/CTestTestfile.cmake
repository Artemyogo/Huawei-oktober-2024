# CMake generated Testfile for 
# Source directory: /Users/artemi/Documents/repos/Huawei-oktober-2024
# Build directory: /Users/artemi/Documents/repos/Huawei-oktober-2024/xcode
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[small_minsk]=] "main" "<")
set_tests_properties([=[small_minsk]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/artemi/Documents/repos/Huawei-oktober-2024/CMakeLists.txt;10;add_test;/Users/artemi/Documents/repos/Huawei-oktober-2024/CMakeLists.txt;0;")
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test([=[main]=] "")
  set_tests_properties([=[main]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/artemi/Documents/repos/Huawei-oktober-2024/CMakeLists.txt;12;add_test;/Users/artemi/Documents/repos/Huawei-oktober-2024/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test([=[main]=] "")
  set_tests_properties([=[main]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/artemi/Documents/repos/Huawei-oktober-2024/CMakeLists.txt;12;add_test;/Users/artemi/Documents/repos/Huawei-oktober-2024/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test([=[main]=] "")
  set_tests_properties([=[main]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/artemi/Documents/repos/Huawei-oktober-2024/CMakeLists.txt;12;add_test;/Users/artemi/Documents/repos/Huawei-oktober-2024/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test([=[main]=] "")
  set_tests_properties([=[main]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/artemi/Documents/repos/Huawei-oktober-2024/CMakeLists.txt;12;add_test;/Users/artemi/Documents/repos/Huawei-oktober-2024/CMakeLists.txt;0;")
else()
  add_test([=[main]=] NOT_AVAILABLE)
endif()
