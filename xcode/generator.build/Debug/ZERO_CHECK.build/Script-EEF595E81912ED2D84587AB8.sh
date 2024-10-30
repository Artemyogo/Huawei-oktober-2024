#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/artemi/Documents/repos/Huawei-oktober-2024/xcode
  make -f /Users/artemi/Documents/repos/Huawei-oktober-2024/xcode/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/artemi/Documents/repos/Huawei-oktober-2024/xcode
  make -f /Users/artemi/Documents/repos/Huawei-oktober-2024/xcode/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/artemi/Documents/repos/Huawei-oktober-2024/xcode
  make -f /Users/artemi/Documents/repos/Huawei-oktober-2024/xcode/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/artemi/Documents/repos/Huawei-oktober-2024/xcode
  make -f /Users/artemi/Documents/repos/Huawei-oktober-2024/xcode/CMakeScripts/ReRunCMake.make
fi

