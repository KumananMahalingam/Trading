Package: spdlog:x64-windows@1.13.0

**Host Environment**

- Host: x64-windows
- Compiler: MSVC 19.44.35228.0
- CMake Version: 4.3.2
-    vcpkg-tool version: 2026-05-27-d5b6777d666efc1a7f491babfcdab37794c1ae3e
    vcpkg-scripts version: cac7d29f24 2026-06-16 (5 hours ago)

**To Reproduce**

`vcpkg install `

**Failure logs**

```
Downloading https://github.com/gabime/spdlog/archive/v1.13.0.tar.gz -> gabime-spdlog-v1.13.0.tar.gz
Successfully downloaded gabime-spdlog-v1.13.0.tar.gz
-- Extracting source C:/vcpkg/downloads/gabime-spdlog-v1.13.0.tar.gz
-- Using source at C:/vcpkg/buildtrees/spdlog/src/v1.13.0-f032863ee1.clean
-- Configuring x64-windows
-- Building x64-windows-dbg
CMake Error at scripts/cmake/vcpkg_execute_build_process.cmake:134 (message):
    Command failed: C:/vcpkg/downloads/tools/cmake-4.3.2-windows/cmake-4.3.2-windows-x86_64/bin/cmake.exe --build . --config Debug --target install -- -v -j13
    Working Directory: C:/vcpkg/buildtrees/spdlog/x64-windows-dbg
    See logs for more information:
      C:\vcpkg\buildtrees\spdlog\install-x64-windows-dbg-out.log

Call Stack (most recent call first):
  C:/Users/mkuma/Trading/cpp/vcpkg_installed/x64-windows/share/vcpkg-cmake/vcpkg_cmake_build.cmake:74 (vcpkg_execute_build_process)
  C:/Users/mkuma/Trading/cpp/vcpkg_installed/x64-windows/share/vcpkg-cmake/vcpkg_cmake_install.cmake:16 (vcpkg_cmake_build)
  buildtrees/versioning_/versions/spdlog/8ee97c0faf23e06508ca097e013e2bea40579d06/portfile.cmake:36 (vcpkg_cmake_install)
  scripts/ports.cmake:206 (include)



```

<details><summary>C:\vcpkg\buildtrees\spdlog\install-x64-windows-dbg-out.log</summary>

```
Change Dir: 'C:/vcpkg/buildtrees/spdlog/x64-windows-dbg'

Run Build Command(s): C:\vcpkg\downloads\tools\ninja-1.13.2-windows\ninja.exe -v -v -j13 install
[1/9] C:/vcpkg/DOWNLO~1/tools/CMAKE-~1.2-W/CMAKE-~1.2-W/bin/cmcldeps.exe RC C:\vcpkg\buildtrees\spdlog\x64-windows-dbg\version.rc CMakeFiles\spdlog.dir\version.rc.res.d CMakeFiles\spdlog.dir\version.rc.res "Note: including file: " "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" C:\PROGRA~2\WI3CF2~1\10\bin\100261~1.0\x64\rc.exe -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -I C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -I C:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include /c65001 /DWIN32 -D_DEBUG /fo CMakeFiles\spdlog.dir\version.rc.res C:\vcpkg\buildtrees\spdlog\x64-windows-dbg\version.rc
[2/9] C:\PROGRA~2\MICROS~2\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\cl.exe   /TP -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -IC:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -external:IC:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include -external:W0 /nologo /DWIN32 /D_WINDOWS /utf-8 /GR /EHsc /MP  /Zc:__cplusplus /MP /MDd /Z7 /Ob0 /Od /RTC1  /wd4251 /wd4275 /utf-8 /showIncludes /FoCMakeFiles\spdlog.dir\src\async.cpp.obj /FdCMakeFiles\spdlog.dir\ /FS -c C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\src\async.cpp
FAILED: [code=2] CMakeFiles/spdlog.dir/src/async.cpp.obj 
C:\PROGRA~2\MICROS~2\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\cl.exe   /TP -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -IC:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -external:IC:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include -external:W0 /nologo /DWIN32 /D_WINDOWS /utf-8 /GR /EHsc /MP  /Zc:__cplusplus /MP /MDd /Z7 /Ob0 /Od /RTC1  /wd4251 /wd4275 /utf-8 /showIncludes /FoCMakeFiles\spdlog.dir\src\async.cpp.obj /FdCMakeFiles\spdlog.dir\ /FS -c C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\src\async.cpp
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2039: 'basic_format_string': is not a member of 'fmt'
C:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include\fmt\format.h(194): note: see declaration of 'fmt'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2065: 'basic_format_string': undeclared identifier
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2275: 'Args': expected an expression instead of a type
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2988: unrecognizable template declaration/definition
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2059: syntax error: '>'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2143: syntax error: missing ';' before '{'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2447: '{': missing function header (old-style formal list?)
[3/9] C:\PROGRA~2\MICROS~2\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\cl.exe   /TP -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -IC:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -external:IC:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include -external:W0 /nologo /DWIN32 /D_WINDOWS /utf-8 /GR /EHsc /MP  /Zc:__cplusplus /MP /MDd /Z7 /Ob0 /Od /RTC1  /wd4251 /wd4275 /utf-8 /showIncludes /FoCMakeFiles\spdlog.dir\src\file_sinks.cpp.obj /FdCMakeFiles\spdlog.dir\ /FS -c C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\src\file_sinks.cpp
FAILED: [code=2] CMakeFiles/spdlog.dir/src/file_sinks.cpp.obj 
C:\PROGRA~2\MICROS~2\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\cl.exe   /TP -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -IC:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -external:IC:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include -external:W0 /nologo /DWIN32 /D_WINDOWS /utf-8 /GR /EHsc /MP  /Zc:__cplusplus /MP /MDd /Z7 /Ob0 /Od /RTC1  /wd4251 /wd4275 /utf-8 /showIncludes /FoCMakeFiles\spdlog.dir\src\file_sinks.cpp.obj /FdCMakeFiles\spdlog.dir\ /FS -c C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\src\file_sinks.cpp
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2039: 'basic_format_string': is not a member of 'fmt'
C:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include\fmt\format.h(194): note: see declaration of 'fmt'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2065: 'basic_format_string': undeclared identifier
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2275: 'Args': expected an expression instead of a type
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2988: unrecognizable template declaration/definition
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2059: syntax error: '>'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2143: syntax error: missing ';' before '{'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2447: '{': missing function header (old-style formal list?)
[4/9] C:\PROGRA~2\MICROS~2\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\cl.exe   /TP -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -IC:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -external:IC:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include -external:W0 /nologo /DWIN32 /D_WINDOWS /utf-8 /GR /EHsc /MP  /Zc:__cplusplus /MP /MDd /Z7 /Ob0 /Od /RTC1  /wd4251 /wd4275 /utf-8 /showIncludes /FoCMakeFiles\spdlog.dir\src\cfg.cpp.obj /FdCMakeFiles\spdlog.dir\ /FS -c C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\src\cfg.cpp
FAILED: [code=2] CMakeFiles/spdlog.dir/src/cfg.cpp.obj 
C:\PROGRA~2\MICROS~2\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\cl.exe   /TP -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -IC:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -external:IC:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include -external:W0 /nologo /DWIN32 /D_WINDOWS /utf-8 /GR /EHsc /MP  /Zc:__cplusplus /MP /MDd /Z7 /Ob0 /Od /RTC1  /wd4251 /wd4275 /utf-8 /showIncludes /FoCMakeFiles\spdlog.dir\src\cfg.cpp.obj /FdCMakeFiles\spdlog.dir\ /FS -c C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\src\cfg.cpp
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2039: 'basic_format_string': is not a member of 'fmt'
C:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include\fmt\format.h(194): note: see declaration of 'fmt'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2065: 'basic_format_string': undeclared identifier
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2275: 'Args': expected an expression instead of a type
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2988: unrecognizable template declaration/definition
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2059: syntax error: '>'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2143: syntax error: missing ';' before '{'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2447: '{': missing function header (old-style formal list?)
[5/9] C:\PROGRA~2\MICROS~2\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\cl.exe   /TP -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -IC:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -external:IC:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include -external:W0 /nologo /DWIN32 /D_WINDOWS /utf-8 /GR /EHsc /MP  /Zc:__cplusplus /MP /MDd /Z7 /Ob0 /Od /RTC1  /wd4251 /wd4275 /utf-8 /showIncludes /FoCMakeFiles\spdlog.dir\src\color_sinks.cpp.obj /FdCMakeFiles\spdlog.dir\ /FS -c C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\src\color_sinks.cpp
FAILED: [code=2] CMakeFiles/spdlog.dir/src/color_sinks.cpp.obj 
C:\PROGRA~2\MICROS~2\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\cl.exe   /TP -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -IC:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -external:IC:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include -external:W0 /nologo /DWIN32 /D_WINDOWS /utf-8 /GR /EHsc /MP  /Zc:__cplusplus /MP /MDd /Z7 /Ob0 /Od /RTC1  /wd4251 /wd4275 /utf-8 /showIncludes /FoCMakeFiles\spdlog.dir\src\color_sinks.cpp.obj /FdCMakeFiles\spdlog.dir\ /FS -c C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\src\color_sinks.cpp
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2039: 'basic_format_string': is not a member of 'fmt'
C:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include\fmt\format.h(194): note: see declaration of 'fmt'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2065: 'basic_format_string': undeclared identifier
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2275: 'Args': expected an expression instead of a type
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2988: unrecognizable template declaration/definition
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2059: syntax error: '>'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2143: syntax error: missing ';' before '{'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2447: '{': missing function header (old-style formal list?)
[6/9] C:\PROGRA~2\MICROS~2\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\cl.exe   /TP -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -IC:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -external:IC:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include -external:W0 /nologo /DWIN32 /D_WINDOWS /utf-8 /GR /EHsc /MP  /Zc:__cplusplus /MP /MDd /Z7 /Ob0 /Od /RTC1  /wd4251 /wd4275 /utf-8 /showIncludes /FoCMakeFiles\spdlog.dir\src\stdout_sinks.cpp.obj /FdCMakeFiles\spdlog.dir\ /FS -c C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\src\stdout_sinks.cpp
FAILED: [code=2] CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.obj 
C:\PROGRA~2\MICROS~2\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\cl.exe   /TP -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -IC:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -external:IC:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include -external:W0 /nologo /DWIN32 /D_WINDOWS /utf-8 /GR /EHsc /MP  /Zc:__cplusplus /MP /MDd /Z7 /Ob0 /Od /RTC1  /wd4251 /wd4275 /utf-8 /showIncludes /FoCMakeFiles\spdlog.dir\src\stdout_sinks.cpp.obj /FdCMakeFiles\spdlog.dir\ /FS -c C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\src\stdout_sinks.cpp
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2039: 'basic_format_string': is not a member of 'fmt'
C:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include\fmt\format.h(194): note: see declaration of 'fmt'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2065: 'basic_format_string': undeclared identifier
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2275: 'Args': expected an expression instead of a type
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2988: unrecognizable template declaration/definition
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2059: syntax error: '>'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2143: syntax error: missing ';' before '{'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2447: '{': missing function header (old-style formal list?)
[7/9] C:\PROGRA~2\MICROS~2\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\cl.exe   /TP -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -IC:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -external:IC:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include -external:W0 /nologo /DWIN32 /D_WINDOWS /utf-8 /GR /EHsc /MP  /Zc:__cplusplus /MP /MDd /Z7 /Ob0 /Od /RTC1  /wd4251 /wd4275 /utf-8 /showIncludes /FoCMakeFiles\spdlog.dir\src\spdlog.cpp.obj /FdCMakeFiles\spdlog.dir\ /FS -c C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\src\spdlog.cpp
FAILED: [code=2] CMakeFiles/spdlog.dir/src/spdlog.cpp.obj 
C:\PROGRA~2\MICROS~2\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\cl.exe   /TP -DFMT_SHARED -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS -IC:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include -external:IC:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include -external:W0 /nologo /DWIN32 /D_WINDOWS /utf-8 /GR /EHsc /MP  /Zc:__cplusplus /MP /MDd /Z7 /Ob0 /Od /RTC1  /wd4251 /wd4275 /utf-8 /showIncludes /FoCMakeFiles\spdlog.dir\src\spdlog.cpp.obj /FdCMakeFiles\spdlog.dir\ /FS -c C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\src\spdlog.cpp
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2039: 'basic_format_string': is not a member of 'fmt'
C:\Users\mkuma\Trading\cpp\vcpkg_installed\x64-windows\include\fmt\format.h(194): note: see declaration of 'fmt'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2065: 'basic_format_string': undeclared identifier
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2275: 'Args': expected an expression instead of a type
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2988: unrecognizable template declaration/definition
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2059: syntax error: '>'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2143: syntax error: missing ';' before '{'
C:\vcpkg\buildtrees\spdlog\src\v1.13.0-f032863ee1.clean\include\spdlog/common.h(369): error C2447: '{': missing function header (old-style formal list?)
ninja: build stopped: subcommand failed.
```
</details>

**Additional context**

<details><summary>vcpkg.json</summary>

```
{
  "name": "trading-lstm-cpp",
  "version": "1.0.0",
  "description": "C++ migration of the Python LSTM trading platform (5-feature single-stream model).",
  "builtin-baseline": "cac7d29f246eb789a84004fbfc11bb00db1aa89b",
  "dependencies": [
    {
      "name": "curl",
      "version>=": "8.6.0"
    },
    {
      "name": "nlohmann-json",
      "version>=": "3.11.3"
    },
    {
      "name": "eigen3",
      "version>=": "3.4.0"
    },
    {
      "name": "xlnt",
      "version>=": "1.5.0"
    },
    {
      "name": "spdlog",
      "version>=": "1.13.0"
    }
  ],
  "overrides": [
    {
      "name": "curl",
      "version": "8.6.0"
    },
    {
      "name": "nlohmann-json",
      "version": "3.11.3"
    },
    {
      "name": "eigen3",
      "version": "3.4.0"
    },
    {
      "name": "xlnt",
      "version": "1.5.0"
    },
    {
      "name": "spdlog",
      "version": "1.13.0"
    }
  ]
}

```
</details>
