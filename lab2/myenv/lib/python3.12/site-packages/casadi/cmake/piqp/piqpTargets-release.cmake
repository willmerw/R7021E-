#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "piqp::piqp_c" for configuration "Release"
set_property(TARGET piqp::piqp_c APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(piqp::piqp_c PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libpiqpc.so"
  IMPORTED_SONAME_RELEASE "libpiqpc.so"
  )

list(APPEND _cmake_import_check_targets piqp::piqp_c )
list(APPEND _cmake_import_check_files_for_piqp::piqp_c "${_IMPORT_PREFIX}/lib/libpiqpc.so" )

# Import target "piqp::piqp" for configuration "Release"
set_property(TARGET piqp::piqp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(piqp::piqp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libpiqp.so"
  IMPORTED_SONAME_RELEASE "libpiqp.so"
  )

list(APPEND _cmake_import_check_targets piqp::piqp )
list(APPEND _cmake_import_check_files_for_piqp::piqp "${_IMPORT_PREFIX}/lib/libpiqp.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
