configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(JFORCE_TEST_DEPENDS
  FileCheck
  jforce-opt
)

add_lit_testsuite(
  check-jforce
  "Running JForce regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${JFORCE_TEST_DEPENDS}
)

set_target_properties(
  check-jforce
  PROPERTIES FOLDER "Tests"
)
