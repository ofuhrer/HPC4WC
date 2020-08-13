add_library (m_utils OBJECT m_utils.f)
target_link_libraries (m_utils PUBLIC MPI::MPI_Fortran)
add_library (utils INTERFACE)
target_sources (utils INTERFACE $<TARGET_OBJECTS:m_utils>)
