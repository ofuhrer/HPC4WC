add_library (m_partitioner OBJECT m_partitioner.f)
target_link_libraries (m_partitioner PUBLIC utils MPI::MPI_Fortran)
add_library (partitioner INTERFACE)
target_sources (partitioner INTERFACE $<TARGET_OBJECTS:m_partitioner>)
