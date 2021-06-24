add_library (m_halo_seq OBJECT halo_seq.f)
add_library (halo_seq INTERFACE)
target_sources (halo_seq INTERFACE $<TARGET_OBJECTS:m_halo_seq>)
target_compile_definitions (halo_seq INTERFACE -D m_halo=m_halo_seq)
