if(RVML_ENABLE_TORCH_IMPORT)
  rvml_add_library(RVMLImporter
    TorchImporter.cpp

    LINK_LIBS
    MLIRIR
    MLIRSupport
  )
  target_include_directories(RVMLImporter SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})
  target_link_libraries(obj.RVMLImporter PRIVATE ${TORCH_LIBRARIES})
  target_link_libraries(RVMLImporter PRIVATE ${TORCH_LIBRARIES})
endif()
