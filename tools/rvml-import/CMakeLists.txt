if(RVML_ENABLE_TORCH_IMPORT)
  rvml_add_tool(rvml-import
    main.cpp

    LINK_LIBS
    RVMLImporter
    RVMLDialect
    MLIRIR
    MLIRSupport
  )
  target_link_libraries(rvml-import PRIVATE ${TORCH_LIBRARIES})
endif()
