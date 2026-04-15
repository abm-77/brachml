set(PYTHON_DIR "${CMAKE_SOURCE_DIR}/python")
set(PYTHON_EXE "${PYTHON_DIR}/.venv/bin/python3")

# train-<model>  : Train (if needed) + quantize + export .pt2.
# import-<model> : Run the MLIR importer on the exported .pt2.

add_custom_target(train-cifar10
  COMMAND ${PYTHON_EXE} ${PYTHON_DIR}/models/cifar10/main.py
  WORKING_DIRECTORY ${PYTHON_DIR}
  COMMENT "Training / exporting cifar10"
  USES_TERMINAL
)

add_custom_target(import-cifar10
  COMMAND ${PYTHON_EXE} -m brachml_import.importer
    ${PYTHON_DIR}/models/cifar10/cifar_core_aten.pt2
    -o ${PYTHON_DIR}/models/cifar10/cifar10.mlir
  WORKING_DIRECTORY ${PYTHON_DIR}
  COMMENT "Importing cifar10 into BrachML MLIR"
  USES_TERMINAL
)
add_dependencies(import-cifar10 train-cifar10)

add_custom_target(train-resnet
  COMMAND ${PYTHON_EXE} ${PYTHON_DIR}/models/resnet/main.py
  WORKING_DIRECTORY ${PYTHON_DIR}
  COMMENT "Training / exporting resnet"
  USES_TERMINAL
)

add_custom_target(import-resnet
  COMMAND ${PYTHON_EXE} -m brachml_import.importer
    ${PYTHON_DIR}/models/resnet/resnet_core_aten.pt2
    -o ${PYTHON_DIR}/models/resnet/resnet.mlir
  WORKING_DIRECTORY ${PYTHON_DIR}
  COMMENT "Importing resnet into BrachML MLIR"
  USES_TERMINAL
)
add_dependencies(import-resnet train-resnet)
