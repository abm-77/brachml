import os
import lit.formats
import lit.util

config.name = "RVML"
config.test_format = lit.formats.ShTest(not False)
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.rvml_obj_root, "tests")

# Tools available to tests.
config.substitutions.append(("%rvml-opt", os.path.join(config.rvml_tools_dir, "rvml-opt")))
config.substitutions.append(("%rvml-import", os.path.join(config.rvml_tools_dir, "rvml-import")))

# Make FileCheck and friends available.
llvm_tools = [
    "FileCheck",
    "count",
    "not",
]
tool_dirs = [config.llvm_tools_dir, config.rvml_tools_dir]
llvm_config = getattr(config, "llvm_config", None)
