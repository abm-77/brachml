import os
import lit.formats
import lit.util

config.name = "BrachML"
config.test_format = lit.formats.ShTest(not False)
config.suffixes = [".mlir", ".py"]
config.excludes = ["generate_models.py", "lit.cfg.py", "lit.site.cfg.py", "models"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.brachml_obj_root, "tests")

# Tools available to tests.
config.substitutions.append(("%brachml-opt", os.path.join(config.brachml_tools_dir, "brachml-opt")))

# Python importer: run via the venv in python/
python_dir = os.path.join(config.brachml_src_root, "python")
python_exe = os.path.join(python_dir, ".venv", "bin", "python3")
config.substitutions.append(("%brachml_import", f"{python_exe} -m brachml_import.importer"))
config.environment["PYTHONPATH"] = python_dir

# Make FileCheck and friends available via PATH.
config.environment["PATH"] = os.pathsep.join([
    config.llvm_tools_dir,
    config.brachml_tools_dir,
    config.environment.get("PATH", ""),
])
