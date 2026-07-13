# -*- Python -*-

import os
import lit.formats

from lit.llvm import llvm_config
from lit.llvm.subst import FindTool, ToolSubst


# Name shown by llvm-lit.
config.name = "JForce"

# RUN lines are shell commands.
config.test_format = lit.formats.ShTest()

# Initially, only treat .mlir files as tests.
config.suffixes = [".mlir"]

# Source tree containing the actual test files.
config.test_source_root = os.path.dirname(__file__)

# Build-tree location used for temporary test output.
config.test_exec_root = os.path.join(
    config.jforce_obj_root,
    "test",
)

# These files/directories are configuration or auxiliary data, not tests.
config.excludes = [
    "CMakeLists.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
    "lit.site.cfg.py.in",
    "Inputs",
    "README.md",
]

# Adds standard substitutions such as FileCheck and not.
llvm_config.use_default_substitutions()

tool_dirs = [
    config.jforce_tools_dir,
    config.llvm_tools_dir,
]

# Define a stable substitution for the project-specific executable.
tools = [
    ToolSubst(
        "%jforce-opt",
        command=FindTool("jforce-opt"),
        unresolved="fatal",
    ),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Also make the tools available to shell commands by name.
llvm_config.with_environment(
    "PATH",
    config.jforce_tools_dir,
    append_path=True,
)

llvm_config.with_environment(
    "PATH",
    config.llvm_tools_dir,
    append_path=True,
)
