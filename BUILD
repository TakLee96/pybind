load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
# load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "main",
    srcs = [ "main.py" ],
    main = "main.py",
    imports = [ "." ],
    data = [ ":mumpy" ],
    python_version = "PY3",
)

pybind_extension(
    name = "mumpy",
    srcs = [ "mumpy.cc" ],
)
