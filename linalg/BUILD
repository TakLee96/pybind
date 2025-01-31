load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

package(default_visibility = ["//visibility:public"])

pybind_library(
    name = "lib",
    srcs = glob([ "*.cc" ]),
    hdrs = glob([ "*.h" ]),
    deps = [
        "@eigen",
    ]
)
