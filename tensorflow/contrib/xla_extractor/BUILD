package(default_visibility = ["//tensorflow:__subpackages__"])

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
    visibility = ["//tensorflow/contrib/xla_extractor:internal"],
)

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

cc_library(
    name = "tf_graph_to_xla_hlo_lib",
    srcs = ["tf_graph_to_xla_hlo_lib.cc"],
    hdrs = ["tf_graph_to_xla_hlo_lib.h"],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:direct_session",
        "//tensorflow/compiler/jit:xla_cpu_device", # friends only
        "//tensorflow/compiler/tf2xla",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/client:client_library", # friends only
        "//tensorflow/compiler/xla/service", # friends only
    ],
)

tf_cc_binary(
    name = "tf_graph_to_xla_hlo",
    srcs = ["tf_graph_to_xla_hlo.cc"],
    deps = [
        ":tf_graph_to_xla_hlo_lib",
    ],
)

py_library(
    name = "xla_extractor",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/xla/service:session_proto_py",
    ],
)

