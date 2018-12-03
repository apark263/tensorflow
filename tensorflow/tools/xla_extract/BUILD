package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "tf_graph_to_xla",
    srcs = ["tf_graph_to_xla.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/compiler/jit:xla_cpu_device",
	"//tensorflow/compiler/tf2xla:xla_compiler",
    ],
)
