/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Usage: graph_execution saved_graph.pbtxt
#include "tensorflow/compiler/xla/tools/tf_graph_to_xla_hlo_lib.h"

#include <stdio.h>
#include <unistd.h>
#include <string>

#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

void RealMain(TfToXlaConverterOptions converter_options)
{
  GraphDef g;
  Status s = LoadTextOrBinaryGraphFile(converter_options.in_graph, &g);
  if (!s.ok())
    LOG(FATAL) << "Loading graph failed: " << s.error_message();

  std::vector<std::unique_ptr<xla::SessionModule>> xla_modules;
  Status compile_status = FindAndCompileLaunchNodes(g, converter_options.target_node, &xla_modules);
  if (!compile_status.ok())
    LOG(FATAL) << "Compilation to XLA failed: " << compile_status.error_message();

  for (unsigned int idx = 0; idx < xla_modules.size(); ++idx) {
      std::ostringstream out_filename;
      out_filename << converter_options.out_prefix << "_" << idx << ".pb";
      if (converter_options.output_as_text)
        out_filename << "txt";

      Status save_status = SaveTextOrBinaryXlaModule(out_filename.str(), *xla_modules[idx]);
      if (!save_status.ok())
        LOG(FATAL) << "Save failed: " << save_status.error_message();
  }
}
}  // namespace tensorflow

int main(int argc, char** argv) {

  tensorflow::TfToXlaConverterOptions converter_options = 
    { .in_graph = "",
      .out_prefix = "xla_graph",
      .target_node = "",
      .verbose = false,
      .output_as_text = false,
    };

  std::vector<tensorflow::Flag> flag_list = {
      {"in_graph", &converter_options.in_graph, "input graph file name"},
      {"out_prefix", &converter_options.out_prefix,
         "prefix for output xla hlo modules protobufs.  Names will be <prefix>_<i>.pb(txt).  Default is \"xla_graph\"."},
      {"target_node", &converter_options.target_node,
         "space separated list of target nodes for capture"},
      {"verbose", &converter_options.verbose, "whether to print extra output"},
      {"output_as_text", &converter_options.output_as_text,
          "whether to write the graph in text protobuf format"}
  };

  const xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parsed_flags_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  if (!parsed_flags_ok) {
    LOG(ERROR) << usage;
    return -1;
  }

  if (converter_options.in_graph.empty()) {
    LOG(ERROR) << "in_graph graph can't be empty.\n" << usage;
    return -1;
  }

  if (converter_options.target_node.empty()) {
    LOG(ERROR) << "no target_node specified.\n" << usage;
    return -1;
  }
  tensorflow::RealMain(converter_options);
  return 0;
}

