#ifndef TENSORFLOW_CONTRIB_TF_GRAPH_TO_XLA_LIB_H
#define TENSORFLOW_CONTRIB_TF_GRAPH_TO_XLA_LIB_H

#include <stdio.h>
#include <unistd.h>
#include <iterator>
#include <string>
#include <tuple>
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"  // for DEVICE_CPU_XLA_JIT
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow/compiler/xla/service/service.h"
namespace tensorflow {

std::vector<XlaCompiler::Argument> BuildXlaArgsFromClientGraph(
    const std::unique_ptr<ClientGraph>& cg);

void InitializeDevices(const SessionOptions& options, DeviceMgr** device_mgr,
                       DeviceSet* dev_set);

xla::HloModuleProto ExtractHloFromGraphDef(const GraphDef& in_graph,
                                           const std::string& fetch);

void RealMain(const std::string& in_graph, const std::string& out_graph,
              const std::string& target_node);

bool xla_extract_via_strings(const string& graph_def_msg,
                               const string& target_node,
                               string& out_graph);
}  // namespace tensorflow

#endif