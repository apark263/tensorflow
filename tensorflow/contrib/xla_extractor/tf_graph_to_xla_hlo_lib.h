#ifndef TENSORFLOW_CONTRIB_TF_GRAPH_TO_XLA_HLO_LIB_H
#define TENSORFLOW_CONTRIB_TF_GRAPH_TO_XLA_HLO_LIB_H

#include "tensorflow/compiler/tf2xla/shape_util.h"  // for TensorShapeToXLAShape
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"  // for DEVICE_CPU_XLA_JIT
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
// Using this just for its friend status with DirectSession
class XlaExtractor {
 public:
  XlaExtractor(DirectSession* session) : session_(session) {}
  ~XlaExtractor() {}

  Status Create(const GraphDef& graph) { return session_->Create(graph); }

  // Status CreatePartitionGraphs(
  //     const std::vector<string>& outputs,
  //     const std::vector<string>& targets,
  //     std::vector<Graph *>* partition_graphs,
  //     FunctionLibraryDefinition** flib_def);

  Status CreatePartitionGraphs(const std::vector<string>& outputs,
                               const std::vector<string>& targets,
                               std::vector<Graph*>* partition_graphs,
                               FunctionLibraryDefinition** flib_def) {
    DebugOptions dbg_opts;
    DirectSession::RunStateArgs rs(dbg_opts);
    DirectSession::ExecutorsAndKeys* executors_and_keys;

    TF_RETURN_IF_ERROR(session_->GetOrCreateExecutors(
        {}, outputs, targets, &executors_and_keys, &rs));

    for (auto& i : executors_and_keys->items) {
      partition_graphs->push_back(i.graph);
    }
    *flib_def = session_->functions_.back()->flib_def.get();
    return Status::OK();
  }

 private:
  DirectSession* session_;
};

Status xla_extract_via_strings(const string& graph_def_msg,
                               const string& target_node,
                               std::vector<string>* xla_mod_strings);

Status SaveTextOrBinaryXlaModule(const string& file_name,
                                 const xla::SessionModule& m);
Status LoadTextOrBinaryGraphFile(const string& file_name, GraphDef* graph_def);
Status FindAndCompileLaunchNodes(
    const GraphDef& g, const std::string& target_node,
    std::vector<std::unique_ptr<xla::SessionModule>>* xla_modules);

struct TfToXlaConverterOptions {
  std::string in_graph;
  std::string out_prefix;
  std::string target_node;
  bool verbose;
  bool output_as_text;
};

}  // namespace tensorflow
#endif