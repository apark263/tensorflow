#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"  // for DEVICE_CPU_XLA_JIT
#include "tensorflow/compiler/tf2xla/shape_util.h"  // for TensorShapeToXLAShape

namespace xla {
namespace swig {

StatusOr<string> xla_extract_via_strings(
    const string& graph_def_msg,
    const string& target_node
);

} // namespace swig
} // namespace xla

namespace tensorflow {
// Using this just for its friend status with DirectSession
class DebugGateway {
public:
  DebugGateway(DirectSession* session): session_(session) {}
  virtual ~DebugGateway() {}

  Status Create(const GraphDef& graph)
  {
    return session_->Create(graph);
  }

  Status CreatePartitionGraphs(
      const std::vector<string>& outputs,
      const std::vector<string>& targets,
      std::vector<Graph *>* partition_graphs,
      FunctionLibraryDefinition** flib_def);

private:
  DirectSession* session_;
};

Status SaveTextOrBinaryXlaModule(const string& file_name, const xla::SessionModule& m);
Status LoadTextOrBinaryGraphFile(const string& file_name, GraphDef* graph_def);
Status FindAndCompileLaunchNodes(const GraphDef& g, const std::string& target_node, std::vector<std::unique_ptr<xla::SessionModule>>* xla_modules);

struct TfToXlaConverterOptions {
  std::string in_graph;
  std::string out_prefix;
  std::string target_node;
  bool verbose;
  bool output_as_text;
};

} // namespace tensorflow
