#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"  // for DEVICE_CPU_XLA_JIT
#include "tensorflow/compiler/tf2xla/shape_util.h"  // for TensorShapeToXLAShape

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


class TfToXlaConverter {
public:

  struct TfToXlaConverterOptions {
    std::string in_graph;
    std::string out_prefix;
    std::string target_node;
    bool verbose;
    bool output_as_text;
    bool dump_arg_mapping;
  };

  TfToXlaConverter(TfToXlaConverterOptions options);

  Status LoadAndPartitionGraphs();
  // Partitioned graphs to be converted to XLA compiled nodes can sometimes contain
  // recv nodes if the partitioning crosses device boundaries. The recv nodes lack 
  // attributes necessary to create XLA arguments.  This function matches up the 
  // corresponding send nodes from the source graph partition so that the attributes
  // are attached to the recv nodes
  Status FindAndCompileLaunchNodes();

  Status CompileAndSaveLaunchNode(
    const Node& launch_node,
    const unsigned int&  partition_index
  );

  static Status MatchSendRecvNodes(std::vector<Graph *>* partition_graphs);
  static Status BuildArgumentsFromNode(const Node& launch_node, std::vector<XlaCompiler::Argument>* args);
  static Status SaveTextOrBinaryXlaModule(const string& file_name, const xla::SessionModule& m);
  static Status LoadTextOrBinaryGraphFile(const string& file_name, GraphDef* graph_def);

private:
  TfToXlaConverterOptions converter_options_;
  XlaCompiler::Options compile_options_;
  GraphDef graph_def_;
  std::vector<Graph *> graphs_list_;
  FunctionLibraryDefinition* flib_def_;
  std::unique_ptr<DirectSession> dsession_;
  DebugGateway dbg_;


  // The taken from 
  struct GuardedCompilation {
    mutex mu;
    Status status GUARDED_BY(mu);
    // Output of the XlaCompiler.
    XlaCompiler::CompilationResult result GUARDED_BY(mu);
  };
};


} // namespace tensorflow