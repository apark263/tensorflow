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
#include "tensorflow/tools/xla_extract/tf_graph_to_xla_lib.h"

namespace tensorflow {

std::vector<XlaCompiler::Argument> BuildXlaArgsFromClientGraph(
    const std::unique_ptr<ClientGraph>& cg) {
  std::vector<XlaCompiler::Argument> xla_args;
  for (const Node* node: cg->graph.nodes()) {
    if (node->type_string() == "XlaLaunch") {
      // iterate over the inputs to this node for the args
      for (const Node* in : node->in_nodes()) {
        auto in_def = in->def();

	XlaCompiler::Argument arg;
	if (in_def.op() == "VarHandleOp") {
	  arg.kind = XlaCompiler::Argument::kResource;
	  arg.resource_kind = XlaResource::kVariable;
	  arg.initialized = true;
	  GetNodeAttr(in_def, "shape", &(arg.shape));
	} else {
	  arg.kind = XlaCompiler::Argument::kParameter;
          std::vector<tensorflow::TensorShape> shape_value;
          GetNodeAttr(in_def, "_output_shapes", &shape_value);
          arg.shape = shape_value[0];
	}
        arg.name = in_def.name();
        GetNodeAttr(in_def, "dtype", &(arg.type));
        xla_args.push_back(std::move(arg));
      }
    }
  }
  return std::move(xla_args);
}
void InitializeDevices(const SessionOptions& options, DeviceMgr** device_mgr,
                       DeviceSet* dev_set) {
                         
  std::vector<Device*> devices;
  Status s = DeviceFactory::AddDevices(options, "/job:localhost/replica:0/task:0", &devices);
  *device_mgr = new DeviceMgr(devices);
  int devices_added = 0;
  for (auto d : (*device_mgr)->ListDevices()) {
    dev_set->AddDevice(d);
    d->op_segment()->AddHold("HOLD");
    if (devices_added == 0) {
      dev_set->set_client_device(d);
    }
    ++devices_added;
  }
}

xla::HloModuleProto ExtractHloFromGraphDef(const GraphDef& in_graph,
                                           const std::string& fetch) {
  Status s;
  SessionOptions sess_options;
  DeviceMgr* device_mgr;
  DeviceSet dev_set;
  InitializeDevices(sess_options, &device_mgr, &dev_set);

  // Local copy for modification
  GraphDef gdef = in_graph;
  GraphExecutionStateOptions ges_options;
  ges_options.device_set = &dev_set;
  ges_options.session_options = &sess_options;
  std::unique_ptr<GraphExecutionState> execution_state;
  s = GraphExecutionState::MakeForBaseGraph(&gdef, ges_options,
                                            &execution_state);
  if (!s.ok()) LOG(FATAL) << "execution state creation failed: " << s.error_message();

  BuildGraphOptions bg_options;
  bg_options.use_function_convention = true;
  bg_options.callable_options.add_fetch(fetch);
  std::unique_ptr<ClientGraph> client_graph;
  s = execution_state->BuildGraph(bg_options, &client_graph);
  if (!s.ok()) LOG(FATAL) << "build graph failed " << s.error_message();

  auto fdef = client_graph->flib_def->ToProto().function(0);
  auto xla_args = BuildXlaArgsFromClientGraph(client_graph);
  xla::HloModuleProto hmod;
  {
    DeviceType device_type(DEVICE_CPU_XLA_JIT);
    XlaCompiler::Options compile_options;
    std::cout<<"1 pass\n";
    compile_options.client = xla::ClientLibrary::LocalClientOrDie();
    std::cout<<"2 pass\n";
    compile_options.device_type = device_type;
    compile_options.flib_def = client_graph->flib_def.get();

    NameAttrList function;
    function.set_name(fdef.signature().name());
    *(function.mutable_attr()) = fdef.attr();

    XlaCompiler compiler(compile_options);
    XlaCompiler::CompilationResult result;

    s = compiler.CompileFunction(XlaCompiler::CompileOptions(), function,
                                 xla_args, &result);
    if (!s.ok()) LOG(FATAL) << "Couldn't compile to xla" << s.error_message();

    LOG(INFO) << "Done Compiling";
    hmod.CopyFrom(result.computation->proto());
  }
  if (device_mgr != nullptr) {
    delete(device_mgr);
  }
  return std::move(hmod);
}

void RealMain(const std::string& in_graph, const std::string& out_graph,
              const std::string& target_node) {
  GraphDef gdef;
  Status s;
  s = ReadTextProto(Env::Default(), in_graph, &gdef);
  if (!s.ok()) LOG(FATAL) << "Loading graphdef failed: " << s.error_message();

  auto hmod = ExtractHloFromGraphDef(gdef, target_node);

  s = WriteTextProto(Env::Default(), out_graph, hmod);
  if (!s.ok()) LOG(FATAL) << "Couldn't write hlo module: " << s.error_message();
  LOG(INFO) << "ALL DONE";
}

bool xla_extract_via_strings(const std::string& graph_def_msg,
                               const std::string& target_node,
                               std::string& out_graph){
GraphDef gdef;
gdef.ParseFromString(graph_def_msg);

auto hmod = ExtractHloFromGraphDef(gdef, target_node);
WriteTextProto(Env::Default(), "xla.pbtxt", hmod);
bool s;
s=tensorflow::protobuf::TextFormat::PrintToString(hmod, &out_graph);
return s;
}

}  // namespace tensorflow