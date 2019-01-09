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
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/tools/xla_extract/tf_graph_to_xla_lib.h"

#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/interpreter/compiler.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"

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
  std::vector<std::unique_ptr<Device>> devices;
  Status s = DeviceFactory::AddDevices(options, "/job:localhost/replica:0/task:0", &devices);
  *device_mgr = new DeviceMgr(std::move(devices));
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
    compile_options.client = xla::ClientLibrary::LocalClientOrDie();
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

    // hlo optimizations

    // getting hlo_module from proto
    xla::StatusOr<xla::ProgramShape> program_shape_status = result.computation->GetProgramShape();
    xla::ProgramShape program_shape = program_shape_status.ValueOrDie();
    xla::HloModuleConfig module_config = xla::HloModuleConfig(program_shape);
    
    xla::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module_status = xla::HloModule::CreateFromProto(hmod, module_config);
    std::unique_ptr<xla::HloModule> hlo_module = std::move(hlo_module_status.ValueOrDie());
    std::cout<<hlo_module->name()<<"\n";

    xla::HloPassPipeline pipeline("Interpreter");
    // adding passes we wish to run
    pipeline.AddPass<xla::LayoutAssignment>(
      hlo_module.get()->mutable_entry_computation_layout(),
      xla::LayoutAssignment::InstructionCanChangeLayout);

    // hlo optimization run
    s = pipeline.Run(hlo_module.get()).status();

    if (!s.ok()) LOG(FATAL) << "Couldn't RunHloOptimization" << s.error_message();

    LOG(INFO) << "Done HLO Optimization\n";
    hmod = hlo_module.get()->ToProto();
  }
  if (device_mgr != nullptr) {
    delete(device_mgr);
  }
  return std::move(hmod);
}

Status xla_extract_via_strings(const std::string& graph_def_msg,
			       const std::string& target_node,
			       std::string* out_graph) {
  GraphDef gdef;
  gdef.ParseFromString(graph_def_msg);
  auto hmod = ExtractHloFromGraphDef(gdef, target_node);
  hmod.SerializeToString(out_graph);

  return Status::OK();
}

}  // namespace tensorflow
