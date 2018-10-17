#include "tensorflow/contrib/xla_extractor/tf_graph_to_xla_hlo_lib.h"
#include <fstream>  // std::ofstream
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

Status xla_extract_via_strings(const string& graph_def_msg,
                               const string& target_node,
                               std::vector<string>* hlo_snapshot_strings) {
  tensorflow::GraphDef g;
  g.ParseFromString(graph_def_msg);

  string serialized_xla;
  std::vector<std::unique_ptr<xla::HloSnapshot>> hlo_snapshots;
  Status compile_status =
      FindAndCompileLaunchNodes(g, target_node, &hlo_snapshots);

  if (!compile_status.ok()) {
    LOG(WARNING) << "Compilation to XLA failed: "
                 << compile_status.error_message();
    return compile_status;
  }

  if (hlo_snapshots.size() == 0) {
    return tensorflow::errors::Internal("Too few xla modules generated");
  }

  hlo_snapshot_strings->resize(hlo_snapshots.size());

  for (unsigned int i = 0; i < hlo_snapshots.size(); ++i) {
    hlo_snapshots[i]->SerializeToString(&(*hlo_snapshot_strings)[i]);
  }

  return Status::OK();
}

DirectSession* CreateSession() {
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions_GlobalJitLevel_ON_1);
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);
  session_options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);

  session_options.config.set_inter_op_parallelism_threads(1);
  session_options.config.set_intra_op_parallelism_threads(1);
  return dynamic_cast<DirectSession*>(NewSession(session_options));
}

Status PopulateXlaArgFromNode(const Node& n, XlaCompiler::Argument* arg,
                              bool is_resource) {
  TF_RETURN_IF_ERROR(GetNodeAttr(n.def(), "dtype", &(arg->type)));
  TF_RETURN_IF_ERROR(GetNodeAttr(n.def(), "shape", &(arg->shape)));
  arg->name = n.name();
  if (is_resource) {
    arg->kind = XlaCompiler::Argument::kResource;
    arg->resource_kind = XlaResource::kVariable;
    arg->initialized = true;
  } else {
    arg->kind = XlaCompiler::Argument::kParameter;
  }
  return Status::OK();
}

Status prefix_internal_nodes(GraphDef* gd) {
  for (int i = 0; i < gd->node_size(); ++i) {
    NodeDef* node_def = gd->mutable_node(i);
    if (node_def->op() == "_Retval") {
      node_def->set_name("r" + node_def->name());
    }
  }
  return Status::OK();
}

bool is_target_node_op(StringPiece node_op) {
  // This covers all of the resource assigns that come from the update ops
  if (node_op.starts_with("Resource")) return true;
  if (node_op.starts_with("Dummy")) return true;
  if (node_op == "NoOp") return true;
  if (node_op == "ControlTrigger") return true;
  if (node_op.starts_with("Assign")) return true;
  return false;
}

void tag_parameters(const std::vector<XlaCompiler::Argument>& args,
                    xla::HloSnapshot* snapshot) {
  auto hlo_module = snapshot->mutable_hlo()->mutable_hlo_module();
  auto entry_comp_id = hlo_module->entry_computation_id();
  xla::HloComputationProto *entry_comp = hlo_module->mutable_computations(entry_comp_id);

  for (auto &hi : *(entry_comp->mutable_instructions())) {
    if (hi.opcode() == "parameter") {
      hi.set_name(args[hi.parameter_number()].name);
      VLOG(1) << hi.parameter_number() << " " << hi.name();
    }
  }
}

Status ClassifyNode(const GraphDef& g, const string& node_name,
                    std::vector<string>& outputs,
                    std::vector<string>& targets) {
  const protobuf::RepeatedPtrField<NodeDef>& all_nodes = g.node();
  for (const NodeDef& node_def : all_nodes) {
    if (node_def.name() == node_name) {
      VLOG(2) << "Found matching node for " << node_name << ": "
              << SummarizeNodeDef(node_def);

      if (is_target_node_op(node_def.op())) {
        targets.push_back(node_name);
      } else {
        outputs.push_back(node_name);
      }
      return Status::OK();
    }
  }
  return errors::NotFound("Unable to find node in graph: ", node_name);
}

Status BuildArgumentsFromNode(const Node& launch_node,
                              std::vector<XlaCompiler::Argument>* args) {
  DataTypeVector constant_dtypes, arg_dtypes, result_dtypes;

  int num_resources = 0;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(launch_node.def(), "Nresources", &num_resources));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(launch_node.def(), "Tconstants", &constant_dtypes));
  TF_RETURN_IF_ERROR(GetNodeAttr(launch_node.def(), "Targs", &arg_dtypes));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(launch_node.def(), "Tresults", &result_dtypes));

  VLOG(1) << "FUNCSIG: " << constant_dtypes.size() << " " << num_resources
          << " " << arg_dtypes.size() - constant_dtypes.size() - num_resources;

  if (constant_dtypes.size() != 0) {
    LOG(FATAL) << "Have constant args that we do not handle yet";
  }

  std::unordered_map<std::string, int> arg_map;
  for (int idx = 0; idx < launch_node.def().input_size(); ++idx) {
    arg_map.emplace(launch_node.def().input(idx), idx);
  }

  std::vector<const Node*> input_nodes;
  input_nodes.resize(launch_node.def().input_size());
  for (auto n : launch_node.in_nodes()) {
    if (arg_map.find(n->name()) != arg_map.end()) {
      int correct_position = arg_map.at(n->name());
      input_nodes[correct_position] = n;
    } else {
      LOG(INFO) << "Unable to find arg_name: " << n->name();
    }
  }

  if (VLOG_IS_ON(1)) {
    for (auto n : input_nodes) {
      VLOG(2) << "LAUNCHARG: " << n->name() << " " << n->id();
    }
  }
  // Now build the arguments
  int input_num = 0;
  args->resize(input_nodes.size());
  for (auto n : input_nodes) {
    bool is_resource = (n->op_def().name() == "VarHandleOp");
    XlaCompiler::Argument* arg = &(*args)[input_num];
    TF_RETURN_IF_ERROR(PopulateXlaArgFromNode(*n, arg, is_resource));
    ++input_num;
  }
  return Status::OK();
}

Status CompileLaunchNode(
    XlaCompiler& compiler, const Node& launch_node,
    std::vector<std::unique_ptr<xla::HloSnapshot>>* xla_modules) {
  const NameAttrList* function;
  TF_RETURN_IF_ERROR(GetNodeAttr(launch_node.def(), "function", &function));

  std::vector<XlaCompiler::Argument> args;
  TF_RETURN_IF_ERROR(BuildArgumentsFromNode(launch_node, &args));

  XlaCompiler::CompilationResult result;
  Status s = compiler.CompileFunction(XlaCompiler::CompileOptions(), *function,
                                      args, &result);

  if (s.ok()) {
    auto s_o_module = result.computation->Snapshot();
    if (s_o_module.ok()) {
      xla_modules->push_back(std::move(s_o_module).ValueOrDie());
      tag_parameters(args, xla_modules->back().get());
      return Status::OK();
    } else {
      VLOG(1) << "Failed in snapshot";
      return errors::Internal("Snapshot failure");
    }
  } else {
    LOG(INFO) << "Failed in compilation: " << s.error_message();
    return errors::Internal("Compilation failure");
  }
}

Status MatchSendRecvNodes(std::vector<Graph*>* partition_graphs) {
  std::unordered_map<string, std::pair<int, Graph*>> recv_map;
  std::unordered_map<string, std::pair<int, Graph*>> send_map;

  // First traverse the partition graphs and store the send and recv node ids by
  // tensorname
  for (Graph* pg_ptr : (*partition_graphs)) {
    for (const Node* node : pg_ptr->op_nodes()) {
      if (node->def().op() == "_Send") {
        string tensornm;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->def(), "tensor_name", &tensornm));
        Node* source_node;
        TF_RETURN_IF_ERROR(node->input_node(0, &source_node));
        send_map.emplace(tensornm, std::make_pair(source_node->id(), pg_ptr));
      }
      if (node->def().op() == "_Recv") {
        string tensornm;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->def(), "tensor_name", &tensornm));
        recv_map.emplace(tensornm, std::make_pair(node->id(), pg_ptr));
      }
    }
  }
  VLOG(1) << "Caching complete";
  int p_idx = 0;
  for (auto iter = recv_map.begin(); iter != recv_map.end(); ++iter) {
    const string& tensor_name = iter->first;

    Graph *dst_graph, *src_graph;
    int dst_node_id, src_node_id;

    std::tie(dst_node_id, dst_graph) = iter->second;
    Node* dst_node = dst_graph->FindNodeId(dst_node_id);
    if (dst_node == nullptr) {
      return errors::NotFound("Node id: ", dst_node_id, " not found in graph");
    }
    if (dst_node->assigned_device_name().find("XLA_CPU") == std::string::npos) {
      VLOG(1) << "Skipping recv node assigned device: "
              << SummarizeNodeDef(dst_node->def());
      continue;
    }
    if (send_map.find(tensor_name) != send_map.end()) {
      std::tie(src_node_id, src_graph) = send_map.at(tensor_name);

      if (VLOG_IS_ON(1)) {
        GraphDef bb;
        src_graph->ToGraphDef(&bb);
        std::ostringstream output_file;
        output_file << "pgraph_blah_" << p_idx++ << ".pbtxt";
        TF_RETURN_IF_ERROR(
            WriteTextProto(Env::Default(), output_file.str(), bb));
      }

      Node* src_node = src_graph->FindNodeId(src_node_id);

      if (src_node == nullptr) {
        return errors::NotFound("Node id: ", src_node_id,
                                " not found in graph");
      }

      TensorShape source_tensor_shape;
      DataType source_data_type;

      if (src_node->type_string() == "Const") {
        Tensor t;
        TF_RETURN_IF_ERROR(GetNodeAttr(src_node->def(), "value", &t));
        source_tensor_shape = t.shape();
        source_data_type = t.dtype();
      } else {
        tensorflow::grappler::GrapplerItem item;
        src_graph->ToGraphDef(&item.graph);
        // Fix any nodes whose name starts with _ (invalid node for graph
        // construction)
        prefix_internal_nodes(&item.graph);
        tensorflow::grappler::GraphProperties properties(item);
        TF_RETURN_IF_ERROR(properties.InferStatically(false));
        const auto props =
            properties.GetOutputProperties(src_node->def().name());
        const OpInfo::TensorProperties& prop = props[0];
        source_data_type = prop.dtype();
        source_tensor_shape = TensorShape(prop.shape());
      }

      VLOG(1) << "For Tensor: " << tensor_name << ", Recv key: " << dst_node_id
              << " Send key: " << src_node_id;
      VLOG(1) << " ## RECV " << SummarizeNodeDef(dst_node->def());
      VLOG(1) << " ## SEND " << SummarizeNodeDef(src_node->def());

      dst_node->AddAttr("dtype", source_data_type);
      dst_node->AddAttr("shape", source_tensor_shape);

      VLOG(1) << " ## After Copy " << SummarizeNodeDef(dst_node->def());

    } else {
      return errors::NotFound("No send node found for tensor '", tensor_name,
                              "' in recv_map");
    }
  }
  VLOG(1) << "Matching complete";
  return Status::OK();
}

Status SaveTextOrBinaryXlaModule(const string& file_name,
                                 const xla::HloSnapshot& m) {
  StringPiece fname(file_name);
  if (fname.ends_with(".pbtxt")) {
    TF_RETURN_IF_ERROR(WriteTextProto(Env::Default(), file_name, m));
  } else {
    TF_RETURN_IF_ERROR(WriteBinaryProto(Env::Default(), file_name, m));
  }
  return Status::OK();
}

Status LoadTextOrBinaryGraphFile(const string& file_name, GraphDef* graph_def) {
  string file_data;
  Status load_file_status =
      ReadFileToString(Env::Default(), file_name, &file_data);
  if (!load_file_status.ok()) {
    errors::AppendToMessage(&load_file_status, " (for file ", file_name, ")");
    return load_file_status;
  }
  // Try to load in binary format first, and then try ascii if that fails.
  Status load_status = ReadBinaryProto(Env::Default(), file_name, graph_def);
  if (!load_status.ok()) {
    if (protobuf::TextFormat::ParseFromString(file_data, graph_def)) {
      load_status = Status::OK();
    } else {
      errors::AppendToMessage(&load_status,
                              " (both text and binary parsing failed for file ",
                              file_name, ")");
    }
  }
  return load_status;
}

Status FindAndCompileLaunchNodes(
    const GraphDef& g, const std::string& target_node,
    std::vector<std::unique_ptr<xla::HloSnapshot>>* xla_modules) {
  std::unique_ptr<DirectSession> dsession(CreateSession());
  tensorflow::XlaExtractor tf_xla_ifc(dsession.get());

  Status s = tf_xla_ifc.Create(g);
  if (!s.ok()) return s;

  /* Classify requested node as target or output*/
  std::vector<string> outputs, targets;

  string item;
  std::stringstream target_stream(target_node);
  while (std::getline(target_stream, item, ' ')) {
    if (!item.empty()) {
      s = ClassifyNode(g, item, outputs, targets);
      if (!s.ok()) return s;
    }
  }
  VLOG(1) << "Node classification complete";
  for (auto& t : targets) VLOG(1) << "targets: " << t;
  for (auto& o : outputs) VLOG(1) << "outputs: " << o;

  std::vector<Graph*> graphs_list;
  FunctionLibraryDefinition* flib_def;

  s = tf_xla_ifc.CreatePartitionGraphs(outputs, targets, &graphs_list,
                                       &flib_def);
  if (!s.ok()) return s;
  VLOG(1) << "Partition Graphs complete";

  s = MatchSendRecvNodes(&graphs_list);
  if (!s.ok()) return s;
  VLOG(1) << "Matching Send and Recv complete";

  DeviceType device_type(DEVICE_CPU_XLA_JIT);
  XlaCompiler::Options compile_options;
  compile_options.client = xla::ClientLibrary::LocalClientOrDie();
  compile_options.device_type = device_type;
  compile_options.allow_cpu_custom_calls = false;
  compile_options.flib_def = flib_def;
  compile_options.graph_def_version = g.versions().producer();
  XlaCompiler compiler(compile_options);

  for (Graph* partition_graph : graphs_list) {
    for (const Node* node : partition_graph->op_nodes()) {
      if (node->type_string() == "XlaLaunch") {
        VLOG(2) << "LaunchNode\n" << SummarizeNode(*node);

        s = CompileLaunchNode(compiler, *node, xla_modules);
        if (!s.ok()) return s;
      }
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
