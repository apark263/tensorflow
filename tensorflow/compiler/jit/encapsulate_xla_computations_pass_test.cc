/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/encapsulate_xla_computations_pass.h"

#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/tf2xla/cc/ops/xla_jit_ops.h"
#include "tensorflow/compiler/tf2xla/test_util.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"
#include "tensorflow/core/util/ptr_util.h"

////////////
/// optimization
#include "tensorflow/compiler/jit/xla_fusion_optimizer.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
///

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass_test_helper.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/compiler/jit/partially_decluster_pass.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/build_xla_ops_pass.h"

#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/legacy_flags/xla_device_flags.h"
#include "tensorflow/compiler/jit/xla_compile_on_demand_op.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"
#include <iostream>
#include <fstream>




#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/client_library.h"

////////////
namespace tensorflow {

static std::unique_ptr<Graph> MakeOuterGraph(
    const FunctionLibraryDefinition& flib_def, const string& function) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(flib_def.ToProto()));

  auto a = ops::Placeholder(scope.WithOpName("A"), DT_INT32);
  auto b = ops::Placeholder(scope.WithOpName("B"), DT_FLOAT);
  auto c = ops::Placeholder(scope.WithOpName("C"), DT_INT32);
  auto d = ops::Placeholder(scope.WithOpName("D"), DT_FLOAT);
  auto u = ops::Placeholder(scope.WithOpName("U"), DT_RESOURCE);
  auto v = ops::Placeholder(scope.WithOpName("V"), DT_RESOURCE);
  auto w = ops::Placeholder(scope.WithOpName("W"), DT_RESOURCE);

  NodeDef def;
  TF_CHECK_OK(
      NodeDefBuilder("launch0", function, &flib_def)
          .Input(a.node()->name(), 0, DT_INT32)
          .Input(b.node()->name(), 0, DT_FLOAT)
          .Input(c.node()->name(), 0, DT_INT32)
          .Input(d.node()->name(), 0, DT_FLOAT)
          .Input(u.node()->name(), 0, DT_RESOURCE)
          .Input(v.node()->name(), 0, DT_RESOURCE)
          .Input(w.node()->name(), 0, DT_RESOURCE)
          .Device("/cpu:0")
          .Attr(EncapsulateXlaComputationsPass::kXlaClusterAttr, "launch0")
          .Attr("_variable_start_index", 4)
          .Finalize(&def));

  Status status;
  Node* launch = scope.graph()->AddNode(def, &status);
  TF_CHECK_OK(status);
  TF_CHECK_OK(scope.DoShapeInference(launch));
  scope.graph()->AddEdge(a.node(), 0, launch, 0);
  scope.graph()->AddEdge(b.node(), 0, launch, 1);
  scope.graph()->AddEdge(c.node(), 0, launch, 2);
  scope.graph()->AddEdge(d.node(), 0, launch, 3);
  scope.graph()->AddEdge(u.node(), 0, launch, 4);
  scope.graph()->AddEdge(v.node(), 0, launch, 5);
  scope.graph()->AddEdge(w.node(), 0, launch, 6);

  auto out0 =
      ops::XlaClusterOutput(scope.WithOpName("Out0"), Output(launch, 0));
  auto out1 =
      ops::XlaClusterOutput(scope.WithOpName("Out1"), Output(launch, 1));
  auto out2 =
      ops::XlaClusterOutput(scope.WithOpName("Out2"), Output(launch, 2));
  auto out3 =
      ops::XlaClusterOutput(scope.WithOpName("Out3"), Output(launch, 3));

  auto consumer0_a = ops::Identity(scope.WithOpName("consumer0_a"), out0);
  auto consumer0_b = ops::Identity(scope.WithOpName("consumer0_b"), out0);
  auto consumer0_c = ops::Identity(scope.WithOpName("consumer0_c"), out0);
  auto consumer1 = ops::Identity(scope.WithOpName("consumer1"), out1);
  auto consumer2 = ops::Identity(scope.WithOpName("consumer2"), out2);
  auto consumer3 = ops::Identity(scope.WithOpName("consumer3"), out3);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(scope.ToGraph(graph.get()));
  return graph;
}

// Makes an encapsulate body graph for use in tests.
static std::unique_ptr<Graph> MakeBodyGraph() {
  Scope scope = Scope::NewRootScope().ExitOnError();

  auto arg0 = ops::_Arg(scope.WithOpName("a_0_arg"), DT_INT32, 0);
  auto arg1 = ops::_Arg(scope.WithOpName("b_0_arg"), DT_FLOAT, 1);
  auto arg2 = ops::_Arg(scope.WithOpName("c_0_arg"), DT_INT32, 2);
  auto arg3 = ops::_Arg(scope.WithOpName("d_0_arg"), DT_FLOAT, 3);

  auto arg4 = ops::_Arg(scope.WithOpName("u_0_arg"), DT_RESOURCE, 4);
  auto arg5 = ops::_Arg(scope.WithOpName("v_0_arg"), DT_RESOURCE, 5);
  auto arg6 = ops::_Arg(scope.WithOpName("w_0_arg"), DT_RESOURCE, 6);

  auto add_attrs = [](Node* node) {
    node->AddAttr(EncapsulateXlaComputationsPass::kXlaClusterAttr, "launch0");
    node->set_requested_device("/cpu:0");
  };

  auto b_identity = ops::Identity(scope.WithOpName("B_identity"), arg1);
  add_attrs(b_identity.node());
  auto read_u = ops::ReadVariableOp(scope.WithOpName("ReadU"), arg4, DT_FLOAT);
  add_attrs(read_u.node());
  auto read_v = ops::ReadVariableOp(scope.WithOpName("ReadV"), arg5, DT_FLOAT);
  add_attrs(read_v.node());
  auto read_w = ops::ReadVariableOp(scope.WithOpName("ReadW"), arg6, DT_FLOAT);
  add_attrs(read_w.node());

  auto e = ops::Add(scope.WithOpName("E"), arg0, arg2);
  add_attrs(e.node());
  auto f = ops::Add(scope.WithOpName("F"), read_v, read_w);
  add_attrs(f.node());
  auto g = ops::Add(scope.WithOpName("G"), f, arg3);
  add_attrs(g.node());

  auto out0 = ops::_Retval(scope.WithOpName("b_identity_0_retval_RetVal"),
                           b_identity, 0);
  auto out1 = ops::_Retval(scope.WithOpName("e_0_retval_RetVal"), e, 1);
  auto out2 = ops::_Retval(scope.WithOpName("g_0_retval_RetVal"), g, 2);
  auto out3 =
      ops::_Retval(scope.WithOpName("readu_0_retval_RetVal"), read_u, 3);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(scope.ToGraph(graph.get()));
  return graph;
}

TEST(EncapsulateXlaComputations, DeterministicEncapsulate) {
  // Test that control edge insertion order doesn't affect the cache key
  // (cluster name) generated by TPU encapsulate pass.
  auto get_serialized_graph = [](bool control_input_reversed,
                                 bool operand_reversed) -> string {
    FunctionLibraryDefinition flib_def(OpRegistry::Global(), {});
    std::unique_ptr<Graph> graph(new Graph(&flib_def));
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto a0 = ops::Placeholder(scope.WithOpName("A0"), DT_INT32);
      auto a1 = ops::Placeholder(scope.WithOpName("A1"), DT_INT32);

      ops::Add e = operand_reversed ? ops::Add(scope.WithOpName("E"), a0, a1)
                                    : ops::Add(scope.WithOpName("E"), a1, a0);

      auto add_attrs = [](Node* node) {
        node->AddAttr(EncapsulateXlaComputationsPass::kXlaClusterAttr,
                      "launch0");
      };
      add_attrs(e.node());

      TF_CHECK_OK(scope.ToGraph(graph.get()));
      auto get_node_in_graph = [&graph](Node* node) {
        return graph->FindNodeId(node->id());
      };
      // Insert control edge in different order. The order should not affect
      // the encapsulated or serialized graph.
      if (!control_input_reversed) {
        graph->AddControlEdge(get_node_in_graph(a0.node()),
                              get_node_in_graph(e.node()), true);
        graph->AddControlEdge(get_node_in_graph(a1.node()),
                              get_node_in_graph(e.node()), true);
      } else {
        graph->AddControlEdge(get_node_in_graph(a1.node()),
                              get_node_in_graph(e.node()), true);
        graph->AddControlEdge(get_node_in_graph(a0.node()),
                              get_node_in_graph(e.node()), true);
      }
    }
    TF_CHECK_OK(EncapsulateXlaComputationsPass::Encapsulate(&graph, &flib_def));
    GraphDef gdef;
    graph->ToGraphDef(&gdef);
    // Before serialization, sort control inputs first to remove
    // nondeterminism.
    SortControlInputs(&gdef);
    string serialized;
    SerializeToStringDeterministic(gdef, &serialized);
    return serialized;
  };

  // Changing the order of control input shouldn't affect the graph generated.
  EXPECT_EQ(get_serialized_graph(/*control_input_reversed=*/true,
                                 /*operand_reversed=*/false),
            get_serialized_graph(/*control_input_reversed=*/false,
                                 /*operand_reversed=*/false));

  // Changing the order of data input should affect the graph generated.
  EXPECT_NE(get_serialized_graph(/*control_input_reversed=*/false,
                                 /*operand_reversed=*/true),
            get_serialized_graph(/*control_input_reversed=*/false,
                                 /*operand_reversed=*/false));
}

TEST(EncapsulateXlaComputations, Encapsulate) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), {});
  std::unique_ptr<Graph> graph(new Graph(&flib_def));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto a = ops::Placeholder(scope.WithOpName("A"), DT_INT32);
    auto b = ops::Placeholder(scope.WithOpName("B"), DT_FLOAT);
    auto c = ops::Placeholder(scope.WithOpName("C"), DT_INT32);
    auto d = ops::Placeholder(scope.WithOpName("D"), DT_FLOAT);
    auto u = ops::Placeholder(scope.WithOpName("U"), DT_RESOURCE);
    auto v = ops::Placeholder(scope.WithOpName("V"), DT_RESOURCE);
    auto w = ops::Placeholder(scope.WithOpName("W"), DT_RESOURCE);

    auto add_attrs = [](Node* node) {
      node->AddAttr(EncapsulateXlaComputationsPass::kXlaClusterAttr, "launch0");
      node->set_requested_device("/cpu:0");
    };

    auto b_identity = ops::Identity(scope.WithOpName("B_identity"), b);
    add_attrs(b_identity.node());

    auto read_u = ops::ReadVariableOp(scope.WithOpName("ReadU"), u, DT_FLOAT);
    add_attrs(read_u.node());
    auto read_v = ops::ReadVariableOp(scope.WithOpName("ReadV"), v, DT_FLOAT);
    add_attrs(read_v.node());
    auto read_w = ops::ReadVariableOp(scope.WithOpName("ReadW"), w, DT_FLOAT);
    add_attrs(read_w.node());

    auto e = ops::Add(scope.WithOpName("E"), a, c);
    add_attrs(e.node());
    auto f = ops::Add(scope.WithOpName("F"), read_v, read_w);
    add_attrs(f.node());
    auto g = ops::Add(scope.WithOpName("G"), f, d);
    add_attrs(g.node());

    auto out0 = ops::XlaClusterOutput(scope.WithOpName("Out0"), b_identity);
    auto out1 = ops::XlaClusterOutput(scope.WithOpName("Out1"), e);
    auto out2 = ops::XlaClusterOutput(scope.WithOpName("Out2"), g);
    auto out3 = ops::XlaClusterOutput(scope.WithOpName("Out3"), read_u);

    auto consumer0_a = ops::Identity(scope.WithOpName("consumer0_a"), out0);
    auto consumer0_b = ops::Identity(scope.WithOpName("consumer0_b"), out0);
    auto consumer0_c = ops::Identity(scope.WithOpName("consumer0_c"), out0);
    auto consumer1 = ops::Identity(scope.WithOpName("consumer1"), out1);
    auto consumer2 = ops::Identity(scope.WithOpName("consumer2"), out2);
    auto consumer3 = ops::Identity(scope.WithOpName("consumer3"), out3);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
  }

  std::unique_ptr<Graph> graph_copy(new Graph(&flib_def));
  CopyGraph(*graph, graph_copy.get());

  TF_ASSERT_OK(EncapsulateXlaComputationsPass::Encapsulate(&graph, &flib_def));

  std::unordered_map<string, Node*> index = BuildNodeIndex(*graph);
  string function = index.at("launch0")->type_string();

  // Tests the outer graph is as expected.
  {
    std::unique_ptr<Graph> outer = MakeOuterGraph(flib_def, function);
    GraphDef expected_def;
    outer->ToGraphDef(&expected_def);

    GraphDef actual_def;
    graph->ToGraphDef(&actual_def);
    TF_EXPECT_GRAPH_EQ_INTERNAL(expected_def, actual_def);
  }

  // Tests the encapsulated body graph is as expected.
  {
    std::unique_ptr<Graph> body = MakeBodyGraph();
    GraphDef expected_body_def;
    body->ToGraphDef(&expected_body_def);

    InstantiationResultForTest result;
    TF_EXPECT_OK(InstantiateFunctionForTest(function, flib_def, &result));

    EXPECT_EQ((DataTypeVector{DT_INT32, DT_FLOAT, DT_INT32, DT_FLOAT,
                              DT_RESOURCE, DT_RESOURCE, DT_RESOURCE}),
              result.arg_types);
    EXPECT_EQ((DataTypeVector{DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT}),
              result.ret_types);
    TF_EXPECT_GRAPH_EQ(expected_body_def, result.gdef);
  }

  // Encapsulates the same computation again, verifies we reuse the same
  // function. Encapsulation should be deterministic to avoid recompilation.
  TF_ASSERT_OK(
      EncapsulateXlaComputationsPass::Encapsulate(&graph_copy, &flib_def));
  std::unordered_map<string, Node*> index_copy = BuildNodeIndex(*graph_copy);
  string function_copy = index_copy.at("launch0")->type_string();
  EXPECT_EQ(function, function_copy);
}

TEST(EncapsulateXlaComputations, BuildXlaLaunchOp) {
  std::unique_ptr<Graph> body_graph = MakeBodyGraph();
  FunctionDefLibrary flib;
  TF_ASSERT_OK(GraphToFunctionDef(*body_graph, "launch0", flib.add_function()));

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);

  std::unique_ptr<Graph> graph = MakeOuterGraph(flib_def, "launch0");
  TF_ASSERT_OK(EncapsulateXlaComputationsPass::BuildXlaLaunchOps(graph.get()));

  Scope scope = Scope::DisabledShapeInferenceScope().ExitOnError();
  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(flib));

  auto a = ops::Placeholder(scope.WithOpName("A"), DT_INT32);
  auto b = ops::Placeholder(scope.WithOpName("B"), DT_FLOAT);
  auto c = ops::Placeholder(scope.WithOpName("C"), DT_INT32);
  auto d = ops::Placeholder(scope.WithOpName("D"), DT_FLOAT);
  auto u = ops::Placeholder(scope.WithOpName("U"), DT_RESOURCE);
  auto v = ops::Placeholder(scope.WithOpName("V"), DT_RESOURCE);
  auto w = ops::Placeholder(scope.WithOpName("W"), DT_RESOURCE);

  NameAttrList function;
  function.set_name("launch0");
  auto launch = ops::XlaLaunch(
      scope.WithOpName("launch0").WithDevice("/cpu:0"),
      std::initializer_list<Input>{}, std::initializer_list<Input>{a, b, c, d},
      std::initializer_list<Input>{u, v, w},
      DataTypeVector{DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT}, function);

  auto consumer0_a =
      ops::Identity(scope.WithOpName("consumer0_a"), launch.results[0]);
  auto consumer0_b =
      ops::Identity(scope.WithOpName("consumer0_b"), launch.results[0]);
  auto consumer0_c =
      ops::Identity(scope.WithOpName("consumer0_c"), launch.results[0]);
  auto consumer1 =
      ops::Identity(scope.WithOpName("consumer1"), launch.results[1]);
  auto consumer2 =
      ops::Identity(scope.WithOpName("consumer2"), launch.results[2]);
  auto consumer3 =
      ops::Identity(scope.WithOpName("consumer3"), launch.results[3]);

  GraphDef expected_def;
  TF_ASSERT_OK(scope.ToGraphDef(&expected_def));

  GraphDef actual_def;
  graph->ToGraphDef(&actual_def);
  TF_EXPECT_GRAPH_EQ(expected_def, actual_def);
}

//////////////////////////////////
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


TEST(EncapsulateXlaComputations, funclibdef) {
GraphDef input_def;
//std::string in_graph="/home/vishal/experiments/xla_tf12_dummy_eg/graph.pbtxt";
std::string in_graph="/home/vishal/learning_rate/out/vgg_19_tf12.pbtxt";
//std::string in_graph="/home/vishal/experiments/xla_tf12_dummy_eg/xla_compile/graph.pbtxt";
Status s = LoadTextOrBinaryGraphFile(in_graph, &input_def);

if (!s.ok()) LOG(FATAL) << "Loading graph failed: " << s.error_message();

//std::cout<<"before:\n"<<graph_def.DebugString();
std::ofstream myfile;
  myfile.open ("graph_before_opt_nofetch.pbtxt");
  myfile << input_def.DebugString();
  myfile.close();

/* initial optimization*/
  // Just record properties of optimized Grappler items.
  RewriterConfig rewriter_config;
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_min_graph_nodes(-1);

 grappler::MetaOptimizer optimizer(nullptr, rewriter_config);
 //XlaFusionOptimizer optimizer;
grappler::GrapplerItem item;
  item.id = "main";
  item.graph = input_def;
  //item.save_op = "vgg_19/final_node"; // guessing should be target node
  //item.save_op = "sparse_softmax_cross_entropy_loss/Mul";
  GraphDef opt1_graph_def;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &opt1_graph_def));

GraphConstructorOptions opts;
std::unique_ptr<Graph> opt1_graph(new Graph(OpRegistry::Global())); // how to define correctly

ConvertGraphDefToGraph(opts, opt1_graph_def, opt1_graph.get());

/*end*/

// add device to configproto session_options
/*DEVICE_CPU_XLA_JIT,DEVICE_XLA_CPU*/
std::string device_name=DEVICE_XLA_CPU;
absl::string_view xla_cpu_device ="/job:worker/replica:0/task:0/device:XLA_CPU_JIT:0"; // should match with registration.compilation_device_name
for (Node* n : opt1_graph.get()->nodes()) {
    n->set_assigned_device_name(string(xla_cpu_device));
  }

/*for (const Edge* e : (opt1_graph.get())->edges()) {
    if (e->IsControlEdge()) {
      std::cout<<" me\n";
    }
  }*/


FunctionDefLibrary flib;
//FunctionLibraryDefinition flib_def((body_graph.get())->op_registry(), flib); same as 2 lines down
TF_ASSERT_OK(GraphToFunctionDef(*opt1_graph, "launch0", flib.add_function()));

FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);


SessionOptions session_options;
session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_global_jit_level(OptimizerOptions::OFF);//ON_1
GraphOptimizationPassOptions test_option;
test_option.graph=&opt1_graph;
test_option.flib_def=&flib_def;
test_option.session_options=&session_options;

XlaOpRegistry::RegisterCompilationKernels();
legacy_flags::XlaDeviceFlags* flags = legacy_flags::GetXlaDeviceFlags();
bool compile_on_demand = flags->tf_xla_compile_on_demand;

XlaOpRegistry::DeviceRegistration registration;
registration.compilation_device_name = device_name;//DEVICE_XLA_CPU; // also match with device_name in XlaDevice:Create function
registration.requires_compilation = !compile_on_demand; //!
registration.enable_jit_by_default = false;//false
registration.compile_resource_ops = true;

std::string name_prefix="";
std::unique_ptr<XlaDevice> device;
XlaDevice::Create("Host", device_name/*DEVICE_XLA_CPU*/, 0
                                      , DEVICE_CPU_XLA_JIT
                                      , session_options, name_prefix,
                                       registration,
                                       /*transfer_as_literal=*/false,
                                       /*use_multiple_streams=*/false,
                                       /*shape_representation_fn=*/{},
                                       /*padded_shape_fn=*/{}, &device);




EncapsulateXlaComputationsPass pass1;
TF_ASSERT_OK(pass1.Run(test_option));
std::cout<<"Main Test: pass1 done\n";

MarkForCompilationPass pass2;
TF_ASSERT_OK(pass2.Run(test_option));
std::cout<<"Main Test: pass2 done\n";

PartiallyDeclusterPass pass3;
TF_ASSERT_OK(pass3.Run(test_option));
std::cout<<"Main Test: pass3 done\n";

EncapsulateSubgraphsPass pass4;
TF_ASSERT_OK(pass4.Run(test_option));
std::cout<<"Main Test: pass4 done\n";

BuildXlaOpsPass pass5;
TF_ASSERT_OK(pass5.Run(test_option));
std::cout<<"Main Test: pass5 done\n";

bool flag_xla_opt=false; //default
GraphDef final_graph_def;
if(flag_xla_opt==true){
GraphDef inter_graph_def;
XlaFusionOptimizer optimizer;
grappler::GrapplerItem item_inter;
item_inter.id = "main";
test_option.graph->get()->ToGraphDef(&inter_graph_def);
item_inter.graph = inter_graph_def;
//item.save_op = "vgg_19/final_node"; // guessing should be target node
//item.save_op = "sparse_softmax_cross_entropy_loss/Mul";
TF_EXPECT_OK(optimizer.Optimize(nullptr, item_inter, &final_graph_def));
}
else{
test_option.graph->get()->ToGraphDef(&final_graph_def);
}


//std::ofstream myfile;
  myfile.open ("graph_after_opt_nofetch.pbtxt");
  myfile << final_graph_def.DebugString();
  myfile.close();

std::cout<<"apple0\n";
const std::vector<XlaCompiler::Argument> empty_args;
  {
XlaCompiler::CompileOptions Coptions;
Coptions.is_entry_computation = true;
Coptions.add_token_input_output = false;
xla::Client* client_=xla::ClientLibrary::LocalClientOrDie();
XlaCompiler::Options options;


//FunctionDefLibrary flib1;
//TF_ASSERT_OK(GraphToFunctionDef(*body_graph, "launch0", flib1.add_function()));

//FunctionLibraryDefinition flib_def1(OpRegistry::Global(), flib);
    options.device_type = DeviceType(DEVICE_CPU_XLA_JIT);//DEVICE_CPU_XLA_JIT
    options.client = client_;
    options.flib_def = test_option.flib_def;
XlaCompiler compiler(options);

std::cout<<"apple1\n";
XlaCompiler::CompilationResult result;
std::cout<<"apple2\n";
std::unique_ptr<Graph> final_graph(new Graph(OpRegistry::Global())); // how to define correctly
    ConvertGraphDefToGraph(opts, final_graph_def, final_graph.get());
    
    compiler.CompileGraph(Coptions, "NoOp", std::move(final_graph),empty_args, &result);

std::cout<<"apple3\n";
auto temp2=result.computation;
std::cout<<typeid(temp2.get()).name()<<"mango\n";
auto temp3=temp2.get();
std::cout<<"huhndskjnadsk\n";
auto temp4 = temp3->IsNull();
//auto temp4=1;
std::cout<<"huhndskjnadsk1\n";
if(temp4){
    std::cout<<"help\n";
}
else{
    std::cout<<"confised\n";
}
std::cout<<temp3->proto().DebugString();
  }
//std::cout<<"after:\n"<<final_graph_def.DebugString();
}


}  // namespace tensorflow
