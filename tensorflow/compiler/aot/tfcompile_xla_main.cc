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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/aot/codegen.h"
#include "tensorflow/compiler/aot/compile.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace tfcompile {

// Flags for the tfcompile binary.  See *.cc file for descriptions.
struct MainFlags {
  string xlacomp;
  string target_triple;
  string target_cpu;
  string target_features;
  string entry_point;
  string cpp_class;
  string out_function_object;
  string out_metadata_object;
  string out_header;

  // C++ codegen options
  bool gen_name_to_index = false;
  bool gen_program_shape = false;
};

// Appends to flag_list a tensorflow::Flag for each field in MainFlags.

void AppendMainFlags(std::vector<Flag>* flag_list, MainFlags* flags) {
  const std::vector<Flag> tmp = {
      {"xlacomp", &flags->xlacomp,
       "Input xla::Computation file.  If the file ends in '.pbtxt' it is expected to "
       "be in the human-readable proto text format, otherwise it is expected "
       "to be in the proto binary format."},
      // Flags controlling the XLA ahead-of-time compilation, that correspond to
      // the fields of xla::cpu::CpuAotCompilationOptions.
      //
      // TODO(toddw): The following flags also need to be supported:
      //   --xla_cpu_llvm_opt_level
      //   --xla_cpu_llvm_cl_opts
      {"target_triple", &flags->target_triple,
       "Target platform, similar to the clang -target flag.  The general "
       "format is <arch><sub>-<vendor>-<sys>-<abi>.  "
       "http://clang.llvm.org/docs/CrossCompilation.html#target-triple."},
      {"target_cpu", &flags->target_cpu,
       "Target cpu, similar to the clang -mcpu flag.  "
       "http://clang.llvm.org/docs/CrossCompilation.html#cpu-fpu-abi"},
      {"target_features", &flags->target_features,
       "Target features, e.g. +avx2, +neon, etc."},
      {"entry_point", &flags->entry_point,
       "Name of the generated function.  If multiple generated object files "
       "will be linked into the same binary, each will need a unique entry "
       "point."},
      {"cpp_class", &flags->cpp_class,
       "Name of the generated C++ class, wrapping the generated function.  The "
       "syntax of this flag is [[<optional_namespace>::],...]<class_name>.  "
       "This mirrors the C++ syntax for referring to a class, where multiple "
       "namespaces may precede the class name, separated by double-colons.  "
       "The class will be generated in the given namespace(s), or if no "
       "namespaces are given, within the global namespace."},
      {"out_function_object", &flags->out_function_object,
       "Output object file containing the generated function for the "
       "TensorFlow model."},
      {"out_header", &flags->out_header, "Output header file name."},
      {"out_metadata_object", &flags->out_metadata_object,
       "Output object file name containing optional metadata for the generated "
       "function."},
      {"gen_name_to_index", &flags->gen_name_to_index,
       "Generate name-to-index data for Lookup{Arg,Result}Index methods."},
      {"gen_program_shape", &flags->gen_program_shape,
       "Generate program shape data for the ProgramShape method."},
  };
  flag_list->insert(flag_list->end(), tmp.begin(), tmp.end());
}

const char kUsageHeader[] =
    "tfcompile_xla performs ahead-of-time compilation of a XLA HLO computation,\n"
    "resulting in an object file compiled for your target architecture, and a\n"
    "header file that gives access to the functionality in the object file.\n"
    "A typical invocation looks like this:\n"
    "\n"
    "   $ tfcompile_xla --xlacomp=computation.pb --cpp_class=\"mynamespace::MyComputation\"\n"
    "\n";

Status ReadProtoFile(const string& fname, protobuf::Message* proto) {
  if (absl::EndsWith(fname, ".pbtxt")) {
    return ReadTextProto(Env::Default(), fname, proto);
  } else {
    return ReadBinaryProto(Env::Default(), fname, proto);
  }
}

Status Main(const MainFlags& flags) {
  // Read and initialize the graph.
  if (flags.xlacomp.empty()) {
    return errors::InvalidArgument("Must specify --xlacomp");
  }
  HloModuleProto xla_hlo_module;
  TF_RETURN_IF_ERROR(ReadProtoFile(flags.xlacomp, &xla_hlo_module));

  xla::XlaComputation xla_computation(xla_hlo_module);

  se::Platform* cpu_platform =
      se::MultiPlatformManager::PlatformWithName("Host").ValueOrDie();
  xla::CompileOnlyClient* client =
      xla::ClientLibrary::GetOrCreateCompileOnlyClient(cpu_platform)
          .ValueOrDie();

  xla::cpu::CpuAotCompilationOptions aot_opts(
      flags.target_triple, flags.target_cpu, flags.target_features,
      flags.entry_point,
      xla::cpu::CpuAotCompilationOptions::RelocationModel::BigPic);

  CompileResult compile_result;
  TF_RETURN_IF_ERROR(CompileXla(client, xla_computation, aot_opts, &compile_result));

  // Write output files.
  Env* env = Env::Default();
  const std::vector<char>& obj = compile_result.aot->object_file_data();
  TF_RETURN_IF_ERROR(
      WriteStringToFile(env, flags.out_function_object,
                        absl::string_view(obj.data(), obj.size())));
  CodegenOpts codegen_opts;
  codegen_opts.gen_name_to_index = flags.gen_name_to_index;
  codegen_opts.gen_program_shape = flags.gen_program_shape;
  codegen_opts.target_triple = flags.target_triple;
  if (flags.cpp_class.empty()) {
    return errors::InvalidArgument("Must specify --cpp_class");
  }
  codegen_opts.gen_hlo_profile_printer_data =
      xla::GetDebugOptionsFromFlags().xla_hlo_profile();
  TF_RETURN_IF_ERROR(ParseCppClass(flags.cpp_class, &codegen_opts.class_name,
                                   &codegen_opts.namespaces));

  MetadataResult metadata_result;
  TF_RETURN_IF_ERROR(
      GenerateMetadata(codegen_opts, compile_result, &metadata_result));
  TF_RETURN_IF_ERROR(WriteStringToFile(env, flags.out_metadata_object,
                                       metadata_result.object_file_data));
  string header;
  TF_RETURN_IF_ERROR(GenerateHeader(codegen_opts, config, compile_result,
                                    metadata_result, &header));
  TF_RETURN_IF_ERROR(WriteStringToFile(env, flags.out_header, header));
  return Status::OK();
}

}  // end namespace tfcompile
}  // end namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::tfcompile::MainFlags flags;
  flags.target_triple = "x86_64-pc-linux";
  flags.out_function_object = "out_model.o";
  flags.out_metadata_object = "out_helper.o";
  flags.out_header = "out.h";
  flags.entry_point = "entry";

  std::vector<tensorflow::Flag> flag_list;
  AppendMainFlags(&flag_list, &flags);
  xla::AppendDebugOptionsFlags(&flag_list);

  tensorflow::string usage = tensorflow::tfcompile::kUsageHeader;
  usage += tensorflow::Flags::Usage(argv[0], flag_list);
  bool parsed_flags_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  QCHECK(parsed_flags_ok) << "\n" << usage;

  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  QCHECK(argc == 1) << "\nERROR: This command does not take any arguments "
                       "other than flags\n\n"
                    << usage;
  tensorflow::Status status = tensorflow::tfcompile::Main(flags);
  if (status.code() == tensorflow::error::INVALID_ARGUMENT) {
    std::cerr << "INVALID ARGUMENTS: " << status.error_message() << "\n\n"
              << usage;
    return 1;
  } else {
    TF_QCHECK_OK(status);
  }
  return 0;
}
