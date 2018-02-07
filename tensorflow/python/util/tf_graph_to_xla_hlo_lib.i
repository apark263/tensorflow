%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/contrib/xla_extractor/tf_graph_to_xla_hlo_lib.h"
%}

%ignoreall
%unignore tensorflow;
%unignore ExtractXlaWithStringInputs;

%{
std::vector<string> ExtractXlaWithStringInputs(string graph_def_string,
                                               string targets_string,
                                               TF_Status* out_status) {
  std::vector<string> result;

  tensorflow::Status extraction_status = 
    tensorflow::xla_extract_via_strings(graph_def_string, targets_string, &result);

  if (!extraction_status.ok()) {
    tensorflow::Set_TF_Status_from_Status(out_status, extraction_status);
    return result;
  }
  Set_TF_Status_from_Status(out_status, tensorflow::Status::OK());
  return result;
}
%}

std::vector<string> ExtractXlaWithStringInputs(string graph_def_string,
                                               string targets_string,
                                               TF_Status* out_status);

%unignoreall