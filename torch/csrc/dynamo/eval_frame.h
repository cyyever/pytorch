#pragma once
#include <stdbool.h>
#include <stdint.h>

#include <torch/csrc/dynamo/extra_state.h>
#include <torch/csrc/utils/python_compat.h>
#ifdef __cplusplus

extern "C" {

PyObject* torch_c_dynamo_eval_frame_init(void);

#endif

// https://docs.python.org/3/c-api/init.html#c._PyFrameEvalFunction
#define THP_EVAL_API_FRAME_OBJECT _PyInterpreterFrame

// We need to be able to return the _PyInterpreterFrame to python so create
// a python binding for it

typedef struct THPPyInterpreterFrame {
  PyObject_HEAD
  THP_EVAL_API_FRAME_OBJECT* frame; // Borrowed reference
  PyObject* locals;
} THPPyInterpreterFrame;

THPPyInterpreterFrame* THPPyInterpreterFrame_New(
    THP_EVAL_API_FRAME_OBJECT* frame);

extern bool is_skip_guard_eval_unsafe;
extern int fullgraph_compiled_frame_count;
extern bool fullgraph_error_on_nested_compile;


void eval_frame_callback_set(PyObject* obj);

int64_t get_current_isolate_recompiles_id(void);

const char* get_frame_name(THP_EVAL_API_FRAME_OBJECT* frame);

PyObject* dynamo_eval_frame_default(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag);

PyObject* dynamo_eval_custom_code(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    PyCodeObject* code,
    const char* trace_annotation,
    int throw_flag);

#ifdef __cplusplus

} // extern "C"

#endif
