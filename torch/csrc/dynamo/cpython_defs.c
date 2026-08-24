#include <torch/csrc/dynamo/cpython_defs.h>
#include <torch/csrc/dynamo/cpython_includes.h>
#include <torch/csrc/dynamo/debug_macros.h>

#if IS_PYTHON_3_16_PLUS

const uint8_t* THP_PyOpcode_Caches = NULL;
int THP_PyOpcode_Caches_size = 0;

void init_THPCaches() {}

#else

// Rename opcode table/metadata symbols to avoid multiple definition conflict
// with the identical definitions in libpython at link time.
#define _PyOpcode_Caches _torch_PyOpcode_Caches
#define _PyOpcode_Jump _torch_PyOpcode_Jump
#define _PyOpcode_Deopt _torch_PyOpcode_Deopt
#define _PyOpcode_num_popped _torch_PyOpcode_num_popped
#define _PyOpcode_num_pushed _torch_PyOpcode_num_pushed
#define _PyOpcode_opcode_metadata _torch_PyOpcode_opcode_metadata
#define _PyOpcode_macro_expansion _torch_PyOpcode_macro_expansion
#define _PyOpcode_OpName _torch_PyOpcode_OpName
#define _PyOpcode_PseudoTargets _torch_PyOpcode_PseudoTargets

#define Py_BUILD_CORE
#define NEED_OPCODE_TABLES // To get _PyOpcode_Deopt, _PyOpcode_Caches

#define NEED_OPCODE_METADATA
#include <internal/pycore_opcode_metadata.h>
#undef NEED_OPCODE_METADATA

#undef NEED_OPCODE_TABLES
#undef Py_BUILD_CORE
#undef _PyOpcode_PseudoTargets
#undef _PyOpcode_OpName
#undef _PyOpcode_macro_expansion
#undef _PyOpcode_opcode_metadata
#undef _PyOpcode_num_pushed
#undef _PyOpcode_num_popped
#undef _PyOpcode_Deopt
#undef _PyOpcode_Jump
#undef _PyOpcode_Caches

// As a simple way to reduce the impact of ABI changes on the CPython side, this
// check forces us to manually re-check that the function didn't change on the
// next major version
#if IS_PYTHON_3_16_PLUS
#error \
    "Please ensure that the functions below still match the CPython implementation for 3.15"
#endif

// e.g. COPY_FIELD(op, o, globals) becomes
// PY_XINCREF((o)->func_globals);
// (op)->func_globals = (o)->func_globals;
#define COPY_FIELD(f1, f2, field) \
  Py_XINCREF((f2)->func_##field); \
  (f1)->func_##field = (f2)->func_##field;

// Not actually copied from CPython, but loosely based on
// https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Objects/funcobject.c
// Makes a new PyFunctionObject copy of `o`, but with the code object fields
// determined from `code`.
// Ensure that all fields defined in the PyFunctionObject struct in
// https://github.com/python/cpython/blob/e715da6db1d1d70cd779dc48e1ba8110c51cc1bf/Include/cpython/funcobject.h
// are accounted for.
PyFunctionObject* _PyFunction_CopyWithNewCode(
    PyFunctionObject* o,
    PyCodeObject* code) {
  PyFunctionObject* op = PyObject_GC_New(PyFunctionObject, &PyFunction_Type);
  if (op == NULL) {
    return NULL;
  }
  Py_XINCREF(code);
  op->func_code = (PyObject*)code;
  Py_XINCREF(code->co_name);
  op->func_name = code->co_name;
  Py_XINCREF(code->co_qualname);
  op->func_qualname = code->co_qualname;
  COPY_FIELD(op, o, globals);
  COPY_FIELD(op, o, builtins);
  COPY_FIELD(op, o, defaults);
  COPY_FIELD(op, o, kwdefaults);
  COPY_FIELD(op, o, closure);
  COPY_FIELD(op, o, doc);
  COPY_FIELD(op, o, dict);
  op->func_weakreflist = NULL;
  COPY_FIELD(op, o, module);
  COPY_FIELD(op, o, annotations);
  COPY_FIELD(op, o, annotate);
  COPY_FIELD(op, o, typeparams);
  op->vectorcall = o->vectorcall;
  op->func_version = o->func_version;
  PyObject_GC_Track(op);
  return op;
}

#if !IS_PYTHON_3_15_PLUS
// https://github.com/python/cpython/blob/fad48ea1816be3125ea51edcdfe2f999d6ade796/Objects/obmalloc.c#L635
void* THP_PyObject_VirtualAlloc(size_t size) {
  PyObjectArenaAllocator arena;
  PyObject_GetArenaAllocator(&arena);
  return arena.alloc(arena.ctx, size);
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L728
static _PyStackChunk* allocate_chunk(
    int size_in_bytes,
    _PyStackChunk* previous) {
  CHECK(size_in_bytes % sizeof(PyObject**) == 0);
  _PyStackChunk* res = THP_PyObject_VirtualAlloc(size_in_bytes);
  if (res == NULL) {
    return NULL;
  }
  res->previous = previous;
  res->size = size_in_bytes;
  res->top = 0;
  return res;
}

#define DATA_STACK_CHUNK_SIZE (16 * 1024)
#define MINIMUM_OVERHEAD 1000

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L2182
static PyObject** push_chunk(PyThreadState* tstate, int size) {
  int allocate_size = DATA_STACK_CHUNK_SIZE;
  while (allocate_size < (int)sizeof(PyObject*) * (size + MINIMUM_OVERHEAD)) {
    allocate_size *= 2;
  }
  _PyStackChunk* new = allocate_chunk(allocate_size, tstate->datastack_chunk);
  if (new == NULL) {
    return NULL;
  }
  if (tstate->datastack_chunk) {
    tstate->datastack_chunk->top =
        tstate->datastack_top - &tstate->datastack_chunk->data[0];
  }
  tstate->datastack_chunk = new;
  tstate->datastack_limit = (PyObject**)(((char*)new) + allocate_size);
  // When new is the "root" chunk (i.e. new->previous == NULL), we can keep
  // _PyThreadState_PopFrame from freeing it later by "skipping" over the
  // first element:
  PyObject** res = &new->data[new->previous == NULL];
  tstate->datastack_top = res + size;
  return res;
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Include/internal/pycore_frame.h#L199
static inline bool THP_PyThreadState_HasStackSpace(
    PyThreadState* tstate,
    size_t size) {
  CHECK(
      (tstate->datastack_top == NULL && tstate->datastack_limit == NULL) ||
      (tstate->datastack_top != NULL && tstate->datastack_limit != NULL));
  return tstate->datastack_top != NULL &&
      size < (size_t)(tstate->datastack_limit - tstate->datastack_top);
}

// https://github.com/python/cpython/blob/051b8a2589ff28f0194c3701b21f729444691752/Python/pystate.c#L2207
_PyInterpreterFrame* THP_PyThreadState_BumpFramePointerSlow(
    PyThreadState* tstate,
    size_t size) {
  if (THP_PyThreadState_HasStackSpace(tstate, size)) {
    _PyInterpreterFrame* res = (_PyInterpreterFrame*)tstate->datastack_top;
    tstate->datastack_top += size;
    return res;
  }
  if (size > INT_MAX / 2) {
    PyErr_NoMemory();
    return NULL;
  }
  return (_PyInterpreterFrame*)push_chunk(tstate, (int)size);
}
#endif // !IS_PYTHON_3_15_PLUS

const uint8_t* THP_PyOpcode_Caches = NULL;
int THP_PyOpcode_Caches_size = 0;
void init_THPCaches() {
  THP_PyOpcode_Caches = _torch_PyOpcode_Caches;
  THP_PyOpcode_Caches_size = sizeof(_torch_PyOpcode_Caches) / sizeof(uint8_t);
}

#endif // IS_PYTHON_3_15_PLUS
