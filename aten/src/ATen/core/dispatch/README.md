This folder contains the c10 dispatcher. This dispatcher is a single point
through which all kernel calls are routed.

This folder contains the following files:
- Dispatcher.h: Main facade interface. Code using the dispatcher should only use this.
- OperatorEntry.h: Represents a single operator in the dispatcher, including its dispatch table.
- DispatchKeyExtractor.h: Extracts the dispatch key from the arguments of an operator call.

The core interface for calling a kernel (KernelFunction.h) lives in ../boxing.
