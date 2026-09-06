---
orphan: true
---

(multiprocessing-doc)=

# Multiprocessing package - torch.multiprocessing

```{eval-rst}
.. automodule:: torch.multiprocessing
```

```{eval-rst}
.. currentmodule:: torch.multiprocessing
```

:::{warning}
If the main process exits abruptly (e.g. because of an incoming signal),
Python's `multiprocessing` sometimes fails to clean up its children.
It's a known caveat, so if you're seeing any resource leaks after
interrupting the interpreter, it probably means that this has just happened
to you.
:::

## Strategy management

```{eval-rst}
.. autofunction:: get_all_sharing_strategies
```

```{eval-rst}
.. autofunction:: get_sharing_strategy
```

```{eval-rst}
.. autofunction:: set_sharing_strategy

```

(multiprocessing-cuda-sharing-details)=

## Sharing CUDA tensors

Sharing CUDA tensors between processes is supported only in Python 3, using
a `spawn` or `forkserver` start methods.

Unlike CPU tensors, the sending process is required to keep the original tensor
as long as the receiving process retains a copy of the tensor. The refcounting is
implemented under the hood but requires users to follow the next best practices.

:::{warning}
If the consumer process dies abnormally to a fatal signal, the shared tensor
could be forever kept in memory as long as the sending process is running.
:::

1. Release memory ASAP in the consumer.

```
## Good
x = queue.get()
# do somethings with x
del x
```

```
## Bad
x = queue.get()
# do somethings with x
# do everything else (producer have to keep x in memory)
```

2. Keep producer process running until all consumers exits. This will prevent
the situation when the producer process releasing memory which is still in use
by the consumer.

```
## producer
# send tensors, do something
event.wait()
```

```
## consumer
# receive tensors and use them
event.set()
```

3. Don't pass received tensors.

```
# not going to work
x = queue.get()
queue_2.put(x)
```

```
# you need to create a process-local copy
x = queue.get()
x_clone = x.clone()
queue_2.put(x_clone)
```

```
# putting and getting from the same queue in the same process will likely end up with segfault
queue.put(tensor)
x = queue.get()
```

## Sharing strategies

This section provides a brief overview into how different sharing strategies
work. Note that it applies only to CPU tensor - CUDA tensors will always use
the CUDA API, as that's the only way they can be shared.

### File descriptor - `file_descriptor`

This is the only supported sharing strategy.

This strategy will use file descriptors as shared memory handles. Whenever a
storage is moved to shared memory, a file descriptor obtained from `shm_open`
is cached with the object, and when it's going to be sent to other processes,
the file descriptor will be transferred (e.g. via UNIX sockets) to it. The
receiver will also cache the file descriptor and `mmap` it, to obtain a shared
view onto the storage data.

This strategy keeps shared-memory file descriptors open. Systems that share
many tensors must configure a sufficiently high file-descriptor limit.

## Spawning subprocesses

:::{note}
Available for Python >= 3.4.

This depends on the `spawn` start method in Python's
`multiprocessing` package.
:::

Spawning a number of subprocesses to perform some function can be done
by creating `Process` instances and calling `join` to wait for
their completion. This approach works fine when dealing with a single
subprocess but presents potential issues when dealing with multiple
processes.

Namely, joining processes sequentially implies they will terminate
sequentially. If they don't, and the first process does not terminate,
the process termination will go unnoticed. Also, there are no native
facilities for error propagation.

The `spawn` function below addresses these concerns and takes care
of error propagation, out of order termination, and will actively
terminate processes upon detecting an error in one of them.

```{eval-rst}
.. automodule:: torch.multiprocessing.spawn
```

```{eval-rst}
.. currentmodule:: torch.multiprocessing.spawn
```

```{eval-rst}
.. autofunction:: spawn
```

```{eval-rst}
.. currentmodule:: torch.multiprocessing

```

```{eval-rst}
.. class:: SpawnContext

   Returned by :func:`~spawn` when called with ``join=False``.

   .. automethod:: join

```

% This module needs to be documented. Adding here in the meantime

% for tracking purposes

## torch.multiprocessing.pool

```{eval-rst}
.. currentmodule:: torch.multiprocessing.pool
```

```{eval-rst}
.. py:module:: torch.multiprocessing.pool
```

```{eval-rst}
.. py:module:: torch.multiprocessing.queue
```

```{eval-rst}
.. py:module:: torch.multiprocessing.reductions
```
