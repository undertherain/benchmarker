# Code from https://github.com/awwong1/torchprof

import functools
import json
import torch.autograd.profiler as tprofiler
from collections import namedtuple, defaultdict, OrderedDict

Trace = namedtuple("Trace", ["path", "leaf", "module"])
Measure = namedtuple("Measure", ["self_cpu_total", "cpu_total", "cuda_total", "occurrences", "param", "input_shape"])

def walk_modules(module, name="", path=()):
    """Generator. Walks through a PyTorch Module and outputs Trace tuples"""
    if not name:
        name = module.__class__.__name__
    named_children = list(module.named_children())
    path = path + (name,)
    yield Trace(path, len(named_children) == 0 , module)
    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from walk_modules(child_module, name=name, path=path)


class Profile(object):
    """Layer by layer profiling of Pytorch models, using the Pytorch autograd profiler.
    """

    def __init__(self, model, enabled=True, use_cuda=False, paths=None):
        self._model = model
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.paths = paths

        self.entered = False
        self.exited = False
        self.traces = ()
        self.trace_profile_events = defaultdict(list)

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("torchprof profiler is not reentrant")
        self.entered = True
        self._forwards = {}  # store the original forward functions
        self.traces = tuple(map(self._hook_trace, walk_modules(self._model)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # remove unnecessary forwards
        self.exited = True

    def __str__(self):
        if self.exited:
            return traces_to_display(
                self.traces, self.trace_profile_events, paths=self.paths
            )
        return "<unfinished torchprof.profile>"

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        [path, leaf, module] = trace
        if (self.paths is not None and path in self.paths) or (
            self.paths is None and leaf
        ):
            _forward = module.forward
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                with tprofiler.profile(use_cuda=self.use_cuda,record_shapes=True) as prof:
                    res = _forward(*args, **kwargs)
                event_list = prof.function_events
                event_list.populate_cpu_children()
                # each profile call should be contained in its own list
                self.trace_profile_events[path].append(event_list)
                return res

            module.forward = wrap_forward
        return trace

    def _remove_hook_trace(self, trace):
        [path, leaf, module] = trace
        if (self.paths is not None and path in self.paths) or (
            self.paths is None and leaf
        ):
            module.forward = self._forwards[path]

    def raw(self):
        if self.exited:
            return (self.traces, self.trace_profile_events)

    def display(self, show_events=False):
        if self.exited:
            return traces_to_display(
                self.traces,
                self.trace_profile_events,
                show_events=show_events,
                paths=self.paths,
            )
        return "<unfinished torchprof.profile>"


def flatten_tree(t, depth=0):
    flat = []
    for name, st in t.items():
        measures = st.pop(None, None)
        flat.append([depth, name, measures])
        flat.extend(flatten_tree(st, depth=depth + 1))
    return flat


def traces_to_display(traces, trace_events, show_events=False, paths=None):
    """Construct human readable output of the profiler traces and events.
    """
    tree = OrderedDict()
    for trace in traces:
        [path, leaf, module] = trace
        current_tree = tree
        # unwrap all of the events, in case model is called multiple times
        events = [te for tevents in trace_events[path] for te in tevents]
        for depth, name in enumerate(path, 1):
            if name not in current_tree:
                current_tree[name] = OrderedDict()
            if depth == len(path) and (
                (paths is None and leaf) or (paths is not None and path in paths)
            ):
                # tree measurements have key None, avoiding name conflict
                if show_events:
                    for event in events:
                        current_tree[name][event.name] = {
                            None: Measure(
                                sum([e.self_cpu_time_total for e in events if e.name == event.name]),
                                sum([e.cpu_time_total for e in events if e.name == event.name]),
                                sum([e.cuda_time_total for e in events if e.name == event.name]),
                                len([e for e in events if e.name == event.name]),
                                str(module),
                                event.input_shapes
                            )._asdict()
                        }
                else:
                    current_tree[name][None] = Measure(
                        sum([e.self_cpu_time_total for e in events]),
                        sum([e.cpu_time_total for e in events]),
                        sum([e.cuda_time_total for e in events]),
                        len(trace_events[path]),
                        str(module),
                        [e.input_shapes for e in events][0]
                    )._asdict()
            current_tree = current_tree[name]
    return tree
