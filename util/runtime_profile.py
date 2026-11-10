"""Opt-in, bounded CPU/CUDA traces of the actual training loop."""
from contextlib import nullcontext
from contextvars import ContextVar
import json
from pathlib import Path
import tempfile

import torch


_recording = ContextVar('owod_runtime_recording', default=False)


def scope(name):
    return torch.profiler.record_function('owod/' + name) if _recording.get() else nullcontext()


class RuntimeProfile:
    """Profile one fixed window, then release the profiler while training continues.

    CPU scope totals include CUDA waits; device totals can overlap. Neither is
    an exclusive fraction of the step. Use the trace to attribute synchronization.
    No model/optimizer/checkpoint state is saved or changed by this recorder.
    """
    def __init__(self, directory, device, *, steps=0, warmup=20, epoch=0, rank=0, metadata=None):
        if steps < 0 or warmup < 0:
            raise ValueError('Runtime profile steps and warmup must be nonnegative')
        if steps and not directory:
            raise ValueError('Runtime profiling requires an explicit output directory')
        self.steps, self.warmup = steps, warmup
        self.device = torch.device(device)
        self.directory = Path(directory or '.') / 'profiles'
        self.epoch, self.rank = epoch, rank
        self.metadata = dict(metadata or {})
        self.profiler = None
        self.count = 0
        self.token = None

    def __enter__(self):
        if not self.steps:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        self.destination = Path(tempfile.mkdtemp(
            prefix=f'epoch{self.epoch:04d}-rank{self.rank}-', dir=self.directory))
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=self.warmup, active=self.steps, repeat=1),
            on_trace_ready=self._save, record_shapes=False, profile_memory=False, with_stack=False)
        self.profiler.__enter__()
        self.token = _recording.set(True)
        return self

    def _save(self, profiler):
        profiler.export_chrome_trace(str(self.destination / 'trace.json'))
        events = profiler.key_averages()
        completed = min(self.steps, max(0, self.count - self.warmup))
        regions = {}
        for event in events:
            if event.key.startswith('owod/'):
                device_us = getattr(event, 'device_time_total', None)
                if device_us is None:
                    device_us = getattr(event, 'cuda_time_total', 0.)
                regions[event.key] = dict(calls=event.count,
                    cpu_total_ms_per_step=event.cpu_time_total / 1000 / completed if completed else None,
                    device_total_ms_per_step=device_us / 1000 / completed if completed else None)
        report = dict(epoch=self.epoch, rank=self.rank, warmup_steps=self.warmup,
            active_steps=completed, requested_active_steps=self.steps, window_complete=completed == self.steps,
            device=str(self.device), torch_version=torch.__version__,
            cuda_events_recorded=any(event.device_type == torch.autograd.DeviceType.CUDA
                                     for event in profiler.events()),
            regions=regions, metadata=self.metadata,
            note='Nested CPU/device totals are not additive percentages; inspect trace.json for overlap and waits.')
        (self.destination / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        (self.destination / 'operators.txt').write_text(
            events.table(sort_by='self_cpu_time_total', row_limit=80), encoding='utf-8')
        print(f'Runtime profile saved: {self.destination}', flush=True)
        if not report['window_complete']:
            print('Runtime profile window was interrupted; partial timings are not a completed measurement.', flush=True)
        if self.device.type == 'cuda' and not report['cuda_events_recorded']:
            print('Runtime profile has no CUDA kernel events; CPU scopes alone cannot establish GPU costs.', flush=True)

    def step(self):
        if self.profiler is not None:
            self.count += 1
            self.profiler.step()
            if self.count >= self.warmup + self.steps:
                self.__exit__(None, None, None)

    def __exit__(self, exc_type, exc, traceback):
        if self.profiler is not None:
            try:
                self.profiler.__exit__(exc_type, exc, traceback)
            finally:
                self.profiler = None
                _recording.reset(self.token)
                self.token = None
