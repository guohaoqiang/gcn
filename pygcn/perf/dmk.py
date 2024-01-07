# Simple modules for measuring timing using host timer and CUDA events.

import time
import torch.cuda

class _HTimers():
  def __init__(self,tobject): self.tobject = tobject
  def __getattr__(self,name): return self.tobject.gethtimer(name)
class _CTimers():
  def __init__(self,tobject): self.tobject = tobject
  def __getattr__(self,name): return self.tobject.getctimer(name)
class _HCTimers():
  def __init__(self,tobject): self.tobject = tobject
  def __getattr__(self,name): return self.tobject.gethctimer(name)

class Timers():
  def __init__(self):
    self.htimers, self.ctimers = {}, {}
    self.h, self.c, self.hc = _HTimers(self), _CTimers(self), _HCTimers(self)
  def gethtimer(self,name):
    if not name in self.htimers: self.htimers[name] = Timer()
    return self.htimers[name]
  def getctimer(self,name):
    if not name in self.ctimers: self.ctimers[name] = CTimer()
    return self.ctimers[name]
  def gethctimer(self,name):
    return HCTimer(self.gethtimer(name), self.getctimer(name))
  def reset(self):
    for t in (self.htimers,self.ctimers):
      for x in t.values(): x.reset();

class HCTimer():
  def __init__(self,hti=None,cti=True):
    self.h = hti if hti else Timer()
    self.c = cti if isinstance(cti,CTimer) else CTimer(cti)
    self.timers = (self.h,self.c)
  def reset(self):
    for x in self.timers: x.reset()
  def __enter__(self):
    for x in self.timers: x.start()
  def __exit__(self,exc_type,exc_value,stack):
    for x in self.timers: x.stop()
    
class Timer():
  def __init__(self):
    self.dur_ns = 0; self.t_start = time.perf_counter_ns(); self.n_calls = 0;
  def reset(self):
    self.dur_ns = 0; self.t_start = None; self.n_calls = 0;
  def start(self):
    self.t_start = time.perf_counter_ns()
    return self;
  def stop(self):
    assert self.t_start != None
    self.dur_ns += time.perf_counter_ns() - self.t_start;
    self.t_start = None
    self.n_calls += 1;
    return self;
  def __enter__(self): self.start()
  def __exit__(self,exc_type,exc_value,stack): self.stop()
  def __format__(self,spec): return format(self.dur_ns,spec)
  def s(self): return self.dur_ns * 1e-9
  def ms(self): return self.dur_ns * 1e-6
  def us(self): return self.dur_ns * 1e-3
  def ns(self): return self.dur_ns
  def avns(self): return self.dur_ns / self.n_calls if self.n_calls else 0
  def avs(self): return self.avns() * 1e-9
  def avms(self): return self.avns() * 1e-6
  def avus(self): return self.avns() * 1e-3


class CTimer():
  def __init__(self,on=True):
    self.dur_ms = 0;
    self.now_events = None
    self.pending_events = [];
    self.free_events = [];
    self.timing_now = False
    self.want_ctiming = on
    self.cuda_init_check = on;
    self.n_calls = 0
  def reset(self):
    self.harvest(all=True); self.dur_ms = 0; self.n_calls = 0;
  def start(self):
    assert not self.timing_now
    self.timing_now = True
    if self.cuda_init_check:
      self.cuda_init_check = False
      if not torch.cuda.is_initialized(): self.want_ctiming = False
    if not self.want_ctiming: return;
    self.harvest()
    self.now_events = (
      self.free_events.pop() if len(self.free_events) else
      tuple( torch.cuda.Event(enable_timing=True,blocking=True) for x in (1,1)))
    self.now_events[0].record();
  def stop(self):
    assert self.timing_now
    self.timing_now = False;
    self.n_calls += 1;
    if not self.want_ctiming: return;
    self.now_events[1].record();
    self.pending_events.append(self.now_events);
  def harvest(self, all = False):
    while ( len(self.pending_events) and
            ( all or self.pending_events[0][1].query() ) ):
      start, stop = self.pending_events[0];
      if all: stop.synchronize()
      self.free_events.append( self.pending_events.pop(0) )
      self.dur_ms += start.elapsed_time(stop);
    assert not all or len(self.pending_events) == 0
  def __enter__(self): self.start()
  def __exit__(self,exc_type,exc_value,stack): self.stop()
  def ms(self): self.harvest(all=True); return self.dur_ms
  def s(self): return self.ms() * 1e-3
  def us(self): return self.ms() * 1e3
  def avms(self): return self.ms()/self.n_calls if self.n_calls else 0
  def avs(self): return self.avms() * 1e-3
  def avus(self): return self.avms() * 1e3
