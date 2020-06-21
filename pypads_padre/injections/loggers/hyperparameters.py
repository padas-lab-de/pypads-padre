import sys

from pypads.app.injections.base_logger import LoggingFunction
from pypads.injections.analysis.call_tracker import LoggingEnv


class HyperParameters(LoggingFunction):

    def __call_wrapped__(self, ctx, *args, _pypads_env: LoggingEnv, _args, _kwargs, **_pypads_hook_params):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()

        def tracer(frame, event, arg):
            if event == 'return':
                params = frame.f_locals.copy()
                key = _pypads_env.callback.__wrapped__.__qualname__ if _pypads_env.call.call_id.is_wrapped() else \
                    _pypads_env.callback.__qualname__
                pads.cache.run_add(key, params)

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            fn = _pypads_env.callback
            if _pypads_env.call.call_id.is_wrapped():
                fn = _pypads_env.callback.__wrapped__
            fn(*_args, **_kwargs)
            self._fn = fn
        finally:
            # deactivate tracer
            sys.setprofile(None)

        return _pypads_env.callback(*_args, **_kwargs)

    def __post__(self, ctx, *args, **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        params = pads.cache.run_get(self._fn.__qualname__)
        for key, param in params.items():
            pads.api.log_param(key, param)
