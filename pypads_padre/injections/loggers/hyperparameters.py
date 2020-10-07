import sys

from pypads.app.injections.injection import InjectionLogger
from pypads.app.env import InjectionLoggerEnv


class HyperParameters(InjectionLogger):

    def __call_wrapped__(self, ctx, *args, _pypads_env: InjectionLoggerEnv, _args, _kwargs, **_pypads_hook_params):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()

        def tracer(frame, event, arg):
            if event == 'return':
                params = frame.f_locals.copy()
                key = str(self)
                from pypads.app.pypads import get_current_pads
                pads = get_current_pads()
                pads.cache.run_add(key, params)

        fn = _pypads_env.callback
        if _pypads_env.call.call_id.is_wrapped():
            fn = _pypads_env.callback.__wrapped__
        try:

            # tracer is activated on next call, return or exception
            sys.setprofile(tracer)
            fn(*_args, **_kwargs)
        finally:
            # deactivate tracer
            sys.setprofile(None)
        _return = super().__call_wrapped__(ctx, _pypads_env=_pypads_env, _args=_args, _kwargs=_kwargs, **_pypads_hook_params)

        return _return

    def __post__(self, ctx, *args, **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        params = pads.cache.run_get(str(self))
        for key, param in params.items():
            pads.api.log_param(key, param)
