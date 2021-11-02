import numpy as np
import itertools as it

class SymbolicArray(np.ndarray):
    def __new__(cls, arr, symbolic=None, **kwargs):
        if isinstance(arr, cls):
            return arr

        arr = np.array(arr, copy=False, **kwargs)
        obj = arr.view(cls)
        obj.symbolic = str(symbolic or np.array2string(arr, separator=',', threshold=np.inf, floatmode='unique'))

        return obj

    def expand(self):
        sym = self.symbolic
        for k,v in self.locals.items():
            sym = sym.replace(k,v)
        return sym

    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        # This attribute should be maintained!
        self.symbolic = getattr(obj, 'symbolic', None)
        self.locals = getattr(obj, 'locals', {})

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        assert method == '__call__'
        return np.core.overrides.array_function_dispatch \
            (lambda *args, **kwargs: args, verify=False, module='numpy') (ufunc) \
            (*inputs, **kwargs)

    def __array_function__(self, func, types, inputs, kwargs):
        assert func.__module__
        obj = SymbolicArray (
            func (
                *(x.view(np.ndarray) if isinstance(x, SymbolicArray) else x for x in inputs),
                **kwargs
            ),
            f"{func.__module__}.{func.__name__}({', '.join(self._symbolic_args(inputs, kwargs))})"
        )

        for x in filter(lambda x: isinstance(x, type(self)), inputs):
            obj.locals |= x.locals

            if obj.symbolic.count(x.symbolic) > 1 and len(x.symbolic) > 5:
                var = f"var{abs(hash(x.symbolic))}"

                # add symbol to memory
                obj.locals[var] = x.symbolic

                # substitute
                obj.symbolic = obj.symbolic.replace(x.symbolic, var)

        return obj

    def _symbolic_args(self, inputs, kwargs): 
        return it.chain (
            (x.symbolic if isinstance(x, type(self)) else repr(x) for x in inputs),
            (f'{x}={repr(y)}' for x,y in kwargs.items()),
        )