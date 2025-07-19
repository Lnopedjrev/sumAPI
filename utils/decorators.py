from functools import wraps
from inspect import iscoroutinefunction


def preparable(func):

    is_coroutine = iscoroutinefunction(func)

    async def async_wrapper(self, **kwargs):
        if self.prepared:
            dummy_object = dict(
                user_id=0,
                original_text="",
                categories=[],
                summary="",
            )
            return await func(self, **dummy_object)
        else:
            return await func(self, **kwargs)

    def wrapper(self, **kwargs):
        if self.prepared:
            dummy_object = dict(
                user_id=0,
                original_text="",
                categories=[],
                summary="",
            )
            return func(self, **dummy_object)
        else:
            return func(self, **kwargs)
    return async_wrapper if is_coroutine else wrapper