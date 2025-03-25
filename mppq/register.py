"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

A register that can register any types or objects.

Example:

.. code-block:: python

    FOO = Registry("FOO")

    class Foo: ...

    # register directly
    FOO.register(name="foo_snake_style")(Foo)

    # use as a decorator
    @FOO.register()
    class Bar: ...

    # update as a dict
    FOO.update(Bar=NewBar, Foo=Foo)

    print(FOO)
    # ┌─────────────────┐
    # │ Register: FOO   │
    # ├─────────────────┤
    # │ foo_snake_style │
    # │ Bar             │
    # └─────────────────┘
"""

import inspect
from typing import (
    Callable,
    Dict,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Optional,
    TypeVar,
    cast,
)

from tabulate import tabulate

T = TypeVar("T", bound=Callable)
F = TypeVar("F", bound=Callable)


class Registry(Generic[T]):
    """A simple registry object to hold objects from others

    Samples::

        FOO = Registry("FOO")

        @FOO.register()
        def foo(): ...

        print(FOO)
        # ┌───────────────┐
        # │ Register: FOO │
        # ├───────────────┤
        # │ foo           │
        # └───────────────┘
    """

    def __init__(
        self,
        name: Optional[str] = None,
        base: Optional[T] = None,
        parent: Optional["Registry[T]"] = None,
        name_style: Optional[Literal["snake_case", "PascalCase"]] = None,
    ) -> None:
        self._bucks: Dict[str, T] = {}
        self._configs: Dict[str, inspect.Signature] = {}
        self._name = name or "<Registry>"
        self._base = base
        self._parent = parent
        if parent is not None:
            self._name = f"{parent.name}.{self.name}"
        self._style = name_style

    @property
    def name(self) -> str:
        """Return the name of the registry."""
        return self._name

    def _legal_name(self, name: str) -> str:
        if not self._style:
            return name
        words = [""]
        for a, b in zip(list(name), list(name.lower())):
            if a != b:
                words.append("")
            words[-1] += b
        refine_words = [""]
        for w in words:
            if len(w) == 1:
                refine_words[-1] += w
            else:
                refine_words.append(w)
        if refine_words[0] == "":
            refine_words = refine_words[1:]
        if self._style == "snake_case":
            return "_".join(refine_words).strip("_")
        elif self._style == "PascalCase":
            return "".join(w.capitalize() for w in refine_words)
        else:
            return name

    def register(self, name: Optional[str] = None):
        """A decorator to register an object.

        Args:
            name (str, optional): The name of the object. If not provided, the name
                of the function of class will be used after transform to lowercase.
        """

        def wrapper(func: F) -> F:
            if not callable(func):
                raise TypeError(
                    "the object to be registered must be a function or Callable,"
                    f" got {type(func)}"
                )
            _name = name or self._legal_name(func.__name__)
            self._bucks[_name] = cast(T, func)
            self._configs[_name] = inspect.signature(func)
            if not inspect.isfunction(func):
                if self._base is not None:
                    assert isinstance(func, type) and isinstance(self._base, type)
                    if not issubclass(func, self._base):
                        raise TypeError(
                            f"the registered object {func} must be the subclass "
                            f"of {self._base}, but its mro is {func.__mro__}"
                        )
            if self._parent is not None:
                self._parent.register(name)(func)
            # forward the signature of the original function
            return cast(F, func)

        return wrapper

    def update(self, obj: Optional[Mapping[str, T]] = None, **kwargs: T) -> None:
        """Update the registry with a mapping or keyword arguments.

        Args:
            obj (Mapping[str, T]): A mapping of name to object.
            **kwargs (T): Keyword arguments of object.
        """
        if obj is not None and not isinstance(obj, Mapping):
            raise TypeError(f"obj must be a mapping, got {type(obj)}")
        if obj is not None:
            for name, func in obj.items():
                self.register(name)(func)
        for name, func in kwargs.items():
            self.register(name)(func)

    def get(self, name: str) -> Optional[T]:
        """Get a registered object by its name."""
        if name in self._bucks:
            functor = self._bucks[name]
            return functor

    def get_config(self, name: str):
        """Get the configuration of an object"""
        return self._configs.get(name)

    def query_name(self, cls: T) -> str:
        """Query registered name of a class"""
        for key, value in self._bucks.items():
            if value is cls:
                return key
        raise KeyError(f"{cls.__name__} is not registered in {self._name}")

    def __getitem__(self, name: str) -> T:
        """Get a registered object by its name."""
        obj = self.get(name)
        if obj is None:
            raise KeyError(f"{name} is not registered in {self._name}")
        return obj

    def __iter__(self) -> Iterator[str]:
        """Return an Iterator for all registered functions"""
        yield from self._bucks.keys()

    def __contains__(self, name: str | T | None) -> bool:
        """Check if a function is registered"""
        if name is None:
            return False
        if isinstance(name, str):
            return name in self._bucks
        return name in self._bucks.values()

    def __repr__(self) -> str:
        title = [f"Register: {self.name}", "Config"]
        members = []
        for i in sorted(self._bucks.keys()):
            members.append([i, self._configs[i]])
        return tabulate(members, title, "simple_grid", maxcolwidths=[None, 150])
