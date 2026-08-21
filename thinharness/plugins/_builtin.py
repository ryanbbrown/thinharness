"""Private identity and configuration policy for built-in plugins."""

from __future__ import annotations

from typing import Any, ClassVar


class _BuiltinPluginMeta(type):
    """Install and protect one fixed name for each built-in plugin hierarchy."""

    _builtin_fixed_name: str
    _builtin_owner_name: str

    def __new__(
        mcls,
        class_name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        *,
        fixed_name: str | None = None,
        **kwargs: Any,
    ) -> _BuiltinPluginMeta:
        inherited_owner = next(
            (base for base in bases if getattr(base, "_builtin_fixed_name", None) is not None),
            None,
        )
        if inherited_owner is not None:
            owner_name = inherited_owner._builtin_owner_name
            inherited_name = inherited_owner._builtin_fixed_name
            if fixed_name is not None or "name" in namespace:
                raise TypeError(
                    f"{owner_name} subclasses cannot override the fixed name {inherited_name!r}"
                )
        elif fixed_name is not None:
            namespace["name"] = fixed_name
            namespace["_builtin_fixed_name"] = fixed_name
            namespace["_builtin_owner_name"] = class_name
        return super().__new__(mcls, class_name, bases, namespace, **kwargs)

    def __setattr__(cls, attribute: str, value: object) -> None:
        if attribute == "name" and (fixed_name := getattr(cls, "_builtin_fixed_name", None)) is not None:
            owner_name = cls._builtin_owner_name
            raise AttributeError(f"{owner_name}.name is fixed to {fixed_name!r}")
        super().__setattr__(attribute, value)

    def __delattr__(cls, attribute: str) -> None:
        if attribute == "name" and (fixed_name := getattr(cls, "_builtin_fixed_name", None)) is not None:
            owner_name = cls._builtin_owner_name
            raise AttributeError(f"{owner_name}.name is fixed to {fixed_name!r}")
        super().__delattr__(attribute)


class _BuiltinPlugin(metaclass=_BuiltinPluginMeta):
    """Protect a built-in plugin's fixed name on instances."""

    name: ClassVar[str]
    _builtin_fixed_name: ClassVar[str]
    _builtin_owner_name: ClassVar[str]

    def __setattr__(self, attribute: str, value: object) -> None:
        if attribute == "name":
            self._raise_fixed_name_error()
        object.__setattr__(self, attribute, value)

    def __delattr__(self, attribute: str) -> None:
        if attribute == "name":
            self._raise_fixed_name_error()
        object.__delattr__(self, attribute)

    def _raise_fixed_name_error(self) -> None:
        plugin_type = type(self)
        owner_name = plugin_type._builtin_owner_name
        fixed_name = plugin_type._builtin_fixed_name
        raise AttributeError(f"{owner_name}.name is fixed to {fixed_name!r}")


class _FrozenBuiltinPlugin(_BuiltinPlugin):
    """Reject changes to built-in plugin configuration after construction."""

    def __setattr__(self, attribute: str, value: object) -> None:
        if attribute == "name":
            self._raise_fixed_name_error()
        if self.__dict__.get("_frozen", False):
            owner_name = type(self)._builtin_owner_name
            raise AttributeError(f"{owner_name} configuration is frozen")
        object.__setattr__(self, attribute, value)

    def __delattr__(self, attribute: str) -> None:
        if attribute == "name":
            self._raise_fixed_name_error()
        if self.__dict__.get("_frozen", False):
            owner_name = type(self)._builtin_owner_name
            raise AttributeError(f"{owner_name} configuration is frozen")
        object.__delattr__(self, attribute)
