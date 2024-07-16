from typing import TYPE_CHECKING, Union
import importlib

if TYPE_CHECKING:
    from .cron_schedule import CronSchedule
    from .interval_schedule import IntervalSchedule
    from .no_schedule import NoSchedule
    from .r_rule_schedule import RRuleSchedule
    from .schedule_types import SCHEDULE_TYPES, construct_schedule, is_schedule_type


_public_api: dict[str, tuple[str, str]] = {
    "CronSchedule": (__spec__.parent, ".cron_schedule"),
    "IntervalSchedule": (__spec__.parent, ".interval_schedule"),
    "NoSchedule": (__spec__.parent, ".no_schedule"),
    "RRuleSchedule": (__spec__.parent, ".r_rule_schedule"),
    "SCHEDULE_TYPES": (__spec__.parent, ".schedule_types"),
    "construct_schedule": (__spec__.parent, ".schedule_types"),
    "is_schedule_type": (__spec__.parent, ".schedule_types"),
}

__all__ = [
    "CronSchedule",
    "IntervalSchedule",
    "NoSchedule",
    "RRuleSchedule",
    "SCHEDULE_TYPES",
    "construct_schedule",
    "is_schedule_type",
]


def __getattr__(attr_name: str) -> object:
    dynamic_attr = _public_api.get(attr_name)
    if dynamic_attr is None:
        return importlib.import_module(f".{attr_name}", package=__name__)

    package, module_name = dynamic_attr

    from importlib import import_module

    if module_name == "__module__":
        return import_module(f".{attr_name}", package=package)
    else:
        module = import_module(module_name, package=package)
        return getattr(module, attr_name)