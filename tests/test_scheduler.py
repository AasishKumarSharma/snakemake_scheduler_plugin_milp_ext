"""Unit tests for the MILP scheduler plugin."""

from typing import Optional, Type

from snakemake_interface_scheduler_plugins.base import SchedulerBase
from snakemake_interface_scheduler_plugins.settings import SchedulerSettingsBase
from snakemake_interface_scheduler_plugins.tests import TestSchedulerBase

from snakemake_scheduler_plugin_milp_ext import Scheduler, SchedulerSettings


class TestMILPScheduler(TestSchedulerBase):
    __test__ = True

    def get_scheduler_cls(self) -> Type[SchedulerBase]:
        return Scheduler

    def get_scheduler_settings(self) -> Optional[SchedulerSettingsBase]:
        return SchedulerSettings()


# Additional specific tests for MILP functionality
def test_scheduler_settings():
    """Test that scheduler settings work correctly."""
    settings = SchedulerSettings(
        system_profile="test_profile.json",
        scheduler_config="test_config.yaml",
        time_limit=60,
        fallback="greedy",
    )

    assert settings.system_profile == "test_profile.json"
    assert settings.scheduler_config == "test_config.yaml"
    assert settings.time_limit == 60
    assert settings.fallback == "greedy"


def test_scheduler_initialization():
    """Test that the scheduler initializes correctly."""
    import logging

    from snakemake_interface_scheduler_plugins.tests import DummyDAG

    dag = DummyDAG()
    settings = SchedulerSettings()
    scheduler = Scheduler(dag, settings, logging.getLogger("test"))

    assert scheduler.settings == settings
    assert hasattr(scheduler, "node_assignments")
    assert hasattr(scheduler, "_job_start_times")
    assert hasattr(scheduler, "_finished_jobs_history")


def test_default_config():
    """Test that default configuration is properly generated."""
    import logging

    from snakemake_interface_scheduler_plugins.tests import DummyDAG

    dag = DummyDAG()
    scheduler = Scheduler(dag, None, logging.getLogger("test"))

    config = scheduler._get_default_config()

    assert "scheduler" in config
    assert "type" in config["scheduler"]
    assert "optimization" in config["scheduler"]
    assert "time_limit_seconds" in config["scheduler"]["optimization"]
