"""
MILP-based job scheduler plugin for Snakemake.

This module provides an enhanced job scheduler that uses mixed-integer linear programming
to optimize job execution across heterogeneous computing resources.

Plugin name: milp-ext (avoiding forbidden name "milp")
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Union

from snakemake_interface_common.io import AnnotatedStringInterface
from snakemake_interface_scheduler_plugins.base import SchedulerBase
from snakemake_interface_scheduler_plugins.interfaces.jobs import JobSchedulerInterface
from snakemake_interface_scheduler_plugins.settings import SchedulerSettingsBase

# Optional imports with graceful fallbacks
PULP_AVAILABLE = False
try:
    import pulp as _pulp  # noqa: F401

    PULP_AVAILABLE = True
except ImportError:
    pass

try:
    import networkx as nx  # noqa: F401

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

BIG_M = 100000  # Big-M constant for conditional constraints

logger = logging.getLogger("snakemake.scheduler")


@dataclass
class SchedulerSettings(SchedulerSettingsBase):
    """Settings for the MILP scheduler plugin."""

    system_profile: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to system profile JSON file describing compute clusters and nodes",
        },
    )

    scheduler_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to scheduler configuration YAML file",
        },
    )

    time_limit: Optional[int] = field(
        default=30,
        metadata={
            "help": "Time limit in seconds for MILP solver",
        },
    )

    fallback: Optional[str] = field(
        default="greedy",
        metadata={
            "help": "Fallback scheduler when MILP fails (greedy or ilp)",
        },
    )


class Scheduler(SchedulerBase):
    """Enhanced job scheduler using MILP optimization for heterogeneous resources."""

    def __post_init__(self) -> None:
        """Initialize MILP-specific attributes after base initialization."""
        self.node_assignments = {}  # Stores job-to-node assignments
        self._job_start_times = {}  # For tracking job execution times
        self._finished_jobs_history = set()  # Track recorded job history

        # Critical path tracking (for extended MILP)
        self.dag_graph = None
        self.critical_path = None
        self.critical_path_length = 0

        # Check if required libraries are available
        if not PULP_AVAILABLE:
            self.logger.warning(
                "PuLP package is not available. Install with 'pip install pulp'. "
                "Scheduler will use basic greedy selection."
            )

        if not NETWORKX_AVAILABLE:
            self.logger.warning(
                "NetworkX package is not available. Install with 'pip install networkx'. "
                "Critical path analysis will be disabled."
            )

    def select_jobs(
        self,
        selectable_jobs: Sequence[JobSchedulerInterface],
        remaining_jobs: Sequence[JobSchedulerInterface],
        available_resources: Mapping[str, Union[int, str]],
        input_sizes: Dict[AnnotatedStringInterface, int],
    ) -> Sequence[JobSchedulerInterface]:
        """Select jobs using MILP optimization."""

        if not selectable_jobs:
            return []

        if len(selectable_jobs) == 1:
            # For single job, just check resources and return it
            job = selectable_jobs[0]
            if self._can_run_job(job, available_resources):
                return [job]
            else:
                return []

        if not PULP_AVAILABLE:
            # Fallback to simple greedy selection
            return self._greedy_select_jobs(selectable_jobs, available_resources)

        try:
            # Convert to list for indexing
            jobs_list = list(selectable_jobs)

            # 1. Load scheduler configuration
            config = self._load_scheduler_config()
            self.logger.debug(f"Loaded scheduler config: {config}")

            # 2. Load system resource profile
            system_profile = self._load_system_profile(config)
            self.logger.debug(f"Loaded system profile: {list(system_profile.keys())}")

            # 3. Extract available nodes from system profile
            available_nodes = self._extract_available_nodes(system_profile)

            # 4. Check for sufficient nodes
            if not available_nodes:
                self.logger.warning(
                    "No nodes available in system profile. Using greedy selection."
                )
                return self._greedy_select_jobs(selectable_jobs, available_resources)

            # 5. Process job requirements
            job_requirements = {}
            for job in jobs_list:
                job_requirements[id(job)] = self._extract_job_requirements(job, config)

            # 6. Create and solve MILP problem
            selected_jobs = self._solve_milp_scheduling(
                jobs_list, job_requirements, available_nodes, config
            )

            return selected_jobs

        except Exception as e:
            self.logger.warning(f"Error in MILP scheduler: {str(e)}")
            self.logger.warning("Falling back to greedy scheduler.")
            return self._greedy_select_jobs(selectable_jobs, available_resources)

    def _can_run_job(
        self,
        job: JobSchedulerInterface,
        available_resources: Mapping[str, Union[int, str]],
    ) -> bool:
        """Check if a job can run with available resources."""
        job_resources = job.scheduler_resources

        # Check basic resources
        for resource, needed in job_resources.items():
            if isinstance(needed, int) and resource in available_resources:
                available = available_resources.get(resource, 0)
                if isinstance(available, int) and needed > available:
                    return False

        return True

    def _greedy_select_jobs(
        self,
        selectable_jobs: Sequence[JobSchedulerInterface],
        available_resources: Mapping[str, Union[int, str]],
    ) -> Sequence[JobSchedulerInterface]:
        """Simple greedy job selection fallback."""
        selected = []
        remaining_resources = dict(available_resources)

        for job in selectable_jobs:
            if self._can_run_job(job, remaining_resources):
                selected.append(job)

                # Update remaining resources
                job_resources = job.scheduler_resources
                for resource, needed in job_resources.items():
                    if isinstance(needed, int) and resource in remaining_resources:
                        available = remaining_resources.get(resource, 0)
                        if isinstance(available, int):
                            remaining_resources[resource] = available - needed

        return selected

    def _solve_milp_scheduling(
        self, jobs_list, job_requirements, available_nodes, config
    ):
        """Solve MILP scheduling problem."""
        if not PULP_AVAILABLE:
            self.logger.error("PuLP package is required for MILP scheduling")
            return []

        import pulp

        # Create MILP problem
        prob = pulp.LpProblem("HeterogeneousScheduler", pulp.LpMinimize)

        # Create variables
        x = {}  # Job assignment variables
        for job in jobs_list:
            job_id = id(job)
            x[job_id] = {}
            for node in available_nodes:
                x[job_id][node["name"]] = pulp.LpVariable(
                    f"job_{job_id}_node_{node['name']}", cat="Binary"
                )

        # Makespan variable
        makespan = pulp.LpVariable("makespan", lowBound=0, cat="Continuous")

        # Objective: minimize makespan
        prob += makespan

        # Constraints
        # 1. Each job must be assigned to exactly one node
        for job in jobs_list:
            job_id = id(job)
            prob += (
                pulp.lpSum([x[job_id][node["name"]] for node in available_nodes]) == 1
            )

        # 2. Node capacity constraints
        for node in available_nodes:
            node_data = node["data"]

            # CPU cores constraint
            prob += (
                pulp.lpSum(
                    [
                        x[id(job)][node["name"]] * job_requirements[id(job)]["cores"]
                        for job in jobs_list
                    ]
                )
                <= node_data["resources"]["cores"]
            )

            # Memory constraint
            prob += (
                pulp.lpSum(
                    [
                        x[id(job)][node["name"]]
                        * job_requirements[id(job)]["memory_mb"]
                        for job in jobs_list
                    ]
                )
                <= node_data["resources"]["memory_mb"]
            )

        # 3. Feature compatibility constraints
        for job in jobs_list:
            job_id = id(job)
            job_features = job_requirements[job_id]["features"]

            for node in available_nodes:
                node_features = node["data"].get("features", [])

                # If job requires features not present on node, prevent assignment
                if job_features and not all(
                    feature in node_features for feature in job_features
                ):
                    prob += x[job_id][node["name"]] == 0

        # 4. Makespan constraint
        for job in jobs_list:
            job_id = id(job)
            for node in available_nodes:
                runtime = self._calculate_runtime_on_node(
                    job_requirements[job_id], node["data"]
                )
                prob += makespan >= runtime * x[job_id][node["name"]]

        # Solve the problem
        time_limit = (
            config.get("scheduler", {})
            .get("optimization", {})
            .get("time_limit_seconds", 30)
        )
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
        prob.solve(solver)

        # Process solution
        if pulp.LpStatus[prob.status] == "Optimal":
            selected_jobs = []

            for job in jobs_list:
                job_id = id(job)
                for node in available_nodes:
                    if pulp.value(x[job_id][node["name"]]) > 0.5:
                        selected_jobs.append(job)
                        break

            self.logger.info(
                f"MILP solution found with makespan: {pulp.value(makespan)}"
            )
            self.logger.info(f"Selected {len(selected_jobs)} jobs")

            return selected_jobs
        else:
            self.logger.warning(
                f"MILP failed with status: {pulp.LpStatus[prob.status]}"
            )
            return []

    def _load_scheduler_config(self):
        """Load scheduler configuration settings."""
        config_path = None

        # Try to get config path from settings
        if (
            self.settings
            and hasattr(self.settings, "scheduler_config")
            and self.settings.scheduler_config
        ):
            config_path = self.settings.scheduler_config

        # Try multiple locations
        possible_paths = [
            config_path,
            "scheduler_config.yaml",
            os.path.expanduser("~/.snakemake/scheduler_config.yaml"),
        ]

        for path in possible_paths:
            if path and os.path.exists(path):
                try:
                    import yaml

                    with open(path, "r") as f:
                        config = yaml.safe_load(f)
                    self.logger.debug(f"Loaded scheduler config from {path}")
                    return config
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load scheduler config from {path}: {e}"
                    )

        # Return default config
        self.logger.warning("Using default configuration")
        return self._get_default_config()

    def _load_system_profile(self, config):
        """Load system resource profile."""
        profile_path = None

        # Try to get profile path from settings first
        if (
            self.settings
            and hasattr(self.settings, "system_profile")
            and self.settings.system_profile
        ):
            profile_path = self.settings.system_profile
        elif "scheduler" in config and "paths" in config["scheduler"]:
            profile_path = config["scheduler"]["paths"]["system_profile"]

        # Try multiple locations
        possible_paths = [
            profile_path,
            "system_profile.json",
            os.path.expanduser("~/.snakemake/system_profile.json"),
        ]

        for path in possible_paths:
            if path and os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        profile = json.load(f)
                    self.logger.debug(f"Loaded system profile from {path}")
                    return profile
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load system profile from {path}: {e}"
                    )

        # Return default profile
        self.logger.warning("Using default system profile")
        return self._get_default_system_profile()

    def _get_default_system_profile(self):
        """Create a default system profile based on available resources."""
        # Try to get total cores from available resources
        total_cores = 4  # Default fallback
        total_memory = 8192  # Default 8GB

        return {
            "clusters": {
                "local": {
                    "nodes": {
                        "default": {
                            "resources": {
                                "cores": total_cores,
                                "memory_mb": total_memory,
                                "gpu_count": 0,
                                "local_storage_mb": 10000,
                            },
                            "features": ["cpu", "x86_64"],
                            "properties": {
                                "cpu_flops": 100000000000,
                                "memory_bandwidth_mbps": 10000,
                                "read_mbps": 1000,
                                "write_mbps": 800,
                            },
                        }
                    }
                }
            }
        }

    def _extract_available_nodes(self, system_profile):
        """Extract available nodes from system profile."""
        available_nodes = []

        for cluster_name, cluster in system_profile.get("clusters", {}).items():
            for node_name, node_data in cluster.get("nodes", {}).items():
                available_nodes.append(
                    {"cluster": cluster_name, "name": node_name, "data": node_data}
                )

        return available_nodes

    def _extract_job_requirements(self, job: JobSchedulerInterface, config):
        """Extract job requirements from job specification."""
        # Get basic resources
        job_resources = job.scheduler_resources

        requirements = {
            "cores": job_resources.get("_cores", job_resources.get("cores", 1)),
            "memory_mb": job_resources.get(
                "mem_mb", job_resources.get("memory_mb", 1000)
            ),
            "runtime_minutes": job_resources.get("runtime", 30),
            "disk_mb": job_resources.get("disk_mb", 0),
            "features": [],
            "resources": {},
            "properties": {},
        }

        # Try to extract extended specification if this is a single job
        if hasattr(job, "rule") and hasattr(job.rule, "params"):
            if hasattr(job.rule.params, "job_specification"):
                job_spec = job.rule.params.job_specification

                # Extract features
                if "features" in job_spec:
                    requirements["features"] = job_spec["features"]

                # Extract additional resources
                if "resources" in job_spec:
                    requirements["resources"].update(job_spec["resources"])

                # Extract properties
                if "properties" in job_spec:
                    requirements["properties"].update(job_spec["properties"])

        return requirements

    def _calculate_runtime_on_node(self, job_req, node_data):
        """Calculate the runtime of a job on a specific node."""
        # Start with base runtime
        runtime = job_req.get("runtime_minutes", 30)

        # Add small I/O overhead if specified
        if "properties" in job_req and "properties" in node_data:
            node_props = node_data["properties"]

            io_time = 0
            if "input_size_mb" in job_req.get("resources", {}):
                input_size = job_req["resources"]["input_size_mb"]
                read_bandwidth = node_props.get("read_mbps", 1000)
                if read_bandwidth > 0:
                    io_time += min(1.0, input_size / read_bandwidth * 0.1)

            if "output_size_mb" in job_req.get("resources", {}):
                output_size = job_req["resources"]["output_size_mb"]
                write_bandwidth = node_props.get("write_mbps", 800)
                if write_bandwidth > 0:
                    io_time += min(1.0, output_size / write_bandwidth * 0.1)

            runtime += io_time

        return runtime

    def _get_default_config(self):
        """Get default scheduler configuration."""
        return {
            "scheduler": {
                "type": "milp",
                "paths": {
                    "system_profile": "system_profile.json",
                    "job_history": "~/.snakemake/job_history",
                },
                "estimation": {
                    "auto_estimate_file_sizes": True,
                    "history": {"enabled": True, "adaptation_weight": 0.7},
                },
                "optimization": {
                    "objective_weights": {"makespan": 0.8, "energy": 0.2},
                    "time_limit_seconds": 30,
                    "fallback": "greedy",
                },
            }
        }
