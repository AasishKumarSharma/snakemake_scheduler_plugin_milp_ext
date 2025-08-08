# Snakemake MILP Optimizer Scheduler Plugin

A **pip-installable** scheduler plugin for [Snakemake](https://snakemake.readthedocs.io/) that uses Mixed-Integer Linear Programming (MILP) to optimally schedule jobs across heterogeneous compute resources.

**Plugin Name**: `milp-ext` (Note: `milp` is reserved for Snakemake's internal scheduler)

## Features

* **Resource-Aware Scheduling**: Considers CPU cores, memory, GPU, custom resources, and I/O constraints
* **Multi-Objective Optimization**: Balances makespan vs. energy consumption with configurable weights
* **Feature Compatibility**: Ensures jobs run only on nodes with required hardware/software features
* **Historical Estimation**: Learns from past executions to refine runtime predictions
* **Graceful Fallbacks**: Falls back to greedy scheduling if MILP fails or times out
* **Plugin Auto-Discovery**: Integrates seamlessly via Snakemake's new plugin interface

## Installation

### From PyPI (when published)

```bash
pip install snakemake-scheduler-plugin-milp-ext
```

### From Source

```bash
git clone https://github.com/AasishKumarSharma/snakemake-scheduler-plugin-milp-ext.git
cd snakemake-scheduler-plugin-milp-ext
pip install .
```

For development:

```bash
pip install -e .[dev]
```

## Requirements

- Python ≥ 3.11
- snakemake-interface-scheduler-plugins ≥ 2.0.0
- pulp ≥ 2.0 (for MILP optimization)
- networkx ≥ 2.5 (for dependency analysis)
- pyyaml ≥ 5.0 (for configuration files)

## Quick Start

1. **Create a System Profile**: `system_profile.json` describes your compute clusters/nodes.

```json
{
  "clusters": {
    "local": {
      "nodes": {
        "cpu_node": {
          "resources": {
            "cores": 8,
            "memory_mb": 16384,
            "gpu_count": 0
          },
          "features": ["cpu", "x86_64", "avx2"],
          "properties": {
            "cpu_flops": 1e11,
            "memory_bandwidth_mbps": 25600
          }
        },
        "gpu_node": {
          "resources": {
            "cores": 16,
            "memory_mb": 32768,
            "gpu_count": 1,
            "gpu_memory_mb": 11264
          },
          "features": ["cpu", "gpu", "cuda", "x86_64"],
          "properties": {
            "cpu_flops": 2e11,
            "gpu_flops": 14e12,
            "memory_bandwidth_mbps": 51200
          }
        }
      }
    }
  }
}
```

2. **Run Snakemake with MILP Optimizer Scheduler**:

```bash
snakemake --scheduler milp-ext \
          --scheduler-milp-ext-system-profile system_profile.json \
          --cores 8
```

Or with plugin settings:

```bash
snakemake --scheduler milp-ext \
          --scheduler-milp-ext-time-limit 60 \
          --scheduler-milp-ext-fallback greedy \
          --cores 8
```

## Job Specification

Enhanced job specifications in your Snakefile:

```python
rule gpu_job:
    output: "results/gpu_output.txt"
    threads: 4
    resources:
        mem_mb=8192,
        runtime=30
    params:
        job_specification={
            "features": ["gpu", "cuda"],
            "resources": {
                "gpu_memory_mb": 4000,
                "input_size_mb": 1000,
                "output_size_mb": 500
            },
            "properties": {
                "gpu_flops": 1e12
            }
        }
    shell:
        "python gpu_processing.py {input} {output}"
```

## Configuration Options

### Plugin Settings (CLI)

- `--scheduler-milp-ext-system-profile`: Path to system profile JSON
- `--scheduler-milp-ext-scheduler-config`: Path to scheduler config YAML
- `--scheduler-milp-ext-time-limit`: MILP solver time limit in seconds (default: 30)
- `--scheduler-milp-ext-fallback`: Fallback scheduler (default: "greedy")

## Testing

Run the test suite:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License.
