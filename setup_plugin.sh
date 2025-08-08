#!/bin/bash

# Setup script for MILP Optimizer Scheduler Plugin
set -e

echo "Setting up MILP Optimizer Scheduler Plugin..."

# Create directory structure
echo "Creating directory structure..."
mkdir -p src/snakemake_scheduler_plugin_milp_ext
mkdir -p tests
mkdir -p examples

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found. Make sure you're in the plugin root directory."
    exit 1
fi

echo "Installing dependencies..."
pip install snakemake-interface-scheduler-plugins>=2.0.0 pulp>=2.0 networkx>=2.5 pyyaml>=5.0 pytest>=6.0 pytest-cov>=3.0

echo "Installing plugin in development mode..."
pip install -e .

echo "Running tests..."
pytest tests/ -v

echo "Testing plugin import..."
python -c "
import snakemake_scheduler_plugin_milp_ext
print('Plugin imports successfully')

from snakemake_scheduler_plugin_milp_ext import Scheduler, SchedulerSettings
print('Scheduler and SchedulerSettings import successfully')

from snakemake_interface_scheduler_plugins.base import SchedulerBase
assert issubclass(Scheduler, SchedulerBase)
print('Scheduler implements correct interface')

settings = SchedulerSettings(time_limit=60)
print(f'Settings work: {settings}')
"

echo "Plugin setup complete!"
echo ""
echo "Next steps:"
echo "1. Navigate to examples/: cd examples/"
echo "2. Run dry-run: snakemake --scheduler milp-ext --scheduler-milp-ext-system-profile system_profile.json --cores 4 --dry-run"
echo "3. Run actual: snakemake --scheduler milp-ext --scheduler-milp-ext-system-profile system_profile.json --cores 4"
echo ""
echo "Available CLI options:"
echo "  --scheduler-milp-ext-system-profile PATH"
echo "  --scheduler-milp-ext-scheduler-config PATH"
echo "  --scheduler-milp-ext-time-limit SECONDS"
echo "  --scheduler-milp-ext-fallback greedy|ilp"
