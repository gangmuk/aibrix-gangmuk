import argparse
import os
import sys
import utils

if __name__ == "__main__":
    gatway_log_file_name = 'gateway-plugins.log.csv'
    success = utils.collect_k8s_logs(
        namespace='aibrix-system',
        deployment_name='aibrix-gateway-plugins',
        output_file=gatway_log_file_name,
        keyword='**@latency_metrics'
    )