# Prefix Shraing Workload Generator

```bash
python generate_realistic_prefix_share_workload.py
```

the workload config is defined inside `generate_realistic_prefix_share_workload.py`


Note that the order of all requests will be shuffled not just within each workload config but for all requests.

Exapmle config
```python
prefix_workload_configs = [
        {
            "prefix_length": 1024,
            "suffix_length": 128,
            "num_samples_per_prefix": 32,
            "num_prefix": 10,
            "rps": 5,
            "randomize_order": True  # Add the randomization parameter
        },
        {
            "prefix_length": 2048,
            "suffix_length": 128,
            "num_samples_per_prefix": 32,
            "num_prefix": 5,
            "rps": 5,
            "randomize_order": False  # Can be set differently per config
        },
        {
            "prefix_length": 4096,
            "suffix_length": 128,
            "num_samples_per_prefix": 32,
            "num_prefix": 10,
            "rps": 5,
            "randomize_order": True  # Add the randomization parameter
        },
    ]
```