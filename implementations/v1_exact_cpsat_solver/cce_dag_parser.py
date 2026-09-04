"""
Workload DAG Parser & Energy-Latency Model for CCE-QOS Compiler.
Parses multi-layer neural network compute graphs (ResNet, Transformer, LLM attention).
"""

import json
from typing import Dict, List, Any


class CCEDAGParser:
    def __init__(self):
        pass

    def load_workload(self, json_path: str) -> Dict[str, Any]:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data

    def extract_tasks_and_deps(self, workload_data: dict) -> tuple:
        tasks = []
        deps = []
        for node in workload_data.get("nodes", []):
            tasks.append({
                "id": node["id"],
                "name": node.get("name", node["id"]),
                "duration": node.get("duration", 1),
                "memory_kb": node.get("memory_kb", 64.0),
                "energy_pj": node.get("energy_pj", 250.0)
            })
            for parent in node.get("parents", []):
                deps.append((parent, node["id"]))
        return tasks, deps
