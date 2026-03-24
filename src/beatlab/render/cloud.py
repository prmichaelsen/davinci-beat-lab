"""Vast.ai cloud GPU instance management for SD rendering."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


class VastAIManager:
    """Manages Vast.ai GPU instances for rendering."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("VASTAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "VASTAI_API_KEY environment variable is required for cloud rendering.\n"
                "Get an API key at https://vast.ai/console/account/"
            )

    def _vastai_cmd(self, *args: str) -> str:
        """Run a vastai CLI command and return stdout."""
        cmd = ["vastai", "--api-key", self.api_key, *args]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"vastai command failed: {result.stderr}")
        return result.stdout

    def find_instance(
        self,
        min_vram_gb: int = 16,
        max_price_hr: float = 2.0,
        min_upload_mbps: float = 100,
    ) -> dict:
        """Search for a suitable GPU instance.

        Returns instance offer dict with id, gpu_name, price, etc.
        """
        # Search for instances with enough VRAM and reasonable price
        output = self._vastai_cmd(
            "search", "offers",
            f"gpu_ram >= {min_vram_gb * 1024}",
            f"dph_total <= {max_price_hr}",
            f"inet_up >= {min_upload_mbps}",
            "reliability >= 0.95",
            "num_gpus = 1",
            "--type", "on-demand",
            "--order", "dph_total",
            "--limit", "5",
            "--raw",
        )

        offers = json.loads(output) if output.strip() else []
        if not offers:
            raise RuntimeError(
                f"No Vast.ai instances found with {min_vram_gb}GB+ VRAM under ${max_price_hr}/hr. "
                "Try increasing max_price_hr or reducing min_vram_gb."
            )
        return offers[0]

    def create_instance(
        self,
        offer_id: int,
        image: str = "ai-dock/comfyui:latest",
        disk_gb: int = 50,
    ) -> str:
        """Create an instance from an offer. Returns instance ID."""
        output = self._vastai_cmd(
            "create", "instance", str(offer_id),
            "--image", image,
            "--disk", str(disk_gb),
            "--raw",
        )
        result = json.loads(output) if output.strip() else {}
        instance_id = result.get("new_contract")
        if not instance_id:
            raise RuntimeError(f"Failed to create instance: {output}")
        return str(instance_id)

    def wait_until_ready(self, instance_id: str, timeout: int = 600) -> dict:
        """Wait until instance is running. Returns instance info."""
        start = time.time()
        while time.time() - start < timeout:
            output = self._vastai_cmd("show", "instance", instance_id, "--raw")
            info = json.loads(output) if output.strip() else {}
            status = info.get("actual_status", "")
            if status == "running":
                return info
            if status in ("exited", "error"):
                raise RuntimeError(f"Instance {instance_id} failed with status: {status}")
            time.sleep(10)
        raise TimeoutError(f"Instance {instance_id} not ready after {timeout}s")

    def get_ssh_info(self, instance_id: str) -> tuple[str, int]:
        """Get SSH host and port for an instance."""
        output = self._vastai_cmd("show", "instance", instance_id, "--raw")
        info = json.loads(output) if output.strip() else {}
        ssh_host = info.get("ssh_host", "")
        ssh_port = info.get("ssh_port", 22)
        if not ssh_host:
            raise RuntimeError(f"No SSH info for instance {instance_id}")
        return ssh_host, ssh_port

    def get_comfyui_url(self, instance_id: str) -> str:
        """Get the ComfyUI URL for a running instance."""
        output = self._vastai_cmd("show", "instance", instance_id, "--raw")
        info = json.loads(output) if output.strip() else {}
        # ComfyUI typically runs on port 8188, mapped to a public port
        ports = info.get("ports", {})
        if "8188/tcp" in ports:
            port_info = ports["8188/tcp"][0]
            host = port_info.get("HostIp", info.get("ssh_host", ""))
            port = port_info.get("HostPort", "8188")
            return f"http://{host}:{port}"
        # Fallback: construct from SSH host
        ssh_host = info.get("ssh_host", "localhost")
        return f"http://{ssh_host}:8188"

    def destroy_instance(self, instance_id: str) -> None:
        """Destroy an instance."""
        self._vastai_cmd("destroy", "instance", instance_id)

    def ssh_run(self, instance_id: str, command: str) -> str:
        """Run a command on the instance via SSH."""
        host, port = self.get_ssh_info(instance_id)
        result = subprocess.run(
            [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-p", str(port), f"root@{host}",
                command,
            ],
            capture_output=True, text=True, timeout=300,
        )
        return result.stdout

    def upload_files(self, instance_id: str, local_dir: str, remote_dir: str) -> None:
        """Upload files to instance via rsync."""
        host, port = self.get_ssh_info(instance_id)
        subprocess.run(
            [
                "rsync", "-avz", "--progress",
                "-e", f"ssh -o StrictHostKeyChecking=no -p {port}",
                f"{local_dir}/",
                f"root@{host}:{remote_dir}/",
            ],
            check=True,
        )

    def download_files(self, instance_id: str, remote_dir: str, local_dir: str) -> None:
        """Download files from instance via rsync."""
        host, port = self.get_ssh_info(instance_id)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync", "-avz", "--progress",
                "-e", f"ssh -o StrictHostKeyChecking=no -p {port}",
                f"root@{host}:{remote_dir}/",
                f"{local_dir}/",
            ],
            check=True,
        )


def estimate_cost(
    frame_count: int,
    fps_render: float = 7.5,
    price_per_hr: float = 1.0,
) -> dict:
    """Estimate render time and cost.

    Args:
        frame_count: Total frames to render.
        fps_render: Estimated frames per second on GPU (A100 SDXL ~5-10 fps).
        price_per_hr: GPU cost per hour.

    Returns:
        Dict with estimated_seconds, estimated_hours, estimated_cost, frames.
    """
    seconds = frame_count / fps_render
    hours = seconds / 3600
    cost = hours * price_per_hr
    return {
        "frames": frame_count,
        "estimated_seconds": round(seconds),
        "estimated_hours": round(hours, 2),
        "estimated_cost_usd": round(cost, 2),
        "fps_render": fps_render,
        "price_per_hr": price_per_hr,
    }
