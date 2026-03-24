"""Vast.ai cloud GPU instance management for SD rendering."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


INSTANCE_STATE_FILE = Path.home() / ".beatlab" / "vast_instance.json"


def _load_instance_state() -> dict | None:
    """Load saved instance state from disk."""
    if INSTANCE_STATE_FILE.exists():
        with open(INSTANCE_STATE_FILE) as f:
            return json.load(f)
    return None


def _save_instance_state(instance_id: str, gpu_name: str, price: float) -> None:
    """Save instance state to disk for reuse."""
    INSTANCE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INSTANCE_STATE_FILE, "w") as f:
        json.dump({
            "instance_id": instance_id,
            "gpu_name": gpu_name,
            "price": price,
            "created_at": time.time(),
        }, f)


def _clear_instance_state() -> None:
    """Remove saved instance state."""
    if INSTANCE_STATE_FILE.exists():
        INSTANCE_STATE_FILE.unlink()


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
        cmd = ["vastai", *args]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"vastai command failed: {result.stderr}")
        if "failed with error" in result.stdout:
            raise RuntimeError(f"vastai error: {result.stdout.strip()}")
        return result.stdout

    def find_instance(
        self,
        min_vram_gb: int = 16,
        max_price_hr: float = 10.0,
    ) -> dict:
        """Search for a suitable GPU instance.

        Returns instance offer dict with id, gpu_name, price, etc.
        """
        # Keep query simple — complex filters cause empty results on Vast.ai CLI
        # Sort by descending FLOPS to get fastest GPU within budget
        query = f"rentable=true dph_total<={max_price_hr} num_gpus=1"
        sort_order = "total_flops-"  # fastest first
        output = self._vastai_cmd(
            "search", "offers",
            query,
            "--order", sort_order,
            "--limit", "5",
            "--raw",
        )

        all_offers = json.loads(output) if output.strip() else []
        # Filter by VRAM in Python (CLI filter is unreliable with compound queries)
        min_ram = min_vram_gb * 1024
        offers = [o for o in all_offers if o.get("gpu_ram", 0) >= min_ram]
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
        )
        # Output may be JSON {"new_contract": id} or text "Started. new_contract: 12345"
        stripped = output.strip()
        try:
            result = json.loads(stripped)
            instance_id = result.get("new_contract")
        except json.JSONDecodeError:
            # Parse from text output
            import re
            match = re.search(r"new_contract:\s*(\d+)", stripped)
            if match:
                instance_id = match.group(1)
            else:
                # Try to find any number that looks like an instance ID
                match = re.search(r"(\d{6,})", stripped)
                instance_id = match.group(1) if match else None

        if not instance_id:
            raise RuntimeError(f"Failed to create instance: {output}")
        return str(instance_id)

    def wait_until_ready(self, instance_id: str, timeout: int = 3600) -> dict:
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

    def get_or_create_instance(self) -> tuple[str, bool]:
        """Reuse a saved running instance or create a new one.

        Returns (instance_id, was_reused).
        """
        state = _load_instance_state()
        if state:
            instance_id = state["instance_id"]
            try:
                output = self._vastai_cmd("show", "instance", instance_id, "--raw")
                info = json.loads(output) if output.strip() else {}
                if info.get("actual_status") == "running":
                    return instance_id, True
            except Exception:
                pass
            _clear_instance_state()

        # No reusable instance — create new
        offer = self.find_instance()
        instance_id = self.create_instance(offer["id"])
        _save_instance_state(
            instance_id,
            offer.get("gpu_name", "unknown"),
            offer.get("dph_total", 0),
        )
        return instance_id, False

    def destroy_instance(self, instance_id: str) -> None:
        """Destroy an instance and clear saved state."""
        self._vastai_cmd("destroy", "instance", instance_id)
        _clear_instance_state()

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
