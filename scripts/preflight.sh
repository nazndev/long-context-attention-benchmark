#!/usr/bin/env bash
# Record the machine. Missing NVIDIA tools are reported; they are not treated as PCIe or Ethernet.
set -u
echo "=== date ==="
date -u
echo "=== uname ==="
uname -a
echo "=== python ==="
python3 -V || true
echo "=== nvidia-smi ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  echo "=== nvidia-smi topo -m ==="
  nvidia-smi topo -m || true
  echo "=== nvidia-smi topo -p2p p ==="
  nvidia-smi topo -p2p p || true
  echo "=== nvidia-smi -q ==="
  nvidia-smi -q || true
else
  echo "nvidia-smi: not installed. This machine cannot be labeled as an NVIDIA GPU host."
fi
echo "=== lscpu ==="
if command -v lscpu >/dev/null 2>&1; then
  lscpu
else
  echo "lscpu: not installed"
  sysctl -n machdep.cpu.brand_string 2>/dev/null || true
fi
echo "=== ip -br link ==="
if command -v ip >/dev/null 2>&1; then
  ip -br link || true
else
  echo "ip: not installed. Interface names and link status only; addresses are omitted."
  ifconfig -a 2>/dev/null | awk '/^[a-z]/ {print} /status:/ {print}' || true
fi
echo "=== ethtool ==="
if [[ -n "${ETH_IFACE:-}" ]]; then
  if command -v ethtool >/dev/null 2>&1; then
    ethtool "${ETH_IFACE}" || true
  else
    echo "ethtool: not installed"
  fi
else
  echo "ETH_IFACE is unset. Refusing to guess a NIC or call it 100 Gb Ethernet."
fi
echo "=== torch collect_env ==="
python3 -m torch.utils.collect_env 2>/dev/null || echo "torch.utils.collect_env: unavailable in the python3 on PATH"
echo "=== preflight finished ==="
