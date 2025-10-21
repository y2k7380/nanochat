#!/usr/bin/env python3
"""
Quick test script to verify single GPU support works correctly.
Run this to check if the modifications are working.

Usage:
    # Test default GPU (GPU 0)
    python test_gpu.py

    # Test specific GPU
    CUDA_VISIBLE_DEVICES=1 python test_gpu.py

    # Test with GPU 2
    CUDA_VISIBLE_DEVICES=2 python test_gpu.py
"""

import os
import sys
import torch

print("=" * 70)
print("GPU Configuration Test")
print("=" * 70)

# Check CUDA availability
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("   ❌ CUDA is not available. This code requires GPU.")
    sys.exit(1)

print(f"   ✅ CUDA is available")
print(f"   Total GPUs in system: {torch.cuda.device_count()}")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (using all GPUs)')}")

# Test compute_init
print(f"\n2. Testing compute_init()...")
from nanochat.common import compute_init, compute_cleanup

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()

print(f"   DDP mode: {ddp}")
print(f"   DDP rank: {ddp_rank}")
print(f"   DDP world size: {ddp_world_size}")
print(f"   Returned device: {device}")
print(f"   Current CUDA device: cuda:{torch.cuda.current_device()}")

# Verify device is set correctly
expected_device_id = 0  # In single GPU mode, should always be 0 (the first visible GPU)
actual_device_id = torch.cuda.current_device()

if actual_device_id == expected_device_id:
    print(f"   ✅ Device correctly set to cuda:{actual_device_id}")
else:
    print(f"   ❌ Device mismatch! Expected cuda:{expected_device_id}, got cuda:{actual_device_id}")

# Test tensor creation
print(f"\n3. Testing tensor creation...")
test_tensor = torch.randn(100, 100).to("cuda")
print(f"   Tensor device: {test_tensor.device}")
print(f"   Matches returned device: {test_tensor.device == device}")

if test_tensor.device == device:
    print(f"   ✅ Tensors are created on correct device")
else:
    print(f"   ❌ Tensor device mismatch!")

# Test dataloader function signature
print(f"\n4. Testing dataloader has device parameter...")
import inspect
from nanochat.dataloader import tokenizing_distributed_data_loader

sig = inspect.signature(tokenizing_distributed_data_loader)
params = list(sig.parameters.keys())
print(f"   Dataloader parameters: {params}")

if 'device' in params:
    print(f"   ✅ Dataloader has 'device' parameter")
else:
    print(f"   ❌ Dataloader missing 'device' parameter")

# Test memory allocation
print(f"\n5. Memory allocation test...")
test_data = torch.randn(1000, 1000).to(device)
allocated_mb = torch.cuda.memory_allocated() / 1024**2
print(f"   Memory allocated on {device}: {allocated_mb:.2f} MB")

if allocated_mb > 0:
    print(f"   ✅ Memory successfully allocated on device")
else:
    print(f"   ⚠️  No memory allocated")

# Test device persistence
print(f"\n6. Testing device persistence...")
x = torch.randn(32, 2048).to(device=device, dtype=torch.int32)
y = torch.randn(32, 2048).to(device=device, dtype=torch.int64)

print(f"   Tensor x device: {x.device}")
print(f"   Tensor y device: {y.device}")

if x.device == device and y.device == device:
    print(f"   ✅ All tensors on correct device")
else:
    print(f"   ❌ Tensor device mismatch")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

all_tests_passed = (
    torch.cuda.is_available() and
    actual_device_id == expected_device_id and
    test_tensor.device == device and
    'device' in params and
    allocated_mb > 0 and
    x.device == device and
    y.device == device
)

if all_tests_passed:
    print("✅ ALL TESTS PASSED - Single GPU support is working correctly!")
    print(f"\nYou are using: {torch.cuda.get_device_name(device)}")
    print(f"Device: {device}")
else:
    print("❌ SOME TESTS FAILED - Please review the output above")

print("=" * 70)

# Cleanup
compute_cleanup()
