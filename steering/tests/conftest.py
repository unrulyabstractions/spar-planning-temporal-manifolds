"""Shared test setup: keep torch's CPU thread count within what the container may use (pbank/cpu.py). Without it the
tiny-model tests take many minutes on a rented box with more cores than its CPU quota."""

from pbank.cpu import limit_torch_threads

limit_torch_threads()
