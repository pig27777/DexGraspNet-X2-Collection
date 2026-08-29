"""Local executable helpers importable by the Isaac runtime workers.

The workers import sibling helpers as ``scripts.<module>``.  Make that
relationship explicit rather than relying on Python namespace-package
resolution, which is not stable across the installed IsaacLab environment.
"""
