"""
Temporary patches for upstream library bugs.

Each patch module targets a specific upstream issue and should be removed
when the upstream fix is released. See CLAUDE.md for the removal plan.
"""

from app.patches.session_crash import apply_session_crash_patches

__all__ = ['apply_session_crash_patches']
