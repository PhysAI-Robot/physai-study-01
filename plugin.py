"""The research plugin: importing it registers every research policy.

A manifest loads it with `plugins: [research.plugin]` (see physai.plugins); core
never imports it by name.
"""

import research.classical_control.so101_visual_servo  # noqa: F401
import research.imitation_learning.lerobot_policy  # noqa: F401
import research.scripted_experts.so101_pick_place_expert  # noqa: F401
