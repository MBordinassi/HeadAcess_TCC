"""Mouse control abstraction over PyAutoGUI."""

from __future__ import annotations

import logging
from typing import Tuple

import pyautogui


class MouseController:
    """Controls system mouse movement and click actions safely."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        pyautogui.FAILSAFE = False
        self._screen_width, self._screen_height = pyautogui.size()

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Current screen resolution."""
        return self._screen_width, self._screen_height

    def move_to(self, x: float, y: float) -> None:
        """Move cursor while clamping to the screen bounds."""
        cx = int(max(0, min(x, self._screen_width - 1)))
        cy = int(max(0, min(y, self._screen_height - 1)))
        pyautogui.moveTo(cx, cy)

    def left_click(self) -> None:
        """Execute left click."""
        pyautogui.click(button="left")
        self._logger.info("mouse_click button=left")

    def right_click(self) -> None:
        """Execute right click."""
        pyautogui.click(button="right")
        self._logger.info("mouse_click button=right")
