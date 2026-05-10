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
        pyautogui.PAUSE = 0
        self._screen_width, self._screen_height = pyautogui.size()
        self._left_is_down = False
        self._right_is_down = False

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

    def left_down(self) -> None:
        """Press and hold left mouse button."""
        if self._left_is_down:
            return
        pyautogui.mouseDown(button="left")
        self._left_is_down = True
        self._logger.info("mouse_down button=left")

    def left_up(self) -> None:
        """Release left mouse button."""
        if not self._left_is_down:
            return
        pyautogui.mouseUp(button="left")
        self._left_is_down = False
        self._logger.info("mouse_up button=left")

    def right_down(self) -> None:
        """Press and hold right mouse button."""
        if self._right_is_down:
            return
        pyautogui.mouseDown(button="right")
        self._right_is_down = True
        self._logger.info("mouse_down button=right")

    def right_up(self) -> None:
        """Release right mouse button."""
        if not self._right_is_down:
            return
        pyautogui.mouseUp(button="right")
        self._right_is_down = False
        self._logger.info("mouse_up button=right")

    def release_all(self) -> None:
        """Release any pressed mouse buttons to avoid stuck drag state."""
        self.left_up()
        self.right_up()
