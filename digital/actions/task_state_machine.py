"""Explicit task-state contract for the experiment 3 sorting loop."""

from __future__ import annotations

from enum import Enum


class SortingState(str, Enum):
    INITIALIZE = "INITIALIZE"
    WAIT_SCENE = "WAIT_SCENE"
    ACQUIRE = "ACQUIRE"
    ALIGN = "ALIGN"
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    RETREAT = "RETREAT"
    PLACE = "PLACE"
    VERIFY = "VERIFY"
    RETURN_HOME = "RETURN_HOME"
    RECORD = "RECORD"
    DONE = "DONE"
    SAFE_STOP = "SAFE_STOP"


ALLOWED_TRANSITIONS = {
    SortingState.INITIALIZE: {SortingState.WAIT_SCENE, SortingState.SAFE_STOP},
    SortingState.WAIT_SCENE: {SortingState.ACQUIRE, SortingState.SAFE_STOP},
    SortingState.ACQUIRE: {
        SortingState.ALIGN, SortingState.DONE, SortingState.SAFE_STOP,
    },
    SortingState.ALIGN: {
        SortingState.APPROACH, SortingState.RETURN_HOME,
        SortingState.RECORD, SortingState.SAFE_STOP,
    },
    SortingState.APPROACH: {
        SortingState.GRASP, SortingState.RETURN_HOME, SortingState.SAFE_STOP,
    },
    SortingState.GRASP: {
        SortingState.RETREAT, SortingState.VERIFY, SortingState.SAFE_STOP,
    },
    SortingState.RETREAT: {SortingState.PLACE, SortingState.SAFE_STOP},
    SortingState.PLACE: {SortingState.VERIFY, SortingState.SAFE_STOP},
    SortingState.VERIFY: {SortingState.RETURN_HOME, SortingState.SAFE_STOP},
    SortingState.RETURN_HOME: {
        SortingState.ACQUIRE, SortingState.RECORD, SortingState.SAFE_STOP,
    },
    SortingState.RECORD: {
        SortingState.ACQUIRE, SortingState.DONE, SortingState.SAFE_STOP,
    },
    SortingState.DONE: set(),
    SortingState.SAFE_STOP: {SortingState.DONE},
}


class SortingStateMachine:
    def __init__(self, reporter=None):
        self.state = SortingState.INITIALIZE
        self.reporter = reporter or (lambda _message: None)
        self.reporter(f"STATE | current={self.state.value}")

    def transition(self, target: SortingState, detail=""):
        target = SortingState(target)
        if target == self.state:
            return
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(
                f"invalid sorting transition {self.state.value} -> {target.value}"
            )
        previous = self.state
        self.state = target
        suffix = f" | {detail}" if detail else ""
        self.reporter(
            f"STATE | previous={previous.value} | current={target.value}{suffix}"
        )

    def fail(self, detail):
        if self.state not in (SortingState.SAFE_STOP, SortingState.DONE):
            self.transition(SortingState.SAFE_STOP, str(detail))
