"""
Kanban board Streamlit component wrapper.
declare_component must be called from a proper module (not a Streamlit page)
to avoid the "module is None" RuntimeError.
"""
import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kanban")
_kanban_func = components.declare_component("kanban_board", path=_COMPONENT_DIR)


def kanban_board(cards_state, reset_counter, key=None):
    """Render the Kanban board and return the current card state (or None)."""
    return _kanban_func(
        cards_state=cards_state,
        reset_counter=reset_counter,
        key=key,
        default=None
    )
