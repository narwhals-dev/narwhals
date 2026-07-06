"""[Vega Event Stream] rabbit hole.

## Important
These docs & types are for me and will be short-lived.

aaaa
>>> {
...     "type": "mousedown",
...     "marktype": "image",
...     "filter": ("event.ctrlKey", "event.button === 0"),
... }

## Notes
- There aren't any docs for Altair, and not that much from Vega-lite on these guys.
- Turns out that expressions can be used here too
- Something about their schema definition has broken the altair side

[Vega Event Stream]: https://vega.github.io/vega/docs/event-streams
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from altair.vegalite.v6.schema._typing import MarkType_T as MarkType
    from typing_extensions import LiteralString, Required

    from narwhals._plan.altair.typing import VegaExpr
else:
    import sys

    if sys.version_info >= (3, 11):
        from typing import LiteralString
    else:
        LiteralString: TypeAlias = Any


EventType: TypeAlias = Literal[
    "click",
    "dblclick",
    "dragenter",
    "dragleave",
    "dragover",
    "keydown",
    "keypress",
    "keyup",
    "mousedown",
    "mousemove",
    "mouseout",
    "mouseover",
    "mouseup",
    "mousewheel",
    "pointerdown",
    "pointermove",
    "pointerout",
    "pointerover",
    "pointerup",
    "timer",
    "touchend",
    "touchmove",
    "touchstart",
    "wheel",
]
"""A [DOM event type], that is supported in a [Vega Event Stream].

[DOM event type]: https://vega.github.io/vega/docs/event-streams/#types
[Vega Event Stream]: https://vega.github.io/vega/docs/event-streams
"""

StreamSelector: TypeAlias = EventType | LiteralString
"""
A [DOM event type], that is supported in a [Vega Event Stream] or an [Event Stream Selector].

[DOM event type]: https://vega.github.io/vega/docs/event-streams/#types
[Vega Event Stream]: https://vega.github.io/vega/docs/event-streams
[Event Stream Selector]: https://vega.github.io/vega/docs/event-streams/#selector
"""


class _BaseEventStream(TypedDict, total=False):
    """Any event stream object may also include the following properties for filtering or modifying an event stream."""

    between: tuple[Stream, Stream]
    """A two-element array of event stream objects, indicating sentinel starting and ending events.

    Only events that occur between these two events will be captured."""

    consume: bool
    """A boolean flag (default `False`) indicating if this stream should consume the event by invoking [`event.preventDefault()`].

    [`event.preventDefault()`]: https://developer.mozilla.org/en-US/docs/Web/API/Event/preventDefault
    """

    debounce: float
    """The minimum time to wait between event occurrence and processing.

    If a new event arrives during a debouncing window, the debounce timer will restart and only the new event will be captured."""

    filter: VegaExpr | Sequence[VegaExpr]
    """One or more filter expressions, each of which must evaluate to a truthy value in order for the event to be captured.

    These expressions may **not** reference signal values, only event properties."""

    markname: str
    """The unique name of a mark set for which to monitor input events.

    Events originating from other marks will be ignored."""

    marktype: MarkType
    """The type or marks (`arc`, `rect`, _etc._) to monitor for input events.

    Events originating from other mark types will be ignored."""

    throttle: float
    """The minimum time in milliseconds between captured events (default `0`).

    New events that arrive within the throttling window will be ignored.
    For timer events, this property determines the interval between timer ticks."""


class EventStreamLhs(_BaseEventStream, TypedDict, total=False):
    source: Literal["view", "scope"]
    """The input event source.

    - For event streams defined in the top-level scope of a Vega specification, this property defaults to `"view"`, which monitors all input events in the current Vega view component (including those targeting the containing Canvas or SVG component itself).
    - For event streams defined within nested scopes, this property defaults to `"scope"`, which limits consideration to only events originating within the group in which the event stream is defined.
    """

    type: Required[EventType]
    """The event type to monitor (e.g., `"click"`, `"keydown"`, `"timer"`)."""


class EventStreamRhs(_BaseEventStream, TypedDict, total=False):
    source: Required[Literal["window"]]
    """The input event source.

    The browser window object.
    """
    type: Required[StreamSelector]


class DerivedStream(_BaseEventStream, TypedDict, total=False):
    """In addition to basic streams, an event stream object can serve as input for a derived event stream.

    ## Examples
    >>> _ = DerivedStream(
    ...     stream={"type": "click", "marktype": "rect"},
    ...     filter="event.shiftKey",
    ...     debounce=500,
    ... )
    """

    stream: Required[Stream]
    """An input event stream to modify with additional parameters."""


class MergedStream(_BaseEventStream, TypedDict, total=False):
    """A set of event streams can also be merged together.

    ## Examples
    >>> _ = MergedStream(
    ...     merge=(
    ...         {"type": "mousedown", "marktype": "symbol"},
    ...         {"type": "mousedown", "marktype": "symbol"},
    ...     )
    ... )
    """

    merge: Required[Sequence[Stream]]
    """An array of event streams to merge into a single stream."""


EventStream: TypeAlias = EventStreamLhs | EventStreamRhs
"""A single [Vega Event Stream object](https://vega.github.io/vega/docs/event-streams/#object).

## Examples
Capture click events on `rect` marks:
>>> _ = {"type": "click", "marktype": "rect"}

Capture resize events on the browser window:
>>> _ = {"type": "resize", "source": "window"}

Capture mousedown events on `image` marks if the control key is pressed and the left mouse button is used:
>>> _ = {"type": "mousedown","marktype": "image", "filter": ("event.ctrlKey", "event.button === 0")}

Capture mousemove events that occur between mousedown and mouseup events:
>>> _ = {"type": "mousemove", "between": ({"type": "mousedown"}, {"type": "mouseup"})}

Capture timer tick events every 2 seconds (2000 milliseconds):
>>> _ = {"type": "timer", "throttle": 2000}


## Important
[`"EventStream"` (JSON)]: https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/altair/vegalite/v6/schema/vega-lite-schema.json#L8658-L8764
[`EventStream` (Python)]: https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/altair/vegalite/v6/schema/core.py#L22180-L22186

[`"EventStream"` (JSON)] is defined via an anonymous union, and [`EventStream` (Python)] understands 0% of that.
"""


Stream: TypeAlias = EventStream | DerivedStream | MergedStream
"""Any [Vega Event Stream object](https://vega.github.io/vega/docs/event-streams/#object)."""
