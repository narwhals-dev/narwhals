"""Pygments lexer for text containing ANSI color codes."""
from __future__ import annotations

import itertools
import re
import typing

import pygments.lexer
import pygments.token


C = pygments.token.Token.C
Color = pygments.token.Token.Color


_ansi_code_to_color = {
    0: 'Black',
    1: 'Red',
    2: 'Green',
    3: 'Yellow',
    4: 'Blue',
    5: 'Magenta',
    6: 'Cyan',
    7: 'White',
    60: 'BrightBlack',
    61: 'BrightRed',
    62: 'BrightGreen',
    63: 'BrightYellow',
    64: 'BrightBlue',
    65: 'BrightMagenta',
    66: 'BrightCyan',
    67: 'BrightWhite',
}

_256_colors = {
    0: '#000000',
    1: '#800000',
    2: '#008000',
    3: '#808000',
    4: '#000080',
    5: '#800080',
    6: '#008080',
    7: '#c0c0c0',
    8: '#808080',
    9: '#ff0000',
    10: '#00ff00',
    11: '#ffff00',
    12: '#0000ff',
    13: '#ff00ff',
    14: '#00ffff',
    15: '#ffffff',
}
_vals = (0, 95, 135, 175, 215, 255)
_256_colors.update({
    16 + i: '#{:02x}{:02x}{:02x}'.format(*rgb)
    for i, rgb in enumerate(itertools.product(_vals, _vals, _vals))
})
_256_colors.update({
    232 + i: '#{0:02x}{0:02x}{0:02x}'.format(10 * i + 8)
    for i in range(24)
})


def _token_from_lexer_state(
    bold: bool,
    faint: bool,
    fg_color: str | None,
    bg_color: str | None,
) -> pygments.token._TokenType:
    """Construct a token given the current lexer state.

    We can only emit one token even though we have a multiple-tuple state.
    To do work around this, we construct tokens like "Bold.Red".
    """
    components: tuple[str, ...] = ()

    if bold:
        components += ('Bold',)

    if faint:
        components += ('Faint',)

    if fg_color:
        components += (fg_color,)

    if bg_color:
        components += ('BG' + bg_color,)

    if len(components) == 0:
        return pygments.token.Text
    else:
        token = Color
        for component in components:
            token = getattr(token, component)
        return token


DEFAULT_STYLE = {
    'Black': '#000000',
    'Red': '#ef2929',
    'Green': '#8ae234',
    'Yellow': '#fce94f',
    'Blue': '#3465a4',
    'Magenta': '#c509c5',
    'Cyan': '#34e2e2',
    'White': '#f5f5f5',
    'BrightBlack': '#676767',
    'BrightRed': '#ff6d67',
    'BrightGreen': '#5ff967',
    'BrightYellow': '#fefb67',
    'BrightBlue': '#6871ff',
    'BrightMagenta': '#ff76ff',
    'BrightCyan': '#5ffdff',
    'BrightWhite': '#feffff',
}


def color_tokens(
    fg_colors: dict[str, str] = DEFAULT_STYLE,
    bg_colors: dict[str, str] = DEFAULT_STYLE,
    enable_256color: bool = False,
) -> dict[pygments.token._TokenType, str]:
    """Return color tokens for a given set of colors.

    Pygments doesn't have a generic "color" token; instead everything is
    contextual (e.g. "comment" or "variable"). That doesn't make sense for us,
    where the colors actually *are* what we care about.

    This function will register combinations of tokens (things like "Red" or
    "Bold.Red.BGGreen") based on the colors passed in.

    You can also define the tokens yourself, but note that the token names are
    *not* currently guaranteed to be stable between releases as I'm not really
    happy with this approach.

    Optionally, you can enable 256-color support by passing
    `enable_256color=True`. This will (very slightly) increase the CSS size,
    but enable the use of 256-color in text. The reason this is optional and
    non-default is that it requires patching the Pygments formatter you're
    using, using the ExtendedColorHtmlFormatterMixin provided by this file.
    For more details on why and how, see the README.

    Usage:

        .. code-block:: python
            from pygments_ansi_color import color_tokens

            class MyStyle(pygments.styles.SomeStyle):
                styles = dict(pygments.styles.SomeStyle.styles)
                styles.update(color_tokens())
    """
    styles: dict[pygments.token._TokenType, str] = {}

    # Validates custom color IDs.
    if not set(fg_colors).issubset(DEFAULT_STYLE):  # pragma: no cover (trivial)
        raise ValueError(
            f'Unrecognized {set(fg_colors).difference(DEFAULT_STYLE)}'
            ' foreground color',
        )
    if not set(bg_colors).issubset(DEFAULT_STYLE):  # pragma: no cover (trivial)
        raise ValueError(
            f'Unrecognized {set(bg_colors).difference(DEFAULT_STYLE)}'
            ' background color',
        )

    # Merge the default colors with the user-provided colors.
    fg_colors = {**DEFAULT_STYLE, **fg_colors}
    bg_colors = {**DEFAULT_STYLE, **bg_colors}

    if enable_256color:
        styles[pygments.token.Token.C.Bold] = 'bold'
        styles[pygments.token.Token.C.Faint] = ''
        for i, color in _256_colors.items():
            styles[getattr(pygments.token.Token.C, f'C{i}')] = color
            styles[getattr(pygments.token.Token.C, f'BGC{i}')] = f'bg:{color}'

        for color, color_value in fg_colors.items():
            styles[getattr(C, color)] = color_value

        for color, color_value in bg_colors.items():
            styles[getattr(C, f'BG{color}')] = f'bg:{color_value}'
    else:
        for bold, faint, fg_color, bg_color in itertools.product(
                (False, True),
                (False, True),
                {None} | set(fg_colors),
                {None} | set(bg_colors),
        ):
            token = _token_from_lexer_state(bold, faint, fg_color, bg_color)
            if token is not pygments.token.Text:
                value: list[str] = []
                if bold:
                    value.append('bold')
                if fg_color:
                    value.append(fg_colors[fg_color])
                if bg_color:
                    value.append('bg:' + bg_colors[bg_color])
                styles[token] = ' '.join(value)

    return styles


class AnsiColorLexer(pygments.lexer.RegexLexer):
    name = 'ANSI Color'
    aliases = ('ansi-color', 'ansi', 'ansi-terminal')
    flags = re.DOTALL | re.MULTILINE

    bold: bool
    fant: bool
    fg_color: str | None
    bg_color: str | None

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        self.reset_state()

    def reset_state(self) -> None:
        self.bold = False
        self.faint = False
        self.fg_color = None
        self.bg_color = None

    @property
    def current_token(self) -> pygments.token._TokenType:
        return _token_from_lexer_state(
            self.bold, self.faint, self.fg_color, self.bg_color,
        )

    def process(
        self,
        match: re.Match[str],
    ) -> typing.Generator[
        tuple[int, pygments.token._TokenType, str],
        None,
        None,
    ]:
        """Produce the next token and bit of text.

        Interprets the ANSI code (which may be a color code or some other
        code), changing the lexer state and producing a new token. If it's not
        a color code, we just strip it out and move on.

        Some useful reference for ANSI codes:
          * http://ascii-table.com/ansi-escape-sequences.php
        """
        # "after_escape" contains everything after the start of the escape
        # sequence, up to the next escape sequence. We still need to separate
        # the content from the end of the escape sequence.
        after_escape = match.group(1)

        # TODO: this doesn't handle the case where the values are non-numeric.
        # This is rare but can happen for keyboard remapping, e.g.
        # '\x1b[0;59;"A"p'
        parsed = re.match(
            r'([0-9;=]*?)?([a-zA-Z])(.*)$',
            after_escape,
            re.DOTALL | re.MULTILINE,
        )
        if parsed is None:
            # This shouldn't ever happen if we're given valid text + ANSI, but
            # people can provide us with utter junk, and we should tolerate it.
            text = after_escape
        else:
            value, code, text = parsed.groups()
            if code == 'm':  # "m" is "Set Graphics Mode"
                # Special case \x1b[m is a reset code
                if value == '':
                    self.reset_state()
                else:
                    try:
                        values = [int(v) for v in value.split(';')]
                    except ValueError:
                        # Shouldn't ever happen, but could with invalid ANSI.
                        values = []

                    while len(values) > 0:
                        value = values.pop(0)
                        fg_color = _ansi_code_to_color.get(value - 30)
                        bg_color = _ansi_code_to_color.get(value - 40)
                        if fg_color:
                            self.fg_color = fg_color
                        elif bg_color:
                            self.bg_color = bg_color
                        elif value == 1:
                            self.bold = True
                        elif value == 2:
                            self.faint = True
                        elif value == 22:
                            self.bold = False
                            self.faint = False
                        elif value == 39:
                            self.fg_color = None
                        elif value == 49:
                            self.bg_color = None
                        elif value == 0:
                            self.reset_state()
                        elif value in (38, 48):
                            try:
                                five = values.pop(0)
                                color = values.pop(0)
                            except IndexError:
                                continue
                            else:
                                if five != 5:
                                    continue
                                if 0 <= color <= 255:
                                    if value == 38:
                                        self.fg_color = f'C{color}'
                                    else:
                                        self.bg_color = f'C{color}'

        yield match.start(), self.current_token, text

    def ignore_unknown_escape(self, match: re.Match[str]) -> typing.Generator[
        tuple[int, pygments.token._TokenType, str],
        None,
        None,
    ]:
        after = match.group(1)
        # mypy prints these out because it uses curses to determine colors
        # http://ascii-table.com/ansi-escape-sequences-vt-100.php
        if re.match(r'\([AB012]', after):
            yield match.start(), self.current_token, after[2:]
        else:
            yield match.start(), self.current_token, after

    tokens = {
        # states have to be native strings
        'root': [
            (r'\x1b\[([^\x1b]*)', process),
            (r'\x1b([^\x1b]*)', ignore_unknown_escape),
            (r'[^\x1b]+', pygments.token.Text),
        ],
    }


class ExtendedColorHtmlFormatterMixin:

    def _get_css_classes(self, token: pygments.token._TokenType) -> str:
        classes = super()._get_css_classes(token)  # type: ignore
        if token[0] == 'Color':
            classes += ' ' + ' '.join(
                self._get_css_class(getattr(C, part))  # type: ignore
                for part in token[1:]
            )
        return classes
