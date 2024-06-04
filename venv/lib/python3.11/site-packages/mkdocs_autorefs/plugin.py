"""This module contains the "mkdocs-autorefs" plugin.

After each page is processed by the Markdown converter, this plugin stores absolute URLs of every HTML anchors
it finds to later be able to fix unresolved references.
It stores them during the [`on_page_content` event hook](https://www.mkdocs.org/user-guide/plugins/#on_page_content).

Just before writing the final HTML to the disc, during the
[`on_post_page` event hook](https://www.mkdocs.org/user-guide/plugins/#on_post_page),
this plugin searches for references of the form `[identifier][]` or `[title][identifier]` that were not resolved,
and fixes them using the previously stored identifier-URL mapping.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence
from urllib.parse import urlsplit

from mkdocs.plugins import BasePlugin
from mkdocs.structure.pages import Page

from mkdocs_autorefs.references import AutorefsExtension, fix_refs, relative_url

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.pages import Page
    from mkdocs.structure.toc import AnchorLink

try:
    from mkdocs.plugins import get_plugin_logger

    log = get_plugin_logger(__name__)
except ImportError:
    # TODO: remove once support for MkDocs <1.5 is dropped
    log = logging.getLogger(f"mkdocs.plugins.{__name__}")  # type: ignore[assignment]


class AutorefsPlugin(BasePlugin):
    """An `mkdocs` plugin.

    This plugin defines the following event hooks:

    - `on_config`
    - `on_page_content`
    - `on_post_page`

    Check the [Developing Plugins](https://www.mkdocs.org/user-guide/plugins/#developing-plugins) page of `mkdocs`
    for more information about its plugin system.
    """

    scan_toc: bool = True
    current_page: str | None = None

    def __init__(self) -> None:
        """Initialize the object."""
        super().__init__()
        self._url_map: dict[str, str] = {}
        self._abs_url_map: dict[str, str] = {}
        self.get_fallback_anchor: Callable[[str], tuple[str, ...]] | None = None

    def register_anchor(self, page: str, identifier: str, anchor: str | None = None) -> None:
        """Register that an anchor corresponding to an identifier was encountered when rendering the page.

        Arguments:
            page: The relative URL of the current page. Examples: `'foo/bar/'`, `'foo/index.html'`
            identifier: The HTML anchor (without '#') as a string.
        """
        self._url_map[identifier] = f"{page}#{anchor or identifier}"

    def register_url(self, identifier: str, url: str) -> None:
        """Register that the identifier should be turned into a link to this URL.

        Arguments:
            identifier: The new identifier.
            url: The absolute URL (including anchor, if needed) where this item can be found.
        """
        self._abs_url_map[identifier] = url

    def _get_item_url(
        self,
        identifier: str,
        fallback: Callable[[str], Sequence[str]] | None = None,
    ) -> str:
        try:
            return self._url_map[identifier]
        except KeyError:
            if identifier in self._abs_url_map:
                return self._abs_url_map[identifier]
            if fallback:
                new_identifiers = fallback(identifier)
                for new_identifier in new_identifiers:
                    with contextlib.suppress(KeyError):
                        url = self._get_item_url(new_identifier)
                        self._url_map[identifier] = url
                        return url
            raise

    def get_item_url(
        self,
        identifier: str,
        from_url: str | None = None,
        fallback: Callable[[str], Sequence[str]] | None = None,
    ) -> str:
        """Return a site-relative URL with anchor to the identifier, if it's present anywhere.

        Arguments:
            identifier: The anchor (without '#').
            from_url: The URL of the base page, from which we link towards the targeted pages.
            fallback: An optional function to suggest alternative anchors to try on failure.

        Returns:
            A site-relative URL.
        """
        url = self._get_item_url(identifier, fallback)
        if from_url is not None:
            parsed = urlsplit(url)
            if not parsed.scheme and not parsed.netloc:
                return relative_url(from_url, url)
        return url

    def on_config(self, config: MkDocsConfig) -> MkDocsConfig | None:
        """Instantiate our Markdown extension.

        Hook for the [`on_config` event](https://www.mkdocs.org/user-guide/plugins/#on_config).
        In this hook, we instantiate our [`AutorefsExtension`][mkdocs_autorefs.references.AutorefsExtension]
        and add it to the list of Markdown extensions used by `mkdocs`.

        Arguments:
            config: The MkDocs config object.

        Returns:
            The modified config.
        """
        log.debug("Adding AutorefsExtension to the list")
        config["markdown_extensions"].append(AutorefsExtension(self))
        return config

    def on_page_markdown(self, markdown: str, page: Page, **kwargs: Any) -> str:  # noqa: ARG002
        """Remember which page is the current one.

        Arguments:
            markdown: Input Markdown.
            page: The related MkDocs page instance.
            kwargs: Additional arguments passed by MkDocs.

        Returns:
            The same Markdown. We only use this hook to keep a reference to the current page URL,
                used during Markdown conversion by the anchor scanner tree processor.
        """
        self.current_page = page.url
        return markdown

    def on_page_content(self, html: str, page: Page, **kwargs: Any) -> str:  # noqa: ARG002
        """Map anchors to URLs.

        Hook for the [`on_page_content` event](https://www.mkdocs.org/user-guide/plugins/#on_page_content).
        In this hook, we map the IDs of every anchor found in the table of contents to the anchors absolute URLs.
        This mapping will be used later to fix unresolved reference of the form `[title][identifier]` or
        `[identifier][]`.

        Arguments:
            html: HTML converted from Markdown.
            page: The related MkDocs page instance.
            kwargs: Additional arguments passed by MkDocs.

        Returns:
            The same HTML. We only use this hook to map anchors to URLs.
        """
        if self.scan_toc:
            log.debug(f"Mapping identifiers to URLs for page {page.file.src_path}")
            for item in page.toc.items:
                self.map_urls(page.url, item)
        return html

    def map_urls(self, base_url: str, anchor: AnchorLink) -> None:
        """Recurse on every anchor to map its ID to its absolute URL.

        This method populates `self.url_map` by side-effect.

        Arguments:
            base_url: The base URL to use as a prefix for each anchor's relative URL.
            anchor: The anchor to process and to recurse on.
        """
        self.register_anchor(base_url, anchor.id)
        for child in anchor.children:
            self.map_urls(base_url, child)

    def on_post_page(self, output: str, page: Page, **kwargs: Any) -> str:  # noqa: ARG002
        """Fix cross-references.

        Hook for the [`on_post_page` event](https://www.mkdocs.org/user-guide/plugins/#on_post_page).
        In this hook, we try to fix unresolved references of the form `[title][identifier]` or `[identifier][]`.
        Doing that allows the user of `autorefs` to cross-reference objects in their documentation strings.
        It uses the native Markdown syntax so it's easy to remember and use.

        We log a warning for each reference that we couldn't map to an URL, but try to be smart and ignore identifiers
        that do not look legitimate (sometimes documentation can contain strings matching
        our [`AUTO_REF_RE`][mkdocs_autorefs.references.AUTO_REF_RE] regular expression that did not intend to reference anything).
        We currently ignore references when their identifier contains a space or a slash.

        Arguments:
            output: HTML converted from Markdown.
            page: The related MkDocs page instance.
            kwargs: Additional arguments passed by MkDocs.

        Returns:
            Modified HTML.
        """
        log.debug(f"Fixing references in page {page.file.src_path}")

        url_mapper = functools.partial(self.get_item_url, from_url=page.url, fallback=self.get_fallback_anchor)
        fixed_output, unmapped = fix_refs(output, url_mapper)

        if unmapped and log.isEnabledFor(logging.WARNING):
            for ref in unmapped:
                log.warning(f"{page.file.src_path}: Could not find cross-reference target '[{ref}]'")

        return fixed_output
