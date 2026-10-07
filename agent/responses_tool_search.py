"""Client-executed tool search on the Responses API, with a plain-loading fallback.

A plugin marks its tool schemas:

* ``tool_search: True`` on the one tool that answers the model's searches. Its handler gets the
  names exactly as the model wrote them and returns ``{"tools": [<function definition>, ...]}``.
* ``defer_loading: True`` on tools declared without being loaded (typically parameters only).
* ``namespace: {"name", "description"}`` on deferred tools to group them. Inside a namespace a tool
  goes by its name without the ``<namespace>__`` prefix; the model calls it ``<namespace>.<name>``.

On the Responses API the declarations never change while tools load, so the cached prefix holds:
a search goes out as ``tool_search_call`` / ``tool_search_output`` items rebuilt from history.
Anywhere else (``plain_tools``) deferred tools stay off the wire until a search in history loaded
them; their definitions then append at the end of the tool array, in the order they first loaded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

SEARCH_MARK = "tool_search"
DEFER_MARK = "defer_loading"
NAMESPACE_MARK = "namespace"
MARKS = (SEARCH_MARK, DEFER_MARK, NAMESPACE_MARK)
WIRE_SEARCH_NAME = "tool_search"
CLIENT_EXECUTION = "client"
NAMESPACE_PREFIX_SEPARATOR = "__"
_WIRE_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _function_of(tool: Any) -> Optional[Dict[str, Any]]:
    function = tool.get("function") if isinstance(tool, dict) else None
    return function if isinstance(function, dict) and isinstance(function.get("name"), str) else None


def _without_marks(function: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in function.items() if key not in MARKS}


def _namespace_of(function: Dict[str, Any]) -> Optional[tuple]:
    mark = function.get(NAMESPACE_MARK)
    if not isinstance(mark, dict):
        return None
    name, description = mark.get("name"), mark.get("description")
    if not (isinstance(name, str) and _WIRE_NAME.fullmatch(name)):
        return None
    return name, description if isinstance(description, str) else ""


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict)
                       and isinstance(part.get("text"), str))
    return ""


def _call_key(value: Any) -> str:
    return value.split("|", 1)[0].strip() if isinstance(value, str) else ""


def _definitions_in(result: Any) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(_text_of(result))
    except (TypeError, ValueError):
        return []
    tools = parsed.get("tools") if isinstance(parsed, dict) else None
    return [tool for tool in tools if isinstance(tool, dict)] if isinstance(tools, list) else []


@dataclass(frozen=True)
class Deferred:
    name: str
    wire: str
    namespace: Optional[str]
    parameters: Dict[str, Any]

    @property
    def called_as(self) -> str:
        return f"{self.namespace}.{self.wire}" if self.namespace else self.wire


@dataclass(frozen=True)
class ToolWire:
    """What a tool array declares for tool search, derived from the plugin's marks."""

    search: Optional[str] = None
    deferred: Mapping[str, Deferred] = field(default_factory=dict)
    namespaces: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def of(cls, tools: Optional[Iterable[Any]]) -> "ToolWire":
        search, deferred, namespaces = None, {}, {}
        taken: set = set()
        for function in filter(None, map(_function_of, tools or [])):
            name = function["name"]
            if function.get(SEARCH_MARK) is True and search is None:
                search = name
            elif function.get(DEFER_MARK) is True and name not in deferred:
                namespace = _namespace_of(function)
                wire, namespace_name = name, None
                if namespace is not None:
                    namespace_name = namespace[0]
                    namespaces.setdefault(namespace_name, namespace[1])
                    short = name.removeprefix(f"{namespace_name}{NAMESPACE_PREFIX_SEPARATOR}")
                    wire = short if short and (namespace_name, short) not in taken else name
                    taken.add((namespace_name, wire))
                parameters = function.get("parameters")
                deferred[name] = Deferred(name, wire, namespace_name,
                                          parameters if isinstance(parameters, dict) else {"type": "object"})
        return cls(search, deferred, namespaces)

    @property
    def native(self) -> bool:
        return self.search is not None

    @property
    def marked(self) -> bool:
        return self.native or bool(self.deferred)

    def aliases(self) -> Dict[str, str]:
        aliases = {tool.called_as: tool.name for tool in self.deferred.values() if tool.called_as != tool.name}
        if self.search is not None:
            aliases[WIRE_SEARCH_NAME] = self.search
        return aliases

    def wire_name(self, name: str) -> str:
        tool = self.deferred.get(name)
        return tool.called_as if tool else name

    def is_search(self, name: Any) -> bool:
        return self.search is not None and name == self.search

    def declarations(self, tools: Iterable[Any], declare_plain: Any) -> List[Dict[str, Any]]:
        """The Responses ``tools`` array: one client search, deferred tools in their namespaces."""
        declared: List[Dict[str, Any]] = []
        groups: Dict[str, Dict[str, Any]] = {}
        for tool in tools:
            function = _function_of(tool)
            if function is None:
                continue
            name = function["name"]
            if self.is_search(name):
                declared.append({"type": "tool_search", "execution": CLIENT_EXECUTION,
                                 "description": function.get("description") or "",
                                 "parameters": function.get("parameters") or {"type": "object", "properties": {}}})
            elif name in self.deferred:
                self._place(self.deferred[name], function.get("description"), declared, groups)
            else:
                declared.extend(declare_plain([{"type": "function", "function": _without_marks(function)}]) or [])
        return declared

    def found(self, result: Any) -> List[Dict[str, Any]]:
        """The ``tool_search_output`` tools for a search result: only tools declared deferred."""
        loaded: List[Dict[str, Any]] = []
        groups: Dict[str, Dict[str, Any]] = {}
        for definition in self._loadable(_definitions_in(result)):
            tool = self.deferred[definition["name"]]
            self._place(tool, definition.get("description"), loaded, groups, definition.get("parameters"))
        return loaded

    def loaded_in(self, messages: Iterable[Any]) -> List[Dict[str, Any]]:
        """Definitions the search tool returned in ``messages``, first load of each name only."""
        search_calls = set()
        loaded: Dict[str, Dict[str, Any]] = {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            for call in (message.get("tool_calls") or []) if message.get("role") == "assistant" else []:
                if isinstance(call, dict) and self.is_search((call.get("function") or {}).get("name")):
                    search_calls.update(_call_key(call.get(key)) for key in ("id", "call_id"))
            if message.get("role") == "tool" and _call_key(message.get("tool_call_id")) in search_calls - {""}:
                for definition in self._loadable(_definitions_in(message.get("content"))):
                    loaded.setdefault(definition["name"], definition)
        return list(loaded.values())

    def _loadable(self, definitions: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        return (definition for definition in definitions if definition.get("name") in self.deferred)

    def _place(self, tool: Deferred, description: Any, into: List[Dict[str, Any]],
               groups: Dict[str, Dict[str, Any]], parameters: Any = None) -> None:
        function: Dict[str, Any] = {"type": "function", "name": tool.wire}
        if isinstance(description, str) and description:
            function["description"] = description
        function["parameters"] = parameters if isinstance(parameters, dict) else tool.parameters
        function[DEFER_MARK] = True
        if tool.namespace is None:
            into.append(function)
            return
        group = groups.get(tool.namespace)
        if group is None:
            group = groups[tool.namespace] = {"type": "namespace", "name": tool.namespace,
                                              "description": self.namespaces.get(tool.namespace, ""), "tools": []}
            into.append(group)
        group["tools"].append(function)

    def loadable_listing(self) -> str:
        lines = []
        for namespace, description in self.namespaces.items():
            names = ", ".join(tool.name for tool in self.deferred.values() if tool.namespace == namespace)
            lines.append(f"- {namespace} ({description}): {names}" if description else f"- {namespace}: {names}")
        loose = ", ".join(tool.name for tool in self.deferred.values() if tool.namespace is None)
        if loose:
            lines.append(f"- {loose}")
        return "Tools you can load:\n" + "\n".join(lines) if lines else ""


def plain_tools(tools: Optional[List[Dict[str, Any]]], messages: Iterable[Any]) -> Optional[List[Dict[str, Any]]]:
    """The tool array for an interface without tool search: deferred tools only once loaded."""
    wire = ToolWire.of(tools)
    if not wire.marked:
        return tools
    declared: List[Dict[str, Any]] = []
    for tool in tools or []:
        function = _function_of(tool)
        if function is None or function["name"] in wire.deferred:
            continue
        plain = _without_marks(function)
        if wire.is_search(function["name"]):
            plain["description"] = "\n\n".join(filter(None, [function.get("description"), wire.loadable_listing()]))
        declared.append({**tool, "function": plain})
    for definition in wire.loaded_in(messages):
        tool = wire.deferred[definition["name"]]
        parameters = definition.get("parameters")
        description = definition.get("description")
        declared.append({"type": "function", "function": {
            "name": tool.name, "description": description if isinstance(description, str) else "",
            "parameters": parameters if isinstance(parameters, dict) else tool.parameters}})
    return declared
