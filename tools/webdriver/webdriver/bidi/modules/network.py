from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Union

from ._module import BidiModule, command


class AuthCredentials(Dict[str, Any]):
    def __init__(self, username: str, password: str):
        dict.__init__(self, type="password", username=username, password=password)


class NetworkBase64Value(Dict[str, Any]):
    def __init__(self, value: str):
        dict.__init__(self, type="base64", value=value)


class NetworkStringValue(Dict[str, Any]):
    def __init__(self, value: str):
        dict.__init__(self, type="string", value=value)


NetworkBytesValue = Union[NetworkStringValue, NetworkBase64Value]


class CookieHeader(Dict[str, Any]):
    def __init__(self, name: str, value: NetworkBytesValue):
        dict.__init__(self, name=name, value=value)


class Header(Dict[str, Any]):
    def __init__(self, name: str, value: NetworkBytesValue):
        dict.__init__(self, name=name, value=value)


class URLPatternPattern(Dict[str, Any]):
    def __init__(
        self,
        protocol: Optional[str] = None,
        hostname: Optional[str] = None,
        port: Optional[str] = None,
        pathname: Optional[str] = None,
        search: Optional[str] = None,
    ):
        dict.__init__(self, type="pattern")

        if protocol is not None:
            self["protocol"] = protocol

        if hostname is not None:
            self["hostname"] = hostname

        if port is not None:
            self["port"] = port

        if pathname is not None:
            self["pathname"] = pathname

        if search is not None:
            self["search"] = search


class URLPatternString(Dict[str, Any]):
    def __init__(self, pattern: str):
        dict.__init__(self, type="string", pattern=pattern)


URLPattern = Union[URLPatternPattern, URLPatternString]


class Network(BidiModule):
    @command
    def add_intercept(
        self, phases: List[str], url_patterns: Optional[List[URLPattern]] = None
    ) -> Mapping[str, Any]:
        params: MutableMapping[str, Any] = {
            "phases": phases,
        }

        if url_patterns is not None:
            params["urlPatterns"] = url_patterns

        return params

    @add_intercept.result
    def _add_intercept(self, result: Mapping[str, Any]) -> Any:
        assert result["intercept"] is not None
        return result["intercept"]

    @command
    def continue_with_auth(
        self,
        request: str,
        action: str,
        credentials: Optional[AuthCredentials] = None
    ) -> Mapping[str, Any]:
        params: MutableMapping[str, Any] = {
            "request": request,
            "action": action,
        }

        if action == "provideCredentials" and credentials is not None:
            params["credentials"] = credentials

        return params

    @command
    def continue_request(self,
                         request: str,
                         body: Optional[NetworkBytesValue] = None,
                         cookies: Optional[List[CookieHeader]] = None,
                         headers: Optional[List[Header]] = None,
                         method: Optional[str] = None,
                         url: Optional[str] = None) -> Mapping[str, Any]:
        params: MutableMapping[str, Any] = {
            "request": request,
        }

        if body is not None:
            params["body"] = body

        if cookies is not None:
            params["cookies"] = cookies

        if headers is not None:
            params["headers"] = headers

        if method is not None:
            params["method"] = method

        if url is not None:
            params["url"] = url

        return params

    @command
    def continue_response(
            self,
            request: str,
            reason_phrase: Optional[str] = None,
            status_code: Optional[int] = None) -> Mapping[str, Any]:
        params: MutableMapping[str, Any] = {
            "request": request,
        }

        if reason_phrase is not None:
            params["reasonPhrase"] = reason_phrase

        if status_code is not None:
            params["statusCode"] = status_code

        # TODO: Add support for missing parameters: body, credentials, headers

        return params

    @command
    def fail_request(self, request: str) -> Mapping[str, Any]:
        params: MutableMapping[str, Any] = {"request": request}
        return params

    @command
    def provide_response(
            self,
            request: str,
            reason_phrase: Optional[str] = None,
            status_code: Optional[int] = None) -> Mapping[str, Any]:
        params: MutableMapping[str, Any] = {
            "request": request,
        }

        if reason_phrase is not None:
            params["reasonPhrase"] = reason_phrase

        if status_code is not None:
            params["statusCode"] = status_code

        # TODO: Add support for missing parameters: body, cookies, headers

        return params

    @command
    def remove_intercept(self, intercept: str) -> Mapping[str, Any]:
        params: MutableMapping[str, Any] = {"intercept": intercept}
        return params
