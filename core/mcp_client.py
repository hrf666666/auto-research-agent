"""
AutoResearcher MCP Client Mixin

Provides MCP (Model Context Protocol) transport layer for ToolRegistry:
- SSE dual-connection transport (GLM platform)
- stdio transport (local npx subprocesses)
- MCP service detection and session management
- Vision tools via zai-mcp-server
"""

import base64
import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.tools")


class MCPClientMixin:
    """Mixin providing MCP transport and tool methods for ToolRegistry.

    Expects the host class to have:
        - self._mcp_sessions: dict
        - self._mcp_available: Optional[dict]
        - self._mcp_detect_lock: threading.Lock
        - self.mcp_available (property): dict
    """

    def _mcp_init_fields(self):
        """Initialize MCP fields — call from host __init__."""
        self._mcp_sessions = {}
        self._mcp_available = None
        self._mcp_detect_lock = threading.Lock()

    def shutdown_mcp(self):
        """Clean up all MCP sessions: close SSE connections, kill stdio subprocesses."""
        for name, session in list(self._mcp_sessions.items()):
            try:
                if session.get("transport") == "stdio":
                    proc = session.get("proc")
                    if proc and proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=5)
                elif session.get("stop_event"):
                    session["stop_event"].set()
            except Exception:
                pass
        self._mcp_sessions.clear()

    @property
    def mcp_available(self) -> dict:
        """Lazy-loaded MCP service availability. Detects on first access."""
        if self._mcp_available is None:
            with self._mcp_detect_lock:
                if self._mcp_available is None:
                    self._mcp_available = self._detect_mcp_services()
        return self._mcp_available or {}

    # ── MCP SSE Dual-Connection Protocol ──

    @staticmethod
    def _parse_sse_response(raw_text: str) -> dict | None:
        """Parse an SSE (Server-Sent Events) response into a JSON-RPC result."""
        data_payload = None
        for line in raw_text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                candidate = line[5:].strip()
                if candidate.startswith("{"):
                    data_payload = candidate
        if data_payload:
            try:
                return json.loads(data_payload)
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _parse_mcp_text(raw_text: str) -> list | None:
        """Parse MCP tool-call result text, handling multi-layer JSON encoding."""
        if not raw_text or not raw_text.strip():
            return None

        try:
            data = json.loads(raw_text)
            if isinstance(data, list):
                return data
            if isinstance(data, str):
                try:
                    inner = json.loads(data)
                    if isinstance(inner, list):
                        return inner
                except json.JSONDecodeError:
                    pass
        except json.JSONDecodeError:
            pass

        text = raw_text.strip()
        if text.startswith('"') and text.endswith('"'):
            try:
                unquoted = json.loads(text)
                if isinstance(unquoted, str):
                    try:
                        data = json.loads(unquoted)
                        if isinstance(data, list):
                            return data
                    except json.JSONDecodeError:
                        pass
                elif isinstance(unquoted, list):
                    return unquoted
            except json.JSONDecodeError:
                pass

        arr_match = re.search(r'\[{.*}\]', raw_text, re.DOTALL)
        if arr_match:
            try:
                data = json.loads(arr_match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        return None

    def _mcp_sse_reader(self, service_name: str, resp, responses: dict,
                        stop_event: threading.Event):
        """Background thread: read SSE events from GET /sse connection."""
        try:
            while not stop_event.is_set():
                try:
                    line = resp.readline()
                except Exception:
                    break
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str.startswith("data:"):
                    data_val = line_str[5:].strip()
                    if data_val.startswith("{"):
                        try:
                            parsed = json.loads(data_val)
                            rpc_id = parsed.get("id")
                            if rpc_id:
                                responses[rpc_id] = parsed
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass
        finally:
            try:
                resp.close()
            except Exception:
                pass

    def _mcp_initialize(self, service_name: str, config: dict) -> dict | None:
        """Initialize an MCP SSE session using dual-connection model."""
        import urllib.request
        import uuid
        from urllib.parse import urlparse

        sse_url = config["url"]
        api_key = config["auth"]
        auth_header = f"Bearer {api_key}" if not api_key.startswith("Bearer ") else api_key

        get_headers = {
            "Authorization": auth_header,
            "Accept": "text/event-stream",
            "User-Agent": "AutoResearcher/1.0",
        }

        message_path = None
        try:
            req = urllib.request.Request(sse_url, headers=get_headers, method="GET")
            resp = urllib.request.urlopen(req, timeout=15)
            while True:
                line = resp.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str.startswith("data:"):
                    data_val = line_str[5:].strip()
                    if "/message" in data_val and "sessionId" in data_val:
                        message_path = data_val
                        break
        except Exception as e:
            logger.debug(f"MCP {service_name}: GET /sse failed: {e}")
            return None

        if not message_path:
            logger.debug(f"MCP {service_name}: no endpoint event in SSE response")
            try:
                resp.close()
            except Exception:
                pass
            return None

        if message_path.startswith("http"):
            message_url = message_path
        else:
            parsed = urlparse(sse_url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            message_url = base + message_path

        logger.debug(f"MCP {service_name}: message endpoint ...{message_url[-50:]}")

        responses = {}
        stop_event = threading.Event()
        reader_thread = threading.Thread(
            target=self._mcp_sse_reader,
            args=(service_name, resp, responses, stop_event),
            daemon=True,
        )
        reader_thread.start()

        msg_headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
            "User-Agent": "AutoResearcher/1.0",
        }

        init_id = str(uuid.uuid4())
        init_payload = json.dumps({
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "AutoResearcher", "version": "1.0"},
            },
        }).encode()

        try:
            req = urllib.request.Request(message_url, data=init_payload, headers=msg_headers, method="POST")
            with urllib.request.urlopen(req, timeout=15):
                pass
        except Exception as e:
            logger.debug(f"MCP {service_name}: initialize POST failed: {e}")
            stop_event.set()
            return None

        for _ in range(30):
            if init_id in responses:
                break
            time.sleep(0.5)

        init_result = responses.pop(init_id, None)
        if not init_result:
            logger.debug(f"MCP {service_name}: no initialize response received")
            stop_event.set()
            return None

        if "error" in init_result:
            logger.debug(f"MCP {service_name}: initialize error: {init_result['error']}")
            stop_event.set()
            return None

        logger.debug(f"MCP {service_name}: initialized OK")

        notif_payload = json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }).encode()
        try:
            notif_req = urllib.request.Request(
                message_url, data=notif_payload, headers=msg_headers, method="POST"
            )
            with urllib.request.urlopen(notif_req, timeout=10):
                pass
        except Exception:
            pass

        return {
            "message_url": message_url,
            "auth": auth_header,
            "responses": responses,
            "stop_event": stop_event,
        }

    def _mcp_call_tool(self, service_name: str, tool_name: str, arguments: dict,
                       timeout: int = 30) -> str | None:
        """Call an MCP tool. Routes to SSE or stdio transport based on config."""
        config = self.mcp_available.get(service_name)
        if not config:
            return None

        if config.get("transport") == "stdio":
            session = self._mcp_sessions.get(service_name)
            if not session:
                session = self._mcp_stdio_initialize(service_name)
                if not session:
                    return None
                self._mcp_sessions[service_name] = session
            return self._mcp_stdio_call(service_name, tool_name, arguments, timeout)

        import urllib.request
        import uuid

        session = self._mcp_sessions.get(service_name)
        if not session:
            session = self._mcp_initialize(service_name, config)
            if not session:
                return None
            self._mcp_sessions[service_name] = session

        message_url = session["message_url"]
        auth_header = session["auth"]
        responses = session["responses"]

        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
            "User-Agent": "AutoResearcher/1.0",
        }

        rpc_id = str(uuid.uuid4())
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }).encode()

        try:
            req = urllib.request.Request(message_url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
        except Exception as e:
            logger.warning(f"MCP {service_name}/{tool_name} POST failed: {e}")
            self._mcp_sessions.pop(service_name, None)
            return None

        deadline = time.time() + timeout
        while time.time() < deadline:
            if rpc_id in responses:
                break
            time.sleep(0.3)

        data = responses.pop(rpc_id, None)
        if not data:
            logger.warning(f"MCP {service_name}/{tool_name}: no response within timeout")
            return None

        if "error" in data:
            err = data["error"]
            logger.warning(f"MCP {service_name}: error {err.get('code')}: {err.get('message')}")
            if err.get("code") in (-401, -32600):
                self._mcp_sessions.pop(service_name, None)
            return None

        result = data.get("result", {})
        content_blocks = result.get("content", [])
        if result.get("isError"):
            err_text = " ".join(
                b.get("text", "") for b in content_blocks if isinstance(b, dict)
            )
            logger.warning(f"MCP {service_name} tool error: {err_text[:200]}")
            return None

        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)

        raw_text = "\n".join(text_parts)
        return raw_text if raw_text.strip() else None

    # ── MCP stdio transport ──

    def _mcp_stdio_initialize(self, service_name: str) -> dict | None:
        """Initialize an MCP stdio session by spawning a subprocess."""
        glm_key = os.environ.get("GLM_CODING_PLAN_API_KEY", "")
        if not glm_key:
            return None

        try:
            proc = subprocess.Popen(
                ["npx", "-y", "@z_ai/mcp-server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "Z_AI_API_KEY": glm_key, "Z_AI_MODE": "ZHIPU"},
            )
        except FileNotFoundError:
            logger.debug("MCP stdio: npx not found, cannot start zai-mcp-server")
            return None
        except Exception as e:
            logger.debug(f"MCP stdio: failed to spawn zai-mcp-server: {e}")
            return None

        init_msg = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "AutoResearcher", "version": "1.0"},
            },
        }) + "\n"

        try:
            proc.stdin.write(init_msg.encode())
            proc.stdin.flush()
            resp_line = proc.stdout.readline()
            if not resp_line:
                proc.kill()
                return None
            resp = json.loads(resp_line.decode())
            if "error" in resp:
                logger.debug(f"MCP stdio: initialize error: {resp['error']}")
                proc.kill()
                return None
        except Exception as e:
            logger.debug(f"MCP stdio: initialize failed: {e}")
            proc.kill()
            return None

        notif_msg = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}) + "\n"
        try:
            proc.stdin.write(notif_msg.encode())
            proc.stdin.flush()
        except Exception:
            pass

        logger.debug(f"MCP stdio {service_name}: initialized OK")
        return {"transport": "stdio", "proc": proc, "next_id": 2, "lock": threading.RLock()}

    def _mcp_stdio_call(self, service_name: str, tool_name: str,
                        arguments: dict, timeout: int = 30) -> str | None:
        """Call an MCP tool via stdio subprocess."""
        session = self._mcp_sessions.get(service_name)
        if not session or session.get("transport") != "stdio":
            return None

        proc = session["proc"]
        lock = session["lock"]

        with lock:
            session["next_id"] += 1
            rpc_id = session["next_id"]

            payload = json.dumps({
                "jsonrpc": "2.0", "id": rpc_id, "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }) + "\n"

            try:
                proc.stdin.write(payload.encode())
                proc.stdin.flush()
            except BrokenPipeError:
                logger.warning(f"MCP stdio {service_name}: process died, attempting restart")
                self._mcp_sessions.pop(service_name, None)
                try:
                    proc.kill()
                except Exception:
                    pass
                return None

        if proc.poll() is not None:
            logger.warning(f"MCP stdio {service_name}: process exited with code {proc.returncode}")
            self._mcp_sessions.pop(service_name, None)
            return None

        try:
            buf = b""
            deadline = time.time() + timeout
            data = None
            while time.time() < deadline:
                line = proc.stdout.readline()
                if not line:
                    break
                buf += line
                try:
                    data = json.loads(buf.decode())
                    if data.get("id") == rpc_id:
                        break
                    else:
                        buf = b""
                        data = None
                        continue
                except json.JSONDecodeError:
                    continue
            else:
                logger.warning(f"MCP stdio {service_name}: timeout reading response")
                return None

            if data is None:
                logger.warning(f"MCP stdio {service_name}: no valid response received")
                return None
        except Exception as e:
            logger.warning(f"MCP stdio {service_name}: read failed: {e}")
            return None

        if "error" in data:
            err = data["error"]
            logger.warning(f"MCP stdio {service_name}: error {err.get('code')}: {err.get('message')}")
            return None

        result = data.get("result", {})
        content_blocks = result.get("content", [])
        if result.get("isError"):
            err_text = " ".join(b.get("text", "") for b in content_blocks if isinstance(b, dict))
            logger.warning(f"MCP stdio {service_name} tool error: {err_text[:200]}")
            return None

        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)

        raw_text = "\n".join(text_parts)
        return raw_text if raw_text.strip() else None

    # ── MCP service detection ──

    def _detect_mcp_services(self) -> dict:
        """Detect which MCP services are available in the current environment."""
        available = {}
        glm_key = os.environ.get("GLM_CODING_PLAN_API_KEY", "")
        if not glm_key:
            logger.info("MCP: No GLM_CODING_PLAN_API_KEY found, MCP services unavailable")
            return available

        try:
            import urllib.request
            req = urllib.request.Request(
                "https://open.bigmodel.cn/api/paas/v4/models",
                headers={"Authorization": f"Bearer {glm_key}"}, method="GET",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            logger.info("MCP: GLM API unreachable, skipping MCP service detection")
            return available

        start = time.time()
        MAX_DETECT_TIME = 30

        sse_services = {
            "web_search_prime": {"url": "https://open.bigmodel.cn/api/mcp/web_search_prime/sse"},
            "web_reader": {"url": "https://open.bigmodel.cn/api/mcp/web_reader/sse"},
            "zread": {"url": "https://open.bigmodel.cn/api/mcp/zread/sse"},
        }

        for name, info in sse_services.items():
            if time.time() - start > MAX_DETECT_TIME:
                logger.warning(f"MCP: Detection time budget exceeded, stopping at {name}")
                break
            config = {"url": info["url"], "auth": glm_key, "transport": "sse"}
            session = self._mcp_initialize(name, config)
            if session:
                available[name] = config
                self._mcp_sessions[name] = session
                logger.info(f"MCP: {name} service available (SSE)")

        if time.time() - start < MAX_DETECT_TIME:
            stdio_session = self._mcp_stdio_initialize("zai_vision")
            if stdio_session:
                available["zai_vision"] = {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@z_ai/mcp-server"],
                    "env_key": "Z_AI_API_KEY",
                }
                self._mcp_sessions["zai_vision"] = stdio_session
                logger.info("MCP: zai_vision service available (stdio)")

        return available

    # ── MCP web tools ──

    def _mcp_web_search(self, query: str, max_results: int = 5) -> str | None:
        """Search the web via MCP web-search-prime service."""
        if "web_search_prime" not in self.mcp_available:
            return None

        raw_text = self._mcp_call_tool(
            "web_search_prime", "web_search_prime",
            {"search_query": query, "content_size": "medium", "location": "cn"},
            timeout=20,
        )
        if not raw_text:
            return None

        results = self._parse_mcp_text(raw_text)
        if not results:
            return None

        if not isinstance(results, list):
            results = [results]

        papers = []
        for r in results[:max_results]:
            if not isinstance(r, dict):
                continue
            title = r.get("title", "Untitled")
            url = r.get("link", r.get("url", ""))
            snippet = (r.get("content", r.get("snippet", "")) or "")[:300]
            if title:
                papers.append({"title": title, "url": url, "snippet": snippet})

        if not papers:
            return None

        result = {"source": "mcp_web_search", "results": papers}
        logger.info(f"MCP: web_search_prime returned {len(papers)} results")
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _mcp_web_fetch(self, url: str, fetch_info: str = "") -> str | None:
        """Fetch a URL via MCP web-reader service."""
        if "web_reader" not in self.mcp_available:
            return None

        raw_text = self._mcp_call_tool(
            "web_reader", "webReader",
            {"url": url, "return_format": "markdown", "retain_images": False, "timeout": 20},
            timeout=25,
        )
        if not raw_text:
            return None

        content = raw_text
        parsed = self._parse_mcp_text(raw_text)
        if isinstance(parsed, list) and parsed:
            first = parsed[0] if isinstance(parsed[0], dict) else {}
            content = first.get("content", raw_text)
        elif isinstance(parsed, dict):
            content = parsed.get("content", raw_text)

        if not content or not content.strip():
            return None

        result = {
            "source": "mcp_web_reader",
            "url": url,
            "title": fetch_info or "Fetched via MCP",
            "content_snippet": content[:3000],
            "fetched_for": fetch_info,
        }
        logger.info(f"MCP: web_reader fetched {len(content)} chars from {url}")
        return json.dumps(result, ensure_ascii=False, indent=2)

    # ── Vision MCP Tools ──

    def _mcp_vision_call(self, tool_name: str, arguments: dict, timeout: int = 60) -> str | None:
        """Call a zai-mcp-server vision tool via MCP (stdio transport)."""
        if "zai_vision" not in self.mcp_available:
            return None

        remapped = dict(arguments)
        if "image_path" in remapped and "image_source" not in remapped:
            remapped["image_source"] = remapped.pop("image_path")

        actual_tool = "analyze_image" if tool_name == "image_analysis" else tool_name
        raw_text = self._mcp_call_tool("zai_vision", actual_tool, remapped, timeout=timeout)
        return raw_text

    def _mcp_diagnose_error_screenshot(self, image_path: str, context: str = "") -> str | None:
        b64_url = self._local_path_to_data_url(image_path)
        if not b64_url:
            return None
        return self._mcp_vision_call("diagnose_error_screenshot", {"image_path": b64_url, "context": context})

    def _mcp_extract_text(self, image_path: str, language_hint: str = "") -> str | None:
        b64_url = self._local_path_to_data_url(image_path)
        if not b64_url:
            return None
        args = {"image_path": b64_url}
        if language_hint:
            args["language_hint"] = language_hint
        return self._mcp_vision_call("extract_text_from_screenshot", args)

    def _mcp_analyze_data_viz(self, image_path: str, focus: str = "") -> str | None:
        b64_url = self._local_path_to_data_url(image_path)
        if not b64_url:
            return None
        args = {"image_path": b64_url}
        if focus:
            args["prompt"] = focus
        return self._mcp_vision_call("analyze_data_visualization", args)

    def _mcp_understand_diagram(self, image_path: str, prompt: str = "") -> str | None:
        b64_url = self._local_path_to_data_url(image_path)
        if not b64_url:
            return None
        args = {"image_path": b64_url}
        if prompt:
            args["prompt"] = prompt
        return self._mcp_vision_call("understand_technical_diagram", args)

    def _mcp_image_analysis(self, image_path: str, prompt: str = "") -> str | None:
        b64_url = self._local_path_to_data_url(image_path)
        if not b64_url:
            return None
        args = {"image_path": b64_url}
        if prompt:
            args["prompt"] = prompt
        return self._mcp_vision_call("image_analysis", args)

    def _local_path_to_data_url(self, file_path: str) -> str | None:
        """Convert a local image file to a base64 data URL for remote MCP servers."""
        try:
            p = Path(file_path)
            if not p.exists():
                logger.warning(f"MCP vision: file not found: {file_path}")
                return None
            img_bytes = p.read_bytes()
            b64_data = base64.b64encode(img_bytes).decode("utf-8")
            ext = p.suffix.lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".webp": "image/webp", ".gif": "image/gif"}
            mime_type = mime_map.get(ext, "image/png")
            return f"data:{mime_type};base64,{b64_data}"
        except Exception as e:
            logger.warning(f"MCP vision: failed to encode {file_path}: {e}")
            return None
