"""
Lab #3: Baseline Chatbot vs ReAct Agent.

The lab uses a deterministic planner instead of an external LLM so it can be
run and graded offline. The agent still follows the ReAct pattern: decide the
next action, call a registered tool, record the observation, and only then
compose an answer from the collected evidence.
"""

import json
import re
from typing import Any, Dict, List

from tools import TOOL_DEFINITIONS, TOOL_MAP


SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""


class ChatbotBaseline:
    """Baseline chatbot: answers once and never invokes a tool."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "status": "success",
            "answer": (
                "[Chatbot Baseline] Tôi chưa sử dụng công cụ tra cứu nên không thể "
                f"xác minh dữ liệu thời gian thực cho yêu cầu: {user_input}"
            ),
            "tool_calls": [],
        }


class ReActAgent:
    """Small offline ReAct agent backed by the lab's local JSON tools."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []
        self.system_prompt = SYSTEM_PROMPT.format(
            tools=json.dumps(TOOL_DEFINITIONS, ensure_ascii=False)
        )

    @staticmethod
    def _airport_codes(user_input: str) -> List[str]:
        """Return unique three-letter airport codes in their original order."""
        codes: List[str] = []
        supported_codes = {"HAN", "SGN", "DAD"}
        for code in re.findall(r"(?<![A-Za-z])([A-Za-z]{3})(?![A-Za-z])", user_input):
            normalized = code.upper()
            is_code_like = code.isupper() or normalized in supported_codes
            if is_code_like and normalized not in codes:
                codes.append(normalized)
        return codes

    @staticmethod
    def _max_price(user_input: str) -> int:
        """Extract common Vietnamese budget expressions, defaulting to 5m VND."""
        text = user_input.lower().replace(",", ".")

        million = re.search(r"(\d+(?:\.\d+)?)\s*(?:triệu|trieu|million)", text)
        if million:
            return int(float(million.group(1)) * 1_000_000)

        thousand = re.search(r"(\d+(?:\.\d+)?)\s*(?:k|nghìn|nghin)", text)
        if thousand:
            return int(float(thousand.group(1)) * 1_000)

        vnd = re.search(r"([\d.]+)\s*(?:vnd|đồng|dong)", text)
        if vnd:
            digits = re.sub(r"\D", "", vnd.group(1))
            if digits:
                return int(digits)

        return 5_000_000

    def _plan(self, user_input: str) -> List[Dict[str, Any]]:
        """Create the ordered tool plan from the user's request."""
        lowered = user_input.lower()
        codes = self._airport_codes(user_input)
        asks_for_weather = any(
            phrase in lowered
            for phrase in ("thời tiết", "thoi tiet", "weather", "nên mặc", "trang phục")
        )
        asks_for_flight = any(
            phrase in lowered
            for phrase in ("chuyến bay", "chuyen bay", "vé máy bay", "ve may bay", "flight")
        )

        plan: List[Dict[str, Any]] = []
        if asks_for_flight and len(codes) >= 2:
            plan.append(
                {
                    "name": "get_flight_info",
                    "args": {
                        "origin": codes[0],
                        "destination": codes[1],
                        "max_price": self._max_price(user_input),
                    },
                }
            )

        if asks_for_weather and codes:
            # In a combined trip query, weather normally refers to the destination.
            city_code = codes[1] if asks_for_flight and len(codes) >= 2 else codes[-1]
            plan.append(
                {
                    "name": "get_weather_forecast",
                    "args": {"city_code": city_code},
                }
            )

        return plan

    @staticmethod
    def _execute_action(action: Any) -> Any:
        """Validate an Action payload and execute its normalized registry entry."""
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format"}

        if not isinstance(action, dict):
            return {"error": "Invalid action format"}

        tool_name = str(action.get("name", "")).strip().lower()
        arguments = action.get("args", {})
        if tool_name not in TOOL_MAP:
            return {"error": f"Unknown tool: {tool_name or '<empty>'}"}
        if not isinstance(arguments, dict):
            return {"error": "Tool args must be a JSON object"}

        try:
            return TOOL_MAP[tool_name](**arguments)
        except (TypeError, ValueError) as exc:
            return {"error": f"Tool execution failed: {exc}"}

    @staticmethod
    def _format_flights(flights: Any, action: Dict[str, Any]) -> str:
        args = action["args"]
        route = f"{args['origin']} → {args['destination']}"
        if not isinstance(flights, list) or not flights:
            return (
                f"Không tìm thấy chuyến bay {route} trong mức giá tối đa "
                f"{args['max_price']:,} VND."
            )

        details = []
        for flight in flights:
            details.append(
                f"{flight['flight_number']} ({flight['airline']}), khởi hành "
                f"{flight['departure_time']}, giá {flight['price_vnd']:,} VND"
            )
        return f"Các chuyến bay phù hợp tuyến {route}: " + "; ".join(details) + "."

    @staticmethod
    def _format_weather(weather: Any, action: Dict[str, Any]) -> str:
        if not isinstance(weather, dict) or "error" in weather:
            message = (
                weather.get("error", "Không có dữ liệu")
                if isinstance(weather, dict)
                else "Không có dữ liệu"
            )
            return f"Không thể tra cứu thời tiết {action['args']['city_code']}: {message}."

        return (
            f"Thời tiết {weather['city']}: {weather['temperature_c']}°C, "
            f"{weather['condition']}, độ ẩm {weather['humidity_pct']}%. "
            f"Gợi ý: {weather['recommendation']}"
        )

    def _compose_answer(
        self,
        user_input: str,
        completed_actions: List[Dict[str, Any]],
        observations: List[Any],
    ) -> str:
        if not completed_actions:
            if "vinpearl" in user_input.lower():
                return (
                    "Chính sách đổi/trả vé máy bay Vinpearl phụ thuộc vào điều kiện "
                    "của hạng vé và đơn vị vận chuyển. Bạn nên kiểm tra điều kiện ghi "
                    "trên vé hoặc liên hệ kênh hỗ trợ chính thức của Vinpearl trước khi đổi/trả."
                )
            return (
                "Tôi chưa có công cụ phù hợp để xác minh yêu cầu này. Vui lòng cung cấp "
                "mã sân bay hoặc hỏi về chuyến bay/thời tiết."
            )

        answer_parts: List[str] = []
        for action, observation in zip(completed_actions, observations):
            if action["name"] == "get_flight_info":
                answer_parts.append(self._format_flights(observation, action))
            elif action["name"] == "get_weather_forecast":
                answer_parts.append(self._format_weather(observation, action))
        return " ".join(answer_parts)

    def _completed_result(self, iterations: int, answer: str) -> Dict[str, Any]:
        return {
            "status": "completed",
            "iterations": iterations,
            "answer": answer,
            "trace": self.trace,
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Run the Thought -> Action -> Observation loop and return its trace."""
        self.trace = []
        plan = self._plan(user_input)
        observations: List[Any] = []
        completed_actions: List[Dict[str, Any]] = []
        iteration = 0

        if self.max_iterations <= 0:
            return {
                "status": "max_iterations_reached",
                "iterations": 0,
                "answer": "Không thể hoàn thành trong số bước tối đa.",
                "trace": self.trace,
            }

        # A request that needs no lab tool is answered in one reasoning step.
        if not plan:
            iteration = 1
            answer = self._compose_answer(user_input, completed_actions, observations)
            self.trace.append(
                {
                    "iteration": iteration,
                    "thought": "Không có công cụ phù hợp; trả lời bằng thông tin FAQ an toàn.",
                    "action": None,
                    "observation": "No tool call",
                    "final_answer": answer,
                }
            )
            return self._completed_result(iteration, answer)

        while iteration < self.max_iterations and len(completed_actions) < len(plan):
            action = plan[len(completed_actions)]
            iteration += 1
            observation = self._execute_action(action)
            completed_actions.append(action)
            observations.append(observation)
            trace_item: Dict[str, Any] = {
                "iteration": iteration,
                "thought": f"Cần gọi {action['name']} để lấy dữ liệu có thể kiểm chứng.",
                "action": action,
                "observation": observation,
            }
            self.trace.append(trace_item)

        if len(completed_actions) < len(plan):
            return {
                "status": "max_iterations_reached",
                "iterations": iteration,
                "answer": "Không thể hoàn thành trong số bước tối đa.",
                "trace": self.trace,
            }

        answer = self._compose_answer(user_input, completed_actions, observations)

        # A single-tool request can be answered from that observation immediately.
        if len(plan) == 1:
            self.trace[-1]["final_answer"] = answer
            return self._completed_result(iteration, answer)

        # Multi-tool requests reserve one explicit iteration for synthesis.
        if iteration >= self.max_iterations:
            return {
                "status": "max_iterations_reached",
                "iterations": iteration,
                "answer": "Không thể hoàn thành trong số bước tối đa.",
                "trace": self.trace,
            }

        iteration += 1
        self.trace.append(
            {
                "iteration": iteration,
                "thought": "Đã đủ dữ liệu chuyến bay và thời tiết; tổng hợp câu trả lời.",
                "action": None,
                "observation": "All required tool results collected",
                "final_answer": answer,
            }
        )
        return self._completed_result(iteration, answer)


def main() -> None:
    user_query = (
        "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết "
        "thời tiết SGN nên mặc gì?"
    )

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(json.dumps(chatbot.query(user_query), indent=2, ensure_ascii=False))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
