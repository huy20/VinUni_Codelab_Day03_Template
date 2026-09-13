"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

OPENAI_MODEL = "gpt-4o-mini"

DEFAULT_MAX_PRICE = 5_000_000
AIRPORT_CODES = ("HAN", "SGN", "DAD")
FLIGHT_KEYWORDS = ("chuyến bay", "chuyen bay", "máy bay", "may bay", "vé", "ve", "flight")
WEATHER_KEYWORDS = (
    "thời tiết", "thoi tiet", "weather",
    "nên mặc", "nen mac", "mặc gì", "mac gi",
    "trang phục", "trang phuc",
)


def call_openai(
    prompt: str,
    system_prompt: str = "",
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> str:
    """Gọi LLM 1 lượt (chỉ dùng khi có OPENAI_API_KEY)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


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


def _extract_airport_codes(text: str) -> list[str]:
    """Trích xuất các mã sân bay (HAN, SGN, DAD) theo thứ tự xuất hiện."""
    codes = []
    for token in re.findall(r"\b[A-Z]{3}\b", text):
        if token in AIRPORT_CODES and token not in codes:
            codes.append(token)
    return codes


def _extract_weather_city(text: str) -> str | None:
    """Chọn mã thành phố đứng ngay sau từ khóa thời tiết (fallback: mã cuối cùng)."""
    lowered = text.lower()
    positions = [
        (m.start(), m.group())
        for m in re.finditer(r"\b[A-Z]{3}\b", text)
        if m.group() in AIRPORT_CODES
    ]
    if not positions:
        return None

    keyword_pos = None
    for keyword in WEATHER_KEYWORDS:
        idx = lowered.find(keyword)
        if idx != -1 and (keyword_pos is None or idx < keyword_pos):
            keyword_pos = idx

    if keyword_pos is not None:
        after = [code for pos, code in positions if pos >= keyword_pos]
        if after:
            return after[0]
    return positions[-1][1]


def _parse_max_price(lowered_text: str) -> int:
    """Đọc ngân sách dạng 'dưới 2 triệu', '1.5 triệu', '500k'."""
    match = re.search(
        r"dưới\s*([0-9]+(?:[.,][0-9]+)?)\s*(triệu|nghìn|ngàn|vnd|tr|k|đ)",
        lowered_text,
    )
    if not match:
        return DEFAULT_MAX_PRICE

    value = float(match.group(1).replace(",", "."))
    unit = match.group(2)
    if unit in ("triệu", "tr"):
        multiplier = 1_000_000
    elif unit in ("nghìn", "ngàn", "k"):
        multiplier = 1_000
    else:
        multiplier = 1
    return int(value * multiplier)


def _build_plan(user_input: str) -> list[dict]:
    """Suy luận (Thought) và chọn danh sách Action cần thực thi cho câu hỏi."""
    lowered = user_input.lower()
    codes = _extract_airport_codes(user_input)
    tasks = []

    if len(codes) >= 2 and any(keyword in lowered for keyword in FLIGHT_KEYWORDS):
        max_price = _parse_max_price(lowered)
        tasks.append({
            "thought": f"Cần tra cứu chuyến bay {codes[0]} -> {codes[1]} với giá tối đa {max_price} VND.",
            "name": "get_flight_info",
            "args": {"origin": codes[0], "destination": codes[1], "max_price": max_price},
        })

    if any(keyword in lowered for keyword in WEATHER_KEYWORDS):
        city = _extract_weather_city(user_input)
        if city:
            tasks.append({
                "thought": f"Cần tra cứu thời tiết và gợi ý trang phục cho {city}.",
                "name": "get_weather_forecast",
                "args": {"city_code": city},
            })

    return tasks


class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def query(self, user_input: str) -> dict:
        # TODO: Trả về câu trả lời 1 lượt (không dùng tool)
        answer = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                answer = call_openai(prompt=user_input)
            except Exception:
                answer = None
        if not answer:
            answer = (
                "[Chatbot Baseline] Tôi không có công cụ tra cứu dữ liệu nên "
                f"không thể xác minh thông tin cho câu hỏi: {user_input}"
            )
        return {"status": "success", "answer": answer, "tool_calls": []}


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    def _parse_action(self, text: str) -> dict | None:
        """TODO 3: Phân tích Thought / Action từ Agent (Trap 2: JSON hỏng)."""
        if not text:
            return None
        marker = "Action:"
        idx = text.find(marker)
        if idx == -1:
            return None

        payload = text[idx + len(marker):].strip()
        for stopper in ("Observation:", "Final Answer:"):
            stop = payload.find(stopper)
            if stop != -1:
                payload = payload[:stop].strip()

        start, end = payload.find("{"), payload.rfind("}")
        if start != -1 and end > start:
            payload = payload[start:end + 1]

        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return {"error": "Invalid JSON format"}

        if not isinstance(data, dict) or "name" not in data:
            return {"error": "Invalid Action format"}
        return {"name": str(data["name"]).strip().lower(), "args": data.get("args") or {}}

    def _execute_action(self, action: dict) -> object:
        """TODO 4: Thực thi Tool trong TOOL_MAP (Trap 1: chuẩn hóa tên tool)."""
        if not action or "name" not in action:
            return {"error": "No action to execute"}

        name = str(action["name"]).strip().lower()
        tool = TOOL_MAP.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}

        try:
            return tool(**action.get("args", {}))
        except TypeError as exc:
            return {"error": f"Invalid arguments for {name}: {exc}"}
        except Exception as exc:
            return {"error": f"Tool {name} failed: {exc}"}

    def _answer_flight(self, observation) -> str:
        if isinstance(observation, dict) and observation.get("error"):
            return f"Không thể tra cứu chuyến bay: {observation['error']}"
        if not observation:
            return "Rất tiếc, không tìm thấy chuyến bay nào phù hợp với ngân sách của bạn."
        flights = "; ".join(
            f"{fl['flight_number']} ({fl['airline']}) khởi hành {fl['departure_time']} "
            f"giá {fl['price_vnd']:,} VND"
            for fl in observation
        )
        return f"Các chuyến bay phù hợp: {flights}."

    def _answer_weather(self, observation) -> str:
        if isinstance(observation, dict) and observation.get("error"):
            return f"Không thể tra cứu thời tiết: {observation['error']}"
        if not observation:
            return "Rất tiếc, không có dữ liệu thời tiết cho thành phố này."
        return (
            f"Thời tiết tại {observation['city']}: {observation['temperature_c']}°C, "
            f"{observation['condition']}, độ ẩm {observation['humidity_pct']}%. "
            f"Gợi ý trang phục: {observation['recommendation']}"
        )

    def _compose_answer(self, observations: list) -> str:
        if not observations:
            return (
                "Chính sách đổi trả vé máy bay Vinpearl: Vinpearl hỗ trợ đổi/hoàn vé theo "
                "điều kiện hạng vé và thời điểm yêu cầu; vui lòng liên hệ tổng đài Vinpearl "
                "để được hỗ trợ chi tiết."
            )

        parts = []
        for action, observation in observations:
            if action["name"] == "get_flight_info":
                parts.append(self._answer_flight(observation))
            elif action["name"] == "get_weather_forecast":
                parts.append(self._answer_weather(observation))
        return "\n".join(parts)

    def run(self, user_input: str) -> dict:
        # TODO 1: Khởi tạo lịch sử conversation / traces
        self.trace = []
        plan = _build_plan(user_input)
        pending = list(plan)
        needs_synthesis = len(plan) > 1
        observations = []
        iteration = 0

        # TODO 2: Vòng lặp ReAct với safeguard max_iterations
        while iteration < self.max_iterations:
            iteration += 1

            if pending:
                task = pending.pop(0)
                action_text = (
                    f"Thought: {task['thought']}\n"
                    f"Action: {json.dumps({'name': task['name'], 'args': task['args']}, ensure_ascii=False)}"
                )
                action = self._parse_action(action_text)

                if not action or action.get("error"):
                    error = action if isinstance(action, dict) else {"error": "Invalid JSON format"}
                    self.trace.append({
                        "iteration": iteration,
                        "thought": task["thought"],
                        "action": None,
                        "observation": error,
                    })
                    continue

                observation = self._execute_action(action)
                observations.append((action, observation))
                step = {
                    "iteration": iteration,
                    "thought": task["thought"],
                    "action": action,
                    "observation": observation,
                }
                self.trace.append(step)

                # TODO 5: nếu 1 tool đã đủ trả lời thì chốt Final Answer ngay bước này
                if not pending and not needs_synthesis:
                    answer = self._compose_answer(observations)
                    step["final_answer"] = answer
                    return {
                        "status": "completed",
                        "iterations": len(self.trace),
                        "trace": self.trace,
                        "answer": answer,
                    }
                continue

            answer = self._compose_answer(observations)
            self.trace.append({
                "iteration": iteration,
                "thought": "Đã có đủ dữ liệu, tổng hợp câu trả lời cho khách hàng.",
                "action": None,
                "observation": None,
                "final_answer": answer,
            })
            return {
                "status": "completed",
                "iterations": len(self.trace),
                "trace": self.trace,
                "answer": answer,
            }

        # TODO: Safeguard khi vượt quá số bước tối đa
        return {
            "status": "max_iterations_reached",
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "iterations": len(self.trace),
            "trace": self.trace,
        }


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
