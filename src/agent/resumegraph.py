"""Resume optimization graph."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from agent.llm import get_chat_model


class ResumeState(TypedDict):
    """State for the resume optimization graph."""

    resume_text: str
    target_role: NotRequired[str]
    issues: NotRequired[list[str]]
    suggestions: NotRequired[list[str]]
    revised_resume: NotRequired[str]


def _target_role_text(state: ResumeState) -> str:
    target_role = state.get("target_role", "").strip()
    if not target_role:
        return "目标岗位未指定，请按 AI 应用开发/大模型应用开发的通用标准优化。"
    return f"目标岗位：{target_role}"


def _parse_bullets(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = cleaned.removeprefix("-").removeprefix("*").strip()
        if cleaned:
            items.append(cleaned)
    return items or [text.strip()]


def analyze_resume(state: ResumeState) -> dict[str, list[str]]:
    """Analyze problems in the resume."""
    model = get_chat_model()
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "你是严格的技术简历评审专家。只输出问题清单，每行一个问题，"
                    "不要写开场白，不要生成修改后的简历。"
                )
            ),
            HumanMessage(
                content=(
                    f"{_target_role_text(state)}\n\n"
                    "请分析下面简历的问题，重点关注：岗位匹配度、项目亮点、技术深度、"
                    "量化结果、表达是否泛泛而谈。\n\n"
                    f"简历文本：\n{state['resume_text']}"
                )
            ),
        ]
    )
    return {"issues": _parse_bullets(str(response.content))}


def suggest_improvements(state: ResumeState) -> dict[str, list[str]]:
    """Suggest improvements based on the detected issues."""
    model = get_chat_model()
    issues = "\n".join(f"- {issue}" for issue in state.get("issues", []))
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "你是技术简历优化顾问。只输出优化建议清单，每行一条建议，"
                    "建议要具体、可执行。"
                )
            ),
            HumanMessage(
                content=(
                    f"{_target_role_text(state)}\n\n"
                    f"原始简历：\n{state['resume_text']}\n\n"
                    f"已发现的问题：\n{issues}"
                )
            ),
        ]
    )
    return {"suggestions": _parse_bullets(str(response.content))}


def rewrite_resume(state: ResumeState) -> dict[str, str]:
    """Rewrite the resume with the suggested improvements."""
    model = get_chat_model()
    issues = "\n".join(f"- {issue}" for issue in state.get("issues", []))
    suggestions = "\n".join(f"- {item}" for item in state.get("suggestions", []))
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "你是技术简历改写专家。输出一版修改后的简历文本。"
                    "保持事实可信，不编造公司、学校、奖项；可以优化表达、结构和技术亮点。"
                )
            ),
            HumanMessage(
                content=(
                    f"{_target_role_text(state)}\n\n"
                    f"原始简历：\n{state['resume_text']}\n\n"
                    f"问题：\n{issues}\n\n"
                    f"优化建议：\n{suggestions}"
                )
            ),
        ]
    )
    return {"revised_resume": str(response.content)}


builder = StateGraph(ResumeState)

builder.add_node("analyze_resume", analyze_resume)
builder.add_node("suggest_improvements", suggest_improvements)
builder.add_node("rewrite_resume", rewrite_resume)

builder.add_edge(START, "analyze_resume")
builder.add_edge("analyze_resume", "suggest_improvements")
builder.add_edge("suggest_improvements", "rewrite_resume")
builder.add_edge("rewrite_resume", END)

graph = builder.compile(name="Resume Agent")
