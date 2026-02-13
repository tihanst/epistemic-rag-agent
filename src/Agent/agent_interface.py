from typing import (
    TypedDict,
    List,
    Literal,
    Optional,
    Dict,
    Any,
    Annotated,
    Tuple,
    Callable,
)
from operator import add
import sys

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

from ..LLM import llm_interface as llm
from ..Config.config import Settings
from ..rag_engine import query_tool as qt
from ..rag_engine import extraction_formatter as ef
from ..LLM.promp_templates import SYSTEM


# Helper functions


def _information_checker(answer: str) -> bool:
    """
    Checks if the answer from the LLM is COMPLETUM or IMPERFECTUM
    Args:
        answer (str): The answer from the LLM
    Returns:
        bool: True if the answer is COMPLETUM, False if the answer is IMPERFECTUM
    """
    if len(answer) < 30:
        if "IMPERFECTUM" in answer:
            return False
    elif len(answer) >= 30:
        ans = answer[:30]
        if "COMPLETUM" in ans:
            return True
    else:
        print(
            f"\n\nRephrase question or try again - Indeterminite answer: {answer}\n\n"
        )
        return False


def append_context_list(old: List[Any], new: List[Any]) -> List[Any]:
    """
    Reducer function for list-like fields under append.
    """
    if old is None:
        return new
    if new is None:
        return old
    return old + new


# For removal of context that failed to produce and informative LLM result because the question was insufficient.
def append_or_remove_context_list(
    old: List[Any], new: Dict[str, List[Any]]
) -> List[Any]:
    """
    Reducer for list-like fields that can both append and remove.
    Convention:
      {"add": [...]}     -> append
      {"remove": [...]}  -> remove if present
      {"set": [...]}     -> replace entirely (optional)
    """
    result = list(old or [])
    if "set" in new:
        # hard replace
        return list(new["set"])
    if "add" in new:
        result.extend(new["add"])
    if "remove" in new:
        result = [x for x in result if x not in new["remove"]]
    return result


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    turn_index: Annotated[int, add]
    context_chunks: Annotated[List[Tuple[Tuple[int, ...], str]], append_context_list]
    active_messages: Annotated[
        List[AnyMessage], add_messages
    ]  # This is for only the 'good' user Q's and good AI A's - remember to add system message at start.
    active_context_chunks: Annotated[
        List[Tuple[Tuple[int, ...], str]], append_context_list
    ]  # This is for only the good context that resulted from COMPLETUM responses
    length_last_added_chunks: Annotated[
        int, lambda _, new: new
    ]  # This is so reducer of form reducer(old, new) simply replaces old with new.
    use_stream_ui: Annotated[Callable[[str], None] | None, lambda old, new: new]
    # The above is the pass through function for Gradio streaming
    user_input: Annotated[Optional[str], lambda _, new: new]
    # Above is transient key used ONLY in Gradio mode to pass the user input to the agent (CLI mode comes right from the command line)


settings = Settings()

model = llm.LLM(llm_api_key=settings.LLM_API_KEY, llm_endpoint=settings.LLM_ENDPOINT)

# Assumes database is being served
query_tool = qt.RAGPipeline()


# Nodes here


# This must handle the case where the loop comes back to it because the llm answered INCOMPLETUM.
# This must remove the context just added to the context_chunks list
# This must also remove the messages that were just added to the messages list
def enter_question(state: AgentState) -> Dict[str, Any]:
    """
    Question entry point for the agent.
    Args:
        state (AgentState): The state of the agent.
    Returns:
        Dict[str, Any]: The updated state of the agent (reducers applied by Langgraph)

    """
    # Dual-mode input source (Gradio or CLI)
    if callable(state.get("use_stream_ui")):
        # Gradio path
        start_question = (state.get("user_input") or "").strip()
        if not start_question:
            raise RuntimeError(
                "Gradio mode: 'user_input' missing. Handler must supply it."
            )
    else:
        # CLI path, read from stdin
        start_question: str = input("\n\nWhat is your question?\nQ:").strip()
        if start_question == "!exit":
            sys.exit(0)
        if not start_question:
            return {}  # Examine later where this path leads (fail/error, hang, or retry?)

    retrieved_start_question = query_tool.execute_extraction(start_question)
    _, list_of_context_chunks = retrieved_start_question

    string_retrieved_start_question = query_tool.format_context_for_llm_excerpts(
        retrieved_start_question
    )  # Uses original indexing from chunking process for excerpt numbering in llm context

    messages = {"messages": [HumanMessage(content=string_retrieved_start_question)]}
    turn_index = {"turn_index": 1}
    context_chunks = {"context_chunks": list_of_context_chunks}
    length_chunks = {"length_last_added_chunks": len(list_of_context_chunks)}

    # In Gradio path, consume user_input so it's not reused accidentally
    if "user_input" in state:
        user_input = {"user_input": None}
        return messages | turn_index | context_chunks | length_chunks | user_input

    # CLI path
    return messages | turn_index | context_chunks | length_chunks


async def llm_call(state: AgentState) -> Dict[str, Any]:
    """
    Asynchronous call to the LLM to generate a response.
    Args:
        state (AgentState): The state of the agent.
    Returns:
        Dict[str, Any]: The updated state of the agent (reducers applied by Langgraph)
    """
    # Note this returns AIMessage or ToolMessage
    if state["use_stream_ui"] is not None:
        answer = await model.call_front_end_aware_agentic_llm(
            state["messages"], on_delta=state["use_stream_ui"]
        )
    else:
        answer = await model.call_agentic_llm(state["messages"])
    # answer = model.call_agentic_llm(state["messages"])
    return {"messages": answer}


def sufficient_information_routing_fn(state: AgentState) -> bool:
    """
    Checks if the LLM response is sufficient to answer the question.
    Args:
        state (AgentState): The state of the agent.
    Returns:
        bool: True if the LLM response is sufficient, False otherwise
    """
    if _information_checker(state["messages"][-1].content):
        return True
    else:
        return False


def follow_up_point(state: AgentState) -> bool:
    """
    Question to check for need to move to follow up step or to send to END node.

    """
    # In Gradio mode, end the turn and let the user type naturally in the UI
    if callable(state.get("use_stream_ui")):
        return False

    # CLI prompt
    follow_up = input(
        "\n\nWould you like to follow up on this? If yes, press enter, if not type !exit.\n"
    )
    if follow_up.strip() == "!exit":
        print("\n\nThank you for using Roman History Agent!\n\n")
        return False
    else:
        return True


# Node if there was sufficient information so add both User and AI response messages to active_messages and context to active_context
def update_active_state_elements(state: AgentState) -> Dict[str, Any]:
    """
    Updates the active messages and active context chunks with the most recent messages and context chunks.
    Args:
        state (AgentState): The state of the agent.
    Returns:
        Dict[str, Any]: The updated state of the agent (reducers applied by Langgraph)
    """
    user_message: AnyMessage = state["messages"][-2]
    try:
        assert isinstance(user_message, HumanMessage), (
            f"Trying to update active_state_message HumanMessage with {type(user_message)}"
        )
    except AssertionError as e:
        print(e)
        sys.exit(1)

    ai_message: AnyMessage = state["messages"][-1]
    try:
        assert isinstance(ai_message, AIMessage), (
            f"Trying to update active_state_message AIMessage with {type(ai_message)}"
        )
    except AssertionError as e:
        print(e)
        sys.exit(1)

    active_messages: Dict[str, List[AnyMessage]] = {
        "active_messages": [user_message, ai_message]
    }
    # Need code to read only the most recently added context chunk to append to active context chunks
    recent_chunk_idx = state["length_last_added_chunks"]
    active_context_chunks = {
        "active_context_chunks": state["context_chunks"][-recent_chunk_idx:]
    }

    return active_messages | active_context_chunks


def follow_up_question(
    state: AgentState,
) -> Dict[
    str, Any
]:  # Need to use the extraction_formatter thin_list_of_redundant before sending to llm
    """
    Follow-up question entry point for the agent.
    Args:
        state (AgentState): The state of the agent.
    Returns:
        Dict[str, Any]: The updated state of the agent (reducers applied by Langgraph)
    """
    # Sanity checks:

    # for item in state.items():
    #     print("\n\n")
    #     print(f"Key: {item[0]}\n\n")
    #     print(f"Value:\n\n")
    #     print(item[1])
    #     print("\n\n\n\n")

    # Gradio / CLI router
    def get_text(prompt: str):
        """
        Get text from the user as entered in Gradio UI (after checking use_stream_ui is set).
        Args:
            prompt (str): The prompt to display to the user.
        """
        if callable(state.get("use_stream_ui")):
            return (state.get("user_input") or "").strip()
        return input(prompt).strip()

    # If we're in Gradio and there is no new user_input yet, ask for it and end the turn.
    if (
        callable(state.get("use_stream_ui"))
        and not (state.get("user_input") or "").strip()
    ):
        ask = AIMessage(
            content="I need a bit more detail to answer. Please clarify your question or add specifics (names, dates, places)."
        )
        return {"messages": ask}

    print(f"Length context_chunks: {len(state['context_chunks'])}\n")
    print(f"Length active_context_chunks: {len(state['active_context_chunks'])}\n")
    print(f"Length messages: {len(state['messages'])}\n")
    print(f"Length active_messages: {len(state['active_messages'])}")
    print(
        f"Length last_added_chunks: {state['length_last_added_chunks']}\n\n--------------------\n\n"
    )

    check_have_had_legit_initial_question = state.get("active_messages") or None

    if check_have_had_legit_initial_question is None:
        # New path that handles both cases:
        new_question = get_text(
            "\n\nInsufficient information to answer question. Please ask a new question.\nQ:"
        )
        if new_question == "!exit":
            return {} if callable(state.get("use_stream_ui")) else sys.exit(0)

        # If check_have_had_legit_initial_question is None then no need to thin list as we are still in first question phase.
        retrieved_new_question = query_tool.execute_extraction(
            new_question
        )  # shape Tuple[str, List[Tuple[Tuple[int, ...], str]]]

        composed_new_question = query_tool.format_context_for_llm_excerpts(
            retrieved_new_question
        )  # Uses excerpts enumerated by original indexing in text chunking process
        new_message = {"messages": HumanMessage(content=composed_new_question)}
        turn_index = {"turn_index": 1}
        context_chunks = {"context_chunks": retrieved_new_question[1]}
        length_chunks = {"length_last_added_chunks": len(retrieved_new_question[1])}
        return new_message | turn_index | context_chunks | length_chunks

    else:
        new_question = get_text("\n\nWhat is your follow up question?\nQ:")
        if new_question.strip() == "!exit":
            return {} if callable(state["use_stream_ui"]) else sys.exit(0)

        retrieved_new_question = query_tool.execute_extraction(
            new_question
        )  # shape Tuple[str, List[Tuple[Tuple[int, ...], str]]]

        query, list_to_be_thinned = retrieved_new_question

        thinned_context_list = ef.thin_list_of_redundant(
            list_to_be_thinned,
            state[
                "active_context_chunks"
            ],  # This needs to be from active_context_chunks
        )

        composed_new_question = query_tool.format_context_for_llm_excerpts(
            (query, thinned_context_list)
        )  # Uses excerpt indexing from original text chunking process
        new_message = {"messages": HumanMessage(content=composed_new_question)}
        turn_index = {"turn_index": 1}
        context_chunks = {"context_chunks": thinned_context_list}
        length_chunks = {"length_last_added_chunks": len(thinned_context_list)}

        # Consume user_input in Gradio mode so it's not reused accidentally
        if "user_input" in state:
            user_input = {"user_input": None}
            return (
                new_message | turn_index | context_chunks | length_chunks | user_input
            )

        return new_message | turn_index | context_chunks | length_chunks


# Building graph here


def ready_for_followup(state: AgentState) -> bool:
    """Are we ready to call the LLM again from follow_up_question?"""
    # CLI: always ready (we synchronously collected text via input())
    if not callable(state.get("use_stream_ui")):
        return True
    # Gradio: only ready if user_input is non-empty (i.e., the user submitted a follow-up)
    return bool((state.get("user_input") or "").strip())



def graph_build():
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("enter_question", enter_question)
    builder.add_node("llm_call", llm_call)
    builder.add_node("update_active_state_elements", update_active_state_elements)
    builder.add_node("follow_up_question", follow_up_question)

    # Entry
    builder.add_edge(START, "enter_question")
    builder.add_edge("enter_question", "llm_call")

    # After LLM: COMPLETUM vs IMPERFECTUM
    builder.add_conditional_edges(
        "llm_call",
        sufficient_information_routing_fn,
        {True: "update_active_state_elements", False: "follow_up_question"},
    )

    # After a good answer:
    builder.add_conditional_edges(
        "update_active_state_elements",
        follow_up_point,
        {True: "follow_up_question", False: END},
    )

    # For follow-up path: only call LLM if we actually have user text (Gradio) or always (CLI)
    builder.add_conditional_edges(
        "follow_up_question",
        ready_for_followup,
        {True: "llm_call", False: END},
    )

    return builder.compile()
