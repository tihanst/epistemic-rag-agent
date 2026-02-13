from __future__ import annotations
from typing import Literal, List, Optional, Dict, Any, Callable
import asyncio


from together import Together
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

from .promp_templates import SYSTEM, PROMPT_START


# For now LLM is meant for 1 call. 'Memnory' can be passed in through messages and last message recived can come through message_received
# When using check if message.received is None (then can use, otherwise need new one).
class LLM:
    """
    Class for interacting with the LLM endpoint
    """

    def __init__(self, llm_endpoint: str, llm_api_key: str):
        """
        Initialize the LLM class.
        Args:
            llm_endpoint (str): The endpoint for the LLM.
            llm_api_key (str): The API key for the LLM.
        Returns:
            None
        """
        self.llm_endpoint = llm_endpoint
        self.llm_api_key = llm_api_key
        self.system_prompt = SYSTEM
        self.user_prompt_start = PROMPT_START
        self.user_full_prompt: Optional[str] = None
        self.message_received: Optional[ToolMessage | AIMessage] = (
            None  # str if an AIMessage content, dict if a ToolMessage
        )
        self.client = Together(api_key=self.llm_api_key)

    def call_llm_direct(
        self,
        prompt: str,
        stream: bool = True,
        temperature: float = 0.7,
        reasoning_effort: Literal["low", "medium", "high"] = "low",
        top_p: float = 0.95,
        top_k: int = 50,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> None:
        """
        Call the LLM endpoint directly.
        Args:
            prompt (str): The prompt to send to the LLM.
            stream (bool): Whether to stream the response.
            temperature (float): The temperature to use for the LLM.
            reasoning_effort (Literal["low", "medium", "high"]): The reasoning effort to use for the LLM.
            top_p (float): The top_p to use for the LLM.
            top_k (int): The top_k to use for the LLM.
            presence_penalty (float): The presence_penalty to use for the LLM.
            frequency_penalty (float): The frequency_penalty to use for the LLM.
        Returns:
            None
        """
        self.user_full_prompt: str = prompt
        response = self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=stream,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        accumulated_response: List[str] = []

        for token in response:
            if getattr(token, "choices", None) and getattr(
                token.choices[0], "delta", None
            ):
                piece = getattr(token.choices[0].delta, "content", None)
                if piece:
                    accumulated_response.append(piece)
                    print(piece, end="", flush=True)

        self.message_received = AIMessage(content="".join(accumulated_response))

    def _convert_messages_to_list_of_dicts(
        self, messages: List[AnyMessage]
    ) -> List[Dict[str, str]]:
        """
        Convert a list of messages to a list of dictionaries.
        Args:
            messages (List[AnyMessage]): The messages to convert.
        Returns:
            List[Dict[str, str]]: The list of dictionaries.

        """
        messages_dict: List[Dict[str, str]] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                messages_dict.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                messages_dict.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                messages_dict.append({"role": "assistant", "content": message.content})
            elif isinstance(
                message, ToolMessage
            ):  # This will need to be fixed as tool content different
                messages_dict.append({"role": "tool", "content": message.content})
            else:
                raise ValueError(f"Unknown message type: {type(message)}")
        return messages_dict

    # Turned into async as the node that calls it must be async because grdio llm call version is async
    async def call_agentic_llm(
        self,
        messages: List[AnyMessage],
        stream: bool = True,
        temperature: float = 0.7,
        reasoning_effort: Literal["low", "medium", "high"] = "low",
        top_p: float = 0.95,
        top_k: int = 50,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> AnyMessage:
        """
        Asynchronous call of the LLM endpoint with a list of messages.
        Args:
            messages (List[AnyMessage]): The messages to send to the LLM.
            stream (bool): Whether to stream the response.
            temperature (float): The temperature to use for the LLM.
            reasoning_effort (Literal["low", "medium", "high"]): The reasoning effort to use for the LLM.
            top_p (float): The top_p to use for the LLM.
            top_k (int): The top_k to use for the LLM.
            presence_penalty (float): The presence_penalty to use for the LLM.
            frequency_penalty (float): The frequency_penalty to use for the LLM.
        Returns:
            AnyMessage: The response from the LLM.

        """
        formatted_messages: List[Dict[str, str]] = (
            self._convert_messages_to_list_of_dicts(messages)
        )
        response = self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=formatted_messages,
            stream=stream,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        # Buffers
        text_parts: List[str] = []
        tool_calls: dict[int, Dict[str, Any]] = {}

        for chunk in response:
            # guard against empty/malformed chunks
            if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            # Non-tool call response text
            if getattr(delta, "content", None) is not None:
                text_parts.append(delta.content)
                print(delta.content, end="", flush=True)

            # Tool calls where arguments arrive in pieces; merge by index
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    i = tc.index
                    buf = tool_calls.setdefault(
                        i,
                        {
                            "id": "",
                            "type": "",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if getattr(tc, "id", None):
                        buf["id"] = tc.id
                    if getattr(tc, "type", None):
                        buf["type"] = tc.type
                    fn = getattr(tc, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            buf["function"]["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            buf["function"]["arguments"] += fn.arguments

        # Decide which message to construct
        if tool_calls:
            # Note that if multiple tool calls are possible, iterate and return a list instead. This is just one
            idx = min(tool_calls.keys())
            tc = tool_calls[idx]
            tc = ToolMessage(
                content=tc["function"]["arguments"],
                tool_call_id=tc["id"] or "",  # required by LangChain ToolMessage
                name=tc["function"]["name"] or None,
            )
            self.message_received = tc
            return tc

        full_text = "".join(text_parts)
        ai_msg = AIMessage(content=full_text)
        self.message_received = ai_msg
        return ai_msg

    async def call_front_end_aware_agentic_llm(
        self,
        messages: List[AnyMessage],
        stream: bool = True,
        temperature: float = 0.7,
        reasoning_effort: Literal["low", "medium", "high"] = "low",
        top_p: float = 0.95,
        top_k: int = 50,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        on_delta: Optional[Callable[[str], None]] = None,  # stream callback to UI
    ) -> AnyMessage:
        """
        Asynchronous call of the LLM endpoint with a list of messages, with awareness of front-end.
        Args:
            messages (List[AnyMessage]): The messages to send to the LLM.
            stream (bool): Whether to stream the response.
            temperature (float): The temperature to use for the LLM.
            reasoning_effort (Literal["low", "medium", "high"]): The reasoning effort to use for the LLM.
            top_p (float): The top_p to use for the LLM.
            top_k (int): The top_k to use for the LLM.
            presence_penalty (float): The presence_penalty to use for the LLM.
            frequency_penalty (float): The frequency_penalty to use for the LLM.
            on_delta (Optional[Callable[[str], None]]): The callback to invoke when a delta is received to notify front-end.
        Returns:
            AnyMessage: The response from the LLM.
        """
        formatted_messages: List[Dict[str, str]] = (
            self._convert_messages_to_list_of_dicts(messages)
        )

        # Accumulators/Buffers live in the outer scope, the worker thread mutates them.
        text_parts: List[str] = []
        tool_calls: dict[int, Dict[str, Any]] = {}
        final_msg_holder: List[
            AnyMessage
        ] = []  # single-element list to bring result back out of the thread

        def _runner():
            response = self.client.chat.completions.create(
                model=self.llm_endpoint,
                messages=formatted_messages,
                stream=stream,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            for chunk in response:
                # guard against empty/malformed chunks
                if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                # Non-tool call response text
                # Stream assistant text response to front end UI
                if getattr(delta, "content", None) is not None:
                    piece = delta.content
                    text_parts.append(piece)
                    if on_delta is not None:
                        # Safe to invoke from this thread; UI adapter should be thread-safe
                        on_delta(piece)

                # Tool calls where arguments arrive in pieces; merge by index
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        i = tc.index
                        buf = tool_calls.setdefault(
                            i,
                            {
                                "id": "",
                                "type": "",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if getattr(tc, "id", None):
                            buf["id"] = tc.id
                        if getattr(tc, "type", None):
                            buf["type"] = tc.type
                        fn = getattr(tc, "function", None)
                        if fn:
                            if getattr(fn, "name", None):
                                buf["function"]["name"] += fn.name
                            if getattr(fn, "arguments", None):
                                buf["function"]["arguments"] += fn.arguments

            # Decide which message to construct
            if tool_calls:
                # Note that if multiple tool calls are possible, iterate and return a list instead. This is just one
                idx = min(tool_calls.keys())
                tc = tool_calls[idx]
                msg = ToolMessage(
                    content=tc["function"]["arguments"],
                    tool_call_id=tc["id"] or "",  # required by LangChain ToolMessage
                    name=tc["function"]["name"] or None,
                )
                final_msg_holder.append(msg)
            else:
                full_text = "".join(text_parts)
                msg = AIMessage(content=full_text)
                final_msg_holder.append(msg)

        # Run the blocking stream off the event loop
        await asyncio.to_thread(_runner)

        # Retrieve and record the final message
        final_msg: AnyMessage = final_msg_holder[0]
        if isinstance(final_msg, ToolMessage):
            self.message_received = final_msg
            return final_msg
        elif isinstance(final_msg, AIMessage):
            self.message_received = final_msg
            return final_msg

        raise ValueError(f"Unexpected message type: {type(final_msg)}")
