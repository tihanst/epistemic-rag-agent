from time import perf_counter
import time
import sys
import os
import threading
from langchain_core.messages import SystemMessage
import asyncio
import traceback
import gradio as gr
from typing import Optional

import psycopg

import src.rag_engine.query_tool as qt
import src.LLM.llm_interface as llm
from src.Config.config import Settings
import src.Agent.agent_interface as ai
from src.LLM.promp_templates import SYSTEM
from src.Config.config import Settings


def single_queries():
    print("Hello welcome to Roman History!")
    settings = Settings()
    query_tool = qt.RAGPipeline()
    model = llm.LLM(
        llm_api_key=settings.LLM_API_KEY, llm_endpoint=settings.LLM_ENDPOINT
    )
    while True:
        query = input("Enter your question: ")
        start = perf_counter()
        if query.strip() == "!exit":
            break
        ans = query_tool.execute_extraction(query)
        ans_string = query_tool.format_context_for_llm_excerpts(ans)
        middle = perf_counter()
        print(f"Time taken for extraction: {middle - start}")
        model.call_llm_direct(ans_string)
        end = perf_counter()
        print(f"\n\nTime taken for LLM: {end - middle}\n\n")


def agentic_flow():
    print("\n\nWelcome to Roman History Agent!\n\n")

    graph = ai.graph_build()

    asyncio.run(
        graph.ainvoke(
            {
                "messages": [SystemMessage(content=SYSTEM)],
                "turn_index": 0,
                "active_messages": [SystemMessage(content=SYSTEM)],
                "context_chunks": [],
                "active_context_chunks": [],
                "use_stream_ui": None,
            }
        )
    )


def agentic_flow_stream_ui_5():
    """
    Gradio UI.
    """
    settings = Settings()
    print("\n\nWelcome to Roman History Agent!\n\n")

    graph = ai.graph_build()  # compiled runnable/graph with some async nodes

    # Cross-thread signal to request shutdown
    shutdown_event = threading.Event()

    with gr.Blocks(title="Roman History Agent") as demo:
        gr.Markdown(
            "# 🏛️ Welcome to Roman History Agent!",
            elem_id="app_title",
        )


        chat = gr.Chatbot(height="75vh", show_label=False, avatar_images=None)
        inp = gr.Textbox(placeholder="Ask about Roman history…", scale=1)

        lg_state = gr.State(
            {
                "messages": [SystemMessage(content=SYSTEM)],
                "turn_index": 0,
                "active_messages": [SystemMessage(content=SYSTEM)],
                "context_chunks": [],
                "active_context_chunks": [],
                "length_last_added_chunks": 0,
                "use_stream_ui": None,
                "user_input": None,
            }
        )
        

        # helper: close Gradio and exit the whole Python process
        def _shutdown_entire_app():
            """
            Close Gradio and exit the whole Python process.
            """
            try:
                demo.close()  # stop this Gradio server
            except Exception:
                pass
            os._exit(0)  # hard exit the interpreter (no dangling tasks)

        async def respond(
            user_text: str, history: list[dict[str, str]] | None, state: dict
        ):
            """
            Respond to user input and update the chat history.
            Args:
                user_text (str): The user's input.
                history (list[dict[str, str]] | None): The chat history.
                state (dict): The state of the chat.
            Returns:
                None

            """
            history = list(history or [])

            # --- Handle !exit BEFORE any graph call ---
            if user_text.strip() == "!exit":
                goodbye = "Thank you for using Roman History Agent!"
                history.append({"role": "assistant", "content": goodbye})
                state["terminated"] = True

                # Immediately update UI: disable the textbox so the user sees it's done
                yield history, state, gr.update(value="", interactive=False)
                shutdown_event.set()
                return

            try:
                # Add user message and empty assistant message
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": ""})
                idx = len(history) - 1

                q: "asyncio.Queue[str]" = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def on_delta(piece: str) -> None:
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, piece)
                    except RuntimeError:
                        pass

                # Inject per-turn fields
                state["use_stream_ui"] = on_delta
                state["user_input"] = user_text

                task = asyncio.create_task(graph.ainvoke(state))

                acc = ""
                # Timeout-polling stream; exits when graph is done and queue drained
                while True:
                    if task.done() and q.empty():
                        break
                    try:
                        piece = await asyncio.wait_for(q.get(), timeout=0.15)
                        acc += piece
                        history[idx] = {"role": "assistant", "content": acc}
                        # important: always yield (history, state)
                        yield history, state, gr.update(value="", interactive=True)
                    except asyncio.TimeoutError:
                        # nothing to stream right now;; keep waiting
                        pass

                new_state = await task
                # final yield for this turn; keep input enabled unless terminated flag is set
                yield (
                    history,
                    new_state,
                    gr.update(
                        value="", interactive=not new_state.get("terminated", False)
                    ),
                )

            except Exception:
                # on error, show a message and keep input enabled
                err = traceback.format_exc(limit=2)
                history.append({"role": "assistant", "content": "⚠️ An error occurred. Please try again."})
                # still return 3
                yield history, state, gr.update(value="", interactive=True)

        inp.submit(
            fn=respond,
            inputs=[inp, chat, lg_state],
            outputs=[chat, lg_state, inp],
        )

    demo.queue()
    demo.launch(
        server_name=settings.GRADIO_SERVER_NAME,
        server_port=settings.GRADIO_SERVER_PORT,
        inbrowser=True,
        prevent_thread_lock=True,
        show_error=True,
    )

    # Wait until the handler signals shutdown (when user types !exit)
    shutdown_event.wait()


    # Now close the server from the main thread, then exit the process
    demo.close()
    sys.exit(0)


def wait_for_postgres(timeout_s: int = 60, sleep_s: float = 1.5) -> None:
    """
    Repeatedly attempts a connection to Postgres until it succeeds or times out.
    Args:
        timeout_s (int, optional): The maximum time to wait for Postgres to be ready, in seconds. Defaults to 60.
        sleep_s (float, optional): The time to sleep between connection attempts, in seconds. Defaults to 1.5.
    Returns:
        None
    """
    settings = Settings()
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    dsn = (
        f"host={settings.PG_HOST} port={settings.PG_PORT} dbname={settings.PG_DATABASE}"
    )
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(dsn, connect_timeout=2) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
            print("✅ Postgres is ready.")
            return
        except Exception as e:
            last_err = e
            print(f"⏳ Waiting for Postgres… ({e})")
            time.sleep(sleep_s)
    raise TimeoutError(f"Postgres not ready after {timeout_s}s") from last_err


def main():
    if len(sys.argv) == 1:
        print(
            "No argument provided. Choose either [single_queries] or [agentic_flow] or [agentic_flow use_stream_ui]\n\n"
        )
    elif sys.argv[1] == "single_queries":
        single_queries()
    elif sys.argv[1] == "agentic_flow":
        if len(sys.argv) > 2:
            if sys.argv[2] == "use_stream_ui":
                agentic_flow_stream_ui_5()  # Gradio path w/async inside
            else:
                print(
                    "Invalid argument provided. Choose either [single_queries] or [agentic_flow]\n\n"
                )
        else:
            agentic_flow()  # Terminal path now using async
    else:
        print(
            "Invalid argument provided. Choose either [single_queries] or [agentic_flow] or [agentic_flow use_stream_ui]\n\n"
        )


# From extraction_formatter as ef apply expand_and_merge_linear and then reduce_and_return (which will be list of merged extracted texts)
# That will be part of a tool call object that does the extraction
if __name__ == "__main__":
    wait_for_postgres()
    main()
    sys.exit(0)
