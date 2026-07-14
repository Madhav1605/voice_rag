import time
import socket
import threading

import ollama

from speech_to_text import speech_to_text
from voice_service import play_text_to_speech
from interrupt_service import watch_for_stop
from prompt_toolkit import prompt

from src.retriever import retrieve_content
from src.reranker import rerank_results
from src.answer_generator import generate_final_answer
from src.config import GROQ_API_KEY

# from src.config import QUERY_REWRITE_WORD_LIMIT
# from src.answer_generator import call_ollama
#
# SKIP_REWRITE_PREFIXES = (
#     "what","why","how","when","where","who","which",
#     "explain","describe","define","summarize","list","compare"
# )
#
# def rewrite_query(query):
#     word_count = len(query.split())
#     query_lower = query.lower().strip()
#     if (
#         word_count > QUERY_REWRITE_WORD_LIMIT
#         or query.endswith("?")
#         or query_lower.startswith(SKIP_REWRITE_PREFIXES)
#     ):
#         return query
#     try:
#         prompt = (
#             "Rewrite this short search query into a complete "
#             "question, staying as close as possible to the "
#             "original wording and meaning. Do NOT introduce new "
#             "topics. Reply with only the rewritten question.\n"
#             f"Query: \"{query}\""
#         )
#         return (call_ollama(prompt,temperature=0)).strip() or query
#     except Exception:
#         return query

REMINDER_THRESHOLD = 2
EXIT_THRESHOLD = 4

EXIT_WORDS = [
    "bye","goodbye","exit","quit",
    "stop","close assistant"
]

IGNORE_PHRASES = [
    "",".",",","...","uh","umm","hmm","okay","ok"
]

GREETING_WORDS = [
    "hello","hi","hey","good morning",
    "good afternoon","good evening","howdy"
]

GREETING_RESPONSE = (
    "Hello! I am your document assistant. "
    "Ask me anything about the loaded documents."
)

# SPEAK WITH INTERRUPT
def speak_with_interrupt(text):
    """TTS + S-key interrupt watcher run in parallel threads."""
    tts_thread = threading.Thread(
        target=play_text_to_speech,
        args=(text,)
    )
    tts_thread.start()
    time.sleep(0.1)
    stop_thread = threading.Thread(target=watch_for_stop)
    stop_thread.start()
    tts_thread.join()
    stop_thread.join()

# SYSTEM STATUS CHECK
def check_ollama_online():
    """Check if local Ollama service is reachable."""
    try:
        ollama.list()
        return True
    except Exception:
        return False

def check_internet_online():
    """Check if internet (Groq path) is reachable."""
    try:
        socket.create_connection(("8.8.8.8",53),timeout=2)
        return True
    except Exception:
        return False

def print_system_status():
    """Print full system readiness at startup."""
    ollama_ok = (check_ollama_online())
    internet_ok = (check_internet_online())
    print("\n" + "=" * 50)
    print("SYSTEM STATUS")
    print("=" * 50)
    print(f"Ollama  (offline LLM) : {'OK' if ollama_ok else 'OFFLINE'}")
    print(f"Internet (Groq path)  : {'OK' if internet_ok else 'OFFLINE'}")
    if not ollama_ok and not internet_ok:
        print("WARNING: Neither path available - answers will fail.")
    elif not ollama_ok:
        print("WARNING: Ollama offline - all queries will use Groq.")
    elif not internet_ok:
        print("WARNING: Internet offline - Groq fallback unavailable.")
    print("=" * 50)

# GREETING CHECK
def is_greeting(query):
    """Return True for greetings - bypass RAG pipeline."""
    query_lower = (query.lower().strip())
    return any(
        query_lower == word or query_lower.startswith(word)
        for word in GREETING_WORDS
    )

def confirm_or_edit_transcript(query):
    """
    Show what STT recognized.

    [Enter] -> accept
    [E]     -> edit (cursor starts at end of recognized text)
    [R]     -> record again

    Returns:
        (query, record_again)
    """
    print("\n Recognized:")
    print(query)

    print("\n[E] Edit   [R] Record Again   [Enter] Send")

    choice = input("> ").strip().lower()

    if choice == "r":
        return None, True

    if choice == "e":
        edited = prompt(
            "Edit: ",
            default=query
        ).strip()

        return edited if edited else query, False

    return query, False

# ANSWER ONE QUERY
def answer_query(query):
    """
    RAG pipeline: retrieve -> rerank -> generate.
    Query rewrite is disabled in voice mode (see comments above).
    Returns (answer_text, citations).
    """
    candidate_chunks = (retrieve_content(query))
    if not candidate_chunks:
        return "I couldn't find anything relevant in the documents.",[]

    rerank_result = (rerank_results(query,candidate_chunks))
    final_chunks = (rerank_result["chunks"])
    if not final_chunks:
        return "I couldn't find anything relevant in the documents.",[]

    result = (generate_final_answer(final_chunks,query))
    return result["answer"],result.get("citations",[])

# MAIN LOOP
def main():
    miss_count = 0

    print_system_status()

    welcome_message = (
        "Hello. I am your document assistant. "
        "Ask me anything about the loaded documents."
    )
    print("\nAssistant:",welcome_message)
    speak_with_interrupt(welcome_message)
    time.sleep(1)

    try:
        while True:
            print("\n🎤 Listening...")
            query = (speech_to_text())

            # No speech
            if not query:
                miss_count += 1
                print(
                    f"No speech detected "
                    f"(Attempt {miss_count}/{EXIT_THRESHOLD})"
                )
                if miss_count == REMINDER_THRESHOLD:
                    reminder = (
                        "I am still listening. "
                        "Please ask your question."
                    )
                    print("\nAssistant:",reminder)
                    speak_with_interrupt(reminder)
                elif miss_count >= EXIT_THRESHOLD:
                    goodbye = (
                        "It seems you are unavailable right now. "
                        "Goodbye."
                    )
                    print("\nAssistant:",goodbye)
                    speak_with_interrupt(goodbye)
                    break
                continue

            # Ignore filler
            if query.strip() in IGNORE_PHRASES:
                continue

            miss_count = 0

            # Transcript confirmation / edit / re-record
            query,record_again = (confirm_or_edit_transcript(query))
            if record_again:
                continue
            if not query:
                continue

            print(f"\nUser: {query}")

            # Exit
            if any(word in query.lower() for word in EXIT_WORDS):
                farewell = ("Goodbye. Have a great day.")
                print("\nAssistant:",farewell)
                speak_with_interrupt(farewell)
                break

            # Greeting bypass
            if is_greeting(query):
                print("\nAssistant:",GREETING_RESPONSE)
                speak_with_interrupt(GREETING_RESPONSE)
                time.sleep(1)
                continue

            # RAG pipeline
            try:
                print("\nThinking...")
                answer,citations = (answer_query(query))
                print("\nAssistant:",answer)
                if citations:
                    print("\nSources:")
                    for citation in citations:
                        print(
                            f"  - {citation['file_name']} "
                            f"(page {citation['page_numbers']})"
                        )
                speak_with_interrupt(answer)
            except Exception as error:
                print(f"\nError: {error}")
                speak_with_interrupt("Sorry, something went wrong.")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nExiting Voice Assistant...")
        speak_with_interrupt("Goodbye.")

if __name__ == "__main__":
    main()