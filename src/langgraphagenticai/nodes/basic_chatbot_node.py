from src.langgraphagenticai.state.state import State
import os
import streamlit as st
from langchain_groq import ChatGroq
from src.langgraphagenticai.ui.uiconfigfile import Config


class BasicChatbotNode:
    """Basic Chatbot node that invokes the LLM and handles decommissioned models."""
    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """Processes the input state and generates a chatbot response.

        If the configured Groq model has been decommissioned, this method will
        attempt to switch to the first working alternative from the UI config
        (`GROQ_MODEL_OPTIONS`) and retry the request.
        """
        try:
            return {"messages": self.llm.invoke(state["messages"])}
        except Exception as e:
            msg = str(e).lower()
            if "decommission" in msg or "model_decommissioned" in msg:
                try:
                    cfg = Config()
                    options = cfg.get_groq_model_options()
                except Exception:
                    options = []

                # Obtain API key from session state first, then env
                api_key = ""
                try:
                    api_key = st.session_state.get("GROQ_API_KEY", "")
                except Exception:
                    api_key = ""
                if not api_key:
                    api_key = os.environ.get("GROQ_API_KEY", "")

                # Try alternatives and collect errors for diagnostics
                alt_errors = []
                for alt in [m.strip() for m in options if m]:
                    try:
                        alt_llm = ChatGroq(api_key=api_key, model=alt)
                        # attempt the invoke with alternative model
                        result = alt_llm.invoke(state["messages"])
                        # switch to the working model for future calls
                        self.llm = alt_llm
                        st.info(f"Model switched to '{alt}' because previous model was decommissioned.")
                        return {"messages": result}
                    except Exception as ie:
                        alt_errors.append(f"{alt}: {ie}")
                        continue

                # If none of the alternatives worked, raise a clear error with details
                details = "; ".join(alt_errors) if alt_errors else "no details available"
                raise ValueError(
                    f"All configured Groq models failed after decommission. Tried: {', '.join(options)}. Details: {details}"
                ) from e

            # re-raise other exceptions
            raise

