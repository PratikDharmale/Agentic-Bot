import os
import streamlit as st
from langchain_groq import ChatGroq
from src.langgraphagenticai.ui.uiconfigfile import Config


class GroqLLM:
    def __init__(self, user_contols_input):
        self.user_controls_input = user_contols_input or {}

    def get_llm_model(self):
        try:
            # Prefer explicit UI-provided key, fall back to environment variable
            groq_api_key = self.user_controls_input.get("GROQ_API_KEY", "")
            env_api_key = os.environ.get("GROQ_API_KEY", "")
            api_key = groq_api_key or env_api_key

            selected_groq_model = self.user_controls_input.get("selected_groq_model")
            custom_model = (self.user_controls_input.get("custom_groq_model") or "").strip()
            # Prefer a custom model string if provided by the user
            selected_model = custom_model or selected_groq_model

            if not api_key:
                st.error("Please enter your GROQ API key in the sidebar or set the GROQ_API_KEY environment variable.")
                raise ValueError("GROQ API key is missing")

            try:
                llm = ChatGroq(api_key=api_key, model=selected_groq_model)
            except Exception as inner_e:
                # Surface provider errors with clearer messages and suggestions
                msg = str(inner_e).lower()
                if "invalid" in msg or "401" in msg:
                    raise ValueError("Graph setup failed: invalid GROQ API key (401). Check your key and try again.") from inner_e

                # Handle decommissioned model errors specifically
                if "decommission" in msg or "model_decommissioned" in msg:
                    try:
                        cfg = Config()
                        options = cfg.get_groq_model_options()
                    except Exception:
                        options = []

                    # Try alternate models from config (exclude the decommissioned selection)
                    suggestions = [m.strip() for m in options if m and m.strip() != (selected_model or "")]
                    alt_errors = []
                    for alt in suggestions:
                        try:
                            alt_llm = ChatGroq(api_key=api_key, model=alt)
                            st.info(f"Model '{selected_groq_model}' is decommissioned — switched to '{alt}'.")
                            return alt_llm
                        except Exception as ie:
                            alt_errors.append(f"{alt}: {ie}")
                            continue

                    suggestion_text = ", ".join(suggestions) if suggestions else "a supported model"
                    details = "; ".join(alt_errors) if alt_errors else "no details available"
                    raise ValueError(
                        f"Graph setup failed: the model '{selected_groq_model}' has been decommissioned. "
                        f"Tried alternatives: {suggestion_text} but none worked. Details: {details}. "
                        f"See https://console.groq.com/docs/deprecations"
                    ) from inner_e

                raise

        except Exception as e:
            raise ValueError(f"Error occurred while creating Groq LLM: {e}")

        return llm