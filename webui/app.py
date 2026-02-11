"""
Gradio web interface for Jetson Speech.

Provides a user-friendly interface for TTS and STT operations.
"""

from typing import Any

try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore


def create_gradio_app():
    """Create the Gradio application."""
    if gr is None:
        raise ImportError("Gradio not installed. Install with: pip install jetson-speech[webui]")

    from jetson_speech import __version__
    from jetson_speech.core.audio import audio_to_bytes
    from jetson_speech.core.engine import Engine
    from jetson_speech.stt.registry import list_stt_backends
    from jetson_speech.tts.registry import list_tts_backends

    # Global engine instance
    engine = Engine()

    # === TTS Tab ===

    def load_tts_backend(backend_name: str, model_size: str) -> str:
        """Load TTS backend."""
        try:
            engine.load_tts_backend(backend_name, model_size=model_size)
            return f"Loaded {backend_name} ({model_size})"
        except Exception as e:
            return f"Error: {e}"

    def get_tts_voices() -> list[str]:
        """Get available voices."""
        if not engine.get_tts_info().get("loaded"):
            return ["(Load backend first)"]
        voices = engine.get_tts_voices()
        return [v.id for v in voices]

    def synthesize_speech(
        text: str,
        voice: str,
        language: str,
        temperature: float,
    ) -> tuple[Any, str]:
        """Synthesize speech from text."""
        if not engine.get_tts_info().get("loaded"):
            return None, "Error: No TTS backend loaded"

        if not text.strip():
            return None, "Error: No text provided"

        try:
            result = engine.synthesize(
                text=text,
                voice=voice,
                language=language,
                temperature=temperature,
            )

            # Return audio as (sample_rate, audio_data) tuple for Gradio
            return (result.sample_rate, result.audio), f"Duration: {result.duration:.2f}s"
        except Exception as e:
            return None, f"Error: {e}"

    # === STT Tab ===

    def load_stt_backend(backend_name: str, model_size: str) -> str:
        """Load STT backend."""
        try:
            engine.load_stt_backend(backend_name, model_size=model_size)
            return f"Loaded {backend_name} ({model_size})"
        except Exception as e:
            return f"Error: {e}"

    def transcribe_audio(audio: tuple[int, Any] | str, language: str | None) -> str:
        """Transcribe audio to text."""
        if not engine.get_stt_info().get("loaded"):
            return "Error: No STT backend loaded"

        try:
            if isinstance(audio, tuple):
                sample_rate, audio_data = audio
                result = engine.transcribe(audio_data, sample_rate, language or None)
            else:
                result = engine.transcribe(audio, language=language or None)

            return f"{result.text}\n\n[Language: {result.language} | Duration: {result.duration:.2f}s]"
        except Exception as e:
            return f"Error: {e}"

    # === Build Interface ===

    with gr.Blocks(title=f"Jetson Speech v{__version__}") as app:
        gr.Markdown(f"# Jetson Speech v{__version__}")
        gr.Markdown("Modular TTS + STT server for Jetson/edge devices")

        with gr.Tabs():
            # TTS Tab
            with gr.TabItem("Text-to-Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        tts_backend = gr.Dropdown(
                            choices=[b["name"] for b in list_tts_backends()],
                            value="qwen",
                            label="Backend",
                        )
                        tts_model_size = gr.Dropdown(
                            choices=["0.6B", "1.7B"],
                            value="0.6B",
                            label="Model Size",
                        )
                        tts_load_btn = gr.Button("Load Backend")
                        tts_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="Text",
                            placeholder="Enter text to synthesize...",
                            lines=5,
                        )
                        with gr.Row():
                            tts_voice = gr.Dropdown(
                                choices=["ryan", "serena", "vivian"],
                                value="ryan",
                                label="Voice",
                            )
                            tts_language = gr.Dropdown(
                                choices=["English", "Chinese", "Japanese", "Korean", "French", "German", "Spanish"],
                                value="English",
                                label="Language",
                            )
                        tts_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Temperature",
                        )
                        tts_generate_btn = gr.Button("Generate Speech", variant="primary")

                tts_audio = gr.Audio(label="Generated Audio", type="numpy")
                tts_info = gr.Textbox(label="Info", interactive=False)

                # TTS event handlers
                tts_load_btn.click(
                    fn=load_tts_backend,
                    inputs=[tts_backend, tts_model_size],
                    outputs=[tts_status],
                ).then(
                    fn=lambda: gr.Dropdown(choices=get_tts_voices()),
                    outputs=[tts_voice],
                )

                tts_generate_btn.click(
                    fn=synthesize_speech,
                    inputs=[tts_text, tts_voice, tts_language, tts_temperature],
                    outputs=[tts_audio, tts_info],
                )

            # STT Tab
            with gr.TabItem("Speech-to-Text"):
                with gr.Row():
                    with gr.Column(scale=1):
                        stt_backend = gr.Dropdown(
                            choices=[b["name"] for b in list_stt_backends()],
                            value="whisper",
                            label="Backend",
                        )
                        stt_model_size = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                            value="base",
                            label="Model Size",
                        )
                        stt_load_btn = gr.Button("Load Backend")
                        stt_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=2):
                        stt_audio = gr.Audio(
                            label="Audio Input",
                            type="numpy",
                            sources=["upload", "microphone"],
                        )
                        stt_language = gr.Dropdown(
                            choices=["", "en", "zh", "ja", "ko", "fr", "de", "es"],
                            value="",
                            label="Language (empty = auto-detect)",
                        )
                        stt_transcribe_btn = gr.Button("Transcribe", variant="primary")

                stt_result = gr.Textbox(label="Transcription", lines=5, interactive=False)

                # STT event handlers
                stt_load_btn.click(
                    fn=load_stt_backend,
                    inputs=[stt_backend, stt_model_size],
                    outputs=[stt_status],
                )

                stt_transcribe_btn.click(
                    fn=transcribe_audio,
                    inputs=[stt_audio, stt_language],
                    outputs=[stt_result],
                )

            # Info Tab
            with gr.TabItem("Info"):
                gr.Markdown("## System Information")

                def get_system_info() -> str:
                    from jetson_speech.config import get_jetson_power_mode, is_jetson

                    info = []
                    info.append(f"**Version:** {__version__}")
                    info.append(f"**Jetson Device:** {'Yes' if is_jetson() else 'No'}")

                    power_mode = get_jetson_power_mode()
                    if power_mode:
                        info.append(f"**Power Mode:** {power_mode}")

                    try:
                        import torch

                        info.append(f"**CUDA Available:** {'Yes' if torch.cuda.is_available() else 'No'}")
                        if torch.cuda.is_available():
                            info.append(f"**GPU:** {torch.cuda.get_device_name(0)}")
                    except ImportError:
                        info.append("**CUDA:** PyTorch not installed")

                    info.append("\n### TTS Backends")
                    for b in list_tts_backends():
                        info.append(f"- {b['name']}")

                    info.append("\n### STT Backends")
                    for b in list_stt_backends():
                        info.append(f"- {b['name']}")

                    return "\n".join(info)

                info_display = gr.Markdown(get_system_info())
                refresh_btn = gr.Button("Refresh")
                refresh_btn.click(fn=get_system_info, outputs=[info_display])

    return app


def launch_webui(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
):
    """Launch the Gradio web UI."""
    app = create_gradio_app()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
    )


if __name__ == "__main__":
    launch_webui()
