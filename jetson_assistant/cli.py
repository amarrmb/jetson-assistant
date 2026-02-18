"""
Command-line interface for Jetson Assistant.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="jetson-assistant",
    help="Modular TTS + STT server for Jetson/edge devices",
    no_args_is_help=True,
)

console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    webui: bool = typer.Option(False, "--webui", help="Enable Gradio web UI"),
    preload_tts: str = typer.Option(None, "--preload-tts", help="Preload TTS backend on startup (e.g., 'qwen')"),
    preload_stt: str = typer.Option(None, "--preload-stt", help="Preload STT backend on startup (e.g., 'whisper')"),
    warmup: bool = typer.Option(True, "--warmup/--no-warmup", help="Warmup models after loading"),
):
    """Start the speech server."""
    import os

    # Pass preload config via environment variables
    if preload_tts:
        os.environ["JETSON_ASSISTANT_PRELOAD_TTS"] = preload_tts
    if preload_stt:
        os.environ["JETSON_ASSISTANT_PRELOAD_STT"] = preload_stt
    if warmup:
        os.environ["JETSON_ASSISTANT_WARMUP"] = "1"

    from jetson_assistant.server.app import run_server

    console.print(f"[green]Starting server on {host}:{port}[/green]")

    if preload_tts:
        console.print(f"[dim]Preloading TTS: {preload_tts}[/dim]")
    if preload_stt:
        console.print(f"[dim]Preloading STT: {preload_stt}[/dim]")

    if webui:
        console.print("[blue]Web UI enabled at /ui[/blue]")

    run_server(host=host, port=port, reload=reload, enable_webui=webui)


@app.command()
def tts(
    text: str = typer.Argument(None, help="Text to synthesize"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Input file (txt, pdf, docx, md)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output WAV file"),
    backend: str = typer.Option("qwen", "--backend", "-b", help="TTS backend to use"),
    voice: str = typer.Option("ryan", "--voice", "-v", help="Voice to use"),
    language: str = typer.Option("English", "--language", "-l", help="Language"),
    model_size: str = typer.Option("0.6B", "--model", "-m", help="Model size"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream playback"),
    play: bool = typer.Option(False, "--play", help="Play after saving"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress"),
):
    """Synthesize speech from text."""
    from jetson_assistant.core.audio import play_audio, save_audio
    from jetson_assistant.core.engine import Engine
    from jetson_assistant.core.text import extract_text

    if not text and not file:
        console.print("[red]Error: Either text or --file is required[/red]")
        raise typer.Exit(1)

    # Get text
    if file:
        if not file.exists():
            console.print(f"[red]Error: File not found: {file}[/red]")
            raise typer.Exit(1)
        text = extract_text(file)
        if not quiet:
            console.print(f"[dim]Extracted {len(text)} characters from {file}[/dim]")

    # Initialize engine
    engine = Engine()

    if not quiet:
        console.print(f"[dim]Loading {backend} backend ({model_size})...[/dim]")

    engine.load_tts_backend(backend, model_size=model_size)

    # Synthesize
    if stream:
        if not quiet:
            console.print("[dim]Streaming...[/dim]")

        result = engine.say(text, voice=voice, language=language, stream=True)

        if output and result:
            save_audio(result.audio, result.sample_rate, output)
            if not quiet:
                console.print(f"[green]Saved: {output}[/green]")
    else:
        if output:
            result = engine.synthesize_file(file, output=output, voice=voice, language=language) if file else None
            if not result:
                result = engine.synthesize(text, voice=voice, language=language)
                save_audio(result.audio, result.sample_rate, output)

            if not quiet:
                console.print(f"[green]Saved: {output} ({result.duration:.1f}s)[/green]")

            if play:
                play_audio(result.audio, result.sample_rate)
        else:
            result = engine.say(text, voice=voice, language=language, stream=False)

            if not quiet:
                console.print(f"[dim]Duration: {result.duration:.1f}s[/dim]")


@app.command()
def stt(
    audio: Path = typer.Argument(..., help="Audio file to transcribe"),
    backend: str = typer.Option("whisper", "--backend", "-b", help="STT backend to use"),
    model_size: str = typer.Option("base", "--model", "-m", help="Model size"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code"),
    timestamps: bool = typer.Option(False, "--timestamps", "-t", help="Show timestamps"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Transcribe audio to text."""
    import json

    from jetson_assistant.core.engine import Engine

    if not audio.exists():
        console.print(f"[red]Error: File not found: {audio}[/red]")
        raise typer.Exit(1)

    # Initialize engine
    engine = Engine()

    console.print(f"[dim]Loading {backend} backend ({model_size})...[/dim]")

    engine.load_stt_backend(backend, model_size=model_size)

    # Transcribe
    console.print(f"[dim]Transcribing {audio}...[/dim]")

    result = engine.transcribe(audio, language=language)

    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
    elif timestamps:
        console.print(f"\n[bold]Language:[/bold] {result.language}")
        console.print(f"[bold]Duration:[/bold] {result.duration:.1f}s\n")

        table = Table()
        table.add_column("Start", style="cyan")
        table.add_column("End", style="cyan")
        table.add_column("Text")

        for segment in result.segments:
            table.add_row(
                f"{segment.start:.2f}s",
                f"{segment.end:.2f}s",
                segment.text,
            )

        console.print(table)
    else:
        console.print(f"\n[bold]{result.text}[/bold]\n")
        console.print(f"[dim]Language: {result.language} | Duration: {result.duration:.1f}s[/dim]")


@app.command()
def benchmark(
    type_: str = typer.Argument("tts", help="Benchmark type (tts or stt)"),
    backends: Optional[str] = typer.Option(None, "--backends", "-b", help="Comma-separated backends"),
    text: str = typer.Option("The quick brown fox jumps over the lazy dog.", "--text", "-t", help="Text for TTS"),
    audio: Optional[Path] = typer.Option(None, "--audio", "-a", help="Audio file for STT"),
    iterations: int = typer.Option(3, "--iterations", "-n", help="Number of iterations"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Run benchmarks."""
    from jetson_assistant.benchmark.runner import run_benchmark

    # Parse backends
    backend_list = backends.split(",") if backends else None

    console.print(f"[bold]Running {type_} benchmark[/bold]")
    console.print(f"[dim]Iterations: {iterations}[/dim]")

    results = run_benchmark(
        benchmark_type=type_,
        backends=backend_list,
        text=text,
        audio_path=str(audio) if audio else None,
        iterations=iterations,
    )

    # Display results
    from jetson_assistant.benchmark.report import format_results

    report = format_results(results, format_type="markdown")
    console.print(report)

    # Save if output specified
    if output:
        output.write_text(report)
        console.print(f"[green]Saved: {output}[/green]")


@app.command()
def voices(
    backend: str = typer.Option("qwen", "--backend", "-b", help="TTS backend"),
):
    """List available voices."""
    from jetson_assistant.tts.registry import get_tts_backend

    try:
        tts = get_tts_backend(backend)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Get voices without loading model (some backends support this)
    voices = tts.get_voices()
    languages = tts.get_languages()

    console.print(f"\n[bold]{backend} TTS Voices[/bold]\n")

    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Gender")
    table.add_column("Description")

    for voice in voices:
        table.add_row(voice.id, voice.name, voice.gender, voice.description)

    console.print(table)

    console.print(f"\n[bold]Languages:[/bold] {', '.join(languages)}")


def _resolve_config(auto: bool = False, config: str = None) -> str:
    """Resolve the config file path.

    Priority:
      1. Explicit --config path (returned as-is)
      2. --auto: detect hardware tier and return its config path
      3. Default: "configs/thor.yaml"

    Returns:
        Path string to the YAML config file.
    """
    if config:
        return config
    if auto:
        from jetson_assistant.hardware import detect_tier

        tier = detect_tier()
        config_path = tier.config
        console.print(f"[dim]Auto-detected: {tier.value} tier -> {config_path}[/dim]")
        return config_path
    return "configs/thor.yaml"


@app.command()
def assistant(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config preset (e.g., configs/thor-sota.yaml)"),
    auto: bool = typer.Option(False, "--auto", help="Auto-detect hardware tier and select config"),
    server: bool = typer.Option(False, "--server", "-s", help="Connect to running server (FAST, recommended)"),
    server_port: int = typer.Option(8080, "--server-port", help="Server port when using --server"),
    local_llm: bool = typer.Option(False, "--local-llm", help="Use local LLM instead of server LLM (slower)"),
    wake_word: str = typer.Option("hey_jarvis", "--wake-word", "-w", help="Wake word to listen for"),
    tts_backend: str = typer.Option("piper", "--tts", help="TTS backend (piper, kokoro, or qwen)"),
    stt_backend: str = typer.Option("whisper", "--stt", help="STT backend (whisper, vllm, or nemotron)"),
    stt_model: str = typer.Option("base", "--stt-model", help="STT model size"),
    llm_backend: str = typer.Option("ollama", "--llm", "-l", help="LLM backend (ollama, vllm, openai, anthropic, simple)"),
    llm_model: str = typer.Option("llama3.2:3b", "--llm-model", "-m", help="LLM model name"),
    llm_host: Optional[str] = typer.Option(None, "--llm-host", help="LLM server URL (default: auto per backend)"),
    stt_host: Optional[str] = typer.Option(None, "--stt-host", help="STT server URL (for vllm backend, e.g. http://localhost:8002/v1)"),
    rag: str = typer.Option(None, "--rag", "-r", help="Enable RAG with collection name (e.g., 'dota2')"),
    voice: str = typer.Option("en_US-amy-medium", "--voice", "-v", help="TTS voice"),
    verbose: bool = typer.Option(False, "--verbose", help="Show timing info"),
    no_wake: bool = typer.Option(False, "--no-wake", help="Skip wake word (always listening)"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable LLM streaming (if pipelining causes issues)"),
    vision: bool = typer.Option(False, "--vision", help="Enable camera vision (use with VLM like llama3.2-vision)"),
    camera_device: int = typer.Option(0, "--camera-device", help="Camera device index"),
    watch_interval: float = typer.Option(5.0, "--watch-interval", help="Vision watch polling interval in seconds"),
    audio_device: Optional[int] = typer.Option(None, "--audio-device", help="Audio input device index (see 'python -m sounddevice')"),
    show_vision: bool = typer.Option(False, "--show-vision", help="Show live camera preview window (requires opencv-python)"),
    stream_vision: int = typer.Option(0, "--stream-vision", help="MJPEG stream port (e.g., 9090). Browse to http://<host>:<port>/"),
    aether_hub: Optional[str] = typer.Option(None, "--aether-hub", help="Aether Hub host[:port] for camera alerts (e.g., localhost:8000)"),
    aether_pin: str = typer.Option("", "--aether-pin", help="Aether Hub PIN for authentication"),
    camera_config: str = typer.Option("~/.assistant_cameras.json", "--camera-config", help="Path to camera config JSON file"),
    watch_cooldown: float = typer.Option(60.0, "--watch-cooldown", help="Seconds between repeated alerts on same camera"),
    knowledge: Optional[str] = typer.Option(None, "--knowledge", "-k", help="Knowledge base collection for lookup_info tool (e.g., 'personal', 'family')"),
    remote_camera_port: int = typer.Option(0, "--remote-camera-port", help="UDP port for remote camera (Aether SFU WebRTC stream, 0=disabled)"),
    external_tools: Optional[str] = typer.Option(None, "--external-tools", help="Comma-separated external tool module paths (e.g., 'reachy_tools,my_pkg.tools')"),
):
    """
    Run voice assistant (like Alexa/Google Home).

    RECOMMENDED: Use --server mode for fast responses!

    Example (with config preset):
        jetson-assistant assistant --config configs/thor-sota.yaml

    Example (config + CLI override):
        jetson-assistant assistant --config configs/thor-sota.yaml --voice am_adam

    Example (fast, server mode):
        # Terminal 1: Start server once
        jetson-assistant serve --port 8080

        # Terminal 2: Run assistant
        jetson-assistant assistant --server --llm-model phi3:mini

    Example (auto-detect hardware):
        jetson-assistant assistant --auto

    Example (slow, loads models each time):
        jetson-assistant assistant --llm ollama --llm-model llama3.2:3b
    """
    from jetson_assistant.assistant.core import AssistantConfig, VoiceAssistant

    console.print("[bold]Jetson Voice Assistant[/bold]\n")

    # Resolve config: explicit --config > --auto > default
    resolved_config = _resolve_config(
        auto=auto,
        config=str(config_file) if config_file is not None else None,
    )

    # Load YAML config preset
    yaml_config: dict = {}
    resolved_path = Path(resolved_config)
    if resolved_path.exists():
        yaml_config = AssistantConfig.from_yaml(str(resolved_path))
        console.print(f"[dim]Loaded config: {resolved_path}[/dim]")
    elif config_file is not None:
        # Explicit --config was given but file doesn't exist
        console.print(f"[red]Error: Config file not found: {config_file}[/red]")
        raise typer.Exit(1)

    # Determine effective values: CLI args override YAML, YAML overrides defaults.
    # We detect which CLI args were explicitly passed by checking against their defaults.
    # typer doesn't expose "was this arg provided", so we compare to the default values
    # defined above and only use CLI values that differ from defaults.
    _cli_defaults = {
        "server": False, "server_port": 8080, "local_llm": False,
        "wake_word": "hey_jarvis", "tts_backend": "piper", "stt_backend": "whisper",
        "stt_model": "base", "llm_backend": "ollama", "llm_model": "llama3.2:3b",
        "llm_host": None, "stt_host": None, "rag": None,
        "voice": "en_US-amy-medium", "verbose": False, "no_wake": False,
        "no_stream": False, "vision": False, "camera_device": 0,
        "watch_interval": 5.0, "audio_device": None, "show_vision": False,
        "stream_vision": 0, "aether_hub": None, "aether_pin": "",
        "camera_config": "~/.assistant_cameras.json", "watch_cooldown": 60.0,
        "knowledge": None, "remote_camera_port": 0,
        "external_tools": None,
    }
    _cli_locals = {
        "server": server, "server_port": server_port, "local_llm": local_llm,
        "wake_word": wake_word, "tts_backend": tts_backend, "stt_backend": stt_backend,
        "stt_model": stt_model, "llm_backend": llm_backend, "llm_model": llm_model,
        "llm_host": llm_host, "stt_host": stt_host, "rag": rag,
        "voice": voice, "verbose": verbose, "no_wake": no_wake,
        "no_stream": no_stream, "vision": vision, "camera_device": camera_device,
        "watch_interval": watch_interval, "audio_device": audio_device,
        "show_vision": show_vision, "stream_vision": stream_vision,
        "aether_hub": aether_hub, "aether_pin": aether_pin,
        "camera_config": camera_config, "watch_cooldown": watch_cooldown,
        "knowledge": knowledge, "remote_camera_port": remote_camera_port,
        "external_tools": external_tools,
    }
    # CLI overrides: only values that differ from their defaults
    cli_overrides = {k: v for k, v in _cli_locals.items() if v != _cli_defaults[k]}

    def _resolve(cli_name: str, config_name: str = None):
        """Get value: CLI override > YAML > default."""
        cfg_key = config_name or cli_name
        if cli_name in cli_overrides:
            return cli_overrides[cli_name]
        if cfg_key in yaml_config:
            return yaml_config[cfg_key]
        return _cli_locals[cli_name]

    # Resolve all values
    server = _resolve("server", "use_server")
    server_port = _resolve("server_port")
    local_llm = _resolve("local_llm")
    wake_word = _resolve("wake_word")
    tts_backend = _resolve("tts_backend")
    stt_backend = _resolve("stt_backend")
    stt_model = _resolve("stt_model")
    llm_backend = _resolve("llm_backend")
    llm_model = _resolve("llm_model")
    llm_host = _resolve("llm_host")
    stt_host = _resolve("stt_host")
    rag = _resolve("rag", "rag_collection")
    voice = _resolve("voice", "tts_voice")
    verbose = _resolve("verbose")
    no_wake = _resolve("no_wake")
    no_stream = _resolve("no_stream")
    vision = _resolve("vision", "vision_enabled")
    camera_device = _resolve("camera_device")
    watch_interval = _resolve("watch_interval", "watch_poll_interval")
    audio_device = _resolve("audio_device", "audio_input_device")
    show_vision = _resolve("show_vision")
    stream_vision = _resolve("stream_vision", "stream_vision_port")
    aether_hub = _resolve("aether_hub")
    aether_pin = _resolve("aether_pin")
    camera_config = _resolve("camera_config", "camera_config_path")
    watch_cooldown = _resolve("watch_cooldown")
    knowledge = _resolve("knowledge", "knowledge_collection")
    remote_camera_port = _resolve("remote_camera_port")
    external_tools_raw = _resolve("external_tools")

    # Parse external_tools: CLI is comma-separated string, YAML is list
    external_tools_list = None
    if external_tools_raw:
        if isinstance(external_tools_raw, str):
            external_tools_list = [s.strip() for s in external_tools_raw.split(",") if s.strip()]
        elif isinstance(external_tools_raw, list):
            external_tools_list = external_tools_raw

    # Handle wake_word_backend from YAML (no direct CLI flag for this)
    wake_word_backend = yaml_config.get("wake_word_backend", "openwakeword")

    # Handle stream_llm from YAML (CLI uses --no-stream, YAML uses stream_llm: false)
    if "stream_llm" in yaml_config and "no_stream" not in cli_overrides:
        no_stream = not yaml_config["stream_llm"]

    # Resolve mode early — PersonaPlex mode skips pipeline loading entirely
    mode = yaml_config.get("mode", "pipeline")

    engine = None

    if mode == "personaplex":
        console.print("[green]PersonaPlex mode:[/green] Full-duplex speech-to-speech")
        console.print("[dim]Skipping pipeline (TTS/STT/LLM) — PersonaPlex handles everything[/dim]\n")
    elif server:
        # Server mode - connect to running server (FAST)
        console.print(f"[green]Server mode:[/green] Connecting to localhost:{server_port}")
        console.print("[dim]Make sure server is running: jetson-assistant serve[/dim]\n")
    else:
        # In-process mode - load models (SLOW)
        console.print("[yellow]In-process mode:[/yellow] Loading models locally")
        console.print("[dim]Tip: Use --server for faster responses![/dim]\n")

        from jetson_assistant.core.engine import Engine

        engine = Engine()

        console.print(f"[dim]Loading TTS ({tts_backend})...[/dim]")
        engine.load_tts_backend(tts_backend)

        if stt_backend not in ("nemotron", "vllm"):
            console.print(f"[dim]Loading STT ({stt_backend}/{stt_model})...[/dim]")
            engine.load_stt_backend(stt_backend, model_size=stt_model, device="cpu")

    # Vision tip
    if vision and llm_model == "llama3.2:3b" and llm_backend == "ollama":
        console.print("[yellow]Tip:[/yellow] For vision, use a VLM model like --llm-model llama3.2-vision:11b")

    # Set default LLM host based on backend
    if llm_host is None:
        if llm_backend == "vllm":
            llm_host = "http://localhost:8000/v1"
        else:
            llm_host = "http://localhost:11434"

    # Set default STT host for vllm backend
    if stt_host is None and stt_backend == "vllm":
        stt_host = "http://localhost:8002/v1"

    # Parse Aether Hub host:port
    aether_hub_host = None
    aether_hub_port = 8000
    if aether_hub:
        if ":" in aether_hub:
            parts = aether_hub.rsplit(":", 1)
            aether_hub_host = parts[0]
            aether_hub_port = int(parts[1])
        else:
            aether_hub_host = aether_hub

    # Handle no_wake: if YAML set wake_word_backend=energy, treat as no_wake
    if no_wake or wake_word_backend == "energy":
        no_wake = True
        wake_word_backend = "energy"

    # Configure assistant
    config = AssistantConfig(
        use_server=server,
        server_port=server_port,
        use_server_llm=not local_llm and llm_backend not in ("vllm",),
        wake_word=wake_word if not no_wake else "energy",
        wake_word_backend="openwakeword" if not no_wake else "energy",
        llm_backend=llm_backend,
        llm_model=llm_model,
        llm_host=llm_host,
        rag_collection=rag,
        tts_backend=tts_backend,
        tts_voice=voice,
        stt_backend=stt_backend,
        stt_model=stt_model,
        stt_host=stt_host,
        verbose=verbose,
        stream_llm=not no_stream,
        vision_enabled=vision or show_vision or stream_vision > 0,
        camera_device=camera_device,
        watch_poll_interval=watch_interval,
        audio_input_device=audio_device,
        show_vision=show_vision,
        stream_vision_port=stream_vision,
        camera_config_path=camera_config,
        watch_cooldown=watch_cooldown,
        aether_hub_host=aether_hub_host,
        aether_hub_port=aether_hub_port,
        aether_pin=aether_pin,
        knowledge_collection=knowledge,
        remote_camera_port=remote_camera_port,
        external_tools=external_tools_list,
        # PersonaPlex fields (from YAML config)
        mode=mode,
        personaplex_audio=yaml_config.get("personaplex_audio", "browser"),
        personaplex_port=yaml_config.get("personaplex_port", 8998),
        personaplex_ssl_dir=yaml_config.get("personaplex_ssl_dir"),
        personaplex_voice=yaml_config.get("personaplex_voice", "NATF1"),
        personaplex_text_prompt=yaml_config.get("personaplex_text_prompt"),
        personaplex_tool_detection=yaml_config.get("personaplex_tool_detection", "prompt"),
        personaplex_dir=yaml_config.get("personaplex_dir", "~/personaplex"),
        personaplex_cpu_cores=yaml_config.get("personaplex_cpu_cores", "4-13"),
        personaplex_hf_repo=yaml_config.get("personaplex_hf_repo", "nvidia/personaplex-7b-v1"),
    )

    # Create and run assistant
    try:
        assistant_instance = VoiceAssistant(engine, config)
    except ConnectionError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]Start the server first:[/yellow]")
        console.print("  jetson-assistant serve --port 8080")
        raise typer.Exit(1)

    console.print(f"\n[green]Assistant ready![/green]")
    if server:
        console.print("[bold]Mode: Server (fast)[/bold]")
    if no_wake:
        console.print("Wake: Always listening (no wake word)")
    else:
        console.print(f"Wake word: '{wake_word.replace('_', ' ')}'")
    console.print(f"LLM: {llm_backend}/{llm_model}")
    console.print(f"STT: {stt_backend}" + (f" ({stt_host})" if stt_host else f"/{stt_model}"))
    console.print(f"TTS voice: {voice}")
    if vision or show_vision or stream_vision > 0:
        console.print(f"Vision: enabled (camera {camera_device})")
    if show_vision:
        console.print("Vision preview: OpenCV window")
    if stream_vision > 0:
        console.print(f"Vision stream: http://0.0.0.0:{stream_vision}/")
    if audio_device is not None:
        console.print(f"Audio input: device {audio_device}")
    if knowledge:
        console.print(f"Knowledge base: {knowledge}")
    if remote_camera_port > 0:
        console.print(f"Remote camera: UDP port {remote_camera_port}")
    if aether_hub_host:
        console.print(f"Aether Hub: {aether_hub_host}:{aether_hub_port}")
    if external_tools_list:
        console.print(f"External tools: {', '.join(external_tools_list)}")
    console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        assistant_instance.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")


@app.command()
def info():
    """Show system information."""
    from jetson_assistant import __version__
    from jetson_assistant.config import get_jetson_power_mode, is_jetson
    from jetson_assistant.stt.registry import list_stt_backends
    from jetson_assistant.tts.registry import list_tts_backends

    console.print(f"\n[bold]Jetson Assistant v{__version__}[/bold]\n")

    # System info
    table = Table(title="System")
    table.add_column("Property")
    table.add_column("Value")

    table.add_row("Jetson Device", "Yes" if is_jetson() else "No")
    power_mode = get_jetson_power_mode()
    if power_mode:
        table.add_row("Power Mode", power_mode)

    # Check CUDA
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        table.add_row("CUDA Available", "Yes" if cuda_available else "No")
        if cuda_available:
            table.add_row("CUDA Device", torch.cuda.get_device_name(0))
    except ImportError:
        table.add_row("CUDA Available", "PyTorch not installed")

    console.print(table)

    # TTS backends
    console.print("\n[bold]TTS Backends[/bold]")
    tts_backends = list_tts_backends()
    if tts_backends:
        for b in tts_backends:
            streaming = "[green]streaming[/green]" if b.get("supports_streaming") else ""
            console.print(f"  - {b['name']} {streaming}")
    else:
        console.print("  [dim]None available[/dim]")

    # STT backends
    console.print("\n[bold]STT Backends[/bold]")
    stt_backends = list_stt_backends()
    if stt_backends:
        for b in stt_backends:
            streaming = "[green]streaming[/green]" if b.get("supports_streaming") else ""
            console.print(f"  - {b['name']} {streaming}")
    else:
        console.print("  [dim]None available[/dim]")

    console.print()


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
