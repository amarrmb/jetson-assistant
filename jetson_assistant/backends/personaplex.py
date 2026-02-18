"""PersonaPlex 7B full-duplex speech-to-speech backend for jetson-assistant."""

import asyncio
import logging
import os
import sys
import time
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PersonaplexBackend:
    """Full-duplex speech-to-speech backend using PersonaPlex (Moshi 7B).

    Supports two audio modes:
    - browser: WebSocket + Opus codec, serves web UI on personaplex_port
    - local: Direct mic/speaker via AudioInput/AudioOutput

    Tool detection: parses text token stream for [tool commands] and dispatches
    to the jetson-assistant ToolRegistry.
    """

    def __init__(self, config):
        """Initialize backend (lazy — does not load model).

        Args:
            config: AssistantConfig with personaplex_* fields set.
        """
        self.config = config
        self._loaded = False
        self._running = False

        # Model components (set by load())
        self._mimi = None
        self._lm_gen = None
        self._text_tokenizer = None

        # Callbacks (set by set_callbacks())
        self._on_state_change: Optional[Callable] = None
        self._on_audio_chunk: Optional[Callable] = None
        self._tool_registry = None

        # State
        self._state = "idle"

    def is_loaded(self) -> bool:
        return self._loaded

    def set_callbacks(
        self,
        on_state_change: Optional[Callable] = None,
        on_audio_chunk: Optional[Callable] = None,
        tool_registry=None,
    ):
        """Set callbacks for state changes, audio chunks, and tool dispatch.

        Args:
            on_state_change: Called with (old_state, new_state) strings.
            on_audio_chunk: Called with (audio_np, sample_rate) during model speech.
            tool_registry: ToolRegistry for dispatching parsed tool commands.
        """
        self._on_state_change = on_state_change
        self._on_audio_chunk = on_audio_chunk
        self._tool_registry = tool_registry

    def load(self):
        """Load Moshi model, Mimi codec, apply FP8 quantization.

        This is expensive (~15s) and requires GPU. Call once at startup.
        """
        import torch
        from huggingface_hub import hf_hub_download

        personaplex_dir = os.path.expanduser(self.config.personaplex_dir)
        if personaplex_dir not in sys.path:
            sys.path.insert(0, personaplex_dir)

        from moshi.models import loaders, LMGen
        import sentencepiece

        device = torch.device("cuda")

        # Load Mimi (audio codec) — only one copy, skip other_mimi
        logger.info("Loading Mimi...")
        mimi_weight = hf_hub_download(self.config.personaplex_hf_repo, loaders.MIMI_NAME)
        self._mimi = loaders.get_mimi(mimi_weight, device)
        self._mimi = self._mimi.half()
        self._mimi.torch_compile_encoder_decoder = True
        self._mimi = torch.compile(self._mimi)
        logger.info("Mimi loaded (FP16 + compiled)")

        # Load text tokenizer
        tokenizer_path = hf_hub_download(self.config.personaplex_hf_repo, loaders.TEXT_TOKENIZER_NAME)
        self._text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        # Load Moshi LM
        logger.info("Loading Moshi LM...")
        moshi_weight = hf_hub_download(self.config.personaplex_hf_repo, loaders.MOSHI_NAME)
        lm = loaders.get_moshi_lm(moshi_weight, device=device)
        lm.eval()

        # Apply FP8 quantization
        logger.info("Applying FP8 quantization...")
        from fp8_quantize import quantize_model, free_bf16_inproj
        quantize_model(lm)
        logger.info("FP8 quantization complete")

        # Resolve voice prompt directory
        voice_prompt_dir = hf_hub_download(self.config.personaplex_hf_repo, "voices.tgz")
        voices_dir = Path(voice_prompt_dir).parent / "voices"
        if not voices_dir.exists():
            import tarfile
            with tarfile.open(voice_prompt_dir, "r:gz") as tar:
                tar.extractall(path=Path(voice_prompt_dir).parent)

        # Create LMGen
        self._lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self._mimi.frame_rate),
            sample_rate=self._mimi.sample_rate,
            device=device,
            frame_rate=self._mimi.frame_rate,
        )

        # Set voice prompt
        voice_file = voices_dir / f"{self.config.personaplex_voice}.pt"
        if voice_file.exists():
            self._lm_gen.load_voice_prompt_embeddings(str(voice_file))
        else:
            voice_wav = voices_dir / f"{self.config.personaplex_voice}.wav"
            if voice_wav.exists():
                self._lm_gen.load_voice_prompt(str(voice_wav))
            else:
                logger.warning("Voice prompt %s not found, using default", self.config.personaplex_voice)

        # Set text prompt
        if self.config.personaplex_text_prompt:
            prompt_text = self.config.personaplex_text_prompt.strip()
            if not prompt_text.startswith("<system>"):
                prompt_text = f"<system> {prompt_text} <system>"
            self._lm_gen.text_prompt_tokens = self._text_tokenizer.encode(prompt_text)

        # Setup streaming
        self._mimi.streaming_forever(1)
        self._lm_gen.streaming_forever(1)

        # Warmup
        logger.info("Warming up...")
        frame_size = int(self._mimi.sample_rate / self._mimi.frame_rate)
        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=torch.float16, device=device)
            codes = self._mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self._lm_gen.step(codes[:, :, c:c + 1])
                if tokens is not None:
                    _ = self._mimi.decode(tokens[:, 1:9])
        torch.cuda.synchronize()

        # Free bf16 copies after warmup
        free_bf16_inproj(lm)
        logger.info("PersonaPlex loaded. GPU memory: %.2f GB", torch.cuda.memory_allocated() / 1e9)

        # Pre-allocate pinned buffer for DtoH
        self._pinned_pcm = torch.empty(1920, dtype=torch.float32, pin_memory=True)
        self._frame_size = frame_size
        self._device = device
        self._loaded = True

    def _set_state(self, new_state: str):
        """Update state and fire callback."""
        if new_state != self._state:
            old = self._state
            self._state = new_state
            if self._on_state_change:
                try:
                    self._on_state_change(old, new_state)
                except Exception as e:
                    logger.error("on_state_change error: %s", e)

    def stop(self):
        """Stop the backend."""
        self._running = False

    def run_browser(self):
        """Run with browser audio (WebSocket + Opus). Blocking.

        Serves PersonaPlex web UI on config.personaplex_port.
        Browser captures mic, streams Opus over WebSocket.
        Server runs inference, streams audio + text back.
        """
        import torch
        import sphn
        import aiohttp
        from aiohttp import web
        from jetson_assistant.backends.tool_parser import ToolParser

        self._running = True
        lock = asyncio.Lock()

        async def handle_chat(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            logger.info("Browser connected from %s", request.remote)

            self._set_state("listening")

            async with lock:
                # Reset streaming state for new session
                self._mimi.reset_streaming()
                self._lm_gen.reset_streaming()

                # Opus codec
                opus_writer = sphn.OpusStreamWriter(self._mimi.sample_rate)
                opus_reader = sphn.OpusStreamReader(self._mimi.sample_rate)

                # Process voice/text system prompts through mimi
                async def is_alive():
                    return not ws.closed

                await self._lm_gen.step_system_prompts_async(self._mimi, is_alive=is_alive)
                self._mimi.reset_streaming()

                # Send handshake
                if not ws.closed:
                    await ws.send_bytes(b"\x00")

                # Tool parser
                def on_tool_detected(tool_info):
                    if self._tool_registry:
                        class _TC:
                            def __init__(self, name, arguments):
                                self.name = name
                                self.arguments = arguments
                        result = self._tool_registry.execute(_TC(tool_info["name"], tool_info["args"]))
                        logger.info("Tool %s -> %s", tool_info["name"], result)

                tool_parser = ToolParser(
                    on_text=lambda t: None,  # Text display handled by WebSocket 0x02 messages
                    on_tool=on_tool_detected,
                )

                close = False

                async def recv_loop():
                    nonlocal close
                    try:
                        async for message in ws:
                            if message.type == aiohttp.WSMsgType.BINARY:
                                data = message.data
                                if len(data) > 0 and data[0] == 1:  # Audio
                                    opus_reader.append_bytes(data[1:])
                            elif message.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                                break
                    finally:
                        close = True

                async def opus_loop():
                    all_pcm_data = None

                    while not close:
                        await asyncio.sleep(0.001)
                        pcm = opus_reader.read_pcm()
                        if pcm.shape[-1] == 0:
                            continue
                        if all_pcm_data is None:
                            all_pcm_data = pcm
                        else:
                            all_pcm_data = np.concatenate((all_pcm_data, pcm))

                        while all_pcm_data is not None and all_pcm_data.shape[-1] >= self._frame_size:
                            chunk = all_pcm_data[:self._frame_size]
                            all_pcm_data = all_pcm_data[self._frame_size:]
                            if all_pcm_data.shape[-1] == 0:
                                all_pcm_data = None

                            chunk_t = torch.from_numpy(chunk).to(device=self._device, dtype=torch.float16)[None, None]
                            codes = self._mimi.encode(chunk_t)

                            for c in range(codes.shape[-1]):
                                tokens = self._lm_gen.step(codes[:, :, c:c + 1])
                                if tokens is None:
                                    continue

                                # Decode audio
                                main_pcm = self._mimi.decode(tokens[:, 1:9])
                                main_pcm = main_pcm.float()
                                self._pinned_pcm.copy_(main_pcm[0, 0], non_blocking=True)
                                torch.cuda.current_stream().synchronize()

                                pcm_np = self._pinned_pcm.detach().numpy()

                                # Feed audio chunk for motion/animation callbacks
                                rms = float(np.sqrt(np.mean(pcm_np ** 2)))
                                if rms > 0.001:
                                    self._set_state("speaking")
                                    if self._on_audio_chunk:
                                        try:
                                            self._on_audio_chunk(pcm_np, self._mimi.sample_rate)
                                        except Exception as e:
                                            logger.error("on_audio_chunk error: %s", e)
                                else:
                                    self._set_state("listening")

                                # Send audio back to browser
                                opus_writer.append_pcm(pcm_np)

                                # Parse text token for tools
                                text_token = tokens[0, 0, 0].item()
                                if text_token not in (0, 3):
                                    _text = self._text_tokenizer.id_to_piece(text_token)
                                    _text = _text.replace("\u2581", " ")
                                    tool_parser.feed(_text)
                                    # Send text to browser
                                    msg = b"\x02" + bytes(_text, encoding="utf8")
                                    await ws.send_bytes(msg)

                    tool_parser.flush()

                async def send_loop():
                    while not close:
                        await asyncio.sleep(0.001)
                        msg = opus_writer.read_bytes()
                        if len(msg) > 0:
                            await ws.send_bytes(b"\x01" + msg)

                # Run recv + opus + send concurrently
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                await ws.close()

            self._set_state("idle")
            logger.info("Browser disconnected")
            return ws

        # Serve web UI + WebSocket
        app = web.Application()
        app.router.add_get("/api/chat", handle_chat)

        # Serve PersonaPlex's existing web UI
        from huggingface_hub import hf_hub_download
        dist_tgz = hf_hub_download(self.config.personaplex_hf_repo, "dist.tgz")
        dist_dir = Path(dist_tgz).parent / "dist"
        if not dist_dir.exists():
            import tarfile
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=Path(dist_tgz).parent)

        async def handle_root(_):
            return web.FileResponse(os.path.join(str(dist_dir), "index.html"))

        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=str(dist_dir), follow_symlinks=True, name="static")

        # SSL
        ssl_context = None
        if self.config.personaplex_ssl_dir:
            from moshi.utils.connection import create_ssl_context
            ssl_context, _ = create_ssl_context(os.path.expanduser(self.config.personaplex_ssl_dir))

        from moshi.utils.connection import get_lan_ip
        host_ip = get_lan_ip()
        protocol = "https" if ssl_context else "http"
        logger.info("PersonaPlex Web UI: %s://%s:%d", protocol, host_ip, self.config.personaplex_port)

        web.run_app(app, port=self.config.personaplex_port, ssl_context=ssl_context)

    def run_local(self):
        """Run with local mic/speaker. Blocking.

        Uses jetson-assistant's AudioInput/AudioOutput for on-device audio.
        """
        import torch
        from jetson_assistant.assistant.audio_io import AudioInput, AudioOutput, AudioConfig
        from jetson_assistant.backends.tool_parser import ToolParser
        import resampy

        self._running = True

        # Reset streaming for new session
        self._mimi.reset_streaming()
        self._lm_gen.reset_streaming()

        # Process system prompts synchronously
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            self._lm_gen.step_system_prompts_async(self._mimi, is_alive=lambda: True)
        )
        loop.close()
        self._mimi.reset_streaming()

        # Audio I/O at Mimi's sample rate (24kHz)
        audio_out = AudioOutput(sample_rate=self._mimi.sample_rate)

        # Tool parser
        def on_tool_detected(tool_info):
            if self._tool_registry:
                class _TC:
                    def __init__(self, name, arguments):
                        self.name = name
                        self.arguments = arguments
                result = self._tool_registry.execute(_TC(tool_info["name"], tool_info["args"]))
                logger.info("Tool %s -> %s", tool_info["name"], result)

        tool_parser = ToolParser(on_text=lambda t: None, on_tool=on_tool_detected)

        # Audio buffer (mic -> model)
        audio_buffer = []
        buffer_lock = threading.Lock()

        def on_mic_chunk(chunk_int16):
            # Resample 16kHz -> 24kHz
            chunk_f32 = chunk_int16.astype(np.float32) / 32768.0
            chunk_24k = resampy.resample(chunk_f32, 16000, self._mimi.sample_rate)
            with buffer_lock:
                audio_buffer.append(chunk_24k)

        audio_in = AudioInput(
            config=AudioConfig(sample_rate=16000),
            device=self.config.audio_input_device,
        )
        audio_in.start(on_mic_chunk)

        logger.info("PersonaPlex local mode running. Speak into microphone.")

        try:
            while self._running:
                # Drain buffer
                with buffer_lock:
                    if not audio_buffer:
                        time.sleep(0.005)
                        continue
                    all_pcm = np.concatenate(audio_buffer)
                    audio_buffer.clear()

                # Process in frame-sized chunks
                offset = 0
                while offset + self._frame_size <= len(all_pcm):
                    frame = all_pcm[offset:offset + self._frame_size]
                    offset += self._frame_size

                    chunk_t = torch.from_numpy(frame).to(device=self._device, dtype=torch.float16)[None, None]
                    codes = self._mimi.encode(chunk_t)

                    for c in range(codes.shape[-1]):
                        tokens = self._lm_gen.step(codes[:, :, c:c + 1])
                        if tokens is None:
                            continue

                        main_pcm = self._mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.float()
                        self._pinned_pcm.copy_(main_pcm[0, 0], non_blocking=True)
                        torch.cuda.current_stream().synchronize()

                        pcm_np = self._pinned_pcm.detach().numpy().copy()

                        # Play audio
                        pcm_int16 = (pcm_np * 32768).astype(np.int16)
                        audio_out.play(pcm_int16, self._mimi.sample_rate)

                        # State + motion callbacks
                        rms = float(np.sqrt(np.mean(pcm_np ** 2)))
                        if rms > 0.001:
                            self._set_state("speaking")
                            if self._on_audio_chunk:
                                self._on_audio_chunk(pcm_np, self._mimi.sample_rate)
                        else:
                            self._set_state("listening")

                        # Text -> tool parsing
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self._text_tokenizer.id_to_piece(text_token)
                            _text = _text.replace("\u2581", " ")
                            tool_parser.feed(_text)

        except KeyboardInterrupt:
            logger.info("Stopping PersonaPlex local mode...")
        finally:
            audio_in.stop()
            audio_out.stop()
            tool_parser.flush()
            self._running = False
            self._set_state("idle")

    def run(self):
        """Run the backend in configured audio mode. Blocking."""
        if not self._loaded:
            self.load()

        if self.config.personaplex_audio == "browser":
            self.run_browser()
        elif self.config.personaplex_audio == "local":
            self.run_local()
        else:
            raise ValueError(f"Unknown personaplex_audio mode: {self.config.personaplex_audio}")
