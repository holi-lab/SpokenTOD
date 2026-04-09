import argparse

from synthesis.local import (
    DEFAULT_MODEL_ID,
    QwenTTSSynthesizer,
    save_wav,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize a single utterance with Qwen3-TTS via the qwen-tts package."
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--emotion",
        default="neutral",
        help="Emotion name used for the system instruction prompt",
    )
    parser.add_argument(
        "--ref-audio",
        required=True,
        help="Reference speaker audio path used for voice cloning",
    )
    parser.add_argument("--output-path", required=True, help="Output WAV path")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local model path",
    )
    parser.add_argument(
        "--device-map",
        default="cuda:0",
        help='Device map forwarded to Qwen3TTSModel.from_pretrained (default: "cuda:0")',
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention backend forwarded to from_pretrained",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Optional transcript for the reference audio",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional synthesis language override",
    )

    args = parser.parse_args()

    synthesizer = QwenTTSSynthesizer.from_pretrained(
        model_id=args.model_id,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    prompt = synthesizer.build_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
    )
    wav, sample_rate = synthesizer.synthesize(
        text=args.text,
        voice_clone_prompt=prompt,
        emotion_name=args.emotion,
        language=args.language,
    )
    save_wav(args.output_path, wav, sample_rate)


if __name__ == "__main__":
    main()
