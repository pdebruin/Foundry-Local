// Live Audio Transcription — Foundry Local Rust SDK Example
//
// Demonstrates real-time microphone-to-text using:
//   Microphone (cpal) → SDK → Core (NativeAOT DLL) → onnxruntime-genai (StreamingProcessor)
//
// Usage:
//   cargo run              # Live microphone transcription (press ENTER to stop)
//   cargo run -- --synth   # Use synthetic 440Hz sine wave instead of microphone

use std::env;
use std::io::{self, Write};
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use foundry_local_sdk::{FoundryLocalConfig, FoundryLocalManager};
use tokio_stream::StreamExt;

const ALIAS: &str = "nemotron";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let use_synth = env::args().any(|a| a == "--synth");

    println!("===========================================================");
    println!("   Foundry Local -- Live Audio Transcription Demo (Rust)");
    println!("===========================================================");
    println!();

    // ── 1. Resolve e2e-test-pkgs path ────────────────────────────────────
    let exe_dir = env::current_exe()?
        .parent()
        .unwrap()
        .to_path_buf();

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let e2e_pkgs = std::path::PathBuf::from(manifest_dir)
        .join("..")
        .join("e2e-test-pkgs");

    let (core_path, model_cache_dir) = if e2e_pkgs.exists() {
        let core = e2e_pkgs
            .canonicalize()
            .expect("Failed to canonicalize e2e-test-pkgs path");
        let models = core.join("models");
        println!("Using e2e-test-pkgs:");
        println!("  Core DLLs: {}", core.display());
        println!("  Models:    {}", models.display());
        (
            core.to_string_lossy().into_owned(),
            models.to_string_lossy().into_owned(),
        )
    } else {
        println!("Using default paths (exe directory)");
        (
            exe_dir.to_string_lossy().into_owned(),
            exe_dir.join("models").to_string_lossy().into_owned(),
        )
    };

    // ── 2. Initialise the manager ────────────────────────────────────────
    let config = FoundryLocalConfig::new("foundry_local_samples")
        .library_path(&core_path)
        .model_cache_dir(&model_cache_dir)
        .additional_setting("Bootstrap", "false");

    let manager = FoundryLocalManager::create(config)?;
    println!("✓ FoundryLocalManager initialized\n");

    // ── 3. Get the nemotron model ────────────────────────────────────────
    let model = manager.catalog().get_model(ALIAS).await?;
    println!("Model: {} (id: {})", model.alias(), model.id());

    if !model.is_cached().await? {
        println!("Downloading model...");
        model
            .download(Some(|progress: &str| {
                print!("\r  {progress}");
                io::stdout().flush().ok();
            }))
            .await?;
        println!();
    }

    println!("Loading model...");
    model.load().await?;
    println!("✓ Model loaded\n");

    // ── 4. Create live transcription session ─────────────────────────────
    let audio_client = model.create_audio_client();
    let session = Arc::new(audio_client.create_live_transcription_session());

    println!("Starting live transcription session...");
    session.start(None).await?;
    println!("✓ Session started\n");

    // ── 5. Start reading transcription results in background ─────────────
    let mut stream = session.get_transcription_stream()?;
    let read_task = tokio::spawn(async move {
        let mut count = 0usize;
        while let Some(result) = stream.next().await {
            match result {
                Ok(r) => {
                    let text = &r.content[0].text;
                    if r.is_final {
                        println!();
                        println!("  [FINAL] {text}");
                        io::stdout().flush().ok();
                    } else if !text.is_empty() {
                        print!("{text}");
                        io::stdout().flush().ok();
                    }
                    count += 1;
                }
                Err(e) => {
                    eprintln!("\n  [ERROR] Stream error: {e}");
                    break;
                }
            }
        }
        count
    });

    if use_synth {
        // ── 6a. Synthetic audio mode ─────────────────────────────────────
        println!("Generating synthetic PCM audio (440Hz sine wave, 3 seconds)...\n");

        println!("===========================================================");
        println!("  PUSHING AUDIO → SDK → Core → onnxruntime-genai");
        println!("===========================================================\n");

        let pcm_data = generate_sine_wave_pcm(16000, 3, 440.0);
        let chunk_size = 16000 / 10 * 2; // 100ms chunks
        let mut chunks_pushed = 0;
        for offset in (0..pcm_data.len()).step_by(chunk_size) {
            let end = std::cmp::min(offset + chunk_size, pcm_data.len());
            session.append(&pcm_data[offset..end], None).await?;
            chunks_pushed += 1;
        }
        println!("Pushed {chunks_pushed} chunks ({} bytes)", pcm_data.len());
    } else {
        // ── 6b. Live microphone mode ─────────────────────────────────────
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("No input audio device available");
        println!("Microphone: {}", device.name().unwrap_or_default());

        // Query the device's default input config and adapt
        let default_config = device.default_input_config()?;
        println!(
            "Device default: {} Hz, {} ch, {:?}",
            default_config.sample_rate().0,
            default_config.channels(),
            default_config.sample_format()
        );

        let device_rate = default_config.sample_rate().0;
        let device_channels = default_config.channels();
        let mic_config: cpal::StreamConfig = default_config.into();

        let session_for_mic = Arc::clone(&session);
        let rt = tokio::runtime::Handle::current();

        // Build the stream with the device's native sample format (f32)
        // and convert to 16kHz/16-bit/mono PCM for the SDK
        let input_stream = device.build_input_stream(
            &mic_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Step 1: Mix to mono if stereo+
                let mono: Vec<f32> = if device_channels > 1 {
                    data.chunks(device_channels as usize)
                        .map(|frame| frame.iter().sum::<f32>() / device_channels as f32)
                        .collect()
                } else {
                    data.to_vec()
                };

                // Step 2: Resample to 16kHz if device rate differs
                let resampled = if device_rate != 16000 {
                    resample(&mono, device_rate, 16000)
                } else {
                    mono
                };

                // Step 3: Convert f32 → i16 → little-endian bytes
                let bytes: Vec<u8> = resampled
                    .iter()
                    .flat_map(|&s| {
                        let clamped = s.clamp(-1.0, 1.0);
                        let sample = (clamped * i16::MAX as f32) as i16;
                        sample.to_le_bytes()
                    })
                    .collect();

                if !bytes.is_empty() {
                    let session_ref = Arc::clone(&session_for_mic);
                    rt.spawn(async move {
                        if let Err(e) = session_ref.append(&bytes, None).await {
                            eprintln!("Append error: {e}");
                        }
                    });
                }
            },
            |err| eprintln!("Microphone stream error: {err}"),
            None,
        )?;

        input_stream.play()?;

        println!();
        println!("===========================================================");
        println!("  LIVE TRANSCRIPTION ACTIVE");
        println!("  Speak into your microphone.");
        println!("  Transcription appears in real-time.");
        println!("  Press ENTER to stop recording.");
        println!("===========================================================");
        println!();

        // Block until user presses ENTER
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;

        drop(input_stream);
        println!("Microphone stopped.");
    }

    // ── 7. Stop session and wait for results ─────────────────────────────
    println!("\nStopping session (flushing remaining audio)...");
    session.stop(None).await?;
    println!("✓ Session stopped\n");

    let result_count = read_task.await?;

    println!("===========================================================");
    println!("  Total transcription results: {result_count}");
    println!("===========================================================");

    // ── 8. Cleanup ───────────────────────────────────────────────────────
    println!("\nUnloading model...");
    model.unload().await?;
    println!("Done.");

    Ok(())
}

/// Generate synthetic PCM audio (sine wave, 16kHz, 16-bit signed little-endian, mono).
fn generate_sine_wave_pcm(sample_rate: i32, duration_seconds: i32, frequency: f64) -> Vec<u8> {
    let total_samples = (sample_rate * duration_seconds) as usize;
    let mut pcm_bytes = vec![0u8; total_samples * 2];

    for i in 0..total_samples {
        let t = i as f64 / sample_rate as f64;
        let sample = (i16::MAX as f64 * 0.5 * (2.0 * std::f64::consts::PI * frequency * t).sin())
            as i16;
        let bytes = sample.to_le_bytes();
        pcm_bytes[i * 2] = bytes[0];
        pcm_bytes[i * 2 + 1] = bytes[1];
    }

    pcm_bytes
}

/// Simple linear-interpolation resampler (e.g. 48kHz → 16kHz).
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return input.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (input.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;
        let s0 = input[idx.min(input.len() - 1)];
        let s1 = input[(idx + 1).min(input.len() - 1)];
        output.push(s0 + (s1 - s0) * frac as f32);
    }
    output
}
