// Live Audio Transcription — Foundry Local Rust SDK Example
//
// Demonstrates real-time audio-to-text using:
//   SDK (FoundryLocalManager) → Core (NativeAOT DLL) → onnxruntime-genai (StreamingProcessor)
//
// Usage:
//   cargo run                       # Generates synthetic 440Hz sine wave
//   cargo run -- path/to/audio.pcm  # Uses raw PCM file (16kHz, 16-bit, mono)

use std::env;
use std::io::{self, Write};

use foundry_local_sdk::{FoundryLocalConfig, FoundryLocalManager};
use tokio_stream::StreamExt;

const ALIAS: &str = "nemotron";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("===========================================================");
    println!("   Foundry Local -- Live Audio Transcription Demo (Rust)");
    println!("===========================================================");
    println!();

    // ── 1. Resolve e2e-test-pkgs path ────────────────────────────────────
    let exe_dir = env::current_exe()?
        .parent()
        .unwrap()
        .to_path_buf();

    // Try to find e2e-test-pkgs relative to the sample directory first,
    // then fall back to the exe directory for the core DLL.
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
    let session = audio_client.create_live_transcription_session();
    // Settings: 16kHz, 16-bit, mono PCM (defaults match)
    assert_eq!(session.settings.sample_rate, 16000);
    assert_eq!(session.settings.channels, 1);
    assert_eq!(session.settings.bits_per_sample, 16);

    println!("Starting live transcription session...");
    session.start().await?;
    println!("✓ Session started\n");

    // ── 5. Start reading transcription results in background ─────────────
    let mut stream = session.get_transcription_stream()?;
    let read_task = tokio::spawn(async move {
        let mut results = Vec::new();
        while let Some(result) = stream.next().await {
            match result {
                Ok(r) => {
                    if r.is_final {
                        println!("  [FINAL] {}", r.text);
                    } else if !r.text.is_empty() {
                        print!("{}", r.text);
                        io::stdout().flush().ok();
                    }
                    results.push(r);
                }
                Err(e) => {
                    eprintln!("\n  [ERROR] Stream error: {e}");
                    break;
                }
            }
        }
        results
    });

    // ── 6. Generate or load PCM audio ────────────────────────────────────
    let pcm_data = if let Some(pcm_path) = env::args().nth(1) {
        println!("Reading PCM audio from: {pcm_path}");
        std::fs::read(&pcm_path)?
    } else {
        println!("Generating synthetic PCM audio (440Hz sine wave, 3 seconds)...");
        generate_sine_wave_pcm(16000, 3, 440.0)
    };

    println!(
        "Audio: {} bytes ({:.1}s at 16kHz/16-bit/mono)\n",
        pcm_data.len(),
        pcm_data.len() as f64 / (16000.0 * 2.0)
    );

    // ── 7. Push audio in chunks (100ms each) ─────────────────────────────
    println!("===========================================================");
    println!("  PUSHING AUDIO → SDK → Core → onnxruntime-genai");
    println!("===========================================================");
    println!();

    let chunk_size = 16000 / 10 * 2; // 100ms of 16-bit mono audio = 3200 bytes
    let mut chunks_pushed = 0;
    for offset in (0..pcm_data.len()).step_by(chunk_size) {
        let end = std::cmp::min(offset + chunk_size, pcm_data.len());
        session.append(&pcm_data[offset..end]).await?;
        chunks_pushed += 1;
    }
    println!("Pushed {chunks_pushed} chunks ({} bytes total)", pcm_data.len());

    // ── 8. Stop session and wait for results ─────────────────────────────
    println!("\nStopping session (flushing remaining audio)...");
    session.stop().await?;
    println!("✓ Session stopped\n");

    let results = read_task.await?;

    // ── 9. Summary ───────────────────────────────────────────────────────
    println!("===========================================================");
    println!("  RESULTS SUMMARY");
    println!("===========================================================");
    println!("Total transcription results: {}", results.len());
    for (i, r) in results.iter().enumerate() {
        println!(
            "  [{i}] text={:?} is_final={} start={:?} end={:?}",
            r.text, r.is_final, r.start_time, r.end_time
        );
    }

    if results.is_empty() {
        println!("  (No transcription results — synthetic audio may not produce recognizable speech)");
    }

    // Verify all results are well-formed
    for r in &results {
        assert_eq!(r.text, r.transcript, "text and transcript must match");
    }
    println!("\n✓ All results well-formed (text == transcript)");

    // ── 10. Cleanup ──────────────────────────────────────────────────────
    println!("\nUnloading model...");
    model.unload().await?;
    println!("Done.");

    Ok(())
}

/// Generate synthetic PCM audio (sine wave, 16kHz, 16-bit signed little-endian, mono).
fn generate_sine_wave_pcm(sample_rate: i32, duration_seconds: i32, frequency: f64) -> Vec<u8> {
    let total_samples = (sample_rate * duration_seconds) as usize;
    let mut pcm_bytes = vec![0u8; total_samples * 2]; // 16-bit = 2 bytes per sample

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
