# Cleanup validation and measured bottlenecks

## Audio scheduling and GPU work — 2026-10-13

GPU DSP no longer dispatches the previous response after its crossfade has ended. The hop containing the fade end still renders both responses; subsequent hops render only the current response. This skips input FFT/mixing work without changing paths, filters, fidelity, latency or output buffering. `GpuAudio` tracks the response sample on the CPU, so no new shader or GPU readback is needed. Existing CPU/GPU crossfade tests now exercise a fade ending inside a hop, as well as partial-hop delivery. **168/168 core checks passed in 53.95 s.**

On the same saved station input (`.context/spectral-renderer/comparison-223907-inputs`), total offline render time was **10.057 s before / 6.725 s after**. Both produced 977,988 stereo frames with the exact same WAV SHA-256, `b0b709a9d02cb9b3e477504c61f6d1a8e5c61705f2b91688ae828dc577d9621a`. This is one before/after pair, not a statistically established 33% gain. A shader-only intermediate measured 7.016 s; the final implementation also avoids dispatch overhead. The prior implementation remains in Git at `98a7ce4`.

Reports now include GPU PCM request-to-callback time and callback-to-consumption delay. The former includes GPU execution, transfer, PCM copy and Unity's callback scheduling; it is **not a GPU kernel timer**. Multiplayer probes also save producer diagnostics and maximum update gaps. The final moving-station deadline run passed (24.75 s) for 886,784 frames with zero underrun/late/overflow or over-budget callbacks. Maximum readback completion latency was 186.2 ms; callback-to-consumption delay was 5.2 microseconds. These figures localize the delay upstream of final publication, without assigning it all to GPU arithmetic.

A full runtime profile identified **1.028 s and 1.047 s `GfxDeviceMetal.WaitForLastPresent` waits** in two standalone clients. Their longest acoustic `LateUpdate` calls were 12.9–13.1 ms. Thus at least these long stalls are display-presentation waits blocking Unity's main loop, where the acoustic producer and readback callbacks run. The captures retain only the last 2,000 frames; they do not account for every earlier miss. Disabling VSync in an experimental build did not make the four-process workload reliable: three producers still filled their observation queues. That experiment was reverted. No queue/buffer sizes or ray budgets were increased to hide missed deadlines.

Same-Mac synthetic multiplayer runs are recorded separately from device tests:

| Run | Connection, capture and replay | Audio delivery |
|---|---|---|
| Two players, first shader-guard build | Both passed | Host missed 48,128 frames; client 101,377 |
| Two players, warm repeat | Both passed; no late/concealed voice packets | Zero missed output frames |
| Two players, final delivery build | Both saved/replayed two voices; 249 received packets each, no late/concealed packets | Zero underrun or late output frames |
| Four players, final dispatch optimization | Host saved/replayed all four voices; three clients overflowed observation queues | 159,643–257,262 missed frames per process |
| Four players, runtime profiling enabled | All saved/replayed four voices; no late/concealed voice packets | Host 0; clients 2,048 / 37,888 / 54,273 missed frames |

The four-player test also exposed timestamp-only recording-folder collisions when clients initialized in the same millisecond. New scene recordings keep the readable timestamp and append a GUID. The original colliding run is excluded from performance comparisons. Functional replay success does not imply clean audio; four players sharing this GPU remain an unsuccessful realtime stress case. Separate-computer internet quality is unmeasured.

Head-only GPU probes hold source, listener position, gain and propagation delay fixed, using 1 kHz / 6 kHz tones and 90°/s / 360°/s yaw. After startup, none produced silence runs; residual sample steps at FFT-hop boundaries were no larger than interior maxima. Relative RMS differences from a per-output-sample HRIR interpolation oracle were 3.98% / 14.44% at 1 kHz and 19.93% / 62.85% at 6 kHz. These quantify a filter-update/timing approximation, not a perceptual score or an independently validated physical oracle: the reference schedules head orientation on the output clock, while the source includes DSP latency. Moving the reference head schedule back by the declared 1,024-sample DSP latency reduces those differences to 1.51% / 5.49% and 7.74% / 25.70%, with identical rendered PCM. Head-pose/audio clock alignment and faster directional interpolation need a consistent physical timing contract before changing the renderer.

The initial physical microphone test skipped because Unity reported no macOS permission. After the user enabled access, refreshing through `Application.RequestUserAuthorization` updated Unity's cached status without restarting. The explicit device test then passed (9.96 s NUnit duration): it received at least one second of canonical microphone samples and verified that stopping closes the opened device even after the selected device name changes. No microphone audio was saved or transmitted. Synthetic multiplayer transport was tested separately; physical microphone-to-peer playback and mouth-to-ear latency remain unmeasured. Follow-up evidence: `.context/audio-reliability/microphone-authorized.json` and `permission-refresh-result.json`. The final Mac development player rebuilt in 13.30 s with zero errors and 1,869 warnings; its fingerprint matches the saved, stopped TrainStation editor scene. Evidence is in `.context/audio-reliability/` and `.context/voice-sync/lan-*`; takes and raw profiler captures remain local.


## Spectral scene default — 2026-10-13

TrainStation, Church and all three 2D acoustic scenes now select Spectral synthesis. New settings and serialized settings without a synthesis field also default to Spectral. Explicitly serialized IR/Spectral selections keep their enum values. The previous IR default was retained for backward compatibility; it was unsuitable for these moving scenes because its frozen-delay crossfades do not provide continuous Doppler. The existing GPU/CPU spectral implementations are reused. The production change is six settings lines; ray budgets, buffering and shaders are unchanged. The existing default-selection test now covers the new default and both explicit selections.

Validation: **168/168 TakeChecks passed** (41.99 s), including CPU/GPU agreement and Doppler. The station's explicit device-deadline test passed for **886,784 frames** after warmup with movement/head turns: **zero underrun, late, overflow, channel-mismatch or over-budget callback counts**. This single approximately 20-second run does not resolve the intermittent scheduling stalls measured below, nor establish head-only artifact-free rendering. The macOS development player rebuilt successfully; the final incremental build took 2.96 s with zero errors and 1,539 warnings. Its fingerprint matches the editor, whose saved TrainStation is stopped and selects Spectral.

The diagnostic findings below describe the preceding IR configuration. Evidence and build/scene verification are in `.context/artifact-audit/spectral-*`; the refreshed standalone is `.context/voice-sync/Acoustics.app`.

## Windowing and live artifacts — 2026-10-13

The current station uses GPU IR synthesis. Its FFT convolution uses **overlap-save**: source history is retained, the transform is sized for the pressure FIR plus HRIR and output hop, and only the final valid hop is emitted. CPU IR uses partitioned overlap-add. These are linear convolution methods, not Hann-windowed STFT reconstruction; adding a Hann window to the existing blocks would change the output. The comparison spectrogram's Hann window is display analysis only. See `GpuAudio.cs`, `GpuAudio.compute`, `ImpulseResponseAudio.cs`, and [overlap-save's valid-output rule](https://www.dsprelated.com/freebooks/sasp/Overlap_Save_Method.html).

Fresh focused checks passed: **14/14 independent time-domain convolution cases** (1.02 s) and **4/4 CPU/GPU IR/spectral response-crossfade cases** (1.12 s). They exercise fractional delays, ear filters, block boundaries and changing responses. They establish agreement with those numerical contracts, not artifact-free physical rendering or realtime delivery.

The explicit `StationDeliversAudioBeforeItsDeadline` test **failed twice**, including a warm repeat, on Unity 6000.4.1f1 / Apple M5 / Metal. Each measured 885,760 device frames after two seconds of warmup, with player/head motion and a moving train. The existing station preset remained unchanged: 44.1 kHz, FFT1024/hop256, response step1024, transition512, binaural GPU IR, one ray/bin, five bounces, and 16,384-frame output lead (~372 ms).

| Run | Underrun frames | Total missing audio | Late frames | Overflow / channel mismatch / over-budget callbacks |
|---|---:|---:|---:|---|
| First | 5,123 | 116.2 ms | 5,123 | 0 / 0 / 0 |
| Warm repeat | 14,343 | 325.2 ms | 14,343 | 0 / 0 / 0 |

Underrun and late counters describe overlapping failures and must not be added. Durations are totals across each approximately 20-second measurement, not single continuous gaps. `AudioManager.PcmQueue.Consume` emits zeros for unavailable frames and advances the device clock; late PCM is then discarded. Those transitions can cause clicks and gaps. The callback itself stayed within budget (maximum ~69–71 microseconds); producer scheduling, GPU completion and publication timing need profiling to isolate the upstream stall. These Editor runs do not establish standalone-player performance.

Both current source clips have zero-valued loop endpoints, so a simple endpoint step is less likely. Current summed output clipping was not measured. Moving-delay phase interference remains a separate question: IR holds each path delay fixed between responses and crossfades old/new output over 512 samples (~11.6 ms), with responses spaced 1024 samples (~23.2 ms). A numerically correct fade can still combine differently phased signals; this audit has not isolated that as the audible cause.

No runtime code or scene settings changed. Next work should isolate the missed producer deadlines, then evaluate moving-delay continuity with deadlines satisfied. Evidence: `.context/artifact-audit/` contains the two deadline results and producer reports, clip measurements, scene preset and focused test results.

The user subsequently localized the artifacts to movement/head turns. An offline probe now isolates motion through the production **GPU audio backend**, supplying a single synthetic unit-gain path rather than tracing the station. It renders a 1 kHz tone for 88,064 frames at 44.1 kHz, using the station's FFT/hop/response/fade sizes above. Delay starts at 50 ms and increases linearly with receiver speed / 343; gain stays constant to isolate delay behavior. GPU readback is awaited, so device underruns cannot explain these results.

| Synthetic motion | Synthesis | RMS error relative to continuous-delay tone |
|---|---|---:|
| Stationary | IR | 0.00048% |
| Receding at 1.7 m/s | IR | 20.93% |
| Receding at 12 m/s | IR | 133.98% |
| Receding at 12 m/s | Spectral | 0.185% |

The oracle is `0.2 * sin(2*pi*1000*((sample-DspLatencySamples)*(1-speed/343)/44100-0.05))`, excluding the first 250 ms. Errors can exceed 100% when phases differ; they are not a perceptual rating. A phase-independent sine/cosine fit over the interior 1.5 seconds confirms the fast IR case has the wrong dominant tone: the expected Doppler frequency is **965.015 Hz**, but **1008.081 Hz** accounts for about **96.3%** of output energy. These frequencies differ by the 43.066 Hz response-update rate. The spectral case's residual after fitting the expected tone is 0.076% RMS. Thus stepped IR delay causes temporal aliasing/sidebands in this controlled case; a crossfade that passes its numerical contract does not make it continuous-delay propagation.

A separate binaural probe holds the path fixed and turns the head at 90 degrees/second. IR output differs by 3.42% RMS from convolution with the same HRTF interpolated at every output sample. This measures the existing filter-update approximation; it does not isolate a perceptible click or establish a physically exact moving-ear reference. Head-only audible artifacts still need separate reproduction under live load.

Motion distortion is therefore a second demonstrated problem alongside underruns. The existing spectral backend already has continuous path-delay evolution and is the implementation to reuse/evaluate for moving audio; changing the scene default alone is not established as a complete fix. IR remains useful for fixed-response convolution. No runtime or scene changes were made during this follow-up. `Motion.cs`, `motion.json`, `motion-analysis.json` and raw rendered/oracle floats are saved in `.context/artifact-audit/`.

## Station and session UI polish — 2026-10-13

TrainStation keeps its editable Unity geometry and now has consistent station materials, carriage/roof/door trim, boarding wayfinding, and a ground surface extending beyond the fog horizon. The ground collider and acoustic surface expand together. Decorative trim adds no acoustic surfaces; the station still has 47. Church and TrainStation share simple player visuals; heads inherit the existing voice transform, so received and replayed head orientation needs no additional runtime updater. Avatar collision now matches the walking controller's narrower body.

The existing uGUI panel groups capture, replay, microphone/network and mouse settings. It hides while walking, displays session/player state, hides connection setup while connected, gives saved runs readable local timestamps, and explains why replay is disabled until saving. Empty join codes cannot be submitted. The R shortcut only runs with the cursor locked, preventing join-code typing from saving a recording. No runtime files or packages were added; two visual materials were added.

Validation: **166/166 TakeChecks passed** (21.02 s); **StationEndToEndTests.RecordingReconstructsAndReplays passed** on the final ground (33.06 s), including boarding/departure and input reconstruction; **ChurchEndToEndTests.ChurchWalksThroughDoorAndReplays passed** (25.37 s). Live UI checks covered initial action availability, online/LAN field switching, join-code editing, duplicate-build prevention, and saving into the live/replay-ready state. Three avatar head-inheritance checks passed after the final hierarchy changes. The macOS development app rebuilt successfully (12.86 s, zero errors, 1,874 shader/deprecation/Pipeline warnings); its build fingerprint matches the editor. Cosmetic changes do not establish improved acoustic fidelity or a performance gain. Scene/build identity changes require new recordings or the matching historical checkout for old input-only takes.

Evidence and previews are in `.context/polish/`. The synthetic/manual preview take was archived there rather than left in the normal saved-run chooser.

## Timestamped voice and Relay — 2026-10-13

The existing microphone, codec and network components now send separate dry Opus streams with source sample indices and estimated session capture timestamps. The host preserves those timestamps; each receiver renders its own acoustics. Pose snapshots retain their original sample time. The wire protocol advances to 4; peers still require matching scene/build inputs. Internet hosting uses Multiplayer Services 2.2.4's Relay API with DTLS and anonymous authentication. LAN remains available. No new runtime source files or second acoustic engine were added.

**166/166 TakeChecks passed** (40.77 s), including codec impulse alignment, exact final sample counts, packet reordering/loss, payload-before-start, buffered speech across microphone restarts, cursor startup/drift and changing resampler ratios. The explicit real-device microphone test passed (8.29 s), without saving or transmitting device audio. The Mac Development Player built with zero errors and 1,869 existing warnings.

The initial player run exposed a Concentus phase-integer overflow when changing fractional resampling rates. The adapter now bounds reduced phase denominators so their product fits signed int, retaining common exact ratios. The regression exercises repeated rate changes at both 44.1 and 48 kHz. SDK 2.2.0 also had an unguarded Editor-only callback in its Player assembly; the 2.2.4 patch fixes it while retaining UTP 2.6.0. Package-cache/vendor code was not patched.

| Standalone run on this Mac | Routing / capture / replay | Voice concealment / late packets | Output underrun frames |
|---|---|---|---|
| Two players, LAN, normal scene settings | Both passed; both distinct voices saved and replayed offline | Host 1 / 1; client 0 / 0 | Host 39,936; client 1,024 |
| Two players, actual internet Relay, normal scene settings | Both passed; 249 packets received per process, both voices saved and replayed offline | Both 0 / 0 | Host 0; client 3,072 |
| Four players, Relay, 1,024-sample response step | All connected; run failed after GPU observation-queue and source-writer overflow | Timing degraded under contention | 30,721–130,048 |
| Four players, Relay, 4,096-sample response step | Host saved/replayed all four voices; three clients failed with full GPU observation queues | Timing degraded under contention | 141,313–276,480 |

The two-player replay position errors were below 1e-5 m, with zero measured final angular error. All eight saved two-player speaker identities matched their distinct synthetic tones. The coarse four-player host recording contained all four expected tones. A functional pass does not mean clean realtime audio: the four-process stress tests failed overall, and four independent computers remain untested. Relay connectivity is verified, but physical microphone/network/DAC synchronization accuracy and conversational latency are not established.

Reducing the station output lead failed the explicit device deadline gate: 4,096 samples missed 10,264 frames after warmup; 8,192 samples missed 130,109 frames (late-output counts overlap and must not be added). These were separate runs with variable machine load, not evidence that a larger buffer causes worse performance. The existing **16,384-sample scene lead is unchanged**, about 372 ms at 44.1 kHz, plus the 100 ms network presentation buffer and other device latency. Renderer scheduling and shared-GPU contention remain performance work; no buffering increase or fidelity reduction is hidden in this change.

Evidence: `.context/voice-sync/` contains core/device/deadline results, the build summary, per-process LAN/Relay results and logs, and `source-tones.json`. Synthetic takes were moved into its `takes/` folder so ordinary recording selection does not treat test tones as user sessions.

## Audio continuity and multiplayer verification — 2026-10-07

**155/155 TakeChecks passed** in 36.38 seconds on Unity 6000.4.1f1 / Apple M5 / Metal. Four new cases compare changing gain, delay and head orientation against a sample-wise crossfade between independently rendered static responses. CPU/GPU and IR/spectral outputs agree within 2e-5, including partial-hop delivery and an unchanged second listener. These detect DSP discontinuities; they do not validate the physical transport model.

The previously benchmarked 530,501-frame station WAV peaks at 0.105865, with no samples above full scale. Both source clips start/end at zero. Its largest adjacent-sample change (0.082844) occurs at sample **90,112**, exactly at a recorded listener teleport. Cuts currently bypass response fades; this is a known hard discontinuity, not evidence that ordinary motion smoothing failed. Cut windows require explicit treatment in training data.

The explicit station device test passed for 885,760 frames after warmup: **zero underrun, late, overflow, channel-mismatch or over-budget callback samples**, with the existing 16,384-frame lead. The earlier failed deadline run remains recorded below; one passing run does not resolve intermittent stalls. The explicit microphone capture/open-close check also passed (7.22 s); it saved and transmitted no device audio.

Standalone tests ran one host plus clients on this same Mac, using distinct synthetic tones, native scene settings, and a separate recording per player. `MultiplayerSmoke` now accepts `--acoustics-players=N` and requires all participants' streams. Its pass status covers networking, recording and offline replay; audio deadline counters are reported separately.

| Local processes | Response step | Networking / recording / replay | Audio delivery |
|---|---:|---|---|
| Host + one client | 1,024 | Both passed; both saved two voices; no concealed/late voice packets | Host missed 1,024 frames (~23 ms); client missed zero |
| Host + three clients | 1,024 | All connected/exchanged voice; two clients failed with full live-observation queues | 33,793–52,317 underrun frames per process; this run failed |
| Host + three clients | 4,096 | All four passed and saved four voices; all 16 participant/stream tone identities matched | Host missed zero output frames; clients missed 362,280–398,382 (~8.2–9.0 s); each process concealed 124–332 voice frames |

Successful replays ran offline with the microphone closed; maximum final position error was below 1e-5 m and angular error below 0.05°. The coarse-response run proves four-player routing/capture/replay, **not clean realtime speech**. Increasing the response step did not cure contention. Four independent computers, actual conversational voice quality and internet connectivity were not tested. Physical reference acceptance remains blocked by the transport failures below.

The Mac player rebuilt with zero errors and 1,869 existing warnings. No runtime renderer or scene defaults changed. Local evidence is under `.context/audio-quality/`: test XML/status files, waveform/cut/clip checks, per-process results and `multiplayer-summary.json`. Synthetic takes were moved into that directory's `takes/` folder, outside the ordinary recordings list.

## GPU response storage and diagnostics — 2026-10-07

The baseline is preserved at `83a62da`. GPU filter construction now skips the staged large FFTs when responses are unchanged, and audio transforms skip the unused previous bank. Dispatches still occur, but those shader stages return before touching FFT scratch. Persistent material spectra are shared by both ears; each response bank keeps its own head blend so crossfades retain the previous orientation. No paths, precision, FFT sizes or configured budgets are dropped.

The same six-source station take now reserves **563,674,197 numeric bytes versus 916,339,093: 38.49% less**. Material filters use **353,354,400 bytes**, half their previous allocation. The remaining extra head-blend bank slightly offsets that saving. These are allocated numeric buffers, not process memory. The run peaks at **744 grouped paths out of 21,546 reserved slots per bank**, so a future compact allocator still has room to improve; none is introduced here.

Three runs retaining separate ear filters, but using the new command submission/diagnostics, took **5.03–5.28 s**, median **5.275 s**. Three final shared-filter runs took **5.16–5.34 s**, median **5.264 s**, for **12.03 seconds of audio**. These sequential controls show similar total cost, not an established speedup. The original HEAD baseline has one 4.81 s run. New WAVs are repeatable; against HEAD, maximum sample difference is **2.98e-8**, relative RMS **8.17e-8** (141.76 dB SNR). Moving ear composition into reduction permits tiny floating-point rounding changes. Direct diagnostic responses remain bit-identical.

`run.json` and `live.json` gain `gpuAudio` response/reuse, active/peak group, capacity and filter-byte fields, delivered in the normal PCM readback. The comparison window displays them in its existing details foldout. The station reuses **52/519 responses**, including its frozen tail; the separate motion probe found only **10/476 (2.1%)** during the recording. Any changed pair/head currently invalidates the whole batch. Existing underrun/late/overflow and callback counters remain the deadline measures.

Command-buffer scopes label direct transport, sampled tracing, comparison, grouping, filter construction, input resampling/FFT, per-ear reduction/IFFT and PCM output. A real Metal System Trace confirmed these labels and recorded **4.034 s of Unity compute-encoder active intervals** during a separately profiled run. That is not a per-kernel breakdown: Metal coalesces stages under one encoder label, intervals may overlap, and profiling increases wall cost. The profiled run is excluded from the timing comparison. Only sanitized aggregate performance data is retained; the raw Instruments metadata contained inherited environment values and was removed.

The physical audit adds `DirectBenchmark.MeasureReciprocity` to the existing benchmark utility. On an asymmetric finite wall, swapping source/listener yields a **75.0% 2D / 93.2% 3D** discrepancy in mean first-order reflected complex pressure at 250 Hz, even at 65,536 rays/bin across eight non-overlapping seed ranges. The mirror controls agree. Raising the ray budget reduces noise but does not remove this bias. These are reciprocity discrepancies, not accuracy percentages against a measured room. Diffuse weighting therefore remains a known model failure; exact specular-to-edge connections remain incomplete. Neither is disguised with a gain correction, and raytraced renders remain excluded from teacher approval even when temporal renders agree.

**150/150 TakeChecks passed in the normal Unity Editor (38.02 s)**, including 11 CPU/GPU parity cases, unchanged large-FFT scratch preservation and the new training-gate regression. The earlier batchmode run passed 148/149; the comparison-camera ground-sweep case failed there and in an isolated batch rerun, then passed in the normal Editor. This records the environment distinction rather than calling the batch suite green. [Full measurements, provenance, seeds and limitations](benchmarks/gpu-improvements-2026-10-07.json).

The explicit station device-deadline test **failed**: after two seconds of warmup, its 885,760-frame interval recorded **4,096 underrun / 3,072 late frames**, zero overflow, channel mismatch or over-budget callbacks. Underrun and late counters overlap. At 44.1 kHz this is about **93 ms of missing output**; lead remains 16,384 frames. Maximum observed pose gap was 5,120 frames (116 ms), and maximum GPU batch completion was 200 ms. The final producer snapshot precedes the last callback and therefore reports only 3,072 underrun frames; the test's atomic queue counters are authoritative. No lead increase or deadline relaxation was made. Station walking/boarding/recording/replay passed separately (33.33 s), as did Church walking/door/replay (25.90 s).

The Mac app at `.context/voice/Acoustics.app` rebuilt successfully and matches the project's multiplayer fingerprint. The final cached build reports zero errors and one warning. The first build emitted 1,877 warnings, including the existing Metal function-analysis warnings; its successful result also counted a Pipeline 30-second request timeout as one error. Scheduling the repeated build outside that API wait removed the tooling error. No standalone multiplayer/device run was repeated for this change.

## GPU audio scratch storage — 2026-10-07

Reduced DSP scratch allocations without changing the transport, audio math, dispatch count or summation order. Shared-memory FFTs retain only the nonredundant input-spectrum bins. Larger FFTs reuse one pair's filter-preparation scratch while keeping the full input spectra needed for listener reduction. Persistent old/new filters remain independent; no ray, group or FFT cap was added. The original code remains at `305c949`.

The same six-source station take (GPU IR synthesis, one binaural listener, 564 triangles, 530,501 frames at 44.1 kHz, FFT 1,024/hop 256, five bounces and batches of eight responses) now reserves **916,339,093 numeric bytes**, down from **1,092,671,557**: **176,332,464 bytes / 16.14% less**. This reports allocated CPU/GPU numeric storage, not peak process RSS. Separate two-source/two-listener allocation probes at FFT sizes 1,024 and 2,048 reduced total DSP storage by 15.97–24.95% across mono and binaural output.

All eight offline renders produced **bit-identical WAV and direct-response hashes**, with identical scene/state/event/source/settings hashes and finite output. Three original runs took 18.44–19.23 s; three candidate runs took 6.99–8.95 s. Restoring the original code gave an **8.49 s** control; restoring the tested candidate gave **6.92 s**. Workstation variability prevents attributing the large initial timing difference to this optimization. Memory reduction is established; a stable throughput improvement or realtime deadline is not.

Unity 6000.4.1f1 / Apple M5 / Metal: **148/148 TakeChecks passed**, including 11 CPU/GPU audio parity cases and a new large-FFT binaural case. Existing cases exercise multiple sources/listeners, head motion, response fades, cuts, inactive listeners, partial-hop delivery and both synthesis modes. [Reports, hashes, allocation probes and test outcomes](benchmarks/audio-scratch-2026-10-07.json). `sharedReport` plus each run's `report` reconstructs its full renderer report. Remaining memory is dominated by worst-case persistent filter banks; grouping/reduction costs, Windows/DirectX and live device deadlines need separate measurements.

The Mac player at `.context/voice/Acoustics.app` rebuilt successfully with zero errors and 1,834 warnings; it was not launched. Compiler warnings remain, including the transport warnings from the existing normal/debug variants.

## 2D ray debugger — 2026-10-07

**Acoustics → 2D Ray Debugger** captures segments from the actual GPU transport into a frozen XY view. Pair/bin selection, a configurable probe ray budget, propagation scrubbing, path filtering, material/weight inspection and pan/zoom work without editing scene assets. One Editor window is added; no package or scene component is required. Direct, mirror, sampled surface/edge and listener legs share the renderer's travel-time calculation. The normal shader variant has no debug buffer or readback. The window probes frozen geometry with zero velocities; it is not an in-flight live-audio capture. [Workflow, layout and limits](simulation-contracts.md#2d-propagation-debugger).

Unity 6000.4.1f1 / Apple M5 / Metal: **147/147 TakeChecks passed** in 12.35 seconds after shader warm-up, including six new debugger cases. These cover unchanged path sampling/arrival values in 2D/3D with BVH on/off, per-medium segment timing, mirror endpoints, muted-pair clearing and return to the normal shader variant. The initial focused run passed five cases; the sixth test incorrectly assumed inactive listeners stop transport tracing (listener gating belongs to DSP). Correcting it to test a muted pair produced the passing full run. Big Room, SampleScene and SmollRoom each captured successfully without dirtying the scene; window scroll, pan and propagation animation were exercised. SampleScene captured seven diffraction legs at 64 rays/bin. Existing Metal compiler warnings remain. No standalone rebuild or performance claim is part of this Editor feature.

Raw local validation: `.context/ray-debugger/core.xml`, `focused-first.xml`, `scene-checks.txt` and `view.png`.

## Integrated edge trial — 2026-10-07

The previous renderer is preserved at annotated tag `checkpoint/pre-integrated-diffraction-2026-10-07` (`5c67dda`). The experiment replaces the separate shortest-edge search with signed rigid-wedge samples inside the GPU path loop. Multiple edges can participate in one path; the existing arrival format and IR/spectral audio backends retain the resulting delay and phase. Four edge-only kernels and their route buffer are removed. [Contract and limits](simulation-contracts.md#integrated-edge-trial).

Validation on Unity 6000.4.1f1 / Apple M5 / Metal: **139 numerical/transport cases pass, plus two independent signed-arrival CPU/GPU audio cases**. The edge checks cover complex half-plane pressure, shadow-boundary continuity, 2D/3D, BVH/brute traversal, two-bend visibility and timing, moving-edge Doppler, transparent geometry and binaural IR/spectral agreement. The first broad run timed out during cold compilation of the planar BVH shader; that case passed after caching. The two new impulse fixtures initially requested an oversized mixer block; correcting the fixture made both pass. The Mac standalone rebuilt successfully with zero errors and 1,090 warnings; it was not launched. Raw outcomes remain in the report. These checks do not establish a full-wave multi-edge reference or complete specular–edge transport.

The same six-source, one-listener station take renders 530,501 stereo frames (~12 seconds), FFT1024/hop256, one ray/bin, five bounces, response step1024 and batches of eight. The final experiment's two warmed runs took **11.916 and 24.013 seconds**; a fresh checkpoint control took **19.620 seconds** under the concurrent workstation load. The older quiet-session checkpoint measurement was 5.806 seconds. This variation does **not** establish a speedup or a realtime deadline. The two experimental WAV hashes match each other exactly, and the fresh checkpoint WAV matches its historical hash. All samples are finite; direct diagnostic hashes and source/scene/state/event hashes are unchanged.

Reserved numeric storage falls from **1,245,290,965 to 1,092,671,557 bytes (12.3%)**, because diffraction now shares bounce slots. More rays still grow per-arrival audio storage. Metal reports register-pressure warnings for the BVH trace kernels as well as potentially-uninitialized-function warnings; compilation and occupancy need attention before claiming a realtime improvement. General impedance materials, exact specular–edge connections, vertex corrections and independent multi-edge physical convergence remain open. The preview is excluded from teacher approval. [Full reports, model references and test outcomes](benchmarks/integrated-edge-2026-10-07.json).

## Transport cleanup — 2026-10-07

All five scenes now use one renderer. Big Room, SmollRoom and SampleScene retain their editable native 2D walls and source/listener objects, with explicit take bindings. The legacy shader, audio backend, precomputation controls, compute helper and geometry exporter were removed; Git revision `05a5477` retains them. `Monitor Live` selects audible or silent capture.

`Geometry.hlsl` owns shared intersection and static-BVH traversal. `Transport.hlsl` contains direct transmission, edge diffraction, first mirror and sampled reflection calculations; `Transport.compute` contains dispatch entry points. Mirror visibility now uses the BVH for other objects and preserves retarded-pose checks on the reflecting object. Descriptive names and formatting replace compressed transport code. Scene export is `SceneToTake`; GPU batch ownership is `TransportSolver`.

Normal GPU playback submits transport and DSP without the extra opaque diagnostic dispatch/readback. The completed PCM readback fences batch reuse; cancellation fences outstanding transport before freeing buffers. Explicit diagnostics and offline response exports retain the probe pass. CPU direct transport and CPU DSP references remain available. No dependency or acoustic-model changes were introduced.

The same six-source station take produced **bit-identical WAV and diagnostic response hashes**. One offline run took **6.98 s before / 5.81 s after**; solve/wait was **1.04 / 0.88 s**, audio work **4.26 / 3.88 s**. Numeric working storage stayed **1,245,290,965 bytes**. These timings include Editor scheduling and writes and do not establish a speedup or realtime deadline guarantee. [Full reports and provenance](benchmarks/transport-cleanup-2026-10-07.json).

Validation: 140/140 core/GPU tests passed; the silent-source playback check passed after the final source-control change. All five scenes compiled their bindings without missing scripts. Station and Church recording/replay each passed in isolation. Runtime/shader code in `Assets/Scripts` and `Assets/Shaders` shrank from 9,360 to 7,643 lines and 39 to 34 files. The final Mac standalone build succeeded with zero errors and 1,301 warnings. The combined EndToEnd selection reproduces the pre-existing Church observation-clock and Station null-path failures. Run the scene fixtures separately while investigating the combined-run failure; do not describe the combined suite as passing. Metal also retains its existing potentially-uninitialized-function shader warnings.

## Dominant-edge diffraction — 2026-10-07

Replaced the wavelength/proximity branch with one explicit, visible source–edge–listener route per pair. Route geometry runs once on the GPU and frequency bins share it. Low-ray-budget sound no longer depends on a random ray happening to hit near a corner. Coplanar triangle seams and collinear 2D joins are excluded; both legs must be clear. The scalar knife-edge magnitude softens the opaque-screen shadow boundary at strength 1 and retains path delay, arrival direction and local Doppler. [Contract and limitations](simulation-contracts.md#edge-diffraction-2026-10-07). The previous renderer remains in Git at `f7bd12a`.

**Unity 6000.4.1f1 / Apple M5 / Metal:** 139/139 numerical checks passed (32.67 s), including 13 new diffraction cases. These cover 2D/3D and BVH/brute traversal, an independently integrated Fresnel reference (<0.4 dB for the tested shadow cases), grazing continuity, ray-budget independence, moving edges, visibility and transparent/same-side geometry. CPU/GPU binaural audio agrees for IR and spectral synthesis with moving edges and head rotation. The standalone app rebuilt with zero errors; compiler warnings remain. It was not relaunched.

**Scene validation limitation:** station walking/recording/replay passed in an isolated run (33.10 s), and Church walking/door/replay passed separately (24.70 s). Combined category runs failed on recording-path/observation-clock state; a fixture serialization experiment did not resolve this and was reverted. The aggregate runner issue remains open. These results do not support claiming a clean combined end-to-end run. Raw failures and passing selections are retained below.

One before/after offline render used identical saved station multiplayer source/pose inputs: six sources, one binaural listener, 564 triangles, 530,501 stereo frames (12.03 s), FFT 1,024/hop 256, response step 1,024, five bounces and batches of eight responses.

| Measurement | Previous proximity model | Explicit edge route |
|---|---:|---:|
| Renderer wall | 6.234 s | 4.744 s |
| Solve/wait bucket | 0.794 s | 0.987 s |
| Audio/wait bucket | 4.775 s | 3.344 s |
| Reserved numeric memory | 1,092.6 MB | 1,245.3 MB |

These are one Editor run each, not medians or isolated GPU timer measurements. No statistical speedup or realtime deadline claim is made. Memory grows **14.0%** because the existing DSP reserves worst-case filters for the extra arrival slots. Scene/state/event/source and direct-diagnostic hashes match; wet output changes as intended and contains only finite samples. PCM upload/readback volumes are unchanged. Both runs use numeric `diffractionFactor=1`, whose meaning changed from proximity radius to pressure strength. [Full reports, provenance and test outcomes](benchmarks/diffraction-2026-10-07.json).

The model remains a single dominant thin-screen magnitude approximation. Thick obstacles needing multiple corners, diffracted reflections, full wedge/impedance physics and complex Fresnel phase are not implemented. The preview remains ineligible as a physical training teacher.

## Microphone, LAN voice and offline multiplayer replay — 2026-10-06

Implemented in the actual Unity project; previous scene/session foundation is preserved at `e605dc7`. Church and TrainStation have four editable player/spawn slots, opt-in microphone controls, Host/Join over direct LAN, owner-driven player/head poses, host-driven actors and nearby door requests. One local head renders all audible sources. Managed Concentus 2.2.2 supplies Opus/Speex; NGO 2.7.0 and Unity Transport 2.6.0 supply transport/clocks. There is no separate network renderer or NetworkObject hierarchy for each prop.

Take v3 stores the exact canonical source samples consumed by the renderer, with source-block availability/loss status, clock epochs and hashes. Solo movement stays compact. LAN takes save actual presented numeric poses plus geometry/clip inputs and per-source PCM; wet output is optional. Completed takes replay from the scene picker with networking and microphone off. Numeric replay feeds the existing queue ahead of playback and respects its capacity. Save drains tails across frames; it no longer tries to push an entire tail into a full output queue. Self-muted pending microphone data cannot stall other audible sources.

**Validation: Unity 6000.4.1f1 / Apple M5 / Metal.** 126/126 numerical cases passed in 39.92 s; 3/3 scene end-to-end cases passed in 36.43 s (one overlaps the numerical suite). Scene cases include walking onto the departing train, door interaction, compact reconstruction and replay. The explicit MacBook Pro Microphone test passed: one second through production capture/resampling, zero microphone PCM saved/transmitted, and correct closure of the originally opened device even after the configured device name changes. Unity reports 7.05 s for that test run and 12.24 s including its fixture lifecycle; domain reload adds wall time.

Two Development Players ran each scene on localhost with synthetic 440/660 Hz voice through production Opus, transport, source history, capture and offline replay. Host captures lasted 18 s, clients 11 s; both peers moved and the shared actor moved. These are one final run per scene, with all prior failed trials retained in the [machine-readable results](benchmarks/multiplayer-2026-10-06.json).

| Scene | Voice packets received, host/client | Concealed voice frames / late packets | Final replay error | Live underrun / late PCM frames |
|---|---:|---|---|---|
| Church | 249 / 248 | 0 / 0 on both peers | 0 position, 0 orientation | 0 / 0 on both peers |
| TrainStation | 249 / 249 | 0 / 0 on both peers | Host < 0.000001 m, client 0; both 0° | Host 0 / 0; client 1 / 1 |

Underrun and late counters overlap; do not add them. The single station frame is about 23 microseconds at 44.1 kHz. Capture counters are measured continuously, not queried after the live renderer has been destroyed. Both peers save nonzero local/remote source files and replay without a server or microphone. Final pose checks supplement the numerical timeline tests; they are not a claim that distributed physics is deterministic.

**GPU upload fix:** two High offline exports exhausted system memory; macOS reported 73,864 MB for the second killed Editor. Repeated updates of the large source-history buffer were replaced with mapped writes of new ranges. The previous PCM readback completes before these unsynchronized writes; the whole next input batch is uploaded before any readers dispatch. History reserves the pending batch as well as filter/path support. This invariant matters because [Unity SubUpdates](https://docs.unity.com/en-us/engine/6000.6/script-reference/unityengine/computebuffermode/subupdates) does not supply automatic write/read synchronization. The managed upload scratch buffer is also removed. Earlier 1,024/2,048-step live trials had queue failures; the final 1,024-step preset passes both scenes without increasing playback lead.

The same saved multiplayer source/pose input now exports **618,701 stereo frames (14.03 s including tail)**: Low step 4,096 in **4.34 s**, High step 256 in **14.60 s**, FFT 1,024/hop 256, five bounces, transition 256. Source hash is identical across both. Low WAV/diagnostic-response hashes are unchanged from before the upload fix. Sampled Editor footprint peaked at **4.8 GB** during the successful run; numeric working buffers are approximately **912 MB**, a different measurement. This is one run, not a statistical benchmark. RMS difference 0.000618 and SNR 29.54 dB describe disagreement between budgets, not physical error or teacher acceptance.

**Remaining limits:** playback lead is still 16,384 samples (371.5 ms at 44.1 kHz), plus DSP/device delay; remote presentation adds 100 ms. Two-machine LAN, four simultaneous players, longer sessions, ADC timing, drift correction and AEC remain open. Managed codec resampling has numeric coverage, but Windows/DirectX and other microphone hardware need their own checks. Existing Metal warnings remain. Saved takes can provide paired training inputs, but spectral transport is still an unvalidated teacher and model training/enrollment are not implemented. Synthetic smoke takes were moved out of the ordinary recordings list; no speech was enrolled in training.

Raw local evidence: `.context/voice/core-results-final.xml`, `e2e-results-final.xml`, `device-results.xml`, host/client results 11 (Church) and 12 (station), `render-pair-result.json`, `render-memory.json`, and `comparison-20261007-013650/`. The standalone app is `.context/voice/Acoustics.app` (TrainStation first, Church included). [Usage](../README.md#microphone-and-lan-multiplayer).

## Shared scene session and Church — 2026-10-06

Implemented `SceneSession` for input, actor stepping, save/replay and cursor ownership. `StationDemo` now only advances the train. `SlidingDoor` is shared by station and church. Renamed the existing recorder/reconstructor to `SceneRecording`/`SceneReplay`, preserving their asset GUIDs; new compact headers use `scene-input-from-init-v1` with player and ordered actor settings. The station-v2 payload reader remains; matching project/build requirements remain strict. No microphone/network runtime or package dependency was added.

Added the editable Church blockout and common Player/TestSpeaker prefabs. Station retains its source clips and authored motion; its player now uses the shared prefab. Church contains 54 registered objects, 612 acoustic triangles, one source and one binaural head. Station remains 53 objects, 564 triangles, two sources and one head. All five project scenes load without missing scripts. Church uses the existing diagnostic chime and provisional material preset; its three-second tail and current transport are not physically validated church labels. Old room/legacy-renderer retirement and material migration remain pending.

Unity **6000.4.1f1, Apple M5, Metal**: **110/110 numerical/format/GPU cases passed** (26.23 s). This includes both input schemas and independent actor-ID/order/settings checks. The separate EndToEnd selection passed **3/3**: Church walking/door entry/replay, station walking aboard/motion/replay, and the existing offline comparison (already counted among the 110). Both scene journeys match exact state/event hashes after isolated reconstruction and final live replay transforms. Test waits remain bounded; generated captures are removed by teardown. Compile and scene-reference checks passed.

Realtime remains **intermittent**: the first explicit 20-second device test measured 882,688 frames and failed with 117,760 new underrun frames and the same number of late frames (~2.67 s, overlapping counters). Overflow, channel mismatch and callback-budget counters were zero. A diagnostic rerun with unchanged runtime/audio settings passed: 886,784 frames and zero misses; maximum async completion 102.5 ms, maximum observation gap 5,121 frames (~116 ms), maximum validation 0.422 ms. The second run does not erase the first failure or establish its cause. Lead stayed 16,384 samples (~371.5 ms); no latency increase, weakened threshold or scheduling fix was applied.

Local evidence: `.context/scene-session/numeric.json`, `e2e.xml`, `realtime-first.xml`, `realtime-diagnostics.xml` and `church.png`. Test timings exclude some domain-reload overhead; individual scene case durations include setup and cannot be added to the root duration. Windows/DirectX, standalone builds, Church physical convergence and actual multiplayer/microphone collection have not been validated by this change. [Multiplayer training capture contract](simulation-contracts.md#turning-multiplayer-captures-into-training-examples).

## Test suite cleanup — 2026-10-06

Replaced the manual menu/batch runners with selectable Unity Test Framework cases in the existing C# files. Removed obsolete legacy FFT tests, redundant comparison runs, shared assertion counters and reflection wrappers. Fixtures own their scenes/files; CPU convolution uses pairwise parameter coverage with dedicated boundary regressions. Three assembly definitions and two friend declarations replace the wrappers; no new C# files or packages. Test C# remains nine files: 161,469 → 162,834 bytes (+0.85% including the new integration coverage).

Unity **6000.4.1f1, Apple M5, Metal**: **107/107 default cases passed**, no failures or unexpected skips. Unity reported **54.86 s** for the suite; Play-mode reload overhead makes elapsed time longer. The 106 numerical/integration cases took **20.90 s**; all 106 passed again in **24.97 s** after final source cleanup. Station recording/reconstruction/replay took **40.70 s including setup**, checking exact state/event hashes after walking aboard, departure, jumping and head movement. Offline comparison is the other default end-to-end case. The explicit realtime case is excluded from this count.

The separately selected realtime test passed in **29.41 s including setup**. After warmup, it measured **885,760 device frames** (~20.1 s at 44.1 kHz) with **zero new underrun, late, overflow, channel-mismatch or over-budget callback counters**. The Game view was active and the train/player moved. Playback lead remained **16,384 samples (~371.5 ms)**. This short run does not establish low latency or eliminate the earlier scheduling failures below; no runtime DSP optimization was made by this cleanup. Existing Metal shader warnings remain.

Local raw results: `.context/test-runner/suite-final.xml` and `realtime-final.xml` (final numerical rerun: `numeric-final.xml`). [Run instructions](../README.md#checks). Tests report individual cases, not the historical assertion totals below. DirectX and standalone validation remain open.

## GPU audio port 2026-10-06

Raytraced audio now defaults to GPU DSP, with **Audio Backend → Cpu Reference** retaining the original CPU algorithms. Scene/settings, the source-event timeline, live/offline scheduling and PCM output stay shared. GPU path grouping, material and directional ear filters, continuous Doppler sampling, path FFTs, per-ear sums/IFFT and response fades no longer require CPU path/spectrum readback. IR and Spectral both work in 2D/3D with multiple sources/listeners. No packages or scene-format changes were added. Previous production code is preserved at `33d8a50`; the older legacy GPU accumulation/IFFT remains in Git and in the legacy scenes.

**Validation:** 2,168 core, 5,458 GPU, 12,487 transport and 44 new audio assertions pass on Unity 6000.4.1f1 / Apple M5 / Metal. The new fixtures compare GPU/CPU IR and Spectral output for mono/binaural, linear/sinc delay, movement, head turns, source stops, receiver fades/cuts, short deliveries, circular-history wrap and larger transforms/group tables. Live/offline output and response-batch invariance pass in both dimensions with two sources and two heads. These are numerical checks, not independent physical experiments. Repeated station output hashes match for every configuration after fixing a shared-memory barrier for small FFT filters.

Same numeric station take: 21.1766 seconds plus a one-second tail, 44.1 kHz, two sources, one binaural listener, 53 objects and 564 triangles. Full-quality runs use FFT 1024 / hop 256, one ray per frequency bin and five bounces. Low/High differ only in response step (2048/512). Three sequential warmed Editor runs per GPU configuration:

| Configuration | GPU median wall | GPU range | Earlier CPU wall (one run) |
|---|---:|---:|---:|
| IR Low | 7.77 s | 6.72–9.62 s | 10.26 s |
| IR High | 16.70 s | 16.10–22.52 s | 31.93 s |
| Spectral Low | 9.26 s | 8.89–10.17 s | — |
| Spectral High | 17.38 s | 15.74–17.67 s | — |
| Spectral smoke (smaller grid) | 5.21 s | 3.92–5.24 s | 115.11 s |

CPU figures are recorded earlier in [the Doppler benchmark](benchmarks/doppler-2026-10-06.json); they were not repeated alongside this run. The smaller spectral smoke uses FFT128/hop64/one bounce. Its acceleration ratio does not establish a full-budget CPU/GPU speedup; no completed full-budget CPU spectral baseline is claimed. GPU and CPU IR Low/High and the matching spectral smoke differ by less than 2.2e-6 peak and 7.8e-5 relative RMS, with identical diagnostic response hashes. All three GPU repetitions match bit for bit within each configuration.

The full station budget reserves about **387 MB of numeric working storage**, mostly worst-case GPU filter/path capacity; this is not process memory. Streaming source/pose uploads and PCM/metadata readback are each about 7.8 MB for the take. Routine full-path readback is gone; small direct diagnostic probes remain. Offline wall includes allocation, dispatch, completion waits and writing. `responsePreparationSeconds` is CPU submission time, not shader execution, and GPU work can finish during the PCM wait. Timing variation includes Editor scheduling and output writes; the report retains all runs rather than selecting the fastest.

**Live:** a 20-second station run with scripted walking/head turns and the moving train recorded and finished without queue exhaustion. The initial implementation's readback per observation was replaced with one readback per known audio batch; DSP can be dispatched immediately after tracing. No compilation ran during this final scene test. Maximum pending observations were four. The interval from the two-second snapshot to just before stopping accumulated **48,144 underrun frames and 48,144 late frames**; these overlap and must not be added. There were no overflows, channel mismatches or callback-budget overruns. The largest observation gap was 12,287 samples (~279 ms). The 16,384-sample lead remains unchanged. This is improved throughput and functioning capture, **not a deadline-safe or low-latency claim**.

Remaining costs include serial grouping per pair, per-bin path reduction, worst-case filter storage and control-thread/readback cadence under graphics load. GPU timer queries and Windows/DirectX/standalone deadline validation remain open. Directional training exports, inference and physical validation of diffuse/diffraction heuristics are still separate work. [Raw reports, source hashes, waveform errors and live counters](benchmarks/gpu-audio-2026-10-06.json).

## Doppler and separate synthesis routes — 2026-10-06

Direct source/listener motion now uses retarded emission time and velocity-derived delay rates. The optional spectral route resamples each path's source time and applies short material/ear filters with overlap-save. The previous scene-IR convolver is retained as `ImpulseResponseAudio` and remains the default. IR snapshot crossfades do not provide continuous Doppler. [Contract and limits](simulation-contracts.md#continuous-doppler-and-synthesis-methods-2026-10-06).

**Validation:** 2,168 core, 5,458 GPU and 12,487 transport assertions pass in Unity 6000.4.1f1 on Apple M5/Metal. Analytic source/listener and normal moving-mirror tone tests pass with crossfades disabled; tests cover linear/sinc resampling, pitch-up alias rejection, material speed and slow two-medium root convergence. Static convolution/long-HRIR oracles pass for both synthesis methods. The previously failing station pose at sample 698,368 now passes.

Single warmed IR renders reuse the same 21.1766-second take (53 objects, 564 triangles, two sources, one head), FFT 1024/block 256, one ray per bin, five bounces, static BVH, batches of 64 and a 44,100-sample tail. Low/High change response cadence only. These are throughput observations, not a controlled speedup comparison with the previous measurements.

| IR render work | Low: step 2048 | High: step 512 |
|---|---:|---:|
| Renderer wall time | 10.257 s | 31.933 s |
| Transport/staging/readback waits | 0.634 s | 2.086 s |
| Response/filter preparation | 7.031 s | 26.545 s |
| Audio synthesis excluding preparation | 1.920 s | 2.520 s |

The full-take **spectral smoke** uses FFT 128/block 64, one bounce, one ray per bin and response step 2048. It completes in **115.105 s**, with 3.152 s preparation and 110.902 s synthesis. This smaller workload verifies the new route end to end; it is not comparable fidelity to the IR runs above. All three WAVs have the expected frame count, finite PCM and matching recorded hashes.

The earlier full-budget spectral trial ended incomplete: a fixed-point material travel-time solve failed at sample 698,368. A response-only probe reproduced the failure; a bracketed solve fixed it. That trial also crossed an overnight interruption, so its wall timing is unusable. No completed final full-budget spectral Low/High benchmark or realtime claim is made. The per-path CPU method remains opt-in; acceleration/complex moving-boundary physics and training-teacher validity remain open. [Raw reports, hashes and validation](benchmarks/doppler-2026-10-06.json). Previous production code remains at `6cc47f1`.

## Renderer correctness repairs — 2026-10-05

This earlier repair established the scene-IR method now retained as `ImpulseResponseAudio`: partitioned linear convolution and a circular source-spectrum history. Full propagation/fractional delay and HRIR support no longer wrap inside a synthesis window. Response fades operate on output samples; unrelated indirect slots are no longer interpreted as velocity. Frequency sampling uses a configurable integer rays-per-bin budget and seed. First-order image-source mirrors are separate from the scattering-controlled diffuse preview. Compact recording failures invalidate the save, and reconstruction preserves initial source/surface activity. [Current contract and remaining limits](simulation-contracts.md#renderer-repairs-2026-10-05).

**Validation:** 2,168 core, 5,458 GPU and 12,275 transport assertions passed in Unity 6000.4.1f1 on Apple M5/Metal. New independent tests cover integer/fractional linear/sinc convolution, full long HRIRs, replaced indirect slots, listener fades, analytic mirror gain/time and shared edges, legacy settings migration, failed saves and activation. Existing streaming/batch/multiple-endpoint checks also pass. Assertions are not independent physical experiments; DirectX remains untested.

Reused the same 21.1766-second station take (53 objects, 564 triangles, two sources, one head), reconstructed before changing its project revision. Both final renders use one ray per bin, five bounces, FFT 1024/block 256, static BVH, 64-response batches and a 44,100-sample tail. Low/High change response cadence only.

| Latest measured render work | Low: step 2048 | High: step 512 |
|---|---:|---:|
| Renderer wall time | 15.173 s | 54.726 s |
| Transport/staging/readback waits | 0.609 s | 2.633 s |
| Response/filter preparation | 10.791 s | 46.284 s |
| Source-block convolution and output synthesis | 3.023 s | 4.475 s |

Filter preparation is included in `audioSeconds`, and is now also reported as `responsePreparationSeconds`. Whole-job wall time includes preparation and file writes; it excludes CLI compilation, the already-completed input reconstruction and final output hashing. Earlier optimized passes measured 14.307 s Low and 35.626 s High. A final empty-partition FFT skip preserved their WAV hashes exactly; these variable timings do not demonstrate a speedup from that change. Both final reports cover observed response support (23,452 samples) with the requested tail. Working numeric buffers are approximately 84 MB, not total process memory.

**Live is still not realtime-safe.** A 25.52-second recorded run with scripted walking/head input saved and reconstructed successfully. Its producer spent 32.98 s in CPU audio work. The instrumented run also compiled its control probe during Play (5.96 s), causing a large observation/completion stall, so its counters are not a clean production latency benchmark. It recorded 1,771,520 underrun frames, 1,038,405 late frames and no callback-budget overruns or overflows. These overlapping counters span the queue lifetime/draining and must not be added or converted to a percentage of take duration. Playback lead remains 16,384 frames (371.5 ms at 44.1 kHz). Filter rebuilding and response delivery require further work; no low-latency claim is made.

Diffuse pressure weights/PDFs, higher-order mirror connections and diffraction remain unvalidated, so training eligibility remains false. [Raw reports, provenance, hashes, test counts and live/replay evidence](benchmarks/renderer-repair-2026-10-05.json). The old implementation remains at `1586cbb`, and the independent audit at `332f8f3`.

## Earlier material travel time and head interpolation — 2026-10-05

Raytraced paths now accumulate travel time and pressure attenuation per traversed medium: `time += distance * IOR / airSpeed`, `gain *= exp(-damping * distance)`. Air absorption applies only to air segments. Closed meshes/2D loops define solid interiors; open sheets have zero thickness. Containment handles native collider geometry, mirrored winding, interior sources and nested solids. All surface materials on one closed object must share its interior IOR/damping. Overlapping solids use the smaller enclosed volume, with object-order ties; this is a deterministic authoring convention, not a mixed-medium physical model. Indirect rays use the incident and destination media for refraction instead of unconditionally exiting into air.

Both shared audio modes now interpolate the paired KEMAR filters on the measured azimuth/elevation grid. The spectral mixer also honors `transitionSamples` with old/new response crossfades at synthesis hops, retaining each response’s own delay/frequency mapping. Cuts reset the transition and Doppler history. TrainStation and its builder enable 512-sample transitions (11.6 ms at 44.1 kHz). World arrival directions stay in the existing 32-byte GPU path; the per-arrival 710-direction GPU search is removed. No new packages or take payloads are required. Solver/DSP provenance versions distinguish the changed output.

**Validation:** Unity 6000.4.1f1 / Metal passed 2,159 core checks, the existing GPU suite and 9,686 spectral checks. New fixtures cover analytic delay/loss through nested slabs; endpoints inside solids; open sheets; moved/inactive volumes; reflected paths within a solid; native 2D/3D box/sphere/capsule exports and mirrored geometry; frequency-dependent speeds; measured-HRIR reproduction; angular boundary/rear-seam continuity; ear symmetry; the exact WOLA transition envelope and cuts; and live/offline/batch agreement with transitions enabled. Counts are numerical assertions, not independent physical scenarios or listening evaluations.

The same reconstructed 21.1766-second station take completed three Low/High comparisons (2,048/512 response steps, otherwise identical settings including 512-sample transitions). Low wall times were **11.167 / 12.059 / 6.352 s**; High **9.696 / 14.845 / 10.674 s**. Output samples were finite, and each quality produced identical WAV hashes across all three runs. These variable Editor measurements establish repeatable output and completion, not a speedup. The second CLI call timed out at 25 seconds while its Unity job continued; its completed manifest and outputs were subsequently verified. [Settings, provenance, timings and hashes](benchmarks/material-head-fixes-2026-10-05.json).

**Remaining limits:** deterministic source-to-listener transmission still follows a straight connection, while indirect rays refract; this is not a general refracted-path boundary-value solver. Diffraction/reflection normalization, exact convolution and moving-boundary acoustics remain approximations. HRTF interpolation is linear over generic measured filters, with elevation clamping outside their coverage. These fixes do not remove the existing realtime buffering and Editor deadline misses documented below.

## Directional spectral renderer — 2026-10-05

TrainStation now selects the shared **GPU raytraced spectral preview** instead of the direct-only baseline. The old 2D shader and the direct CPU/GPU mode remain intact. The new path adapts reflection/scattering, transmission/refraction, material/air loss and edge diffraction to both segments and 3D triangles, retaining arrival direction through per-ear KEMAR filtering. FFT/hop, rays, bounces and response cadence are configurable. Delay-derived frequency reconstruction passes approaching/receding tone checks; it remains a frozen-geometry approximation, not a retarded-time moving-boundary solver.

The source/event timeline, native object extraction, static/moving BVH, asynchronous batches and compact recording workflow are shared. There is no new package or external dependency. Source spectra are shared across listeners; delay/frequency/ear coefficients are prepared once per response. `responses.bin` remains direct diagnostic probes; raytraced WAVs include the richer transport. Neither those probes nor this preview establish a physically converged training dataset.

**Validation:** 1,417 core checks, the existing Metal FFT/direct/BVH/batch suite, and 6,416 spectral numerical checks passed. The latter cover versioned material round trips, 2D/3D transmission/reflection/diffraction, head rotation, moving/static BVH agreement, identity gain/latency across FFT/hop choices, approaching/receding Doppler, and multi-source/listener live/offline agreement with irregular observations, teleport cuts, sound events and tails. GPU batch sizes one/four produce byte-identical spectral WAVs in the fixtures. DirectX has not been tested.

A new **21.1766-second compact station recording** (`station-20261005-223835-007`) includes the automatic train departure. The matching scene contains 53 objects, 564 triangles, two sources and one binaural head at 44.1 kHz. Both renders use 512 rays/pair, five bounces, FFT 1,024 / hop 256, static BVH, 64-response batches and a one-second output tail. Low/High change response step only: 2,048 / 512 samples. These are temporal quality settings, not converged ray-budget references.

| Timed work | Low | High |
|---|---:|---:|
| Renderer wall time | 6.358 s | 8.669 s |
| Preparation | 0.376 s | 0.413 s |
| Motion evaluation | 0.044 s | 0.192 s |
| Solve, staging and readback waits | 0.483 s | 1.681 s |
| Response preparation and audio synthesis | 5.025 s | 5.945 s |
| WAV/probe writes | 0.418 s | 0.418 s |

Compact-input reconstruction took **1.132 s**, shared by both runs. **The entire reconstruction + comparison job took 16.584 s**, including warmup, both renders, final hashes and sample comparison. Individual renderer wall times end before final hashes. These are single-run measurements, not medians; caches were warm. CPU spectral synthesis dominates this workload. The output difference has RMS 0.000937 and peak 0.01544; this measures disagreement between response cadences, not physical error. Training eligibility remains false. [Raw timings, settings, hashes and live diagnostics](benchmarks/spectral-station-2026-10-05.json) preserve the measurement.

**Live limitation:** explicit nonblocking readback polling reduces delayed delivery, and the station lead is now 16,384 frames (**371.5 ms**) plus FFT latency (**23.2 ms**), path/HRIR and device latency. A 7.3-second stationary run had no deadline misses. The moving 21.2-second run had 8,192 late/underrun frames (**0.186 s**) and a 371.5 ms maximum observation gap; no overflow or callback overrun. A separately instrumented 40.2-second followup had 77,824 missed frames and a 1.045 s observation gap, including an in-Play native property inspection and stop. Do not treat the live path as underrun-free or low-latency production audio. Improving main-thread scheduling/synthesis cost is still required; a larger playback buffer is a latency tradeoff, not a compute optimization.

At that revision the phase/material coefficients and diffraction were geometric heuristics, and WOLA was not exact arbitrary-length convolution. The repair above replaces its DSP with partitioned linear convolution; the physical-model limits remain. See [the operating contract](../README.md#render-offline) for supported controls and limitations. Original pre-integration code is preserved at `6e83274`; subsequent commits retain this combined path.


## 2026-10-05 static geometry BVH

The current BVH now indexes **only geometry whose pose has stayed fixed**. The first position/rotation change removes an object's primitives from the tree for the rest of that solver session; moving geometry uses the existing exact direct intersection tests. Stopping or returning to the initial pose does not put it back. The tree rebuilds only when membership changes, and every GPU batch shares one tree. This replaces the previous per-response moving-bound refits and node snapshots, without adding scene tags, recording fields, dependencies or another traversal mode.

Both station recordings finish with **408 static and 156 moving triangles**. Each performs only **two tree builds**: initialization and train departure. Continued motion and later stops cause no more rebuilds. The older full BVH needed 11,425 updates at High in the three-lap run. Pose comparisons and direct tests of moving triangles still have a cost; eliminating refits does not eliminate all motion work.

Measured code: **`2e1951f`**. The prior implementation remains in Git at `3b9ffb6` and the earlier optimization commits below. Reused the same takes, quality steps, hardware, harness and three-repetition/warmup method described in the next section; GPU capacity remains 64. The machine was on battery during this pass and even the fresh brute-force baselines were substantially slower than the earlier session. The cause of that timing shift was not isolated. Compare methods **within this pass**, not the absolute times across sessions, and do not infer duration scaling by comparing these two take lengths.

Median complete offline render seconds, including validation and output files:

| Case | Short brute force | Short static BVH | Three-lap brute force | Three-lap static BVH |
|---|---:|---:|---:|---:|
| CPU Low | 0.420 | 0.403 | 5.208 | 5.144 |
| CPU High | 0.945 | **0.759** | 12.613 | **11.637** |
| GPU Low | 0.411 | 0.399 | **4.861** | 5.196 |
| GPU High | 0.779 | 0.737 | 11.382 | 10.527 |

CPU High total time fell **19.7% on the short take and 7.7% on the three-lap take**. The short ranges were 0.901–0.967 s versus 0.758–0.760 s; long ranges were 12.404–12.975 s versus 11.564–11.725 s. CPU solve time fell from 0.235 to 0.081 s short and 3.022 to 2.084 s long. Other work limits the overall improvement: the long static-BVH CPU High run spent 4.560 s evaluating motion and 2.378 s writing output.

GPU results are mixed. Long Low regressed; High median totals improved, but the long High ranges overlap (10.525–11.476 s brute versus 10.502–10.568 s static BVH), and file-writing variation contributes. The option therefore stays **off by default**, available as **Static geometry BVH** in render settings and **Direct Settings → Use Bvh** on the manager. No live-latency improvement is claimed by these offline runs.

The GPU High numeric-buffer estimate is **2,935,164 bytes** with this static split, compared with 2,860,130 for brute force and 5,130,253 for the earlier BVH that stored a tree per response frame. This is buffer accounting, not total process memory.

**1,417 core checks plus Metal GPU checks passed.** Added fixtures cover first motion inside a batch, mixed fixed/moving geometry in 2D/3D, returns/stops, moving endpoints, activity changes and a formerly static object moving late until no static geometry remains. Warmed CPU solves still allocate no managed memory. All four new benchmark groups preserve the original same-backend response/WAV bytes exactly, at both qualities, including the complete three-lap outputs.

[Raw samples](benchmarks/station-2026-10-05.csv) now contain **132 measured renders**, including 48 new samples under `static-short-brute`, `static-short-bvh`, `static-long-brute` and `static-long-bvh`. Full local reports and one output per case are under `.context/traversal-benchmark/` with those names. The reproduction call below still applies; `useBvh=true` now selects this static/moving split. `bvhStaticPrimitives` reports final membership, and `bvhUpdates` now counts tree builds including initialization, rather than per-pose refits.

## 2026-10-05 station offline benchmarks

This earlier pass measured the **full moving-geometry BVH**, since replaced by the static split above. **Keep offline GPU batching enabled; keep BVH optional.** Batching reduced the short station recording's high-quality GPU render from **1.020 s to 0.361 s (2.83×)** with byte-identical responses and audio. BVH helped CPU rendering during the short, mostly stationary take, but its moving-geometry maintenance cost outweighed traversal savings in the three-lap take. The station still defaults to CPU with `useBvh=false`; offline GPU renders use the existing 64-response batch setting. No scene objects or sound assets changed for these measurements.

### Workload and timing

Measured in the actual project's Unity **6000.4.1f1** Editor on **Apple M5 / Metal**, in Edit mode. Both recordings use the real TrainStation geometry: **53 objects, 564 triangles, two sources, one binaural listener**, 44,100 Hz stereo, linear fractional delay, 64-sample transitions and no appended tail. The take fixes the scene, recorded motion and source events for every run.

| Recording under `Recordings/` | Duration | Output frames | Low responses | High responses |
|---|---:|---:|---:|---:|
| `station-20261005-195909-093` | 12.701315 s | 560,128 | 274 | 2,188 |
| `station-20261005-192306-273` | 95.805556 s | 4,225,025 | 2,064 | 16,505 |

Low uses a **2,048-sample response step** (46.44 ms); High uses **256 samples** (5.805 ms). Both use the same direct/opaque transport and binaural DSP. These labels change response cadence, not reflection fidelity, sample rate or ray count; neither is an approved reflection-training reference.

Each case has one discarded warmup and three measured repetitions, alternating case order. Tables show median **whole-render wall seconds**, including opening/checking the take, preparation, motion evaluation, tracing, GPU staging/readback waits, DSP, writing WAV/responses and final output hashes/reports. File caches are warm. The harness report and duplicate-output deletion happen outside this timer. Component times below come from the median-total repetition; they are not independent medians.

Inputs here are existing full diagnostic takes. **Reconstructing compact inputs into a numeric take is a separate, untimed preparation step**, shared by Low and High. No gameplay/physics rerun, visual rendering or live audio deadline is included. These results measure offline throughput, not live latency. At CPU High, only 20.62% of short-take frames and 2.74% of long-take frames contain nonzero audio under the direct/opaque model. Do not extrapolate their DSP cost to continuously audible scenes or reflections. Larger geometry and DirectX remain unmeasured.

### Whole-render results

Short recording, seconds:

| Implementation | CPU Low | CPU High | GPU Low | GPU High |
|---|---:|---:|---:|---:|
| Original brute force, synchronous GPU | 0.201 | 0.452 | 0.313 | 1.020 |
| BVH, full hierarchy refit | 0.207 | 0.410 | 0.314 | 0.979 |
| BVH, refit affected branches | 0.193 | 0.384 | 0.283 | 0.933 |
| Brute force, GPU batches of 64 | 0.204 | 0.449 | 0.205 | **0.361** |
| BVH, GPU batches of 64 | 0.198 | **0.390** | 0.213 | 0.423 |

The final brute-force GPU High range was **0.358–0.371 s**, versus **0.995–1.175 s** originally. Its solve/staging/readback time fell from 0.636 s to 0.035 s. Batching removes thousands of individual waits; it does not change intersection physics. CPU execution still solves one response at a time. The small Low CPU differences are within observed run variation; one BVH Low repetition took 0.502 s, retained in the raw data.

Three-lap recording, median seconds (minimum–maximum):

| Backend | Brute force | BVH |
|---|---:|---:|
| CPU Low | **0.928** (0.921–0.953) | 0.982 (0.953–0.983) |
| CPU High | **2.462** (2.454–2.672) | 2.598 (2.582–2.663) |
| GPU Low, batch 64 | 1.126 (0.933–1.130) | 1.019 (0.973–1.038) |
| GPU High, batch 64 | **2.007** (1.973–2.023) | 3.057 (2.729–3.144) |

BVH High regressed 5.5% on CPU. High solve time, including BVH maintenance, rose from 0.689 to 0.819 s on CPU and from 0.226 to 0.910 s on GPU. GPU BVH also stages a hierarchy per response frame; its reported numeric working buffers grow from 2,860,130 to 5,130,253 bytes. These are buffer estimates, not total process memory. Low GPU totals overlap substantially and include variable file-writing costs, so they do not establish a BVH improvement. There was no synchronous-GPU baseline run for the long take.

### Remaining costs

Final brute-force GPU High, seconds:

| Work | Short take | Three-lap take |
|---|---:|---:|
| Open/validate inputs | 0.016 | 0.034 |
| Prepare renderer | 0.015 | 0.050 |
| Evaluate recorded motion | 0.116 | 0.861 |
| Solve, stage and wait for readback | 0.035 | 0.226 |
| Audio DSP | 0.082 | 0.094 |
| Write responses/WAV | 0.047 | 0.378 |
| Final hashes/report/disposal | 0.024 | 0.178 |
| Total, including other loop/setup work | **0.361** | **2.007** |

Motion evaluation is the largest measured component after batching. Output writing and hashing also matter. Optimize those only against the same recordings and exact-output checks. BVH build/update and upload/dispatch counters are nested diagnostics, not extra times to add to the table. With asynchronous batches, `dispatchReadbackSeconds` measures submission work; it is **not GPU kernel execution time**. `solveSeconds` includes the completion wait. The total does not require estimating GPU execution from CPU submission timestamps.

### Correctness and preserved history

All optimized short-run `responses.bin` and stereo WAVs match the baseline **byte-for-byte within each backend**, at both qualities. Long-run BVH and brute-force outputs also match exactly. Every repetition matches its own case's hashes. CPU versus GPU retains the original small numerical differences: short-take RMS error 7.22–7.87e-8, maximum 2.53e-6; long-take RMS 2.45–2.50e-8, maximum 2.18e-6. Cross-backend output is not claimed byte-identical.

**1,387 core checks plus Metal FFT/direct/BVH/batched/live-offline checks passed.** Coverage includes 2D/3D, moving/rotating/inactive/empty geometry, multiple source/listener pairs, zero-allocation warmed CPU solves and batch sizes 1/3/8/64 preserving responses and audio with cuts, gain/stop events, tails, multiple listeners and both delay filters. The Editor was left in Edit mode with TrainStation clean and the manager Idle.

| Git revision | Preserved implementation |
|---|---|
| `877c21c` | Original renderer plus repeatable benchmark harness; before BVH/batching changes. |
| `7c892fc` | CPU/HLSL BVH and brute-force correctness comparisons. |
| `2c9ffb6` | Refit only hierarchy branches affected by moved geometry. |
| `0d6e11b` | Offline GPU response batching with exact-output checks. |
| `fdb6ec8` | Keep one inspectable output per benchmark case; measured code used for long runs. |

Old source is kept in Git, without duplicate runtime files. [Raw measured samples](benchmarks/station-2026-10-05.csv) preserve all **84 measured renders from this pass**, full revisions, input/output hashes, settings that vary and timing components, alongside the later static-BVH pass. Full local reports and one output per case live under `.context/traversal-benchmark/{baseline,bvh,sparse,batched-brute,batched-bvh,long-brute,long-bvh}/`; recordings and binary outputs are not committed.

### Reproduction

In the graphics-enabled Editor, stop Play and invoke [DirectBenchmark.Run](../Assets/Editor/DirectBenchmark.cs) from an Editor script/native Unity CLI wrapper. It uses immutable recorded geometry rather than the open scene. For the measured settings:

```csharp
DirectBenchmark.Run(
    "Recordings/station-20261005-192306-273",
    ".context/benchmarks/new-run", // Must not exist.
    "<current git rev-parse HEAD>",
    new TakeRender.Settings {
        listenerOutput = TakeRender.ListenerOutput.Binaural,
        delayFilter = TakeRender.DelayFilter.Linear,
        transitionSamples = 64,
        maximumResponsesPerBatch = 64,
        useBvh = false
    }, lowStep: 2048, highStep: 256, repetitions: 3);
```

The async Editor operation writes progress to `benchmark.json`; wait for `complete=true` and an empty `error`. Switch `useBvh` for the traversal comparison. Set `maximumResponsesPerBatch=1` for the original synchronous GPU scheduling. For a new compact recording, reconstruct once using its matching scene/code revision, then supply the generated numeric take to this harness. Do not launch a second Editor against the already-open project. The harness requires a graphics device; CPU-only `-nographics` jobs remain supported by `TakeBatch.Render` separately.

## 2026-10-05 compact capture cleanup

- Removed unreferenced `TrainVisualLoader` and `AudioConvolve`, plus the duplicate FFT scene/component. Automated FFT/direct GPU checks remain.
- Reduced `ComputeHelper` from 553 to 101 lines. Legacy manager/audio code now lives in explicit partial files, retaining serialized component references; current Inspectors omit legacy controls.
- Station capture writes input states, sample/tick timing, source events, identity metadata and small diagnostics by default. Full poses/geometry/PCM/output audio are optional diagnostic reference data.
- `StationReplay` reconstructs in Edit mode and is shared by Inspector rendering and batch jobs. It uses public scene/controller/audio preparation APIs and the same initialized simulation step. Explicit ground queries replace reliance on Play-only collision callbacks.
- 397 core checks and Metal FFT/direct/batched/live-versus-offline checks passed. The three-lap, 4,801-tick recording reconstructed byte-identical states/events and stereo audio across 4,225,025 frames; its five replay files occupied 39,176 bytes. Artifact: `Recordings/station-20261005-192306-273/`.
- A clean Git checkout at `2ad3c7e` and a fresh batch Editor (`-nographics`, no live audio) rebuilt a normal six-file recording. State/event hashes and all 896,001 non-silent stereo frames matched the original Editor's offline reconstruction exactly. The generated comparison opened successfully in the native comparison window. Inputs: `Recordings/station-20261005-194749-220/`; job, output and equality report: `.context/cleanup/checkout-job-final.json`, `checkout-render-final/`, `checkout-comparison-final.json`.
- That recovery test exposed unstable compiler module IDs across checkout directories. New Editor recordings validate relative project paths and file contents instead; older recordings and Player builds retain their original identity checks. Matching Git history supplies the files, but automatic checkout is still not implemented.
- Batch parsing now preserves absent/null comparison and reference jobs. Unity's inline JSON serializer previously instantiated those optional objects, triggering unrequested work. The parser uses the already-installed Json.NET 3.2.2 package, now declared directly. Regression checks cover optional jobs; the final batch run produced only `low/` and `high/`, with no reference render.
- A separate replay cadence test measured 4.90 and 9.99 FPS with identical final scene positions. The test deliberately throttles the Editor; it checks simulation accuracy, not uninterrupted output at those cadences. Artifact: `Recordings/station-20261005-193400-102/`.
- The default station now uses CPU tracing and 8,192 lead frames (185.8 ms at 44.1 kHz), down from 24,000 (544.2 ms). A 32.6-second standalone Mac run recorded only six compact files, with zero underrun/late/overflow samples. Maximum observation gap: 1,025 samples; maximum direct solve: 0.996 ms. Built app and report: `.context/cleanup/Station.app`, `.context/cleanup/player-result.json`.
- The background Editor remained near 10 FPS even with higher requested frame-rate caps. The three-lap capture had one 1,024-sample deadline miss and a 9,215-sample observation gap. Replay can also miss deadlines at this cadence or during script compilation. The smaller lead is not a guarantee against Editor stalls. Standalone measurements are reported separately.

## Earlier measurements

Historical measurements from 2026-09-29: Unity 6000.4.1f1, Apple M5 CPU/GPU, Metal, using an isolated validation project. At that time the main project was still on Unity 2022.3.62f3; it has since migrated to Unity 6000.4.1f1. The numbers below are unchanged, were not rerun for the documentation consolidation, and do not establish Windows/DirectX results. See the [README](../README.md) for current operation and [design contract](simulation-contracts.md) for planned work.

## Correctness

- 263 CPU checks passed, including input mutation rejection, immutable observed knots, bounded long-session history, analytic fractional-delay FIR accuracy, live/offline equality, recorder wrap/overflow, concurrent capture finalization, and reference acceptance/rejection.
- GPU FFT, transformed 2D/3D direct fixtures, batched asynchronous completion/disposal and live/offline equality with both delay filters passed on Metal.
- CPU and GPU Play-mode walkthroughs completed with editable native objects, moving listener/occluder, source events, saved input/output files and subsequent offline comparisons. Disabling the manager released its session and output queue.
- The actual Editor CLI completed a three-budget comparison. An exact-agreement reference gate rejected the moving take while leaving its completed measurement outputs available.
- No runtime files, packages or assembly boundaries were added for this cleanup. The existing implementation files were extended.

## Scene validation

The warmed mesh/material fixture performed 1,000 validations with zero managed allocations. Its total duration was about 3.2 ms. Snapshots compare shared mesh data and authored shape/material inputs; they do not regenerate scene geometry.

The earlier walkthrough spent 778 ms validating 26,632 polls, including duplicate audio timestamps. The final CPU walkthrough spent **2.76 ms** across **130** validations; 28,451 duplicate polls serviced pending rendering without revalidating or rewriting accepted poses. Maximum validation cost was 43.8 microseconds. Preparation, including snapshot warmup, is reported separately (35.4 ms in this run). These are observations from comparable three-second runs, not a general speedup guarantee.

## Live output

Both walkthroughs used one source/listener, 60 triangles, 44,100 Hz audio, a 256-sample response step and a 4,096-sample monitoring lead (92.9 ms). They recorded 129 distinct pose observations each. Both reported zero underrun, late, overflow and over-budget callback samples/counts at producer stop; output capture finalized without overflow.

The CPU run used linear delay filtering; the GPU run exercised a 16-sample-radius causal windowed-sinc filter. Their DSP times therefore represent different configurations. GPU maximum submit-to-consumption completion time was 14.74 ms. Output WAVs preserve component PCM before downstream Unity mixer/device processing, and callback records include the DSP clock relative to the take origin.

## Direct traversal throughput

Controlled all-miss fixture: 16 source/listener pairs, static triangles outside every path, warmed backend, eight measured calls per case. CPU and synchronous GPU calls each solve one response frame. GPU batches contain eight frames per call; their column divides elapsed batch time by eight. All GPU figures include readback/wait overhead. This measures throughput; batch latency and live scheduling must be assessed separately.

| Triangles | CPU ms/response | GPU sync ms/response | GPU batch ms/response |
|---:|---:|---:|---:|
| 64 | 0.209 | 0.388 | 0.051 |
| 4096 | 13.010 | 3.633 | 0.204 |
| 65536 | 207.502 | 25.780 | 2.939 |

At the time of these earlier measurements, traversal scanned every primitive per pair. The near-linear scaling motivated the optional BVH now measured above; large-scene BVH scaling has not yet been rerun. Batching reduces dispatch/readback overhead but does not change the brute-force traversal algorithm. Configurable FFT/STFT, validated indirect responses and converged ray-budget training references remain future work.

## Reproduction

Run `TakeChecks.Run` or `TakeChecks.RunGpu` with Unity's `-executeMethod` flag; omit `-nographics` for GPU checks. The walkthrough and controlled traversal automation, logs, CSV, takes, monitored PCM, offline renders and CLI outputs are retained locally under `.context/validation/cleanup/`. The temporary automation is outside the runtime codebase.

Unity Editor SearchDatabase logged its previously observed startup indexing exception during Play-mode entry. The capture, finalization and comparison checks above completed. This issue is recorded in the retained logs; no project upgrade was made during these cleanup measurements. The subsequent main-project migration is separate from this report.
