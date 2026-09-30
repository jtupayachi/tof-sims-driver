# # #!/usr/bin/env python3

# # import os
# # import io
# # import json
# # import base64
# # import requests
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import sys
# # from pathlib import Path


# # # ============================================================
# # # Local Qwen endpoint
# # # ============================================================

# # API_URL = "http://localhost:4000/v1/chat/completions"
# # MODEL = "ornl-qwen3-8-27b"

# # # Leave blank and fill locally if needed
# # API_KEY = ""


# # # ============================================================
# # # Input / output
# # # ============================================================

# # INPUT_FILE = "raw_spectrum.parquet"
# # OUTPUT_PLOT = "raw_spectrum_6PPD_annotated.png"
# # OUTPUT_JSON = "raw_spectrum_6PPD_analysis.json"


# # # ============================================================
# # # Load spectrum
# # # ============================================================

# # def load_spectrum(path):
# #     """
# #     Expected columns:

# #         mz
# #         intensity

# #     or

# #         mass
# #         counts
# #     """

# #     if path.endswith(".parquet"):
# #         df = pd.read_parquet(path)
# #     elif path.endswith(".csv"):
# #         df = pd.read_csv(path)
# #     else:
# #         raise ValueError("Use CSV or Parquet input.")

# #     # Normalize column names
# #     df.columns = [c.lower().strip() for c in df.columns]

# #     if "mz" in df.columns:
# #         mz_col = "mz"
# #     elif "mass" in df.columns:
# #         mz_col = "mass"
# #     elif "channel" in df.columns:
# #         mz_col = "channel"
# #     else:
# #         raise ValueError(
# #             f"Could not find m/z column. Columns: {list(df.columns)}"
# #         )

# #     if "intensity" in df.columns:
# #         intensity_col = "intensity"
# #     elif "counts" in df.columns:
# #         intensity_col = "counts"
# #     else:
# #         raise ValueError(
# #             f"Could not find intensity column. Columns: {list(df.columns)}"
# #         )

# #     mz = df[mz_col].to_numpy(dtype=float)
# #     intensity = df[intensity_col].to_numpy(dtype=float)

# #     # Remove invalid values
# #     mask = np.isfinite(mz) & np.isfinite(intensity)

# #     mz = mz[mask]
# #     intensity = intensity[mask]

# #     # Sort by m/z
# #     order = np.argsort(mz)

# #     mz = mz[order]
# #     intensity = intensity[order]

# #     # Normalize
# #     if intensity.max() > 0:
# #         intensity = intensity / intensity.max()

# #     return mz, intensity


# # # ============================================================
# # # Create spectrum plot
# # # ============================================================

# # def make_spectrum_plot(mz, intensity, filename):
# #     """
# #     Creates a plot visually similar to the supplied reference:
# #     black spectrum, thin vertical colored peak markers,
# #     minimal axes, and blank space on the right for the key.
# #     """

# #     fig, ax = plt.subplots(figsize=(12, 6))

# #     # Spectrum
# #     ax.plot(
# #         mz,
# #         intensity,
# #         color="black",
# #         linewidth=0.65
# #     )

# #     # Peak detection for visualization
# #     # This is intentionally conservative.
# #     threshold = 0.015

# #     peak_idx = []

# #     for i in range(1, len(intensity) - 1):
# #         if (
# #             intensity[i] > intensity[i - 1]
# #             and intensity[i] >= intensity[i + 1]
# #             and intensity[i] > threshold
# #         ):
# #             peak_idx.append(i)

# #     # Small red markers above peaks
# #     for i in peak_idx:
# #         ax.plot(
# #             [mz[i], mz[i]],
# #             [intensity[i] + 0.01, intensity[i] + 0.04],
# #             color="red",
# #             linewidth=1.2
# #         )

# #     # --------------------------------------------------------
# #     # Leave blank space on right for LLM annotation/key
# #     # --------------------------------------------------------

# #     xmax = max(mz) * 1.15
# #     ax.set_xlim(0, xmax)

# #     ax.set_xlabel("m/z", fontsize=18)
# #     ax.set_ylabel("Normalized intensity", fontsize=18)

# #     ax.tick_params(axis="both", labelsize=12)

# #     ax.spines["top"].set_visible(False)
# #     ax.spines["right"].set_visible(False)

# #     plt.tight_layout()

# #     fig.savefig(
# #         filename,
# #         dpi=300,
# #         bbox_inches="tight"
# #     )

# #     plt.close(fig)


# # # ============================================================
# # # Encode image for Qwen Vision
# # # ============================================================

# # def encode_image(filename):

# #     with open(filename, "rb") as f:
# #         encoded = base64.b64encode(f.read()).decode("utf-8")

# #     return f"data:image/png;base64,{encoded}"


# # # ============================================================
# # # Ask Qwen to analyze spectrum
# # # ============================================================

# # def analyze_with_qwen(image_data):

# #     prompt = r"""
# # You are analyzing a mass spectrum.

# # The known target compound is:

# # 6PPD
# # N-(1,3-dimethylbutyl)-N'-phenyl-p-phenylenediamine

# # Analyze the supplied spectrum image.

# # IMPORTANT:
# # - Do not modify or invent measured peaks.
# # - Identify prominent observed m/z peaks.
# # - Determine which observed peaks could support the presence of 6PPD.
# # - Distinguish strong evidence from weak/ambiguous evidence.
# # - Do not claim identification from a single peak.
# # - Consider the molecular ion and characteristic fragment ions where visible.
# # - If a requested diagnostic ion is not visible, explicitly say so.
# # - Use the plotted spectrum itself as the source of observed m/z values.

# # Return ONLY valid JSON in this format:

# # {
# #   "compound": "6PPD",
# #   "assessment": "consistent|not_consistent|ambiguous",
# #   "diagnostic_peaks": [
# #     {
# #       "mz": 0,
# #       "observed": true,
# #       "confidence": 0.0,
# #       "role": "molecular_ion|fragment|supporting|unknown",
# #       "comment": ""
# #     }
# #   ],
# #   "other_prominent_peaks": [
# #     {
# #       "mz": 0,
# #       "comment": ""
# #     }
# #   ],
# #   "missing_expected_features": [],
# #   "summary": ""
# # }

# # Use integer m/z values when possible.
# # Confidence must be between 0 and 1.
# # """


# #     headers = {
# #         "Content-Type": "application/json"
# #     }

# #     if API_KEY.strip():
# #         headers["Authorization"] = f"Bearer {API_KEY}"

# #     payload = {
# #         "model": MODEL,
# #         "messages": [
# #             {
# #                 "role": "system",
# #                 "content": (
# #                     "You are a mass-spectrometry analyst. "
# #                     "Return precise, conservative interpretations."
# #                 )
# #             },
# #             {
# #                 "role": "user",
# #                 "content": [
# #                     {
# #                         "type": "text",
# #                         "text": prompt
# #                     },
# #                     {
# #                         "type": "image_url",
# #                         "image_url": {
# #                             "url": image_data
# #                         }
# #                     }
# #                 ]
# #             }
# #         ],
# #         "temperature": 0.1,
# #         "max_tokens": 4000
# #     }

# #     response = requests.post(
# #         API_URL,
# #         headers=headers,
# #         json=payload,
# #         timeout=300
# #     )

# #     response.raise_for_status()

# #     result = response.json()

# #     content = result["choices"][0]["message"]["content"]

# #     # Remove accidental markdown fences
# #     content = content.replace("```json", "")
# #     content = content.replace("```", "")
# #     content = content.strip()

# #     return json.loads(content)


# # # ============================================================
# # # Overlay Qwen annotations
# # # ============================================================

# # def annotate_spectrum(mz, intensity, analysis, filename):

# #     fig, ax = plt.subplots(figsize=(12, 6))

# #     ax.plot(
# #         mz,
# #         intensity,
# #         color="black",
# #         linewidth=0.65
# #     )

# #     # Peak detection
# #     threshold = 0.015

# #     peak_idx = []

# #     for i in range(1, len(intensity) - 1):

# #         if (
# #             intensity[i] > intensity[i - 1]
# #             and intensity[i] >= intensity[i + 1]
# #             and intensity[i] > threshold
# #         ):
# #             peak_idx.append(i)

# #     # Original red peak markers
# #     for i in peak_idx:

# #         ax.plot(
# #             [mz[i], mz[i]],
# #             [
# #                 intensity[i] + 0.01,
# #                 intensity[i] + 0.04
# #             ],
# #             color="red",
# #             linewidth=1.2
# #         )

# #     # --------------------------------------------------------
# #     # Qwen annotations
# #     # --------------------------------------------------------

# #     diagnostic = analysis.get("diagnostic_peaks", [])

# #     colors = {
# #         "molecular_ion": "blue",
# #         "fragment": "magenta",
# #         "supporting": "green",
# #         "unknown": "gray"
# #     }

# #     for peak in diagnostic:

# #         target_mz = float(peak["mz"])

# #         # Find closest measured m/z
# #         idx = np.argmin(np.abs(mz - target_mz))

# #         observed_mz = mz[idx]
# #         observed_intensity = intensity[idx]

# #         role = peak.get("role", "unknown")
# #         color = colors.get(role, "gray")

# #         # Colored marker
# #         ax.plot(
# #             [observed_mz, observed_mz],
# #             [
# #                 observed_intensity + 0.015,
# #                 observed_intensity + 0.07
# #             ],
# #             color=color,
# #             linewidth=2.0
# #         )

# #         # Label
# #         ax.text(
# #             observed_mz,
# #             observed_intensity + 0.075,
# #             str(round(observed_mz)),
# #             ha="center",
# #             va="bottom",
# #             fontsize=10,
# #             color=color
# #         )

# #     # --------------------------------------------------------
# #     # Assessment
# #     # --------------------------------------------------------

# #     assessment = analysis.get(
# #         "assessment",
# #         "ambiguous"
# #     )

# #     ax.text(
# #         0.97,
# #         0.94,
# #         f"6PPD: {assessment}",
# #         transform=ax.transAxes,
# #         ha="right",
# #         va="top",
# #         fontsize=13
# #     )

# #     # --------------------------------------------------------
# #     # Blank area for key
# #     # --------------------------------------------------------

# #     xmax = max(mz) * 1.20

# #     ax.set_xlim(0, xmax)

# #     ax.set_xlabel("m/z", fontsize=18)
# #     ax.set_ylabel("Normalized intensity", fontsize=18)

# #     ax.tick_params(axis="both", labelsize=12)

# #     ax.spines["top"].set_visible(False)
# #     ax.spines["right"].set_visible(False)

# #     plt.tight_layout()

# #     fig.savefig(
# #         filename,
# #         dpi=300,
# #         bbox_inches="tight"
# #     )

# #     plt.close(fig)


# # # ============================================================
# # # Main
# # # ============================================================

# # # def main():

# # #     print("Loading spectrum...")
# # #     mz, intensity = load_spectrum(INPUT_FILE)

# # #     print("Creating initial spectrum...")
# # #     make_spectrum_plot(
# # #         mz,
# # #         intensity,
# # #         "spectrum_for_qwen.png"
# # #     )

# # #     print("Sending spectrum to Qwen...")
# # #     image_data = encode_image(
# # #         "spectrum_for_qwen.png"
# # #     )

# # #     analysis = analyze_with_qwen(
# # #         image_data
# # #     )

# # #     print("\nQwen analysis:")
# # #     print(json.dumps(
# # #         analysis,
# # #         indent=2
# # #     ))

# # #     # Save analysis
# # #     with open(
# # #         OUTPUT_JSON,
# # #         "w"
# # #     ) as f:

# # #         json.dump(
# # #             analysis,
# # #             f,
# # #             indent=2
# # #         )

# # #     print("\nCreating annotated spectrum...")

# # #     annotate_spectrum(
# # #         mz,
# # #         intensity,
# # #         analysis,
# # #         OUTPUT_PLOT
# # #     )

# # #     print(f"\nSaved: {OUTPUT_PLOT}")
# # #     print(f"Saved: {OUTPUT_JSON}")



# # def main():

# #     if len(sys.argv) < 2:
# #         print("Usage:")
# #         print("  python3 check2.py <spectrum_directory>")
# #         sys.exit(1)

# #     input_dir = Path(sys.argv[1]).resolve()

# #     if not input_dir.is_dir():
# #         raise FileNotFoundError(
# #             f"Input directory does not exist: {input_dir}"
# #         )

# #     print(f"Input directory: {input_dir}")

# #     # Find spectrum files
# #     parquet_files = sorted(input_dir.glob("*.parquet"))
# #     npz_files = sorted(input_dir.glob("*.npz"))

# #     print(f"Found {len(parquet_files)} Parquet files")
# #     print(f"Found {len(npz_files)} NPZ files")

# #     if not parquet_files and not npz_files:
# #         raise FileNotFoundError(
# #             f"No .parquet or .npz files found in {input_dir}"
# #         )

# #     # --------------------------------------------------------
# #     # Use the first Parquet file for now
# #     # --------------------------------------------------------

# #     if parquet_files:

# #         INPUT_FILE = parquet_files[0]

# #         print(f"Loading Parquet:")
# #         print(f"  {INPUT_FILE}")

# #         mz, intensity = load_spectrum(
# #             str(INPUT_FILE)
# #         )

# #     else:
# #         INPUT_FILE = npz_files[0]

# #         print(f"Loading NPZ:")
# #         print(f"  {INPUT_FILE}")

# #         mz, intensity = load_spectrum(
# #             str(INPUT_FILE)
# #         )

# #     # --------------------------------------------------------
# #     # Output files go into the input directory
# #     # --------------------------------------------------------

# #     OUTPUT_PLOT = input_dir / "spectrum_for_qwen.png"
# #     OUTPUT_PLOT_ANNOTATED = input_dir / "spectrum_6PPD_annotated.png"
# #     OUTPUT_JSON = input_dir / "spectrum_6PPD_analysis.json"

# #     print("Creating initial spectrum...")

# #     make_spectrum_plot(
# #         mz,
# #         intensity,
# #         str(OUTPUT_PLOT)
# #     )

# #     print(f"Created: {OUTPUT_PLOT}")

# #     print("Sending spectrum to Qwen...")

# #     image_data = encode_image(
# #         str(OUTPUT_PLOT)
# #     )

# #     analysis = analyze_with_qwen(
# #         image_data
# #     )

# #     print("\nQwen analysis:")
# #     print(json.dumps(
# #         analysis,
# #         indent=2
# #     ))

# #     with open(
# #         OUTPUT_JSON,
# #         "w"
# #     ) as f:
# #         json.dump(
# #             analysis,
# #             f,
# #             indent=2
# #         )

# #     print(f"\nSaved analysis: {OUTPUT_JSON}")

# #     print("Creating annotated spectrum...")

# #     annotate_spectrum(
# #         mz,
# #         intensity,
# #         analysis,
# #         str(OUTPUT_PLOT_ANNOTATED)
# #     )

# #     print(f"Saved annotated spectrum: {OUTPUT_PLOT_ANNOTATED}")


# # if __name__ == "__main__":
# #     main()

# # sk-118ac647ad3540a02ee84435e6e0a0e7bd76ea085c5c247e
# #!/usr/bin/env python3

# import sys
# import json
# import base64
# from pathlib import Path

# import requests
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# # ============================================================
# # Local Qwen endpoint
# # ============================================================

# API_URL = "http://localhost:4000/v1/chat/completions"
# MODEL = "ornl-qwen3-8-27b"
# API_KEY = ""


# # ============================================================
# # Load spectrum
# # ============================================================

# def load_spectrum(path):
#     path = Path(path)

#     if path.suffix.lower() == ".parquet":
#         df = pd.read_parquet(path)

#     elif path.suffix.lower() == ".csv":
#         df = pd.read_csv(path)

#     elif path.suffix.lower() == ".npz":
#         data = np.load(path)

#         print("NPZ keys:", list(data.keys()))

#         if "mass" in data:
#             mz = data["mass"]
#         elif "mz" in data:
#             mz = data["mz"]
#         else:
#             raise ValueError(
#                 f"Could not find mass/mz in NPZ: {list(data.keys())}"
#             )

#         if "counts" in data:
#             intensity = data["counts"]
#         elif "intensity" in data:
#             intensity = data["intensity"]
#         else:
#             raise ValueError(
#                 f"Could not find counts/intensity in NPZ: {list(data.keys())}"
#             )

#         return clean_spectrum(mz, intensity)

#     else:
#         raise ValueError("Use CSV, Parquet, or NPZ input.")

#     df.columns = [str(c).lower().strip() for c in df.columns]

#     print("Columns:", list(df.columns))
#     print("Shape:", df.shape)

#     # --------------------------------------------------------
#     # Correct spectrum columns
#     # --------------------------------------------------------

#     if "mass" in df.columns:
#         mz_col = "mass"
#     elif "mz" in df.columns:
#         mz_col = "mz"
#     else:
#         raise ValueError(
#             f"No mass/mz column found. Columns: {list(df.columns)}"
#         )

#     if "counts" in df.columns:
#         intensity_col = "counts"
#     elif "intensity" in df.columns:
#         intensity_col = "intensity"
#     else:
#         raise ValueError(
#             f"No counts/intensity column found. Columns: {list(df.columns)}"
#         )

#     mz = df[mz_col].to_numpy(dtype=float)
#     intensity = df[intensity_col].to_numpy(dtype=float)

#     return clean_spectrum(mz, intensity)


# # ============================================================
# # Clean spectrum
# # ============================================================

# def clean_spectrum(mz, intensity):

#     mask = (
#         np.isfinite(mz)
#         & np.isfinite(intensity)
#         & (intensity >= 0)
#     )

#     mz = mz[mask]
#     intensity = intensity[mask]

#     order = np.argsort(mz)

#     mz = mz[order]
#     intensity = intensity[order]

#     if len(mz) == 0:
#         raise ValueError("Spectrum contains no valid data.")

#     max_intensity = intensity.max()

#     if max_intensity > 0:
#         normalized = intensity / max_intensity
#     else:
#         normalized = intensity

#     print(f"Number of points: {len(mz):,}")
#     print(f"m/z range: {mz.min():.6f} - {mz.max():.6f}")
#     print(f"Maximum counts: {max_intensity:g}")

#     return mz, normalized


# # ============================================================
# # Detect peaks numerically
# # ============================================================

# def detect_peaks(mz, intensity, threshold=0.01):

#     peaks = []

#     for i in range(1, len(intensity) - 1):

#         if (
#             intensity[i] > intensity[i - 1]
#             and intensity[i] >= intensity[i + 1]
#             and intensity[i] >= threshold
#         ):
#             peaks.append(i)

#     return np.array(peaks, dtype=int)


# # ============================================================
# # Create spectrum plot
# # ============================================================

# def make_spectrum_plot(mz, intensity, filename):

#     peak_idx = detect_peaks(
#         mz,
#         intensity,
#         threshold=0.01
#     )

#     fig, ax = plt.subplots(figsize=(14, 7))

#     ax.plot(
#         mz,
#         intensity,
#         color="black",
#         linewidth=0.6
#     )

#     # Mark detected peaks
#     for i in peak_idx:

#         ax.plot(
#             [mz[i], mz[i]],
#             [
#                 intensity[i] + 0.01,
#                 intensity[i] + 0.04
#             ],
#             color="red",
#             linewidth=1.0
#         )

#     ax.set_xlabel(
#         "m/z",
#         fontsize=18
#     )

#     ax.set_ylabel(
#         "Normalized intensity",
#         fontsize=18
#     )

#     ax.tick_params(
#         axis="both",
#         labelsize=12
#     )

#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     # Give Qwen some whitespace
#     xmin = max(0, mz.min())
#     xmax = mz.max()

#     ax.set_xlim(
#         xmin,
#         xmax * 1.05
#     )

#     ax.set_ylim(
#         0,
#         min(1.15, max(1.05, intensity.max() * 1.10))
#     )

#     plt.tight_layout()

#     fig.savefig(
#         filename,
#         dpi=250,
#         bbox_inches="tight"
#     )

#     plt.close(fig)

#     print(f"Detected {len(peak_idx)} peaks above threshold.")


# # ============================================================
# # Encode image
# # ============================================================

# def encode_image(filename):

#     with open(filename, "rb") as f:
#         encoded = base64.b64encode(
#             f.read()
#         ).decode("utf-8")

#     return f"data:image/png;base64,{encoded}"


# # ============================================================
# # Ask Qwen
# # ============================================================

# def analyze_with_qwen(image_data):

#     prompt = r"""
# You are analyzing a mass spectrum.

# Known target compound:

# 6PPD
# N-(1,3-dimethylbutyl)-N'-phenyl-p-phenylenediamine

# Analyze ONLY the supplied spectrum.

# Important rules:

# 1. Do not invent peaks.
# 2. Use the plotted spectrum as the source for observed m/z values.
# 3. Identify prominent observed peaks.
# 4. Identify peaks that could support 6PPD.
# 5. Distinguish strong evidence from weak or ambiguous evidence.
# 6. Do not claim compound identification from a single peak.
# 7. Consider the molecular ion and characteristic fragment ions.
# 8. If an expected diagnostic ion is not visible, explicitly state that.
# 9. Be conservative.
# 10. Use integer m/z values when appropriate.

# Return ONLY valid JSON:

# {
#   "compound": "6PPD",
#   "assessment": "consistent|not_consistent|ambiguous",

#   "diagnostic_peaks": [
#     {
#       "mz": 0,
#       "observed": true,
#       "confidence": 0.0,
#       "role": "molecular_ion|fragment|supporting|unknown",
#       "comment": ""
#     }
#   ],

#   "other_prominent_peaks": [
#     {
#       "mz": 0,
#       "comment": ""
#     }
#   ],

#   "missing_expected_features": [],

#   "summary": ""
# }

# Confidence must be between 0 and 1.
# """

#     headers = {
#         "Content-Type": "application/json"
#     }

#     if API_KEY.strip():
#         headers["Authorization"] = f"Bearer {API_KEY}"

#     payload = {
#         "model": MODEL,

#         "messages": [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a conservative mass-spectrometry analyst. "
#                     "Never invent observed peaks."
#                 )
#             },

#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": prompt
#                     },

#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": image_data
#                         }
#                     }
#                 ]
#             }
#         ],

#         "temperature": 0.1,
#         "max_tokens": 4000
#     }

#     print("Calling Qwen...")

#     response = requests.post(
#         API_URL,
#         headers=headers,
#         json=payload,
#         timeout=300
#     )

#     response.raise_for_status()

#     result = response.json()

#     content = result["choices"][0]["message"]["content"]

#     content = content.replace(
#         "```json",
#         ""
#     )

#     content = content.replace(
#         "```",
#         ""
#     )

#     content = content.strip()

#     return json.loads(content)


# # ============================================================
# # Annotate spectrum
# # ============================================================

# def annotate_spectrum(
#     mz,
#     intensity,
#     analysis,
#     filename
# ):

#     fig, ax = plt.subplots(
#         figsize=(14, 7)
#     )

#     ax.plot(
#         mz,
#         intensity,
#         color="black",
#         linewidth=0.6
#     )

#     # --------------------------------------------------------
#     # Original peak detection
#     # --------------------------------------------------------

#     peak_idx = detect_peaks(
#         mz,
#         intensity,
#         threshold=0.01
#     )

#     for i in peak_idx:

#         ax.plot(
#             [mz[i], mz[i]],
#             [
#                 intensity[i] + 0.01,
#                 intensity[i] + 0.04
#             ],
#             color="red",
#             linewidth=1.0
#         )

#     # --------------------------------------------------------
#     # Qwen diagnostic peaks
#     # --------------------------------------------------------

#     colors = {
#         "molecular_ion": "blue",
#         "fragment": "magenta",
#         "supporting": "green",
#         "unknown": "gray"
#     }

#     diagnostic = analysis.get(
#         "diagnostic_peaks",
#         []
#     )

#     for peak in diagnostic:

#         if not peak.get("observed", False):
#             continue

#         try:
#             target_mz = float(
#                 peak["mz"]
#             )
#         except (KeyError, ValueError, TypeError):
#             continue

#         idx = np.argmin(
#             np.abs(mz - target_mz)
#         )

#         observed_mz = mz[idx]
#         observed_intensity = intensity[idx]

#         role = peak.get(
#             "role",
#             "unknown"
#         )

#         color = colors.get(
#             role,
#             "gray"
#         )

#         # Marker
#         ax.plot(
#             [observed_mz, observed_mz],
#             [
#                 observed_intensity + 0.015,
#                 observed_intensity + 0.08
#             ],
#             color=color,
#             linewidth=2.0
#         )

#         # Label
#         ax.text(
#             observed_mz,
#             observed_intensity + 0.085,
#             str(round(observed_mz)),
#             ha="center",
#             va="bottom",
#             fontsize=10,
#             color=color
#         )

#     # --------------------------------------------------------
#     # Assessment
#     # --------------------------------------------------------

#     assessment = analysis.get(
#         "assessment",
#         "ambiguous"
#     )

#     ax.text(
#         0.98,
#         0.95,
#         f"6PPD: {assessment}",
#         transform=ax.transAxes,
#         ha="right",
#         va="top",
#         fontsize=14
#     )

#     # --------------------------------------------------------
#     # Axes
#     # --------------------------------------------------------

#     ax.set_xlabel(
#         "m/z",
#         fontsize=18
#     )

#     ax.set_ylabel(
#         "Normalized intensity",
#         fontsize=18
#     )

#     ax.tick_params(
#         axis="both",
#         labelsize=12
#     )

#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     ax.set_xlim(
#         max(0, mz.min()),
#         mz.max() * 1.05
#     )

#     ax.set_ylim(
#         0,
#         min(1.20, max(1.05, intensity.max() * 1.12))
#     )

#     plt.tight_layout()

#     fig.savefig(
#         filename,
#         dpi=300,
#         bbox_inches="tight"
#     )

#     plt.close(fig)


# # ============================================================
# # Main
# # ============================================================

# def main():

#     if len(sys.argv) < 2:

#         print(
#             "Usage:\n"
#             "  python3 check2.py <spectrum_directory>"
#         )

#         sys.exit(1)

#     input_dir = Path(
#         sys.argv[1]
#     ).resolve()

#     if not input_dir.is_dir():

#         raise FileNotFoundError(
#             f"Input directory does not exist: {input_dir}"
#         )

#     print(
#         f"Input directory: {input_dir}"
#     )

#     # --------------------------------------------------------
#     # Find the ACTUAL spectrum
#     # --------------------------------------------------------

#     spectrum_file = (
#         input_dir / "raw_spectrum.parquet"
#     )

#     if spectrum_file.exists():

#         print(
#             f"Using spectrum:\n"
#             f"  {spectrum_file}"
#         )

#     else:

#         parquet_files = sorted(
#             input_dir.glob("*.parquet")
#         )

#         spectrum_candidates = [
#             f for f in parquet_files
#             if "spectrum" in f.name.lower()
#         ]

#         if spectrum_candidates:

#             spectrum_file = spectrum_candidates[0]

#         else:

#             raise FileNotFoundError(
#                 "Could not find raw_spectrum.parquet"
#             )

#     # --------------------------------------------------------
#     # Load
#     # --------------------------------------------------------

#     mz, intensity = load_spectrum(
#         spectrum_file
#     )

#     # --------------------------------------------------------
#     # Output
#     # --------------------------------------------------------

#     output_plot = (
#         input_dir /
#         "spectrum_for_qwen.png"
#     )

#     output_plot_annotated = (
#         input_dir /
#         "spectrum_6PPD_annotated.png"
#     )

#     output_json = (
#         input_dir /
#         "spectrum_6PPD_analysis.json"
#     )

#     # --------------------------------------------------------
#     # Initial plot
#     # --------------------------------------------------------

#     print(
#         "Creating initial spectrum..."
#     )

#     make_spectrum_plot(
#         mz,
#         intensity,
#         output_plot
#     )

#     print(
#         f"Created: {output_plot}"
#     )

#     # --------------------------------------------------------
#     # Qwen
#     # --------------------------------------------------------

#     image_data = encode_image(
#         output_plot
#     )

#     analysis = analyze_with_qwen(
#         image_data
#     )

#     print("\nQwen analysis:")
#     print(
#         json.dumps(
#             analysis,
#             indent=2
#         )
#     )

#     # --------------------------------------------------------
#     # Save JSON
#     # --------------------------------------------------------

#     with open(
#         output_json,
#         "w"
#     ) as f:

#         json.dump(
#             analysis,
#             f,
#             indent=2
#         )

#     print(
#         f"\nSaved analysis: {output_json}"
#     )

#     # --------------------------------------------------------
#     # Annotated plot
#     # --------------------------------------------------------

#     print(
#         "Creating annotated spectrum..."
#     )

#     annotate_spectrum(
#         mz,
#         intensity,
#         analysis,
#         output_plot_annotated
#     )

#     print(
#         f"Saved annotated spectrum: "
#         f"{output_plot_annotated}"
#     )


# if __name__ == "__main__":
#     main()

import sys
import json
import base64
import os
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Optional / MS libraries
# ============================================================

try:
    import pyopenms as oms
    PYOPENMS_AVAILABLE = True
except Exception as e:
    oms = None
    PYOPENMS_AVAILABLE = False
    PYOPENMS_IMPORT_ERROR = str(e)


try:
    import matchms
    from matchms import Spectrum
    from matchms.importing import load_spectra
    from matchms.similarity import CosineGreedy
    MATCHMS_AVAILABLE = True
except Exception as e:
    matchms = None
    Spectrum = None
    load_spectra = None
    CosineGreedy = None
    MATCHMS_AVAILABLE = False
    MATCHMS_IMPORT_ERROR = str(e)


# ============================================================
# Local Qwen endpoint
# ============================================================

API_URL = "http://localhost:4000/v1/chat/completions"
MODEL = "ornl-qwen3-8-27b"

# Set with:
# export QWEN_API_KEY="your-key"
API_KEY = os.getenv("QWEN_API_KEY", "")

# Number of Qwen calls per spectrum range
N_ITERATIONS = 30

# Requested analysis ranges
RANGES = [
    (0, 150),
    (150, 300),
    (300, 800),
]

# Basic numerical peak threshold
PEAK_THRESHOLD = 0.01

# pyOpenMS peak processing
PYOPENMS_MIN_INTENSITY = 0.01

# Mass tolerance for computational matching.
#
# IMPORTANT:
# Tune this to the actual mass accuracy of your instrument.
#
# 0.01 Da is deliberately conservative for the first version.
MATCH_TOLERANCE_DA = 0.01

# Number of strongest computational peaks supplied to Qwen
MAX_PEAKS_FOR_QWEN = 150

# Optional local reference library.
#
# Put MSP / MGF / mzML / JSON spectra here.
#
# Example:
#
#   processed/
#       6PPD_neg1/
#           itax/
#               references/
#                   6PPD.msp
#
REFERENCE_DIR_NAME = "references"


# ============================================================
# Utility
# ============================================================

def json_safe(value):
    """
    Convert NumPy / Python objects into JSON-safe values.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    return value


# ============================================================
# Load spectrum
# ============================================================

def load_spectrum(path):

    path = Path(path)

    if path.suffix.lower() == ".parquet":

        df = pd.read_parquet(path)

    elif path.suffix.lower() == ".csv":

        df = pd.read_csv(path)

    elif path.suffix.lower() == ".npz":

        data = np.load(path)

        print("NPZ keys:", list(data.keys()))

        if "mass" in data:
            mz = data["mass"]

        elif "mz" in data:
            mz = data["mz"]

        else:
            raise ValueError(
                f"Could not find mass/mz in NPZ: {list(data.keys())}"
            )

        if "counts" in data:
            intensity = data["counts"]

        elif "intensity" in data:
            intensity = data["intensity"]

        else:
            raise ValueError(
                f"Could not find counts/intensity in NPZ: {list(data.keys())}"
            )

        return clean_spectrum(mz, intensity)

    else:

        raise ValueError(
            "Use CSV, Parquet, or NPZ input."
        )

    df.columns = [
        str(c).lower().strip()
        for c in df.columns
    ]

    print("Columns:", list(df.columns))
    print("Shape:", df.shape)

    # --------------------------------------------------------
    # Correct spectrum columns
    # --------------------------------------------------------

    if "mass" in df.columns:

        mz_col = "mass"

    elif "mz" in df.columns:

        mz_col = "mz"

    else:

        raise ValueError(
            f"No mass/mz column found. Columns: {list(df.columns)}"
        )

    if "counts" in df.columns:

        intensity_col = "counts"

    elif "intensity" in df.columns:

        intensity_col = "intensity"

    else:

        raise ValueError(
            f"No counts/intensity column found. "
            f"Columns: {list(df.columns)}"
        )

    mz = df[mz_col].to_numpy(dtype=float)
    intensity = df[intensity_col].to_numpy(dtype=float)

    return clean_spectrum(mz, intensity)


# ============================================================
# Clean spectrum
# ============================================================

def clean_spectrum(mz, intensity):

    mask = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & (intensity >= 0)
    )

    mz = mz[mask]
    intensity = intensity[mask]

    order = np.argsort(mz)

    mz = mz[order]
    intensity = intensity[order]

    if len(mz) == 0:
        raise ValueError(
            "Spectrum contains no valid data."
        )

    max_intensity = intensity.max()

    if max_intensity > 0:

        normalized = intensity / max_intensity

    else:

        normalized = intensity

    print(
        f"Number of points: {len(mz):,}"
    )

    print(
        f"m/z range: "
        f"{mz.min():.6f} - {mz.max():.6f}"
    )

    print(
        f"Maximum counts: "
        f"{max_intensity:g}"
    )

    return mz, normalized


# ============================================================
# pyOpenMS
# ============================================================

def run_pyopenms_peak_processing(mz, intensity):

    result = {
        "available": PYOPENMS_AVAILABLE,
        "status": "not_run",
        "error": None,
        "profile_points": int(len(mz)),
        "centroided_points": 0,
        "centroided_peaks": [],
        "deisotoped_peaks": [],
    }

    if not PYOPENMS_AVAILABLE:

        result["status"] = "unavailable"

        result["error"] = PYOPENMS_IMPORT_ERROR

        return result

    try:

        # ----------------------------------------------------
        # Create OpenMS spectrum
        # ----------------------------------------------------

        spectrum = oms.MSSpectrum()

        spectrum.set_peaks(
            (
                np.asarray(mz, dtype=float),
                np.asarray(intensity, dtype=float)
            )
        )

        spectrum.setMSLevel(1)

        # ----------------------------------------------------
        # Peak picking / centroiding
        # ----------------------------------------------------
        #
        # The input appears to be profile-like data.
        # PeakPickerHiRes converts profile data into
        # centroided peaks.
        #

        picked = oms.MSSpectrum()

        picker = oms.PeakPickerHiRes()

        try:

            picker.pick(
                spectrum,
                picked
            )

        except Exception:

            # Some pyOpenMS versions expose the operation
            # through pickExperiment rather than pick.
            experiment_in = oms.MSExperiment()
            experiment_out = oms.MSExperiment()

            experiment_in.addSpectrum(spectrum)

            picker.pickExperiment(
                experiment_in,
                experiment_out,
                True
            )

            if experiment_out.size() > 0:

                picked = experiment_out[0]

        centroid_mz, centroid_intensity = (
            picked.get_peaks()
        )

        centroid_mz = np.asarray(
            centroid_mz,
            dtype=float
        )

        centroid_intensity = np.asarray(
            centroid_intensity,
            dtype=float
        )

        if len(centroid_mz) == 0:

            raise RuntimeError(
                "pyOpenMS returned zero centroided peaks."
            )

        # Normalize
        if centroid_intensity.max() > 0:

            centroid_norm = (
                centroid_intensity
                / centroid_intensity.max()
            )

        else:

            centroid_norm = centroid_intensity

        centroided_peaks = []

        for m, inten in zip(
            centroid_mz,
            centroid_norm
        ):

            if inten >= PYOPENMS_MIN_INTENSITY:

                centroided_peaks.append(
                    {
                        "mz": float(m),
                        "relative_intensity": float(inten),
                    }
                )

        result[
            "centroided_points"
        ] = int(len(centroid_mz))

        result[
            "centroided_peaks"
        ] = centroided_peaks

        # ----------------------------------------------------
        # Deisotoping
        # ----------------------------------------------------
        #
        # OpenMS Deisotoper identifies isotope patterns and
        # charge states when enough evidence is available.
        #
        # API signatures differ somewhat across versions,
        # so we attempt the operation conservatively.
        #

        deisotoped = picked.copy()

        try:

            # Common OpenMS interface.
            #
            # Parameters:
            #   tol
            #   fragment_tolerance
            #   keep_only_deisotoped
            #   min_charge
            #   max_charge
            #
            oms.Deisotoper.deisotopeAndSingleCharge(
                deisotoped,
                0.01,
                False,
                1,
                3,
                True,
                2,
                6,
                True,
                True,
                True,
                True,
                True,
                False,
                0,
                False
            )

            deis_mz, deis_intensity = (
                deisotoped.get_peaks()
            )

            deis_mz = np.asarray(
                deis_mz,
                dtype=float
            )

            deis_intensity = np.asarray(
                deis_intensity,
                dtype=float
            )

            if len(deis_mz) > 0:

                if deis_intensity.max() > 0:

                    deis_norm = (
                        deis_intensity
                        / deis_intensity.max()
                    )

                else:

                    deis_norm = deis_intensity

                deisotoped_peaks = []

                for m, inten in zip(
                    deis_mz,
                    deis_norm
                ):

                    if inten >= PYOPENMS_MIN_INTENSITY:

                        deisotoped_peaks.append(
                            {
                                "mz": float(m),
                                "relative_intensity":
                                    float(inten),
                            }
                        )

                result[
                    "deisotoped_peaks"
                ] = deisotoped_peaks

        except Exception as e:

            result[
                "deisotoping_error"
            ] = str(e)

        result["status"] = "completed"

        return json_safe(result)

    except Exception as e:

        result["status"] = "failed"

        result["error"] = str(e)

        return json_safe(result)


# ============================================================
# Numerical peak detection
# ============================================================

def detect_peaks(
    mz,
    intensity,
    threshold=PEAK_THRESHOLD
):

    peaks = []

    for i in range(
        1,
        len(intensity) - 1
    ):

        if (
            intensity[i]
            > intensity[i - 1]
            and intensity[i]
            >= intensity[i + 1]
            and intensity[i]
            >= threshold
        ):

            peaks.append(i)

    return np.array(
        peaks,
        dtype=int
    )


def numerical_peak_table(
    mz,
    intensity,
    xmin,
    xmax
):

    range_mz, range_intensity = (
        get_range(
            mz,
            intensity,
            xmin,
            xmax
        )
    )

    if len(range_mz) == 0:
        return []

    peak_idx = detect_peaks(
        range_mz,
        range_intensity
    )

    rows = []

    for i in peak_idx:

        rows.append(
            {
                "mz": float(range_mz[i]),
                "relative_intensity":
                    float(range_intensity[i])
            }
        )

    rows.sort(
        key=lambda x: x["relative_intensity"],
        reverse=True
    )

    return rows[:MAX_PEAKS_FOR_QWEN]


# ============================================================
# Restrict spectrum to range
# ============================================================

def get_range(
    mz,
    intensity,
    xmin,
    xmax
):

    mask = (
        (mz >= xmin)
        & (mz < xmax)
    )

    return (
        mz[mask],
        intensity[mask]
    )


# ============================================================
# Create spectrum plot
# ============================================================

def make_spectrum_plot(
    mz,
    intensity,
    filename,
    xmin,
    xmax
):

    range_mz, range_intensity = (
        get_range(
            mz,
            intensity,
            xmin,
            xmax
        )
    )

    if len(range_mz) == 0:

        raise ValueError(
            f"No spectrum points in range "
            f"{xmin}-{xmax}"
        )

    peak_idx = detect_peaks(
        range_mz,
        range_intensity,
        threshold=PEAK_THRESHOLD
    )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        range_mz,
        range_intensity,
        color="black",
        linewidth=0.6
    )

    # --------------------------------------------------------
    # Numerical peaks
    # --------------------------------------------------------

    for i in peak_idx:

        y = range_intensity[i]

        ax.plot(
            [
                range_mz[i],
                range_mz[i]
            ],
            [
                y + 0.01,
                y + 0.04
            ],
            color="red",
            linewidth=1.0
        )

    ax.set_xlabel(
        "m/z",
        fontsize=18
    )

    ax.set_ylabel(
        "Normalized intensity",
        fontsize=18
    )

    ax.tick_params(
        axis="both",
        labelsize=12
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(
        xmin,
        xmax
    )

    upper = max(
        1.05,
        float(range_intensity.max()) * 1.10
    )

    ax.set_ylim(
        0,
        min(1.15, upper)
    )

    ax.set_title(
        f"6PPD Spectrum: "
        f"{xmin}-{xmax} m/z",
        fontsize=18
    )

    plt.tight_layout()

    fig.savefig(
        filename,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Detected {len(peak_idx)} peaks "
        f"in {xmin}-{xmax} m/z."
    )

    print(
        f"Created: {filename}"
    )


# ============================================================
# matchms
# ============================================================

def load_reference_library(reference_dir):

    result = {
        "available": MATCHMS_AVAILABLE,
        "status": "not_run",
        "directory": str(reference_dir),
        "files": [],
        "spectra": [],
        "error": None,
    }

    if not MATCHMS_AVAILABLE:

        result["status"] = "unavailable"

        result["error"] = MATCHMS_IMPORT_ERROR

        return result, []

    reference_dir = Path(
        reference_dir
    )

    if not reference_dir.exists():

        result["status"] = "no_reference_directory"

        return result, []

    files = []

    for suffix in [
        "*.msp",
        "*.mgf",
        "*.mzML",
        "*.mzml",
        "*.json",
        "*.mzXML",
        "*.mzxml",
    ]:

        files.extend(
            sorted(reference_dir.glob(suffix))
        )

    files = sorted(
        set(files)
    )

    result["files"] = [
        str(x)
        for x in files
    ]

    if not files:

        result["status"] = "no_reference_files"

        return result, []

    all_spectra = []

    try:

        for file in files:

            print(
                f"Loading reference spectra: "
                f"{file}"
            )

            spectra = list(
                load_spectra(
                    str(file)
                )
            )

            all_spectra.extend(
                spectra
            )

        result["status"] = "completed"

        result["spectrum_count"] = (
            len(all_spectra)
        )

        return (
            json_safe(result),
            all_spectra
        )

    except Exception as e:

        result["status"] = "failed"

        result["error"] = str(e)

        return (
            json_safe(result),
            []
        )


def create_matchms_query(
    mz,
    intensity,
    xmin=None,
    xmax=None
):

    if xmin is not None:

        mask = (
            (mz >= xmin)
            & (mz < xmax)
        )

        mz = mz[mask]
        intensity = intensity[mask]

    peak_idx = detect_peaks(
        mz,
        intensity,
        threshold=PEAK_THRESHOLD
    )

    if len(peak_idx) == 0:

        return Spectrum(
            mz=np.array([]),
            intensities=np.array([])
        )

    query_mz = mz[peak_idx]
    query_intensity = intensity[peak_idx]

    return Spectrum(
        mz=np.asarray(
            query_mz,
            dtype=float
        ),
        intensities=np.asarray(
            query_intensity,
            dtype=float
        ),
        metadata={
            "id": "query_6PPD"
        }
    )


def spectrum_metadata_dict(spectrum):

    try:

        metadata = spectrum.metadata

        if hasattr(
            metadata,
            "items"
        ):

            return {
                str(k): json_safe(v)
                for k, v in metadata.items()
            }

    except Exception:
        pass

    try:

        return {
            str(k): json_safe(
                spectrum.get(k)
            )
            for k in spectrum.metadata.keys()
        }

    except Exception:

        return {}


def run_matchms(
    query_mz,
    query_intensity,
    reference_spectra
):

    result = {
        "available": MATCHMS_AVAILABLE,
        "status": "not_run",
        "matches": [],
        "error": None,
    }

    if not MATCHMS_AVAILABLE:

        result["status"] = "unavailable"

        result["error"] = MATCHMS_IMPORT_ERROR

        return result

    if not reference_spectra:

        result["status"] = "no_reference_spectra"

        return result

    try:

        query = Spectrum(
            mz=np.asarray(
                query_mz,
                dtype=float
            ),
            intensities=np.asarray(
                query_intensity,
                dtype=float
            ),
            metadata={
                "id": "query_6PPD"
            }
        )

        similarity = CosineGreedy(
            tolerance=MATCH_TOLERANCE_DA,
            noise_cutoff=PEAK_THRESHOLD,
        )

        hits = []

        for index, reference in enumerate(
            reference_spectra
        ):

            score = similarity.pair(
                reference,
                query
            )

            if isinstance(
                score,
                dict
            ):

                similarity_score = float(
                    score.get(
                        "score",
                        0.0
                    )
                )

                matched_peaks = int(
                    score.get(
                        "matches",
                        0
                    )
                )

            else:

                similarity_score = float(
                    score[0]
                )

                matched_peaks = int(
                    score[1]
                )

            metadata = (
                spectrum_metadata_dict(
                    reference
                )
            )

            hits.append(
                {
                    "reference_index": index,
                    "score": similarity_score,
                    "matched_peaks":
                        matched_peaks,
                    "metadata": metadata,
                }
            )

        hits.sort(
            key=lambda x: (
                x["score"],
                x["matched_peaks"]
            ),
            reverse=True
        )

        result["matches"] = hits[:20]

        result["status"] = "completed"

        result["tolerance_da"] = (
            MATCH_TOLERANCE_DA
        )

        return json_safe(result)

    except Exception as e:

        result["status"] = "failed"

        result["error"] = str(e)

        return result


# ============================================================
# Encode image
# ============================================================

# ============================================================
# Build Qwen prompt
# ============================================================

def build_prompt(
    xmin,
    xmax,
    iteration,
    previous_answer=None,
    numerical_peaks=None,
    pyopenms_result=None,
    matchms_result=None,
):

    previous_section = ""

    if previous_answer is not None:

        previous_json = json.dumps(
            previous_answer,
            indent=2
        )

        previous_section = f"""

ITERATIVE VALIDATION

This is analysis iteration {iteration}.

A previous Qwen analysis is provided below.

Do NOT blindly accept it.

Re-check the supplied spectrum and the computational
results independently.

Correct unsupported m/z values, incorrect ion assignments,
incorrect isotope assignments, or unsupported compound
identifications.

Previous Qwen analysis:

{previous_json}

"""

    numerical_section = json.dumps(
        numerical_peaks or [],
        indent=2
    )

    pyopenms_section = json.dumps(
        pyopenms_result or {},
        indent=2
    )

    matchms_section = json.dumps(
        matchms_result or {},
        indent=2
    )

    prompt = f"""

You are analyzing a mass spectrum for:

6PPD

N-(1,3-dimethylbutyl)-N'-phenyl-p-phenylenediamine

The supplied image contains ONLY:

{xmin}-{xmax} m/z

This analysis combines:

1. The plotted mass spectrum.
2. Numerical peak detection.
3. pyOpenMS computational MS processing.
4. matchms reference-spectrum matching, when available.
5. Your independent mass-spectrometry interpretation.

Your job is NOT simply to agree with the libraries.

You must independently validate their outputs against the
visible spectrum and identify disagreements.

============================================================
COMPUTATIONAL PEAK DETECTION
============================================================

Numerically detected peaks:

{numerical_section}

============================================================
pyOpenMS RESULTS
============================================================

{pyopenms_section}

============================================================
matchms RESULTS
============================================================

{matchms_section}

============================================================
INTERPRETATION WORKFLOW
============================================================

Follow this order:

1. OBSERVE THE SPECTRUM

Identify clearly visible prominent peaks and approximate
m/z values.

2. VALIDATE NUMERICAL PEAKS

Check whether the numerically detected peaks correspond to
actual visible spectral features.

3. VALIDATE pyOpenMS

Assess whether centroided peaks and any isotope/deisotoping
results are consistent with the displayed spectrum.

Do not automatically accept a pyOpenMS assignment.

4. VALIDATE matchms

If reference matches are available, determine whether the
reported spectral similarity actually supports the proposed
compound.

Consider:

- similarity score
- number of matched peaks
- reference identity
- whether the matching peaks are chemically meaningful
- whether the reference spectrum is actually comparable
- whether the match is based on a few generic peaks

A high similarity score alone is NOT sufficient for compound
identification.

5. ASSIGN IONS

Classify important observed peaks as appropriate:

- molecular ion
- protonated molecular ion
- deprotonated molecular ion
- adduct
- isotope
- fragment
- neutral-loss fragment
- background/common ion
- unknown

6. INTERPRET FRAGMENTATION

Look for groups of ions that could originate from the same
component.

7. EVALUATE 6PPD

Determine whether the combined evidence supports 6PPD.

8. IDENTIFY OTHER COMPONENTS

Look for other chemical components supported by multiple ions
or a reference-spectrum match.

9. RESOLVE DISAGREEMENTS

If Qwen, pyOpenMS, and matchms disagree, explicitly identify
the disagreement and choose the interpretation best supported
by the actual spectrum.

10. FINAL ASSESSMENT

Give an assessment for THIS RANGE ONLY.

============================================================
IMPORTANT RULES
============================================================

1. Do not invent peaks.

2. Do not invent exact m/z values.

3. The plotted spectrum is the authority for whether a peak
   is actually observed.

4. Numerical algorithms are supporting evidence, not ground
   truth.

5. pyOpenMS output is supporting computational evidence, not
   ground truth.

6. matchms similarity is supporting evidence, not proof.

7. Do not identify a compound from one generic peak.

8. Do not assume the base peak is the molecular ion.

9. Consider isotope and adduct explanations.

10. If an expected ion is outside this displayed range,
    report "outside displayed range".

11. If an expected ion should be inside this range but is not
    visible, report it as missing.

12. Do not call a compound "strong" unless the spectrum
    provides sufficient independent evidence.

13. Distinguish:

    - directly observed peak
    - computationally detected peak
    - pyOpenMS-supported assignment
    - reference-library-supported identification
    - Qwen interpretation
    - tentative compound assignment
    - unsupported speculation

14. Other components should be reported when supported.

15. If a component cannot be reliably identified, use
    "unknown component".

16. Re-check everything independently on every iteration.

17. Do not provide hidden chain-of-thought.

18. Return ONLY valid JSON.

{previous_section}

============================================================
REQUIRED JSON
============================================================

{{
  "compound": "6PPD",
  "range": "{xmin}-{xmax}",
  "iteration": {iteration},

  "assessment":
    "consistent|not_consistent|ambiguous",

  "observed_ions": [
    {{
      "mz": 0,
      "relative_intensity": 0.0,
      "ion_type":
        "molecular_ion|protonated_molecular_ion|deprotonated_molecular_ion|adduct|isotope|fragment|neutral_loss_fragment|background|unknown",
      "assignment": "",
      "confidence": 0.0,
      "comment": ""
    }}
  ],

  "pyopenms_validation": {{
    "validated": true,
    "agreement": "agree|partial|disagree|not_available",
    "comment": ""
  }},

  "matchms_validation": {{
    "validated": true,
    "agreement": "agree|partial|disagree|not_available",
    "best_reference": "",
    "score": 0.0,
    "matched_peaks": 0,
    "comment": ""
  }},

  "sixppd_evidence": [
    {{
      "mz": 0,
      "observed": true,
      "ion_type":
        "molecular_ion|fragment|supporting|unknown",
      "confidence": 0.0,
      "comment": ""
    }}
  ],

  "other_components": [
    {{
      "name": "",
      "confidence": 0.0,
      "identification_level":
        "tentative|possible|strong",
      "supporting_ions": [],
      "comment": ""
    }}
  ],

  "other_prominent_peaks": [
    {{
      "mz": 0,
      "comment": ""
    }}
  ],

  "library_disagreements": [],

  "missing_expected_features": [],

  "outside_range_features": [],

  "summary": ""
}}

Confidence must be between 0 and 1.

Only report observed ions that are actually supported by the
displayed spectrum.

If an ion cannot be assigned reliably, use:

"ion_type": "unknown"

Do not fabricate a compound name.
"""

    return prompt


# ============================================================
# Save JSON immediately
# ============================================================

def save_results(
    results,
    output_json
):

    temp_json = Path(
        str(output_json) + ".tmp"
    )

    with open(
        temp_json,
        "w"
    ) as f:

        json.dump(
            json_safe(results),
            f,
            indent=2
        )

    temp_json.replace(
        output_json
    )


# ============================================================
# Extract Qwen content safely
# ============================================================

def extract_qwen_content(
    api_result
):

    if not isinstance(
        api_result,
        dict
    ):

        return None

    choices = api_result.get(
        "choices",
        []
    )

    if not choices:

        return None

    first = choices[0]

    if not isinstance(
        first,
        dict
    ):

        return None

    message = first.get(
        "message",
        {}
    )

    if not isinstance(
        message,
        dict
    ):

        return None

    content = message.get(
        "content"
    )

    if content is None:

        return None

    if isinstance(
        content,
        str
    ):

        return content

    if isinstance(
        content,
        list
    ):

        text_parts = []

        for item in content:

            if isinstance(
                item,
                dict
            ):

                text_value = item.get(
                    "text"
                )

                if text_value:

                    text_parts.append(
                        str(text_value)
                    )

        if text_parts:

            return "\n".join(
                text_parts
            )

    return str(content)


# ============================================================
# Parse Qwen JSON
# ============================================================

def parse_qwen_json(
    content
):

    if not content:

        return None, True

    cleaned = content.strip()

    cleaned = cleaned.replace(
        "```json",
        ""
    )

    cleaned = cleaned.replace(
        "```JSON",
        ""
    )

    cleaned = cleaned.replace(
        "```",
        ""
    )

    cleaned = cleaned.strip()

    try:

        return (
            json.loads(cleaned),
            False
        )

    except json.JSONDecodeError:

        start = cleaned.find(
            "{"
        )

        end = cleaned.rfind(
            "}"
        )

        if (
            start >= 0
            and end > start
        ):

            candidate = cleaned[
                start:end + 1
            ]

            try:

                return (
                    json.loads(candidate),
                    False
                )

            except json.JSONDecodeError:

                pass

        return (
            None,
            True
        )


# ============================================================
# One Qwen call
# ============================================================

# ============================================================
# Annotate spectrum
# ============================================================

def annotate_spectrum(
    mz,
    intensity,
    analysis,
    filename,
    xmin,
    xmax
):

    range_mz, range_intensity = (
        get_range(
            mz,
            intensity,
            xmin,
            xmax
        )
    )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        range_mz,
        range_intensity,
        color="black",
        linewidth=0.6
    )

    # --------------------------------------------------------
    # Original numerical peaks
    # --------------------------------------------------------

    peak_idx = detect_peaks(
        range_mz,
        range_intensity,
        threshold=PEAK_THRESHOLD
    )

    for i in peak_idx:

        y = range_intensity[i]

        ax.plot(
            [
                range_mz[i],
                range_mz[i]
            ],
            [
                y + 0.01,
                y + 0.04
            ],
            color="red",
            linewidth=1.0
        )

    # --------------------------------------------------------
    # Qwen diagnostic peaks
    # --------------------------------------------------------

    colors = {

        "molecular_ion":
            "blue",

        "protonated_molecular_ion":
            "blue",

        "deprotonated_molecular_ion":
            "blue",

        "fragment":
            "magenta",

        "supporting":
            "green",

        "isotope":
            "orange",

        "adduct":
            "cyan",

        "unknown":
            "gray"
    }

    if not isinstance(
        analysis,
        dict
    ):

        analysis = {}

    annotation_count = 0

    # --------------------------------------------------------
    # Combine 6PPD evidence with observed ions
    # --------------------------------------------------------

    diagnostic = []

    for ion in analysis.get(
        "observed_ions",
        []
    ):

        if isinstance(
            ion,
            dict
        ):

            diagnostic.append(
                {
                    "mz": ion.get("mz"),
                    "observed": True,
                    "role":
                        ion.get(
                            "ion_type",
                            "unknown"
                        )
                }
            )

    for peak in analysis.get(
        "sixppd_evidence",
        []
    ):

        if isinstance(
            peak,
            dict
        ):

            diagnostic.append(
                {
                    "mz": peak.get("mz"),
                    "observed":
                        peak.get(
                            "observed",
                            False
                        ),
                    "role":
                        peak.get(
                            "ion_type",
                            "supporting"
                        )
                }
            )

    # Remove duplicate m/z values
    seen = set()

    for peak in diagnostic:

        if not peak.get(
            "observed",
            False
        ):

            continue

        try:

            target_mz = float(
                peak["mz"]
            )

        except (
            KeyError,
            ValueError,
            TypeError
        ):

            continue

        if (
            target_mz < xmin
            or target_mz >= xmax
        ):

            continue

        rounded_key = round(
            target_mz,
            4
        )

        if rounded_key in seen:

            continue

        seen.add(
            rounded_key
        )

        idx = np.argmin(
            np.abs(
                range_mz - target_mz
            )
        )

        observed_mz = range_mz[idx]

        observed_intensity = (
            range_intensity[idx]
        )

        role = peak.get(
            "role",
            "unknown"
        )

        color = colors.get(
            role,
            "gray"
        )

        # Marker
        ax.plot(
            [
                observed_mz,
                observed_mz
            ],
            [
                observed_intensity + 0.015,
                observed_intensity + 0.08
            ],
            color=color,
            linewidth=2.0
        )

        # Label
        ax.text(
            observed_mz,
            observed_intensity + 0.085,
            str(
                round(
                    observed_mz
                )
            ),
            ha="center",
            va="bottom",
            fontsize=10,
            color=color
        )

        annotation_count += 1

    # --------------------------------------------------------
    # Assessment
    # --------------------------------------------------------

    assessment = analysis.get(
        "assessment",
        "ambiguous"
    )

    ax.text(
        0.98,
        0.95,
        f"6PPD: {assessment}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlabel(
        "m/z",
        fontsize=18
    )

    ax.set_ylabel(
        "Normalized intensity",
        fontsize=18
    )

    ax.tick_params(
        axis="both",
        labelsize=12
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(
        xmin,
        xmax
    )

    upper = max(
        1.05,
        float(
            range_intensity.max()
        ) * 1.12
    )

    ax.set_ylim(
        0,
        min(1.20, upper)
    )

    ax.set_title(
        f"6PPD Spectrum: "
        f"{xmin}-{xmax} m/z",
        fontsize=18
    )

    plt.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Annotated {annotation_count} "
        f"diagnostic peaks."
    )

    print(
        f"Saved: {filename}"
    )


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "  python3 check2.py "
            "<spectrum_directory>"
        )

        sys.exit(1)

    input_dir = Path(
        sys.argv[1]
    ).resolve()

    if not input_dir.is_dir():

        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{input_dir}"
        )

    print(
        f"Input directory: "
        f"{input_dir}"
    )

    # --------------------------------------------------------
    # Find actual spectrum
    # --------------------------------------------------------

    spectrum_file = (
        input_dir /
        "raw_spectrum.parquet"
    )

    if spectrum_file.exists():

        print(
            "Using spectrum:\n"
            f"  {spectrum_file}"
        )

    else:

        parquet_files = sorted(
            input_dir.glob(
                "*.parquet"
            )
        )

        spectrum_candidates = [
            f
            for f in parquet_files
            if "spectrum"
            in f.name.lower()
        ]

        if spectrum_candidates:

            spectrum_file = (
                spectrum_candidates[0]
            )

        else:

            raise FileNotFoundError(
                "Could not find "
                "raw_spectrum.parquet"
            )

    # --------------------------------------------------------
    # Load spectrum
    # --------------------------------------------------------

    mz, intensity = load_spectrum(
        spectrum_file
    )

    # --------------------------------------------------------
    # pyOpenMS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING pyOpenMS")
    print("=" * 70)

    pyopenms_result = (
        run_pyopenms_peak_processing(
            mz,
            intensity
        )
    )

    print(
        "pyOpenMS status:",
        pyopenms_result.get(
            "status"
        )
    )

    if pyopenms_result.get(
        "centroided_points"
    ):

        print(
            "Centroided peaks:",
            pyopenms_result[
                "centroided_points"
            ]
        )

    # --------------------------------------------------------
    # matchms reference library
    # --------------------------------------------------------

    reference_dir = (
        input_dir /
        REFERENCE_DIR_NAME
    )

    print()
    print("=" * 70)
    print("LOADING matchms REFERENCE LIBRARY")
    print("=" * 70)

    reference_result, reference_spectra = (
        load_reference_library(
            reference_dir
        )
    )

    print(
        "matchms status:",
        reference_result.get(
            "status"
        )
    )

    print(
        "Reference spectra:",
        reference_result.get(
            "spectrum_count",
            0
        )
    )

    # --------------------------------------------------------
    # Output JSON
    # --------------------------------------------------------

    output_json = (
        input_dir /
        "spectrum_6PPD_qwen_results.json"
    )

    # --------------------------------------------------------
    # Initialize JSON immediately
    # --------------------------------------------------------

    results = {

        "compound": "6PPD",

        "model": MODEL,

        "api_url": API_URL,

        "iterations_per_range":
            N_ITERATIONS,

        "peak_threshold":
            PEAK_THRESHOLD,

        "match_tolerance_da":
            MATCH_TOLERANCE_DA,

        "spectrum_file":
            str(spectrum_file),

        "libraries": {

            "pyopenms": {
                "available":
                    PYOPENMS_AVAILABLE,
                "status":
                    pyopenms_result.get(
                        "status"
                    ),
                "error":
                    pyopenms_result.get(
                        "error"
                    )
            },

            "matchms": reference_result

        },

        "ranges": {}

    }

    save_results(
        results,
        output_json
    )

    print(
        "\nInitialized results file:"
        f"\n  {output_json}"
    )

    # ========================================================
    # Process each requested range
    # ========================================================

    for xmin, xmax in RANGES:

        range_name = (
            f"{xmin}-{xmax}"
        )

        print()
        print("#" * 70)
        print(
            f"PROCESSING RANGE: "
            f"{range_name} m/z"
        )
        print("#" * 70)

        # ----------------------------------------------------
        # Range-specific files
        # ----------------------------------------------------

        output_plot = (
            input_dir /
            f"spectrum_{xmin}_{xmax}.png"
        )

        output_annotated = (
            input_dir /
            f"spectrum_{xmin}_{xmax}_annotated.png"
        )

        # ----------------------------------------------------
        # Numerical peak table
        # ----------------------------------------------------

        numerical_peaks = (
            numerical_peak_table(
                mz,
                intensity,
                xmin,
                xmax
            )
        )

        # ----------------------------------------------------
        # matchms for this range
        # ----------------------------------------------------

        range_mz, range_intensity = (
            get_range(
                mz,
                intensity,
                xmin,
                xmax
            )
        )

        matchms_result = run_matchms(
            range_mz,
            range_intensity,
            reference_spectra
        )

        # ----------------------------------------------------
        # Initialize range in JSON immediately
        # ----------------------------------------------------

        results[
            "ranges"
        ][range_name] = {

            "range_min":
                xmin,

            "range_max":
                xmax,

            "spectrum_image":
                str(output_plot),

            "annotated_image":
                str(output_annotated),

            "numerical_peaks":
                numerical_peaks,

            "pyopenms":
                pyopenms_result,

            "matchms":
                matchms_result,

            "calls": [],

            "final_analysis":
                None,

            "status":
                "running"

        }

        save_results(
            results,
            output_json
        )

        # ----------------------------------------------------
        # Create range-specific spectrum
        # ----------------------------------------------------

        print(
            f"\nCreating "
            f"{range_name} spectrum..."
        )

        make_spectrum_plot(
            mz,
            intensity,
            output_plot,
            xmin,
            xmax
        )

        # ----------------------------------------------------
        # Encode only THIS range
        # ----------------------------------------------------

      

        # ----------------------------------------------------
        # Iterative Qwen analysis
        # ----------------------------------------------------

        previous_answer = None

        for iteration in range(
            1,
            N_ITERATIONS + 1
        ):

            call_record = call_qwen(
    xmin,
    xmax,
    iteration,
    previous_answer,
    numerical_peaks,
    pyopenms_result,
    matchms_result
)

            # =================================================
            # SAVE IMMEDIATELY AFTER EVERY CALL
            # =================================================

            results[
                "ranges"
            ][range_name][
                "calls"
            ].append(
                call_record
            )

            results[
                "ranges"
            ][range_name][
                "last_completed_iteration"
            ] = iteration

            save_results(
                results,
                output_json
            )

            print(
                f"\nSaved Qwen call "
                f"{iteration} immediately:"
            )

            print(
                f"  {output_json}"
            )

            # ------------------------------------------------
            # Use parsed answer for next iteration
            # ------------------------------------------------

            parsed_answer = (
                call_record.get(
                    "parsed_answer"
                )
            )

            if isinstance(
                parsed_answer,
                dict
            ):

                previous_answer = (
                    parsed_answer
                )

            # ------------------------------------------------
            # Continue after API failure
            # ------------------------------------------------

            if call_record.get(
                "api_error"
            ):

                print(
                    "This Qwen call had "
                    "an error."
                )

                print(
                    "Continuing to next "
                    "iteration."
                )

        # ----------------------------------------------------
        # Determine final answer
        # ----------------------------------------------------

        final_analysis = None

        calls = results[
            "ranges"
        ][range_name][
            "calls"
        ]

        # Last successfully parsed response
        for call in reversed(
            calls
        ):

            candidate = call.get(
                "parsed_answer"
            )

            if isinstance(
                candidate,
                dict
            ):

                final_analysis = (
                    candidate
                )

                break

        results[
            "ranges"
        ][range_name][
            "final_analysis"
        ] = final_analysis

        results[
            "ranges"
        ][range_name][
            "status"
        ] = (
            "completed"
            if final_analysis is not None
            else "failed"
        )

        # ----------------------------------------------------
        # Save before annotation
        # ----------------------------------------------------

        save_results(
            results,
            output_json
        )

        # ----------------------------------------------------
        # Annotate final result
        # ----------------------------------------------------

        if final_analysis is not None:

            print()

            print(
                f"Creating final annotation "
                f"for {range_name}..."
            )

            annotate_spectrum(
                mz,
                intensity,
                final_analysis,
                output_annotated,
                xmin,
                xmax
            )

            results[
                "ranges"
            ][range_name][
                "annotation_status"
            ] = "completed"

        else:

            print(
                f"No valid Qwen analysis "
                f"available for {range_name}."
            )

            results[
                "ranges"
            ][range_name][
                "annotation_status"
            ] = "skipped"

        # ----------------------------------------------------
        # Save again after annotation
        # ----------------------------------------------------

        save_results(
            results,
            output_json
        )

        print()
        print(
            f"Finished range "
            f"{range_name}"
        )

    # ========================================================
    # Final save
    # ========================================================

    results[
        "status"
    ] = "completed"

    save_results(
        results,
        output_json
    )

    print()
    print("=" * 70)
    print("ALL RANGES COMPLETE")
    print("=" * 70)

    print(
        f"Results:\n"
        f"  {output_json}"
    )

    for xmin, xmax in RANGES:

        print(
            f"\n{xmin}-{xmax} m/z:"
        )

        print(
            f"  spectrum_"
            f"{xmin}_{xmax}.png"
        )

        print(
            f"  spectrum_"
            f"{xmin}_{xmax}_annotated.png"
        )

        print(
            f"  Qwen calls: "
            f"{N_ITERATIONS}"
        )







# ============================================================
# Qwen iteration stages
# ============================================================

def get_iteration_stage(iteration):

    if 1 <= iteration <= 10:

        return {
            "stage": "peak_ion_validation",
            "objective": """
Focus ONLY on validating the observed spectral peaks and
their possible ion assignments.

Do not make a final compound identification.

Evaluate:

- observed m/z values
- relative intensities
- numerical peak detection
- pyOpenMS centroided peaks
- isotope patterns
- possible adducts
- molecular-ion candidates
- fragment-ion candidates
- neutral-loss relationships
- background/common ions
- unknown ions

Challenge previous Qwen assignments.

Correct inaccurate m/z values and unsupported ion assignments.

The main goal is to establish a reliable list of observed
ions before discussing compound identity.
"""
        }

    elif 11 <= iteration <= 20:

        return {
            "stage": "compound_library_validation",
            "objective": """
Focus on compound identification and disagreement between
computational evidence and chemical interpretation.

Use the validated peak/ion information from earlier iterations.

Evaluate:

- 6PPD evidence
- other possible chemical components
- molecular-ion evidence
- characteristic fragments
- isotope/adduct evidence
- pyOpenMS assignments
- matchms reference-library matches
- similarity scores
- number of matched peaks
- whether matched peaks are chemically meaningful
- disagreements between Qwen, pyOpenMS, and matchms

Do NOT accept a matchms hit simply because its similarity
score is high.

A compound identification should require chemically meaningful
support from multiple ions or a strong reference-spectrum match.

Explicitly identify disagreements and unsupported assignments.
"""
        }

    elif 21 <= iteration <= 30:

        return {
            "stage": "final_consensus",
            "objective": """
Focus on final consensus.

Re-evaluate the entire evidence set:

- observed spectrum peaks
- numerical peak detection
- pyOpenMS results
- isotope/deisotoping information
- matchms reference matches
- previous Qwen analyses
- disagreements identified during iterations 1-20

Resolve disagreements conservatively.

Separate:

1. directly observed ions
2. computationally supported ions
3. plausible ion assignments
4. reference-library-supported identifications
5. tentative compound identifications
6. unsupported speculation

Produce the final range-level assessment for 6PPD and any
other components.

Do not introduce new unsupported peaks or compounds merely
because they appeared in an earlier Qwen response.
"""
        }

    raise ValueError(
        f"Iteration must be between 1 and 30: {iteration}"
    )


# ============================================================
# Build Qwen prompt
#
# IMPORTANT:
# Qwen receives NO IMAGE.
# ============================================================

def build_prompt(
    xmin,
    xmax,
    iteration,
    previous_answer=None,
    numerical_peaks=None,
    pyopenms_result=None,
    matchms_result=None,
):

    stage_info = get_iteration_stage(
        iteration
    )

    previous_section = ""

    if previous_answer is not None:

        previous_json = json.dumps(
            previous_answer,
            indent=2
        )

        previous_section = f"""

============================================================
PREVIOUS QWEN ANALYSIS
============================================================

This is a previous Qwen analysis.

Do NOT blindly accept it.

Treat it as a hypothesis that must be independently checked
against the numerical and computational evidence.

Correct:

- incorrect m/z values
- unsupported peaks
- incorrect ion assignments
- incorrect isotope assignments
- unsupported compound identifications
- unsupported match interpretations

Previous Qwen analysis:

{previous_json}

============================================================
END PREVIOUS ANALYSIS
============================================================
"""

    numerical_section = json.dumps(
        numerical_peaks or [],
        indent=2
    )

    pyopenms_section = json.dumps(
        pyopenms_result or {},
        indent=2
    )

    matchms_section = json.dumps(
        matchms_result or {},
        indent=2
    )

    prompt = f"""

You are a conservative mass-spectrometry analyst.

You are analyzing:

6PPD
N-(1,3-dimethylbutyl)-N'-phenyl-p-phenylenediamine

Displayed spectrum range:

{xmin}-{xmax} m/z

IMPORTANT:

You are operating in TEXT-ONLY mode.

You are NOT receiving the spectrum image.

Do NOT infer visual features that are not represented in the
numerical data supplied below.

The numerical peak list, pyOpenMS results, and matchms results
are the available computational evidence.

============================================================
CURRENT ITERATION
============================================================

Iteration: {iteration}

Stage:

{stage_info["stage"]}

Objective:

{stage_info["objective"]}

============================================================
NUMERICAL PEAK DETECTION
============================================================

The original spectrum was processed numerically.

Detected peaks:

{numerical_section}

These values come from the original spectrum data, not from
image interpretation.

============================================================
pyOpenMS RESULTS
============================================================

{pyopenms_section}

Use pyOpenMS as computational evidence.

Do not automatically accept every pyOpenMS result.

Check whether the reported centroiding, isotope, and
deisotoping information is chemically plausible.

============================================================
matchms RESULTS
============================================================

{matchms_section}

Use matchms as reference-spectrum evidence.

Do not treat a similarity score as proof of identity.

Consider:

- similarity score
- number of matched peaks
- reference identity
- whether the matched ions are distinctive
- whether the reference spectrum is comparable
- whether the match could result from common/background peaks

============================================================
MASS-SPECTROMETRY RULES
============================================================

1. Never invent an observed peak.

2. Never invent an exact m/z value.

3. Only use m/z values supplied by the numerical spectrum or
   computational results.

4. Do not infer peaks from the name of the compound.

5. Do not assume the highest-intensity peak is the molecular
   ion.

6. Do not identify a compound from one generic peak.

7. Prefer multiple chemically meaningful ions.

8. Consider molecular ions, fragments, neutral losses,
   isotopes, adducts, and background ions.

9. Distinguish observed ions from interpreted ions.

10. Distinguish computational evidence from chemical
    interpretation.

11. A matchms hit is evidence, not proof.

12. pyOpenMS output is evidence, not proof.

13. If an expected feature lies outside the displayed range,
    report it as "outside displayed range".

14. If an expected feature should be inside the range but is
    not represented in the numerical peak list, report it as
    missing only when justified.

15. Do not claim another compound unless sufficient evidence
    exists.

16. If another component cannot be reliably identified, use
    "unknown component".

17. Do not expose hidden chain-of-thought.

18. Return ONLY valid JSON.

{previous_section}

============================================================
REQUIRED OUTPUT
============================================================

Return exactly this JSON structure:

{{
  "compound": "6PPD",
  "range": "{xmin}-{xmax}",
  "iteration": {iteration},
  "stage": "{stage_info["stage"]}",

  "assessment":
    "consistent|not_consistent|ambiguous",

  "observed_ions": [
    {{
      "mz": 0.0,
      "relative_intensity": 0.0,
      "ion_type":
        "molecular_ion|protonated_molecular_ion|deprotonated_molecular_ion|adduct|isotope|fragment|neutral_loss_fragment|background|unknown",
      "assignment": "",
      "confidence": 0.0,
      "comment": ""
    }}
  ],

  "pyopenms_validation": {{
    "validated": true,
    "agreement":
      "agree|partial|disagree|not_available",
    "comment": ""
  }},

  "matchms_validation": {{
    "validated": true,
    "agreement":
      "agree|partial|disagree|not_available",
    "best_reference": "",
    "score": 0.0,
    "matched_peaks": 0,
    "comment": ""
  }},

  "sixppd_evidence": [
    {{
      "mz": 0.0,
      "observed": true,
      "ion_type":
        "molecular_ion|fragment|supporting|unknown",
      "confidence": 0.0,
      "comment": ""
    }}
  ],

  "other_components": [
    {{
      "name": "",
      "confidence": 0.0,
      "identification_level":
        "tentative|possible|strong",
      "supporting_ions": [],
      "comment": ""
    }}
  ],

  "other_prominent_peaks": [
    {{
      "mz": 0.0,
      "comment": ""
    }}
  ],

  "library_disagreements": [],

  "missing_expected_features": [],

  "outside_range_features": [],

  "summary": ""
}}

Confidence must be between 0 and 1.

For iterations 1-10, prioritize peak and ion validation.

For iterations 11-20, prioritize compound and library
validation.

For iterations 21-30, prioritize final consensus.

Do not use image-based reasoning.

Return ONLY JSON.
"""

    return prompt


# ============================================================
# One Qwen call
#
# IMPORTANT:
# No image is sent.
# ============================================================

def call_qwen(
    xmin,
    xmax,
    iteration,
    previous_answer,
    numerical_peaks,
    pyopenms_result,
    matchms_result
):

    stage_info = get_iteration_stage(
        iteration
    )

    prompt = build_prompt(
        xmin,
        xmax,
        iteration,
        previous_answer,
        numerical_peaks,
        pyopenms_result,
        matchms_result,
    )

    headers = {
        "Content-Type": "application/json"
    }

    if API_KEY.strip():

        headers["Authorization"] = (
            f"Bearer {API_KEY}"
        )

    # --------------------------------------------------------
    # TEXT ONLY
    #
    # There is deliberately NO image_url here.
    # --------------------------------------------------------

    payload = {

        "model": MODEL,

        "messages": [

            {
                "role": "system",

                "content": (
                    "You are a conservative "
                    "mass-spectrometry analyst. "
                    "You are operating in text-only "
                    "mode. Do not use or assume "
                    "vision capabilities. "
                    "Validate numerical, pyOpenMS, "
                    "and matchms evidence independently. "
                    "Never invent observed peaks. "
                    "Return only the requested JSON."
                )
            },

            {
                "role": "user",

                "content": prompt
            }

        ],

        "temperature": 0.1,

        "max_tokens": 5000
    }

    call_record = {

        "iteration":
            iteration,

        "stage":
            stage_info["stage"],

        "range":
            f"{xmin}-{xmax}",

        "model":
            MODEL,

        "vision_used":
            False,

        "prompt":
            prompt,

        "request":
            payload,

        "raw_api_response":
            None,

        "raw_answer":
            None,

        "parsed_answer":
            None,

        "parse_error":
            False,

        "api_error":
            None
    }

    print()
    print("=" * 70)

    print(
        f"Qwen call "
        f"{iteration}/{N_ITERATIONS}"
        f" | range {xmin}-{xmax}"
    )

    print(
        f"Stage: {stage_info['stage']}"
    )

    print(
        f"Vision: DISABLED"
    )

    print("=" * 70)

    try:

        response = requests.post(

            API_URL,

            headers=headers,

            json=payload,

            timeout=300
        )

        call_record[
            "http_status"
        ] = response.status_code

        response.raise_for_status()

        try:

            api_result = response.json()

        except Exception:

            api_result = {
                "raw_text_response":
                    response.text
            }

        # ----------------------------------------------------
        # Save complete API response
        # ----------------------------------------------------

        call_record[
            "raw_api_response"
        ] = api_result

        content = extract_qwen_content(
            api_result
        )

        if content is None:

            call_record[
                "api_error"
            ] = (
                "Qwen returned no "
                "message content."
            )

            print(
                "WARNING: Qwen returned "
                "no content."
            )

            return call_record

        call_record[
            "raw_answer"
        ] = content

        parsed, parse_error = (
            parse_qwen_json(
                content
            )
        )

        call_record[
            "parsed_answer"
        ] = parsed

        call_record[
            "parse_error"
        ] = parse_error

        if parse_error:

            call_record[
                "api_error"
            ] = (
                "Qwen returned content, "
                "but it could not be parsed "
                "as JSON."
            )

            print(
                "WARNING: JSON parsing failed."
            )

        else:

            print(
                "Qwen response parsed successfully."
            )

        return call_record

    except requests.exceptions.RequestException as e:

        call_record[
            "api_error"
        ] = str(e)

        print(
            f"Qwen request failed: {e}"
        )

        return call_record

    except Exception as e:

        call_record[
            "api_error"
        ] = str(e)

        print(
            f"Unexpected Qwen error: {e}"
        )

        return call_record
    
    
# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()