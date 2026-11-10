"""Feature metadata for the 41 CICFlowMeter-derived features used by the DNN ensemble.

Each entry in FEATURE_INFO maps the exact feature name (as stored in models/feature_names.json)
to a dict containing:
  - description: plain-English meaning based on standard CICFlowMeter definitions (for LLM prompt).
  - implication: concise security interpretation under 15 words (for LLM prompt).
  - note: optional developer-only context (dataset statistics, medians, or caveats).

Only 'description' and 'implication' are provided to the LLM via describe_for_prompt().
"""

from typing import Any

# Registry: exact feature name -> {description, implication, optional note}
FEATURE_INFO: dict[str, dict[str, str]] = {
    "Protocol": {
        "description": (
            "Transport-layer protocol number for the flow (e.g., 6 = TCP, 17 = UDP)."
        ),
        "implication": (
            "Predominance of TCP can be consistent with targeted application or connection flooding."
        ),
        "note": "In training data, DoS/DDoS are nearly 100% TCP, while Benign has ~28% UDP.",
    },
    "Flow Duration": {
        "description": (
            "Total duration of the network flow in microseconds, from first to last packet."
        ),
        "implication": (
            "Very short or very long durations can indicate high-rate flooding or slow-rate attacks."
        ),
        "note": "DoS median ~62 ms (short flooding), DDoS median ~1.4 s (slow/hold attacks).",
    },
    "Tot Fwd Pkts": {
        "description": (
            "Total number of packets transmitted in the forward direction from source to destination."
        ),
        "implication": (
            "High forward packet counts with few backward packets are consistent with one-sided flooding."
        ),
        "note": "Similar medians across classes; differentiates mainly in upper percentiles.",
    },
    "Tot Bwd Pkts": {
        "description": (
            "Total number of packets transmitted in the backward direction from destination to source."
        ),
        "implication": (
            "Near-zero backward packet count is consistent with unanswered DoS flooding traffic."
        ),
        "note": "DoS median 0 backward packets vs Benign 2 and DDoS 4.",
    },
    "Fwd Pkt Len Max": {
        "description": (
            "Maximum size in bytes of a single packet observed in the forward direction."
        ),
        "implication": (
            "Near-zero maximum forward packet size is consistent with header-only flooding."
        ),
        "note": "DoS median 0 bytes vs Benign 64 bytes and DDoS 20 bytes.",
    },
    "Fwd Pkt Len Min": {
        "description": (
            "Minimum size in bytes of a single packet observed in the forward direction."
        ),
        "implication": (
            "Zero minimum forward packet size can be consistent with payload-free traffic."
        ),
        "note": "Median 0 across all classes; 95th percentile Benign is 47 bytes vs 0 for attacks.",
    },
    "Fwd Pkt Len Mean": {
        "description": (
            "Mean size in bytes of packets transmitted in the forward direction."
        ),
        "implication": (
            "Low mean forward packet length is consistent with payload-free flooding traffic."
        ),
        "note": "DoS median 0 bytes vs Benign 44 bytes and DDoS 6.7 bytes.",
    },
    "Bwd Pkt Len Max": {
        "description": (
            "Maximum size in bytes of a single packet observed in the backward direction."
        ),
        "implication": (
            "Zero maximum backward packet length is consistent with unanswered DoS flows."
        ),
        "note": "DoS median 0 bytes vs Benign 147 bytes and DDoS 935 bytes.",
    },
    "Bwd Pkt Len Min": {
        "description": (
            "Minimum size in bytes of a single packet observed in the backward direction."
        ),
        "implication": (
            "Zero minimum backward length is consistent with flows lacking return traffic."
        ),
        "note": "Median 0 across all classes; DoS 95th percentile is also 0.",
    },
    "Bwd Pkt Len Mean": {
        "description": (
            "Mean size in bytes of packets transmitted in the backward direction."
        ),
        "implication": (
            "Near-zero mean backward packet length is consistent with one-sided DoS flooding."
        ),
        "note": "DoS median near 0 vs Benign 110 bytes and DDoS 234 bytes.",
    },
    "Flow Byts/s": {
        "description": (
            "Total bytes transferred in both directions divided by flow duration in seconds."
        ),
        "implication": (
            "Very high or very low byte rate can indicate volumetric flooding or stalled flows."
        ),
        "note": "Benign median ~1190 B/s, DoS median ~0 B/s (stalled/unanswered), DDoS ~649 B/s.",
    },
    "Flow Pkts/s": {
        "description": (
            "Total packets transferred in both directions divided by flow duration in seconds."
        ),
        "implication": (
            "Very high rates are consistent with packet flooding; very low rates with slow or stalled flows."
        ),
        "note": "DoS median ~44 pkts/s vs Benign ~18 pkts/s and DDoS ~5 pkts/s.",
    },
    "Flow IAT Mean": {
        "description": (
            "Mean inter-arrival time in microseconds between consecutive packets in both directions."
        ),
        "implication": (
            "Very low mean IAT is consistent with rapid injection; very high with slow-rate traffic."
        ),
        "note": "DoS median ~37 ms vs Benign ~72 ms and DDoS ~236 ms.",
    },
    "Flow IAT Std": {
        "description": (
            "Standard deviation of inter-arrival times between consecutive packets in both directions."
        ),
        "implication": (
            "Low IAT variability can be consistent with machine-generated uniform flooding traffic."
        ),
        "note": "DDoS median std ~3 ms vs Benign ~22 ms and DoS ~12 ms.",
    },
    "Flow IAT Max": {
        "description": (
            "Maximum inter-arrival time in microseconds between consecutive packets in both directions."
        ),
        "implication": (
            "Context dependent; read together with flow duration and mean IAT."
        ),
        "note": "DDoS median 1.4 s vs DoS 44 ms and Benign 199 ms.",
    },
    "Fwd IAT Std": {
        "description": (
            "Standard deviation of inter-arrival times between consecutive forward-direction packets."
        ),
        "implication": (
            "Very low forward IAT variability is consistent with uniformly timed packet injection."
        ),
        "note": "DDoS median forward IAT std ~200 us vs Benign ~673 us and DoS ~7.7 ms.",
    },
    "Bwd IAT Tot": {
        "description": (
            "Total inter-arrival time accumulated across backward-direction packets in microseconds."
        ),
        "implication": (
            "Near-zero total backward IAT is consistent with absent or minimal return traffic."
        ),
        "note": "DoS median near 0 due to 0 backward packets; DDoS ~8.5 ms and Benign ~627 us.",
    },
    "Bwd IAT Mean": {
        "description": (
            "Mean inter-arrival time in microseconds between consecutive backward-direction packets."
        ),
        "implication": (
            "Near-zero backward IAT mean is consistent with minimal or absent return traffic."
        ),
        "note": "DoS median near 0 due to absence of backward packets.",
    },
    "Bwd IAT Std": {
        "description": (
            "Standard deviation of inter-arrival times between backward-direction packets."
        ),
        "implication": (
            "Context dependent; near-zero in flows with few or no backward packets."
        ),
        "note": "Median near 0 for Benign and DoS; DDoS median is ~4.3 ms.",
    },
    "Bwd IAT Max": {
        "description": (
            "Maximum inter-arrival time in microseconds between backward-direction packets."
        ),
        "implication": (
            "Near-zero maximum backward IAT is consistent with absent or minimal return traffic."
        ),
        "note": "DoS median near 0 vs Benign ~466 us and DDoS ~7.8 ms.",
    },
    "Bwd IAT Min": {
        "description": (
            "Minimum inter-arrival time in microseconds between backward-direction packets."
        ),
        "implication": (
            "Near-zero minimum backward IAT is consistent with flows that have few return packets."
        ),
        "note": "DoS median near 0 vs Benign ~1 us and DDoS ~20 us.",
    },
    "Fwd PSH Flags": {
        "description": (
            "Count of packets with the TCP PSH flag set in forward direction; 0 for UDP."
        ),
        "implication": (
            "Indicates forward immediate push requests; interpret alongside transport protocol and payload."
        ),
        "note": "Near-zero median across classes; 95th percentile Benign is 1 vs 0 for attacks.",
    },
    "Fwd URG Flags": {
        "description": (
            "Count of packets with the TCP URG flag set in forward direction; 0 for UDP."
        ),
        "implication": (
            "Flags urgent forward data; non-zero counts can indicate anomalous packet craft."
        ),
        "note": "Near-zero across all classes in train data.",
    },
    "Bwd Pkts/s": {
        "description": (
            "Number of backward-direction packets divided by flow duration in seconds."
        ),
        "implication": (
            "Zero backward packet rate is consistent with unanswered DoS flooding traffic."
        ),
        "note": "DoS median 0 pkts/s vs Benign 4.1 pkts/s and DDoS 2.6 pkts/s.",
    },
    "Pkt Len Min": {
        "description": (
            "Minimum size in bytes of any single packet in the flow."
        ),
        "implication": (
            "Zero minimum packet length can be consistent with header-only flooding traffic."
        ),
        "note": "All classes median 0; 95th percentile Benign is 47 vs 0 for DoS/DDoS.",
    },
    "Pkt Len Mean": {
        "description": (
            "Mean packet length in bytes across all packets in both directions (average packet size)."
        ),
        "implication": (
            "Near-zero mean packet length is consistent with payload-free DoS flooding."
        ),
        "note": (
            "Equivalent to paper's Pkt Size Avg (dropped during preprocessing due to 0.95 correlation). "
            "DoS median near 0 vs Benign 68 and DDoS 123."
        ),
    },
    "Pkt Len Var": {
        "description": (
            "Variance of packet lengths in bytes across all packets in both directions."
        ),
        "implication": (
            "Near-zero variance with low packet lengths is consistent with uniform flooding packets."
        ),
        "note": "DoS median near 0 vs Benign ~4121 and DDoS ~110146.",
    },
    "FIN Flag Cnt": {
        "description": (
            "Total count of packets with the TCP FIN flag set in both directions."
        ),
        "implication": (
            "Marks connection termination attempts; interpret alongside flow duration and active state."
        ),
        "note": "Near-zero across all classes in train data.",
    },
    "RST Flag Cnt": {
        "description": (
            "Total count of packets with the TCP RST flag set in both directions."
        ),
        "implication": (
            "Elevated RST count can indicate aborted or refused connections during active sessions."
        ),
        "note": "DDoS median 1 vs Benign and DoS median 0.",
    },
    "PSH Flag Cnt": {
        "description": (
            "Total count of packets with the TCP PSH flag set in both directions."
        ),
        "implication": (
            "Few PSH flags despite many packets can indicate payload-free flooding traffic."
        ),
        "note": "Median 0 for Benign/DoS and 1 for DDoS; 95th percentile is 1 for all classes.",
    },
    "ACK Flag Cnt": {
        "description": (
            "Total count of packets with the TCP ACK flag set in both directions."
        ),
        "implication": (
            "Elevated ACK count is consistent with established connections under high-rate interaction."
        ),
        "note": "DoS median 1 vs Benign and DDoS median 0.",
    },
    "URG Flag Cnt": {
        "description": (
            "Total count of packets with the TCP URG flag set in both directions."
        ),
        "implication": (
            "Counts urgent pointer usage; non-zero values can indicate anomalous packet craft."
        ),
        "note": "Near-zero across all classes; DoS 95th percentile is 1.",
    },
    "Down/Up Ratio": {
        "description": (
            "Ratio of download packets to upload packets in the flow."
        ),
        "implication": (
            "Zero ratio is consistent with unanswered DoS flows; near-one ratio with DDoS."
        ),
        "note": "DDoS median 1.0 (symmetric exchange) vs DoS and Benign median 0.0.",
    },
    "Init Fwd Win Byts": {
        "description": (
            "Initial TCP window bytes in forward direction; typically -1 when unrecorded."
        ),
        "implication": (
            "Anomalous or absent forward window values can flag scanning or attack tools."
        ),
        "note": "DoS median 225 vs Benign/DDoS 8192; ~30% of Benign has unrecorded (-1).",
    },
    "Init Bwd Win Byts": {
        "description": (
            "Initial TCP window bytes in backward direction; typically -1 when unrecorded."
        ),
        "implication": (
            "Typically -1 when unrecorded; read together with transport protocol and direction."
        ),
        "note": "-1 means no window value recorded; appears in ~69% DoS and ~42% Benign/DDoS.",
    },
    "Fwd Seg Size Min": {
        "description": (
            "Minimum forward segment size in bytes, reflecting smallest payload segment observed."
        ),
        "implication": (
            "Minimum forward segment size; interpret together with payload length and rate."
        ),
        "note": "Medians 20-32 bytes across classes; context-dependent baseline.",
    },
    "Active Mean": {
        "description": (
            "Mean duration in microseconds of active periods before flow becomes idle."
        ),
        "implication": (
            "Mean active burst duration; interpret together with flow duration and rate."
        ),
        "note": "Median identical across classes; differentiates only in multi-burst benign flows.",
    },
    "Active Std": {
        "description": (
            "Standard deviation of active period durations in microseconds."
        ),
        "implication": (
            "Active burst timing variability; reflects consistency of activity cycles."
        ),
        "note": "Median identical across classes at sentinel floor.",
    },
    "Active Min": {
        "description": (
            "Minimum duration in microseconds of any single active period."
        ),
        "implication": (
            "Shortest active burst duration; interpret alongside overall flow duration."
        ),
        "note": "Median identical across classes at sentinel floor.",
    },
    "Idle Mean": {
        "description": (
            "Mean duration in microseconds of idle periods between packet bursts."
        ),
        "implication": (
            "Mean idle interval duration; reflects pause behavior between packet bursts."
        ),
        "note": "Median identical across classes; broad tail across all three classes.",
    },
    "Idle Std": {
        "description": (
            "Standard deviation of idle period durations in microseconds."
        ),
        "implication": (
            "Idle interval timing variability; reflects regularity of pauses between bursts."
        ),
        "note": "Median identical across classes; low standalone discrimination.",
    },
}

_EXPECTED_COUNT: int = 41


def describe(name: str) -> dict[str, Any]:
    """Return the full FEATURE_INFO entry (including optional note) for a feature name.

    Raises KeyError with a clear message if the name is not in the registry.
    """
    if name not in FEATURE_INFO:
        raise KeyError(
            f"Feature '{name}' is not in FEATURE_INFO. "
            f"Known features ({len(FEATURE_INFO)}): {sorted(FEATURE_INFO.keys())}"
        )
    return FEATURE_INFO[name]


def describe_for_prompt(name: str) -> dict[str, str]:
    """Return only description and implication for a feature for use in LLM prompts.

    Excludes developer-only notes and statistical metadata.
    """
    entry = describe(name)
    return {
        "description": entry["description"],
        "implication": entry["implication"],
    }


def check_coverage(
    feature_names: list[str],
) -> tuple[list[str], list[str]]:
    """Compare a list of feature names against FEATURE_INFO.

    Returns:
        (missing, extra): features present in feature_names but not in
        FEATURE_INFO, and features in FEATURE_INFO but not in feature_names.
    """
    registry_keys = set(FEATURE_INFO.keys())
    names_set = set(feature_names)

    missing = sorted(names_set - registry_keys)
    extra = sorted(registry_keys - names_set)
    return missing, extra
