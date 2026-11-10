# Model and data hash manifest

SHA-256 hashes of the files the frozen Phase 1 results were produced with. All README, audit and report numbers belong to these exact files.

Recorded 10 Nov 2026. On that day dnn_Benign.pt was overwritten by an accidental retrain; the retrained file is byte-identical to the one evaluated (hash recorded in artifacts\eval\hashes_at_eval.txt before the evaluation run). The original Benign training history and curves were overwritten, so the files in artifacts\ are the regenerated ones.

| File | SHA-256 |
|---|---|
| `models\dnn_Benign.pt` | `94921844BD310FA4E07CDDB5BC1095C2C2ABA0AEE0D5F72659818707A2AFF35C` |
| `models\dnn_DoS.pt` | `DA4525520BB16F6259A7F55EFF5E67D7534D3551680B7D65FD2A4A8BDAFB1995` |
| `models\dnn_DDoS.pt` | `F3CBE94C1B1F649073FC28B2485787EE82A306A199F422E7560AC443A7B43970` |
| `models\scaler.joblib` | `41E77134166B966397175D40E1AC52B6886AFB086AEC62D5F10B1688E3C6479E` |
| `models\feature_names.json` | `951076CC748824C51FC3D786367750B8173B4F75F94076C3084977A953FCB1BE` |
| `models\temperatures.json` | `89FD84BAA2A8802A62A02A224A6BA6A0B4A52E687AEF768340CE4EC0D7145A1E` |
| `data\kb\chunks.jsonl` | `876E30B4F9E3C6943CCC5C584B631F2316D20E10D27E012151BA8ED44A006D15` |
| `data\processed\test.parquet` | `06E58BF92021D6E500AAB4925B7899A08ADCE77D86B2CE2C795F561AD0FC5111` |

## Verify (PowerShell, from the project root)

```powershell
Get-Content MODEL_MANIFEST.md | ForEach-Object { if ($_ -match '^\| `([^`]+)` \| `([0-9A-F]{64})` \|') { $h = (Get-FileHash $Matches[1] -Algorithm SHA256).Hash; "{0,-5} {1}" -f $(if ($h -eq $Matches[2]) {"OK"} else {"DIFF"}), $Matches[1] } }
```

Every line should start with OK. Any DIFF means the file changed after the results were produced.