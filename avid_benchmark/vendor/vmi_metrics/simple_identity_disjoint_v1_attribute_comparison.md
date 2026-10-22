# Simple split attribute-distribution comparison

- Source attribute bank: `avid_benchmark/vendor/vmi_metrics/simple_attribute_bank_qwen_v3.json`
- Original validation manifest: `avid_benchmark/vendor/vmi_metrics/simple_sub01_current_val80_manifest.csv`
- New split seed: `AVID-simple-identity-disjoint-v1`
- New split selection is independent of reconstruction performance.

| Attribute | Value | Original train | Original val | Identity-disjoint train | Identity-disjoint val |
|---|---|---:|---:|---:|---:|
| family | letter | 50 (15.62%) | 16 (20.00%) | 54 (16.88%) | 12 (15.00%) |
| family | single_digit | 26 (8.12%) | 4 (5.00%) | 24 (7.50%) | 6 (7.50%) |
| family | digit_sequence | 20 (6.25%) | 10 (12.50%) | 24 (7.50%) | 6 (7.50%) |
| family | shape | 189 (59.06%) | 36 (45.00%) | 180 (56.25%) | 45 (56.25%) |
| family | object | 35 (10.94%) | 14 (17.50%) | 38 (11.88%) | 11 (13.75%) |
| color | red | 100 (31.25%) | 20 (25.00%) | 96 (30.00%) | 24 (30.00%) |
| color | yellow | 95 (29.69%) | 25 (31.25%) | 96 (30.00%) | 24 (30.00%) |
| color | blue | 98 (30.62%) | 24 (30.00%) | 98 (30.62%) | 24 (30.00%) |
| color | other or multicolored | 27 (8.44%) | 11 (13.75%) | 30 (9.38%) | 8 (10.00%) |
| position | top | 40 (12.50%) | 5 (6.25%) | 36 (11.25%) | 9 (11.25%) |
| position | bottom | 34 (10.62%) | 11 (13.75%) | 36 (11.25%) | 9 (11.25%) |
| position | left | 40 (12.50%) | 5 (6.25%) | 36 (11.25%) | 9 (11.25%) |
| position | right | 38 (11.88%) | 7 (8.75%) | 36 (11.25%) | 9 (11.25%) |
| position | center | 168 (52.50%) | 52 (65.00%) | 176 (55.00%) | 44 (55.00%) |

## Visual-identity separation

| Split | Unique train identities | Unique val identities | Overlapping identities | Val stimuli whose identity occurs in train |
|---|---:|---:|---:|---:|
| Original | 81 | 49 | 35 | 66/80 |
| Identity-disjoint v1 | 75 | 20 | 0 | 0/80 |

## Held-out validation identities

- **letter:** `S`, `C`
- **single_digit:** `8`, `5`
- **digit_sequence:** `829`, `286`
- **shape:** `semicircle (straight edge at top)`, `heart`, `horizontal rectangle`
- **object:** `goldfish`, `toothbrush`, `jeans`, `airplane`, `keyboard`, `golf ball`, `knife`, `bed`, `chicken`, `sink`, `hatchet`
