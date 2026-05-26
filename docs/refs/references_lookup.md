# STRIDE paper references — source lookup

Generated 2026-05-26. Use alongside `references.bib`.

| # | Title (short) | DOI / URL | Status |
|---|---------------|-----------|--------|
| 1 | AMD DGF (HPG 2024) | https://gpuopen.com/download/publications/DGF.pdf | direct PDF |
| 2 | Google Draco | https://github.com/google/draco | repo |
| 3 | meshoptimizer | https://github.com/zeux/meshoptimizer | repo |
| 4 | Edgebreaker (Rossignac 1999) | 10.1109/2945.764870 | IEEE — paywalled |
| 5 | Touma–Gotsman (GI 1998) | https://graphicsinterface.org/wp-content/uploads/gi1998-4.pdf | open PDF |
| 6 | Isenburg–Alliez parallelogram (Vis 2002) | 10.1109/VISUAL.2002.1183768 | IEEE — paywalled |
| 7 | Cohen-Or multi-way (TR Tel Aviv 2002) | https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=04c844dafaf5e01f6da8adf37500573ead30d590 | CiteSeerX mirror (UNC TR-02-050 also exists) |
| 8 | TFAN (Mamou 2009) | 10.1002/cav.319 | Wiley — paywalled |
| 9 | AMD DGF blog | https://gpuopen.com/learn/introducing-amd-dgf-supercompression/ | open |
| 10 | Corto | https://github.com/cnr-isti-vclab/corto | repo |
| 11 | Stewart, tunneling strips (GI 2001) | 10.20380/GI2001.11 — https://graphicsinterface.org/wp-content/uploads/gi2001-11.pdf | fixed (was -12) |
| 12 | Devillers–Gandoin (Vis 2000) | 10.1109/VISUAL.2000.885711 | fixed (paper had .885693) |
| 13 | Krivograd (CAI 2008) | https://www.cai.sk/ojs/index.php/cai/article/view/318 | open journal |
| 14 | Peng/Kim/Kuo survey (JVCIR 2005) | 10.1016/j.jvcir.2005.03.001 | Elsevier — paywalled |
| 15 | Maglo survey (CSUR 2015) | 10.1145/2693443 | ACM — DL link |
| 16 | Taubin–Rossignac topo surgery (TOG 1998) | 10.1145/274363.274365 | ACM DL |
| 17 | Alliez–Desbrun valence (EG 2001) | 10.1111/1467-8659.00541 | Wiley |
| 18 | Khodakovsky near-optimal conn (GMOD 2002) | 10.1006/gmod.2002.0575 | fixed |
| 19 | Gumhold–Strasser RT conn (SIG 1998) | 10.1145/280814.280836 | ACM DL |
| 20 | Deering geometry compression (SIG 1995) | 10.1145/218380.218391 | ACM DL |
| 21 | Sorkine high-pass (SGP 2003) | 10.2312/SGP/SGP03/042-051 | EG DL |
| 22 | Witten/Neal/Cleary arith coding (CACM 1987) | 10.1145/214762.214771 | ACM DL |
| 23 | Martin range coding (1979) | https://www.drdobbs.com/cpp/range-coder/184409765 | **no canonical PDF** |
| 24 | Duda ANS (arXiv 1311.2540) | https://arxiv.org/abs/1311.2540 | arXiv |
| 25 | Rice–Plaunt (IEEE 1971) | 10.1109/TCOM.1971.1090789 | IEEE — paywalled |
| 26 | Teuhola clustered bit-vectors (IPL 1978) | 10.1016/0020-0190(78)90024-8 | Elsevier |
| 27 | Hoppe vertex caching (SIG 1999) | 10.1145/311535.311565 | ACM DL |
| 28 | Forsyth linear-speed cache opt | https://tomforsyth1000.github.io/papers/fast_vert_cache_opt.html | open |
| 29 | OpenCTM | https://openctm.sourceforge.net | open |
| 30 | KHR_draco_mesh_compression | https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_draco_mesh_compression | open |
| 31 | Geometry images (SIG 2002) | 10.1145/566654.566589 | ACM DL |
| 32 | Progressive meshes (SIG 1996) | 10.1145/237170.237216 | ACM DL |
| 33 | Progressive geometry compression (SIG 2000) | 10.1145/344779.344932 | ACM DL |
| 34 | Nanite deep dive (SIG 2021) | https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf | open PDF |

## Import to Zotero

1. Zotero -> File -> Import...
2. Pick `docs/refs/references.bib`
3. Tick "Place imports into new collection" -> name it `STRIDE-VC-v6`
4. After import, right-click collection -> **Find Available PDFs** (resolves DOIs, pulls open PDFs).
5. Paywalled IEEE/ACM/Wiley entries need institutional access; Zotero will mark them as "no PDF found".

## Verification round 2026-05-26 — fixed

- [7] Cohen-Or TR — original tau.ac.il URL was 404; swapped to CiteSeerX mirror. UNC TR-02-050 also hosts a copy.
- [11] Stewart GI 2001 — wrong file (`gi2001-12.pdf` is "Truly Selective Refinement"); fixed to `gi2001-11.pdf` + added GI DOI `10.20380/GI2001.11`.
- [12] Devillers-Gandoin — DOI was wrong; correct is `.885711` (verified via HAL + IEEE Xplore lookup).
- [18] Khodakovsky GMOD 2002 — DOI was wrong; correct is `10.1006/gmod.2002.0575` (Elsevier/Inria HAL match).

## Still requires user judgment

- [7] author spelling — paper has "R. Cohen, R. Ironi"; CiteSeerX entry shows "Ram Cohen, Roman Ironi" (different transliterations). bib now uses likely full names.
- [23] Martin 1979 range coding — no canonical online source; Dr. Dobb's reprint is closest.

## Cross-checked and confirmed correct

[1][2][3] URLs ; [4] Edgebreaker 10.1109/2945.764870 ; [5] Touma-Gotsman gi1998-4.pdf ; [6] Isenburg-Alliez 10.1109/VISUAL.2002.1183768 ; [8] TFAN 10.1002/cav.319 ; [9][10] URLs ; [16] Taubin-Rossignac 10.1145/274363.274365 ; [21] Sorkine 10.2312/SGP/SGP03/042-051 ; [31] Geometry images TOG variant `10.1145/566654.566589` resolves (SIGGRAPH variant `10.1145/566570.566589` also valid).

## Paper v6 reference list

No DOIs are embedded in the paper's reference list itself (name+venue+year only), so the corrections above are bib-only — paper text needs no edits.
