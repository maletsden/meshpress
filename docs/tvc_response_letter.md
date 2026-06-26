# Response to Editor — The Visual Computer

**Manuscript:** STRIDE: STRIp-walked Triangulated Residual Integer Decoder for Per-Meshlet GPU Mesh Compression

**Submission ID:** fdf7ac85-185d-40a5-aee3-b333186c198f

**Authors:** Denys Maletskyi, Yaroslav Vyklyuk, Fengping Li

---

We thank the editor for the constructive pre-review feedback on source-code and
data transparency. We fully support the journal's emphasis on reproducibility
and have addressed every point. A summary of the changes precedes the
point-by-point responses; all section/table references are to the revised
manuscript.

## Summary of changes

1. Added **§8 "Code and Data Availability"** to the manuscript, with the public
   repository URL, license, archival DOI, and a description of the released
   artifacts.
2. Added an **availability statement to the abstract** and an artifacts note in
   the experimental setup.
3. Released the **full source code** of STRIDE (encoder, CUDA decoder, bitstream
   specification, benchmark harness, and table/figure-regeneration scripts) in
   the public **MeshPress** GitHub repository under the MIT license.
4. **Archived the repository on Zenodo with a permanent DOI**, cited as the
   archival reference in §8.
5. Published the **test-mesh corpus with a dataset README** documenting each
   mesh's source, size, and license/provenance.
6. Documented **dependencies and requirements** (`requirements.txt`, GPU/CuPy
   notes) and **key-algorithm descriptions** in the repository README and
   `docs/bitstream_spec.md`.
7. Added a **citation block to the GitHub README** naming the article title and
   *The Visual Computer* as the publishing journal.
8. Strengthened journal-scope fit by **citing recent and prior Visual Computer
   connectivity-coding papers** [35–37] in §2.1.

## Point-by-point responses

### Comment 1 — Publish full source code (with DOI) and datasets, with descriptions in the manuscript and abstract

> "We strongly recommend that you publish the full source code of the proposed
> algorithm (with DOI link) uploaded to github, along with the associated data
> sets (including readme files), and provide detailed descriptions (with open
> source link) in the revised manuscript and abstract section ... This should
> cover dependencies and requirements, descriptions and implementations of key
> algorithms, and other relevant details."

**Response.** Done. The complete implementation is public at
**https://github.com/maletsden/meshpress** (MIT license) and is permanently
archived on **Zenodo, DOI: [INSERT ZENODO DOI]**.

The release contains:

- the STRIDE encoder and CUDA GPU decoder;
- the formal bitstream specification (`docs/bitstream_spec.md`);
- the eight-mesh test corpus with a dataset README (`assets/README.md`)
  documenting each mesh's source, triangle count, and license/provenance;
- the raw benchmark CSV files and the scripts that regenerate every table and
  figure in the paper;
- `requirements.txt` plus documented dependencies (Python packages; CuPy + an
  NVIDIA GPU for the CUDA decode-speed benchmarks; scikit-learn for the adaptive
  patch encoders).

We added **§8 "Code and Data Availability"** giving the repository URL, license,
archival DOI, and an enumeration of the released artifacts, and we added an
**availability statement to the abstract** ("All code, the bitstream
specification, raw benchmark CSV files, reproducible scripts, and the test-mesh
corpus are publicly available; see §8."). Key-algorithm descriptions appear both
in the manuscript (§3 method, §4 GPU decoder) and in the repository README and
bitstream specification.

### Comment 2 — Add a citation format to the GitHub description naming the article title and journal

> "We want to add a citation format to your github description document stating
> the title of the article and the journal name of The Visual Computer to
> improve visibility."

**Response.** Done. The repository README now contains a **"How to cite"**
section with a plain-text citation and a BibTeX entry that name the article
title and *The Visual Computer* (Springer) as the publishing journal. Volume,
issue, and page numbers will be completed at camera-ready once assigned.

### Comment 3 — Compare with recent work published in The Visual Computer

> "Please pay attention to comparing and analyzing your work in relation to
> relevant studies recently published in our journal, so as to enhance the
> relevance of your manuscript to the journal's scope."

**Response.** Done. We surveyed the journal's mesh-connectivity-coding corpus
and added three Visual Computer references to the Related Work section (§2.1),
positioning STRIDE's per-meshlet connectivity coding against them:

- Szymczak, *Optimized Edgebreaker encoding for large and regular triangle
  meshes*, The Visual Computer (2003) [35];
- Coors and Rossignac, *Delphi: geometry-based connectivity prediction in
  triangle mesh compression*, The Visual Computer (2004) [36];
- Balreira and da Silveira, *A lossless triangular-matrix representation of mesh
  connectivity*, The Visual Computer (2024) [37].

These situate STRIDE's strip-walked, per-meshlet connectivity within the
journal's Edgebreaker/Delphi lineage and a current (2024) connectivity-coding
result, clarifying the scope fit.

---

We believe these revisions fully address the transparency and scope requests and
make the work straightforward to reproduce. We are happy to provide any further
detail the editor requires.