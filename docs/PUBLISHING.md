# Public release

Canonical repository: https://github.com/xiaokj37/cuhkx-falcons

Original Team Falcons contributions use Apache-2.0. `THIRD_PARTY_NOTICES.md` identifies upstream components and their retained licenses. The public snapshot is separate from the private development history and the committee's signed verification package.

Keep the README, requirements, environment, training guide, source map and notices together. GitHub Actions performs CPU-only source and portability checks; it does not download CUHK-X data or launch training.

The selected checkpoint is a separate artifact, identified by size and SHA-256 in `models/README.md`. Add a public model URL only after the exact asset is published under its applicable weight/data terms. Do not distribute the committee's full verification archive as a model download: it contains personal materials.

Dataset archives, decoded caches, labels, original answer CSVs, credentials and private documents are excluded. The technical report and Zenodo metadata are prepared separately. A report DOI or Zenodo record URL should be added only after it actually exists.
